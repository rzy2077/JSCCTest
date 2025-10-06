import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
from PIL import Image

# 确保在无头模式下运行
matplotlib.use('Agg')

# --- 关键配置 ---
IMAGE_SIZE = (64, 64)  # 目标处理尺寸
BATCH_SIZE = 32
CHANNEL_DIM = 8
SNR_DB = 10
EPOCHS = 10

# --- 数据集选择开关 ---
# 选择: 'cifar_upsampled' (32x32 -> 64x64) 或 'imagenet_64' (模拟 64x64)
# DATASET_CHOICE = 'cifar_upsampled'


DATASET_CHOICE = 'imagenet_64'


# ----------------- 编码器和解码器类 (64x64 结构) -----------------
# 保持不变，结构已适配 64x64 -> 8x8
class Encode(tf.keras.Model):
    def __init__(self, c):
        super().__init__()
        # 1. 64x64 -> 32x32
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            strides=2)
        # 2. 32x32 -> 16x16
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            strides=2)
        # 3. 16x16 -> 8x8
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            strides=2)
        # 4. 8x8 -> 8x8
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            strides=1)
        # 5. 8x8 -> C 维度的信道特征
        self.conv5 = tf.keras.layers.Conv2D(filters=c, kernel_size=[3, 3], padding='same', activation=tf.nn.relu,
                                            strides=1)

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        output = self.conv5(x)
        return output


class Decode(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 1. 8x8 -> 8x8
        self.dconv1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=1)
        # 2. 8x8 -> 16x16
        self.dconv2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)
        # 3. 16x16 -> 32x32
        self.dconv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)
        # 4. 32x32 -> 64x64
        self.dconv4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)
        # 5. 64x64 -> 64x64
        self.dconv5 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.sigmoid, strides=1)

    def call(self, input):
        x = self.dconv1(input)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        output = self.dconv5(x)
        return output


# ----------------- DJSCC 模型封装 (未修改) -----------------
class DJSCC_Model(tf.keras.Model):
    def __init__(self, c, snr_db):
        super().__init__()
        self.encoder = Encode(c)
        self.decoder = Decode()

        self.snr_db = snr_db
        snr_linear = 10 ** (self.snr_db / 10.0)
        self.noise_stddev = tf.sqrt(1 / (2 * snr_linear))

    def call(self, inputs, training=False):
        encoded_output = self.encoder(inputs)

        if training:
            noise = tf.random.normal(shape=tf.shape(encoded_output), mean=0.0, stddev=self.noise_stddev)
            noisy_output = encoded_output + noise
        else:
            noisy_output = encoded_output

        decoded_output = self.decoder(noisy_output)

        return decoded_output


# --- 普通 JSSC 对比代码 (不变) ---
def classical_jssc_process(image_array, snr_db):
    ber = 0.5 * 10 ** (-snr_db / 10.0)
    image = Image.fromarray((image_array.squeeze() * 255).astype(np.uint8))
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=70)
    compressed_stream = buf.getvalue()

    bit_stream = np.unpackbits(np.frombuffer(compressed_stream, dtype=np.uint8))
    num_bits = len(bit_stream)
    num_errors = int(num_bits * ber)
    error_indices = np.random.choice(num_bits, num_errors, replace=False)
    bit_stream[error_indices] = 1 - bit_stream[error_indices]

    try:
        corrupted_bytes = np.packbits(bit_stream).tobytes()
        corrupted_buf = io.BytesIO(corrupted_bytes)
        reconstructed_image = Image.open(corrupted_buf)
        reconstructed_array = np.array(reconstructed_image).astype('float32') / 255.0
        if len(reconstructed_array.shape) == 2:
            reconstructed_array = np.stack([reconstructed_array] * 3, axis=-1)
    except:
        print("Warning: Decompression failed due to severe channel errors.")
        reconstructed_array = np.zeros(IMAGE_SIZE + (3,), dtype=np.float32)

    return reconstructed_array


# ----------------- 数据加载与预处理函数 -----------------

def preprocess_cifar_upsample(x_data):
    """加载 CIFAR-10 (32x32)，归一化并上采样到 64x64。"""
    x_data = tf.cast(x_data, tf.float32) / 255.0
    x_data = tf.image.resize(x_data, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    return x_data


def preprocess_imagenet_64(x_data):
    """加载 64x64 图像，归一化。"""
    # 假设输入已经是 64x64 (或者已经由 load_imagenet_64 调整过)
    x_data = tf.cast(x_data, tf.float32) / 255.0
    return x_data


def load_imagenet_64():
    """
    **占位符函数**：用于模拟加载一个小型、预处理好的 64x64 ImageNet 样本数据集。
    在真实场景中，您需要使用 tfds.load('imagenet_resized/64x64') 或您自己的数据集。

    这里为了代码的可运行性，我们**生成随机数据**来模拟 64x64 ImageNet。
    若要使用真实数据，请替换此逻辑。
    """
    print("WARNING: 正在使用随机数据模拟 ImageNet 64x64 数据集，请替换为真实数据！")
    # 模拟 10,000 张训练图像和 1,000 张测试图像
    n_train = 10000
    n_test = 1000

    x_train = np.random.randint(0, 256, size=(n_train, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    x_test = np.random.randint(0, 256, size=(n_test, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)

    # 返回 NumPy 数组，后续将被 tf.data.Dataset 转换为张量
    return (x_train, None), (x_test, None)


def load_dataset(dataset_choice):
    """根据选择加载并预处理数据集。"""
    if dataset_choice == 'cifar_upsampled':
        print(f"--- 1. 加载 CIFAR-10 并上采样到 {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} ---")
        (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
        preprocess_fn = preprocess_cifar_upsample
    elif dataset_choice == 'imagenet_64':
        print(f"--- 1. 模拟加载 64x64 ImageNet 数据集 ---")
        (x_train, _), (x_test, _) = load_imagenet_64()  # 使用模拟数据
        preprocess_fn = preprocess_imagenet_64
    else:
        raise ValueError(f"未知的数据集选择: {dataset_choice}")

    # 将 NumPy 数组转换为 TensorFlow Dataset
    train_ds = tf.data.Dataset.from_tensor_slices(x_train) \
        .map(preprocess_fn) \
        .map(map_to_autoencoder_format) \
        .shuffle(1024) \
        .batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(x_test) \
        .map(preprocess_fn) \
        .map(map_to_autoencoder_format) \
        .batch(BATCH_SIZE)

    return train_ds, test_ds


# --- 映射函数 ---
def map_to_autoencoder_format(x):
    return x, x


if __name__ == "__main__":
    # 1. 加载和预处理数据 (根据 DATASET_CHOICE 决定)
    try:
        train_ds, test_ds = load_dataset(DATASET_CHOICE)
    except Exception as e:
        print(f"数据加载失败: {e}")
        # 如果加载失败，终止程序
        exit()

    # 2. 实例化 DJSCC 模型
    djscc_model = DJSCC_Model(c=CHANNEL_DIM, snr_db=SNR_DB)

    # 3. 编译模型
    djscc_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss=tf.keras.losses.MeanSquaredError())

    # 4. 训练模型
    dataset_name = DATASET_CHOICE.replace('_', ' ').title()
    print(f"\n--- 2. 开始训练 DJSCC 模型 ({dataset_name} 输入), 信噪比 (SNR) 为 {SNR_DB} dB ---")
    djscc_model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=test_ds)

    # 5. 可视化结果
    print("\n--- 3. 开始可视化重建结果 ---")

    # 从测试数据集中获取一批图像
    test_images_x, _ = next(iter(test_ds))
    test_images_tensor = test_images_x
    n_images = min(test_images_tensor.shape[0], 8)  # 展示 8 张图像

    # 使用 DJSCC 模型进行预测
    reconstructed_images = djscc_model.predict(test_images_tensor[:n_images])

    # 使用传统 JSSC 方法进行重建
    print("开始使用传统 JSSC 方法进行重建...")
    classical_reconstructions = []
    for i in range(n_images):
        img_np = test_images_tensor[i].numpy()
        reconstructed = classical_jssc_process(img_np, snr_db=SNR_DB)
        classical_reconstructions.append(reconstructed)
    classical_reconstructions = np.stack(classical_reconstructions, axis=0)
    test_images_display = test_images_tensor.numpy()[:n_images]

    # 绘制对比图表
    plt.figure(figsize=(15, 8))

    for i in range(n_images):
        # 原始图像 (第1行)
        ax = plt.subplot(3, n_images, i + 1)
        plt.imshow(test_images_display[i])
        if i == 0: ax.set_title(f"Original ({dataset_name})")
        plt.axis("off")

        # DJSCC 重建图像 (第2行)
        ax = plt.subplot(3, n_images, n_images + i + 1)
        plt.imshow(reconstructed_images[i])
        if i == 0: ax.set_title("DJSCC")
        plt.axis("off")

        # 传统 JSSC 重建图像 (第3行)
        ax = plt.subplot(3, n_images, 2 * n_images + i + 1)
        if classical_reconstructions[i].shape == test_images_display[i].shape:
            plt.imshow(classical_reconstructions[i])
        else:
            # 兼容性处理，裁剪到 64x64
            plt.imshow(classical_reconstructions[i][:IMAGE_SIZE[0], :IMAGE_SIZE[1], :])
        if i == 0: ax.set_title("Classical JSSC")
        plt.axis("off")

    plt.suptitle(f"DJSCC ({dataset_name} 64x64) vs. Classical JSSC at SNR = {SNR_DB} dB")
    output_filename = f"djscc_vs_classical_snr_{SNR_DB}db_{DATASET_CHOICE}_64x64.png"
    plt.savefig(output_filename)
    print(f"\n对比结果已保存为文件: {output_filename}")