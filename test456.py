import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
from PIL import Image

# 确保在无头模式下运行
matplotlib.use('Agg')

# --- 关键配置：从 CIFAR-10 读取，使用其原始 32x32 尺寸 ---
IMAGE_SIZE = (32, 32)  # 目标处理尺寸改为 32x32
BATCH_SIZE = 32
CHANNEL_DIM = 8 # 保持信道维度，但信道特征图尺寸更大
SNR_DB = 10
EPOCHS = 10


# ----------------- 编码器和解码器类 (调整为 32x32 结构) -----------------
class Encode(tf.keras.Model):
    def __init__(self, c):
        super().__init__()
        # 1. 32x32 -> 16x16 (降采样)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=2)
        # 2. 16x16 -> 8x8 (降采样)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=2)
        # 3. 8x8 -> 8x8 (保持尺寸，增加特征)
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=1)
        # 4. 8x8 -> C 维度的信道特征 (最终信道符号维度 8x8xC)
        self.conv4 = tf.keras.layers.Conv2D(filters=c, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=1)

    def call(self, input):
        x = self.conv1(input)  # 32x32 -> 16x16
        x = self.conv2(x)      # 16x16 -> 8x8
        x = self.conv3(x)      # 8x8 -> 8x8
        output = self.conv4(x) # 8x8 -> 8x8 (C channels)
        return output


class Decode(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 逆向操作，从 8x8 的信道特征重建 32x32 图像
        # 1. 8x8 -> 8x8
        self.dconv1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=1)
        # 2. 8x8 -> 16x16 (升采样)
        self.dconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)
        # 3. 16x16 -> 32x32 (升采样)
        self.dconv3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)
        # 4. 32x32 -> 32x32 (最终输出)
        self.dconv4 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.sigmoid, strides=1)

    def call(self, input):
        x = self.dconv1(input)  # 8x8 -> 8x8
        x = self.dconv2(x)      # 8x8 -> 16x16
        x = self.dconv3(x)      # 16x16 -> 32x32
        output = self.dconv4(x) # 32x32 -> 32x32 (3 channels)
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


# --- 普通 JSSC 对比代码 (未修改) ---
def classical_jssc_process(image_array, snr_db):
    # 此函数对任何尺寸的图像都适用，因为 PIL 库会处理尺寸差异
    ber = 0.5 * 10 ** (-snr_db / 10.0)

    # 1. 源编码 (JPEG压缩)
    # image_array 维度应为 (32, 32, 3)
    image = Image.fromarray((image_array.squeeze() * 255).astype(np.uint8))
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=70)
    compressed_stream = buf.getvalue()

    # 2. 信道模拟 (引入随机误码)
    bit_stream = np.unpackbits(np.frombuffer(compressed_stream, dtype=np.uint8))
    num_bits = len(bit_stream)
    num_errors = int(num_bits * ber)
    error_indices = np.random.choice(num_bits, num_errors, replace=False)
    bit_stream[error_indices] = 1 - bit_stream[error_indices]

    # 3. 解码
    try:
        corrupted_bytes = np.packbits(bit_stream).tobytes()
        corrupted_buf = io.BytesIO(corrupted_bytes)
        reconstructed_image = Image.open(corrupted_buf)
        reconstructed_array = np.array(reconstructed_image).astype('float32') / 255.0
        if len(reconstructed_array.shape) == 2:
            reconstructed_array = np.stack([reconstructed_array] * 3, axis=-1)
    except:
        print("Warning: Decompression failed due to severe channel errors.")
        reconstructed_array = np.zeros_like(image_array.squeeze())

    return reconstructed_array


# ----------------- 训练与可视化 (主要修改部分) -----------------
def preprocess(x_data):
    """归一化图像。移除缩放步骤，因为 IMAGE_SIZE 已经是 32x32"""
    x_data = tf.cast(x_data, tf.float32) / 255.0
    # 原始 CIFAR-10 数据已经是 32x32，无需缩放
    return x_data

# --- 修复函数：将预处理后的图像映射为 (输入 X, 目标 Y) 对 ---
def map_to_autoencoder_format(x):
    return x, x


if __name__ == "__main__":
    # 1. 加载和预处理 CIFAR-10 数据
    print("加载和预处理 CIFAR-10 数据集 (32x32)...")
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    # 将 NumPy 数组转换为 TensorFlow Dataset
    train_ds = tf.data.Dataset.from_tensor_slices(x_train) \
        .map(preprocess) \
        .map(map_to_autoencoder_format) \
        .shuffle(1024) \
        .batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(x_test) \
        .map(preprocess) \
        .map(map_to_autoencoder_format) \
        .batch(BATCH_SIZE)

    # 2. 实例化 DJSCC 模型
    djscc_model = DJSCC_Model(c=CHANNEL_DIM, snr_db=SNR_DB)

    # 3. 编译模型
    djscc_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss=tf.keras.losses.MeanSquaredError())

    # 4. 训练模型
    print(f"开始训练 DJSCC 模型 (CIFAR-10 32x32 输入), 信噪比 (SNR) 为 {SNR_DB} dB...")
    djscc_model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=test_ds)

    # 5. 可视化结果
    print("开始可视化重建结果...")

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
    plt.figure(figsize=(15, 6))

    for i in range(n_images):
        # 原始图像 (第1行)
        ax = plt.subplot(3, n_images, i + 1)
        plt.imshow(test_images_display[i])
        if i == 0: ax.set_title("Original")
        plt.axis("off")

        # DJSCC 重建图像 (第2行)
        ax = plt.subplot(3, n_images, n_images + i + 1)
        plt.imshow(reconstructed_images[i])
        if i == 0: ax.set_title("DJSCC")
        plt.axis("off")

        # 传统 JSSC 重建图像 (第3行)
        ax = plt.subplot(3, n_images, 2 * n_images + i + 1)
        # 确保尺寸匹配 32x32
        if classical_reconstructions[i].shape == test_images_display[i].shape:
            plt.imshow(classical_reconstructions[i])
        else:
            # 兼容性处理，如果 JPEG 解压输出尺寸略有不同
            plt.imshow(classical_reconstructions[i][:32, :32, :])
        if i == 0: ax.set_title("Classical JSSC")
        plt.axis("off")

    plt.suptitle(f"DJSCC (CIFAR-10 32x32) vs. Classical JSSC at SNR = {SNR_DB} dB")
    output_filename = f"djscc_vs_classical_snr_{SNR_DB}db_cifar10_32x32.png"
    plt.savefig(output_filename)
    print(f"对比结果已保存为文件: {output_filename}")