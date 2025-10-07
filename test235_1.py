import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
from PIL import Image
import os
# 导入 tensorflow_datasets 用于加载 ImageNet/高分辨率数据集
import tensorflow_datasets as tfds

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
matplotlib.use('Agg')

# --- 关键配置 (升级) ---
IMAGE_SIZE = (224, 224)  # <--- 升级到 224x224
BATCH_SIZE = 8  # <--- 显著降低 Batch Size (GPU 内存限制)
CHANNEL_DIM = 8
# Fixed DJSCC 模型的训练目标 SNR (保持不变)
FIXED_SNR_DB = -3
# *******************************************************************
# ********* 修改点 1: 定义多组测试 SNR 列表 *********
# *******************************************************************
TEST_SNRS_DB = [10.0, 5.0, 0.0, -5.0, -10.0]  # 要对比的 5 个 SNR 值
ADAPTIVE_MIN_SNR = 0.0
ADAPTIVE_MAX_SNR = 20.0
EPOCHS = 5  # <--- 初始测试 Epoch (保持较低以快速运行)


# ----------------- 编码器和解码器类 (保持不变) -----------------
# 目标：224 -> 112 -> 56 -> 28 -> 14 (最终特征图尺寸)

class Encoder(tf.keras.Model):
    def __init__(self, c, is_adaptive=False):
        super().__init__()
        self.is_adaptive = is_adaptive

        # 图像处理层 (224x224 -> 14x14)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=2)  # 224->112
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=2)  # 112->56
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=2)  # 56->28
        self.conv4_down = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                                                 activation=tf.nn.relu, strides=2)  # 28->14
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=1)  # 14->14

        # --- Adaptive DJSCC 特有的 SNR 融合层 ---
        if self.is_adaptive:
            self.snr_dense = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)  # 匹配 conv5 后的通道数
            self.snr_conv = tf.keras.layers.Conv2D(filters=256, kernel_size=[1, 1], padding='same',
                                                   activation=tf.nn.relu)

        # 最终编码层 (将特征降到 C 维, 14x14x8)
        self.conv6_final = tf.keras.layers.Conv2D(filters=c, kernel_size=[3, 3], padding='same',
                                                  activation=tf.nn.relu, strides=1)

    def call(self, inputs):
        if self.is_adaptive:
            image, snr_value = inputs
        else:
            image = inputs
            snr_value = None

        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4_down(x)
        x = self.conv5(x)

        # Adaptive DJSCC: SNR 融合
        if self.is_adaptive and snr_value is not None:
            snr_emb = self.snr_dense(snr_value)
            # 广播到 14x14x256
            snr_emb_conv = self.snr_conv(tf.expand_dims(tf.expand_dims(snr_emb, 1), 1))
            x = x + snr_emb_conv

        output = self.conv6_final(x)
        return output


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 逆向操作 (14x14 -> 224x224)
        self.dconv1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=1)  # 14->14
        self.dconv2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)  # 14->28
        self.dconv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)  # 28->56
        self.dconv4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)  # 56->112
        self.dconv5 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.sigmoid, strides=2)  # 112->224

    def call(self, input):
        x = self.dconv1(input)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        output = self.dconv5(x)
        return output


class FixedDJSCC_Model(tf.keras.Model):
    def __init__(self, c, fixed_snr_db):
        super().__init__()
        self.fixed_snr_db = fixed_snr_db
        self.encoder = Encoder(c, is_adaptive=False)
        self.decoder = Decoder()

    def call(self, inputs, training=False):
        image = inputs[0] if isinstance(inputs, tuple) else inputs
        batch_size = tf.shape(image)[0]
        target_snr_db = self.fixed_snr_db
        snr_db = tf.fill(dims=(batch_size, 1), value=tf.cast(target_snr_db, tf.float32))

        encoded_output = self.encoder(image)

        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_stddev = tf.sqrt(1.0 / (2.0 * snr_linear))
        noise_stddev = tf.expand_dims(tf.expand_dims(noise_stddev, 1), 1)

        noise = tf.random.normal(shape=tf.shape(encoded_output), mean=0.0, stddev=noise_stddev)
        noisy_output = encoded_output + noise

        decoded_output = self.decoder(noisy_output)
        return decoded_output


class AdaptiveDJSCC_Model(tf.keras.Model):
    def __init__(self, c, min_snr_db, max_snr_db):
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.encoder = Encoder(c, is_adaptive=True)
        self.decoder = Decoder()

    def call(self, inputs, training=False):

        if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[0], tf.Tensor):
            image, snr_db_tensor_raw = inputs
        elif isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[0], tuple):
            image, snr_db_tensor_raw = inputs[0]
        else:
            image = inputs
            snr_db_tensor_raw = tf.constant(self.min_snr_db, dtype=tf.float32)

        batch_size = tf.shape(image)[0]

        if training:
            snr_db = tf.random.uniform(
                shape=(batch_size, 1),
                minval=self.min_snr_db,
                maxval=self.max_snr_db,
                dtype=tf.float32
            )
            encoder_input = (image, snr_db)
        else:
            snr_val_scalar = tf.squeeze(tf.gather(snr_db_tensor_raw, 0))
            snr_db = tf.fill(dims=(batch_size, 1), value=tf.cast(snr_val_scalar, tf.float32))
            encoder_input = (image, snr_db)

        encoded_output = self.encoder(encoder_input)

        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_stddev = tf.sqrt(1.0 / (2.0 * snr_linear))
        noise_stddev = tf.expand_dims(tf.expand_dims(noise_stddev, 1), 1)

        noise = tf.random.normal(shape=tf.shape(encoded_output), mean=0.0, stddev=noise_stddev)
        noisy_output = encoded_output + noise

        decoded_output = self.decoder(noisy_output)

        return decoded_output


# --- Classical JSSC (更新以兼容 224x224) ---
def classical_jssc_process(image_array, snr_db):
    ber = 0.5 * 10 ** (-snr_db / 10.0)
    ber = max(1e-6, min(ber, 0.5))

    # 兼容高分辨率: PIL Image.fromarray 可以自动处理 224x224
    image = Image.fromarray((image_array.squeeze() * 255).astype(np.uint8))
    buf = io.BytesIO()
    # JPEG 压缩质量 70
    image.save(buf, format='JPEG', quality=70)
    compressed_stream = buf.getvalue()

    bit_stream = np.unpackbits(np.frombuffer(compressed_stream, dtype=np.uint8))
    num_bits = len(bit_stream)

    # 模拟误码
    error_mask = np.random.rand(num_bits) < ber
    bit_stream[error_mask] = 1 - bit_stream[error_mask]

    try:
        corrupted_bytes = np.packbits(bit_stream).tobytes()
        corrupted_buf = io.BytesIO(corrupted_bytes)
        reconstructed_image = Image.open(corrupted_buf)
        reconstructed_array = np.array(reconstructed_image).astype('float32') / 255.0

        if len(reconstructed_array.shape) == 2:
            reconstructed_array = np.stack([reconstructed_array] * 3, axis=-1)

        # 确保尺寸正确 (对于 ImageNet, 压缩后的尺寸可能不完全是 224x224，需要裁剪/填充)
        H, W, C = reconstructed_array.shape
        if H != IMAGE_SIZE[0] or W != IMAGE_SIZE[1]:
            # 确保输出是目标尺寸
            reconstructed_array = tf.image.resize(reconstructed_array, IMAGE_SIZE).numpy()

    except Exception as e:
        # 解码失败，返回黑色图像
        print(f"Warning: Decompression failed due to severe channel errors ({e}).")
        reconstructed_array = np.zeros(IMAGE_SIZE + (3,), dtype=np.float32)

    return reconstructed_array


# ----------------- 训练与可视化 (主要执行逻辑) -----------------

def preprocess(x_data):
    """归一化并缩放图像。"""
    # 假设 TFDS 返回的是字典 {'image': ..., 'label': ...}
    image = x_data['image'] if isinstance(x_data, dict) else x_data

    # 缩放到目标尺寸并归一化
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def map_fixed_format(x):
    """ Fixed DJSCC Dataset format: (Image) -> (Image, Image)"""
    return x, x


def map_adaptive_format(x):
    """ Adaptive DJSCC Dataset format: (Image) -> ((Image, SNR_Placeholder), Image)"""
    snr_placeholder = tf.constant(0.0, dtype=tf.float32)
    return (x, snr_placeholder), x


if __name__ == "__main__":
    # 1. 加载和预处理 ImageNet-like 数据 (使用 tfds 替代 CIFAR-10)
    print(f"加载和预处理 ImageNet-like 数据集 ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]})...")

    DATASET_NAME = 'imagenet_v2'

    try:
        data, info = tfds.load(DATASET_NAME, split=['test[:90%]', 'test[90%:]'], with_info=True,
                               shuffle_files=True)
        print(f"✅ 成功加载 {DATASET_NAME} 数据集。")

        train_data = data[0].map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1024)
        test_data = data[1].map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    except Exception as e:
        print(f"\n!!! 警告: ImageNet 数据集加载失败 ({e}). 请确保您的环境已正确配置 TFDS 并下载了所需数据集。")
        print("!!! 临时回退到 CIFAR-10，但会将其缩放至 224x224 以测试新模型结构。")
        (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()


        def preprocess_cifar_scaled(x_data):
            x_data = tf.cast(x_data, tf.float32) / 255.0
            x_data = tf.image.resize(x_data, IMAGE_SIZE)
            return x_data


        train_data = tf.data.Dataset.from_tensor_slices(x_train).map(preprocess_cifar_scaled).shuffle(1024)
        test_data = tf.data.Dataset.from_tensor_slices(x_test).map(preprocess_cifar_scaled)

        # 调整 Epochs，因为 CIFAR-10 训练更快
        EPOCHS = 10

    # Fixed DJSCC 数据集
    fixed_train_ds = train_data.map(map_fixed_format).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    fixed_test_ds = test_data.map(map_fixed_format).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Adaptive DJSCC 数据集
    adaptive_train_ds = train_data.map(map_adaptive_format).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    adaptive_test_ds = test_data.map(map_adaptive_format).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # ------------------ A & B: 训练模型 (逻辑不变) ------------------

    # 1. Adaptive DJSCC
    adaptive_model = AdaptiveDJSCC_Model(c=CHANNEL_DIM,
                                         min_snr_db=ADAPTIVE_MIN_SNR, max_snr_db=ADAPTIVE_MAX_SNR)
    # 使用 1e-4 的学习率进行快速测试
    adaptive_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss=tf.keras.losses.MeanSquaredError())

    print(f"\n--- 1. 开始训练 Adaptive DJSCC (SNR {ADAPTIVE_MIN_SNR}-{ADAPTIVE_MAX_SNR} dB, Epochs={EPOCHS}) ---")
    adaptive_model.fit(adaptive_train_ds, epochs=EPOCHS, validation_data=adaptive_test_ds, verbose=1)

    # 2. Fixed DJSCC
    fixed_model = FixedDJSCC_Model(c=CHANNEL_DIM, fixed_snr_db=FIXED_SNR_DB)
    # 使用 1e-4 的学习率进行快速测试
    fixed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss=tf.keras.losses.MeanSquaredError())

    print(f"\n--- 2. 开始训练 Fixed DJSCC (SNR {FIXED_SNR_DB} dB, Epochs={EPOCHS}) ---")
    fixed_model.fit(fixed_train_ds, epochs=EPOCHS, validation_data=fixed_test_ds, verbose=1)

    # ------------------ C. 多 SNR 对比可视化 ------------------
    print(f"\n--- 3. 开始对单张图片在 {len(TEST_SNRS_DB)} 个 SNR 下进行对比 ---")

    # *******************************************************************
    # ********* 修改点 2: 提取单张测试图片 *********
    # *******************************************************************
    # 从测试集中获取第一张图片
    test_image_original = next(iter(fixed_test_ds))[0][0:1]  # [1, 224, 224, 3]

    # 定义行标签和 SNR 列表
    n_snrs = len(TEST_SNRS_DB)
    row_labels = ["Adaptive DJSCC", f"Fixed DJSCC ({FIXED_SNR_DB}dB)", "Classical JSSC"]

    # 绘制结果图
    # 3 行 (模型) x (n_snrs + 1) 列 (原始图 + n_snrs 个 SNR)
    fig = plt.figure(figsize=(3 * (n_snrs + 1) + 1, 8))
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.05, hspace=0.1, wspace=0.1)

    # 在第一列显示原始图像（重复 3 次以对应 3 行）
    for r in range(3):
        ax = fig.add_subplot(3, n_snrs + 1, r * (n_snrs + 1) + 1)
        plt.imshow(test_image_original.numpy().squeeze())
        plt.axis("off")

        if r == 0:
            ax.set_title("Original", fontsize=12)

        # 添加行标签
        ax.text(-0.15, 0.5, row_labels[r],
                verticalalignment='center',
                horizontalalignment='right',
                transform=ax.transAxes,
                fontsize=12,
                fontweight='bold')

    # *******************************************************************
    # ********* 修改点 3: 循环测试和绘图 *********
    # *******************************************************************
    for col_idx, snr_db in enumerate(TEST_SNRS_DB):

        # --- 1. Adaptive DJSCC 预测 ---
        test_snr_tensor = tf.fill(dims=(1, 1), value=tf.constant(snr_db, dtype=tf.float32))
        adaptive_reconstruction = adaptive_model.predict((test_image_original, test_snr_tensor), verbose=0)

        # --- 2. Fixed DJSCC 预测 ---
        fixed_reconstruction = fixed_model.predict(test_image_original, verbose=0)

        # --- 3. Classical JSSC 预测 ---
        img_np = test_image_original[0].numpy()
        classical_reconstruction = classical_jssc_process(img_np, snr_db=snr_db)
        # 将 numpy 数组转为 (1, H, W, C) 格式
        classical_reconstruction = np.expand_dims(classical_reconstruction, axis=0)

        # 图像列表 (按行)
        reconstructions = [adaptive_reconstruction, fixed_reconstruction, classical_reconstruction]

        # 在当前 SNR 列绘制结果
        for row_idx in range(3):
            ax = fig.add_subplot(3, n_snrs + 1, row_idx * (n_snrs + 1) + col_idx + 2)

            img_to_show = reconstructions[row_idx].squeeze()
            if img_to_show.ndim == 2 or img_to_show.shape[-1] == 1:
                img_to_show = np.stack([np.squeeze(img_to_show)] * 3, axis=-1)

            plt.imshow(img_to_show)
            plt.axis("off")

            # 在第一行添加 SNR 标题
            if row_idx == 0:
                ax.set_title(f"SNR={snr_db}dB", fontsize=12, fontweight='bold')

    plt.suptitle(
        f"Single Image Multi-SNR Comparison ({IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} - Fixed Model Target: {FIXED_SNR_DB}dB)",
        fontsize=16, y=0.95)
    output_filename = f"multi_snr_comparison_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.png"
    plt.savefig(output_filename)
    print(f"\n对比结果已保存为文件: {output_filename}")