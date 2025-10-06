import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 确保在无头模式下运行
matplotlib.use('Agg')

# --- 关键配置 ---
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
CHANNEL_DIM = 8
FIXED_SNR_DB = 10.0  # Fixed DJSCC 的训练 SNR
TEST_SNR_DB = 5.0  # 三重对比测试的 SNR (通常选一个挑战性的低值)
ADAPTIVE_MIN_SNR = 0.0  # Adaptive DJSCC 训练范围
ADAPTIVE_MAX_SNR = 20.0
EPOCHS = 15  # 增加 Epoch 以帮助 Adaptive 模型收敛


# ----------------- 编码器和解码器类 (用于两种模型) -----------------

class Encoder(tf.keras.Model):
    def __init__(self, c, is_adaptive=False):
        super().__init__()
        self.is_adaptive = is_adaptive

        # 图像处理层 (32x32 -> 8x8)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=2)  # 32->16
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=2)  # 16->8
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=1)  # 8->8

        # --- Adaptive DJSCC 特有的 SNR 融合层 ---
        if self.is_adaptive:
            # 用于处理 SNR 标量的层
            self.snr_dense = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
            # 1x1 卷积将 SNR 嵌入广播到特征图尺寸 (8x8x128)
            self.snr_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same',
                                                   activation=tf.nn.relu)

        # 最终编码层 (将特征降到 C 维)
        self.conv4 = tf.keras.layers.Conv2D(filters=c, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=1)

    def call(self, inputs):
        # inputs 可能是 image (Fixed) 或 (image, snr_value) (Adaptive)
        if self.is_adaptive:
            image, snr_value = inputs
        else:
            image = inputs
            snr_value = None  # Fixed model doesn't use this

        # 1. 处理图像
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)

        # 2. Adaptive DJSCC: SNR 融合
        if self.is_adaptive and snr_value is not None:
            # 扩展 SNR 嵌入以匹配特征图尺寸
            snr_emb = self.snr_dense(snr_value)
            # (B, 128) -> (B, 1, 1, 128) -> (B, 8, 8, 128)
            snr_emb_conv = self.snr_conv(tf.expand_dims(tf.expand_dims(snr_emb, 1), 1))

            # 融合：将 SNR 特征加到图像特征上
            x = x + snr_emb_conv

        # 3. 最终编码
        output = self.conv4(x)
        return output


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 逆向操作 (8x8 -> 32x32)
        self.dconv1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=1)
        self.dconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)
        self.dconv3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.relu, strides=2)
        self.dconv4 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(3, 3), padding='same',
                                                      activation=tf.nn.sigmoid, strides=1)

    def call(self, input):
        x = self.dconv1(input)
        x = self.dconv2(x)
        x = self.dconv3(x)
        output = self.dconv4(x)
        return output


# ----------------- Fixed DJSCC 模型封装 (仅处理一个固定 SNR) -----------------
class FixedDJSCC_Model(tf.keras.Model):
    def __init__(self, c, fixed_snr_db):
        super().__init__()
        self.fixed_snr_db = fixed_snr_db
        self.encoder = Encoder(c, is_adaptive=False)
        self.decoder = Decoder()

    def call(self, inputs, training=False):
        image = inputs[0] if isinstance(inputs, tuple) else inputs
        batch_size = tf.shape(image)[0]

        # 1. 确定 SNR (训练/非训练都使用 fixed_snr_db)
        target_snr_db = self.fixed_snr_db
        snr_db = tf.fill(dims=(batch_size, 1), value=tf.cast(target_snr_db, tf.float32))

        # 2. 编码
        encoded_output = self.encoder(image)

        # 3. 信道噪声
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_stddev = tf.sqrt(1.0 / (2.0 * snr_linear))
        # (B, 1) -> (B, 1, 1, 1)
        noise_stddev = tf.expand_dims(tf.expand_dims(noise_stddev, 1), 1)

        noise = tf.random.normal(shape=tf.shape(encoded_output), mean=0.0, stddev=noise_stddev)
        noisy_output = encoded_output + noise

        # 4. 解码
        decoded_output = self.decoder(noisy_output)

        return decoded_output


# ----------------- Adaptive DJSCC 模型封装 (处理随机/指定 SNR) -----------------
class AdaptiveDJSCC_Model(tf.keras.Model):
    def __init__(self, c, min_snr_db, max_snr_db):
        super().__init__()
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db
        self.encoder = Encoder(c, is_adaptive=True)
        self.decoder = Decoder()

    def call(self, inputs, training=False):

        # Keras fit/evaluate/predict 输入处理:
        if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[0], tf.Tensor):
            # 预测模式: (image_tensor, snr_tensor)
            image, snr_db_tensor_raw = inputs
        elif isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[0], tuple):
            # 训练/验证模式 (Keras .fit): inputs[0] = (image, snr_placeholder)
            image, snr_db_tensor_raw = inputs[0]
        else:
            # 强制构建或其他异常情况，默认使用图像，SNR 标量
            image = inputs
            snr_db_tensor_raw = tf.constant(self.min_snr_db, dtype=tf.float32)

        batch_size = tf.shape(image)[0]

        # 1. 确定 SNR 张量 (B x 1 形状)
        if training:
            # 训练: 随机 SNR
            snr_db = tf.random.uniform(
                shape=(batch_size, 1),
                minval=self.min_snr_db,
                maxval=self.max_snr_db,
                dtype=tf.float32
            )
            encoder_input = (image, snr_db)
        else:
            # 测试/预测: 使用输入的 snr_db_tensor_raw
            # 鲁棒地提取第一个元素的 SNR 值，确保 Rank 兼容性

            # 使用 tf.gather 无论 snr_db_tensor_raw 是 (B,) 还是 (B, 1) 都能安全提取
            snr_val_scalar = tf.squeeze(tf.gather(snr_db_tensor_raw, 0))

            snr_db = tf.fill(dims=(batch_size, 1), value=tf.cast(snr_val_scalar, tf.float32))
            encoder_input = (image, snr_db)

        # 2. 编码
        encoded_output = self.encoder(encoder_input)

        # 3. 信道噪声
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_stddev = tf.sqrt(1.0 / (2.0 * snr_linear))
        # (B, 1) -> (B, 1, 1, 1)
        noise_stddev = tf.expand_dims(tf.expand_dims(noise_stddev, 1), 1)

        noise = tf.random.normal(shape=tf.shape(encoded_output), mean=0.0, stddev=noise_stddev)
        noisy_output = encoded_output + noise

        # 4. 解码
        decoded_output = self.decoder(noisy_output)

        return decoded_output


# --- 普通 JSSC 对比代码 (不变) ---
def classical_jssc_process(image_array, snr_db):
    ber = 0.5 * 10 ** (-snr_db / 10.0)
    ber = max(1e-6, min(ber, 0.5))  # 限制 BER 在合理范围

    image = Image.fromarray((image_array.squeeze() * 255).astype(np.uint8))
    buf = io.BytesIO()
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

        # 处理灰度图转换为 RGB
        if len(reconstructed_array.shape) == 2:
            reconstructed_array = np.stack([reconstructed_array] * 3, axis=-1)

        # 确保尺寸正确
        if reconstructed_array.shape[0] > IMAGE_SIZE[0] or reconstructed_array.shape[1] > IMAGE_SIZE[1]:
            reconstructed_array = reconstructed_array[:IMAGE_SIZE[0], :IMAGE_SIZE[1], :]
        # 确保颜色通道正确
        if reconstructed_array.shape[-1] < 3:
            reconstructed_array = np.pad(reconstructed_array,
                                         ((0, 0), (0, 0), (0, 3 - reconstructed_array.shape[-1])),
                                         mode='constant')

    except Exception as e:
        # 解码失败，返回黑色图像
        print(f"Warning: Decompression failed due to severe channel errors ({e}).")
        reconstructed_array = np.zeros(IMAGE_SIZE + (3,), dtype=np.float32)

    return reconstructed_array


# ----------------- 训练与可视化 (主要执行逻辑) -----------------

def preprocess(x_data):
    """归一化图像。"""
    x_data = tf.cast(x_data, tf.float32) / 255.0
    return x_data


def map_fixed_format(x):
    """ Fixed DJSCC Dataset format: (Image) -> (Image, Image)"""
    return x, x


def map_adaptive_format(x):
    """ Adaptive DJSCC Dataset format: (Image) -> ((Image, SNR_Placeholder), Image)"""
    # 占位符现在是 Rank 0 标量，Batching 后会变成 Rank 1 (B,)，更容易处理
    snr_placeholder = tf.constant(0.0, dtype=tf.float32)
    return (x, snr_placeholder), x


if __name__ == "__main__":
    # 1. 加载和预处理 CIFAR-10 数据
    print("加载和预处理 CIFAR-10 数据集 (32x32)...")
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    # 训练数据集 (用于 Fixed 和 Adaptive 训练)
    train_data = tf.data.Dataset.from_tensor_slices(x_train).map(preprocess).shuffle(1024)
    test_data = tf.data.Dataset.from_tensor_slices(x_test).map(preprocess)

    # Fixed DJSCC 数据集
    fixed_train_ds = train_data.map(map_fixed_format).batch(BATCH_SIZE)
    fixed_test_ds = test_data.map(map_fixed_format).batch(BATCH_SIZE)

    # Adaptive DJSCC 数据集
    adaptive_train_ds = train_data.map(map_adaptive_format).batch(BATCH_SIZE)
    # Adaptive DJSCC 测试集 (注意: 这里的占位符是用于 Keras 结构匹配，实际 SNR 在 predict 时传入)
    adaptive_test_ds = test_data.map(map_adaptive_format).batch(BATCH_SIZE)

    # ------------------ A. 训练 Adaptive DJSCC (SNR 0-20 dB) ------------------
    adaptive_model = AdaptiveDJSCC_Model(c=CHANNEL_DIM,
                                         min_snr_db=ADAPTIVE_MIN_SNR, max_snr_db=ADAPTIVE_MAX_SNR)
    adaptive_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=tf.keras.losses.MeanSquaredError())

    # 强制构建 Adaptive 模型 (输入是 ((Image, SNR), Target))
    sample_adaptive_x, _ = next(iter(adaptive_train_ds))
    # 这里的 snr_placeholder 是 (B,) 形状
    adaptive_model(sample_adaptive_x)

    print(f"\n--- 1. 开始训练 Adaptive DJSCC (SNR {ADAPTIVE_MIN_SNR}-{ADAPTIVE_MAX_SNR} dB) ---")
    adaptive_model.fit(adaptive_train_ds, epochs=EPOCHS, validation_data=adaptive_test_ds, verbose=1)

    # ------------------ B. 训练 Fixed DJSCC (SNR 10 dB) ------------------
    fixed_model = FixedDJSCC_Model(c=CHANNEL_DIM, fixed_snr_db=FIXED_SNR_DB)
    fixed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss=tf.keras.losses.MeanSquaredError())

    # 强制构建 Fixed 模型
    sample_image_batch, _ = next(iter(fixed_train_ds))
    fixed_model(sample_image_batch)

    print(f"\n--- 2. 开始训练 Fixed DJSCC (SNR {FIXED_SNR_DB} dB) ---")
    fixed_model.fit(fixed_train_ds, epochs=EPOCHS, validation_data=fixed_test_ds, verbose=1)

    # ------------------ C. 三重对比可视化 (在 TEST_SNR_DB 下) ------------------
    print(f"\n--- 3. 开始在 SNR = {TEST_SNR_DB} dB 下进行三重对比 ---")

    # 获取测试图像
    test_images_x, _ = next(iter(fixed_test_ds))
    test_images_tensor = test_images_x
    n_images = min(test_images_tensor.shape[0], 6)

    # 构建 Adaptive 模型测试输入：([Image], [TEST_SNR_DB])
    # 注意：这里构造的张量 shape 是 (N, 1)
    test_snr_tensor_for_predict = tf.fill(dims=(n_images, 1), value=tf.constant(TEST_SNR_DB, dtype=tf.float32))
    adaptive_test_images_only = test_images_tensor[:n_images]

    # 1. Fixed DJSCC 预测
    fixed_reconstructions = fixed_model.predict(adaptive_test_images_only)

    # 2. Adaptive DJSCC 预测 (传入图像和目标 SNR)
    adaptive_reconstructions = adaptive_model.predict((adaptive_test_images_only, test_snr_tensor_for_predict))

    # 3. Classical JSSC 预测
    classical_reconstructions = []
    for i in range(n_images):
        img_np = adaptive_test_images_only[i].numpy()
        reconstructed = classical_jssc_process(img_np, snr_db=TEST_SNR_DB)
        classical_reconstructions.append(reconstructed)
    classical_reconstructions = np.stack(classical_reconstructions, axis=0)

    # 绘制对比图表 (四行: Original, Adaptive DJSCC, Fixed DJSCC, Classical JSSC)
    plt.figure(figsize=(15, 10))
    test_images_display = adaptive_test_images_only.numpy()

    for i in range(n_images):
        # 原始图像 (第1行)
        ax = plt.subplot(4, n_images, i + 1)
        plt.imshow(test_images_display[i])
        if i == 0: ax.set_title("Original")
        plt.axis("off")

        # Adaptive DJSCC (第2行)
        ax = plt.subplot(4, n_images, n_images + i + 1)
        plt.imshow(adaptive_reconstructions[i])
        if i == 0: ax.set_title("Adaptive DJSCC")
        plt.axis("off")

        # Fixed DJSCC (第3行)
        ax = plt.subplot(4, n_images, 2 * n_images + i + 1)
        plt.imshow(fixed_reconstructions[i])
        if i == 0: ax.set_title(f"Fixed DJSCC ({FIXED_SNR_DB}dB)")
        plt.axis("off")

        # 传统 JSSC (第4行)
        ax = plt.subplot(4, n_images, 3 * n_images + i + 1)
        # 确保尺寸匹配 32x32
        if classical_reconstructions[i].shape == test_images_display[i].shape:
            plt.imshow(classical_reconstructions[i])
        else:
            # 兼容性处理，以防 JSSC 输出尺寸不完全一致
            plt.imshow(classical_reconstructions[i][:IMAGE_SIZE[0], :IMAGE_SIZE[1], :])
        if i == 0: ax.set_title("Classical JSSC")
        plt.axis("off")

    plt.suptitle(f"Triple Comparison on CIFAR-10 (32x32) at Test SNR = {TEST_SNR_DB} dB")
    output_filename = f"triple_djscc_comparison_snr_{TEST_SNR_DB}db_32x32.png"
    plt.savefig(output_filename)
    print(f"\n对比结果已保存为文件: {output_filename}")