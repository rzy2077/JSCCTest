import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
from PIL import Image

# 确保在无头模式下运行
matplotlib.use('Agg')

# --- 关键配置 ---
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
CHANNEL_DIM = 8
FIXED_SNR_DB = 10.0  # Fixed DJSCC 的训练 SNR
TEST_SNR_DB = 5.0  # 三种方法进行对比测试的 SNR
ADAPTIVE_MIN_SNR = 0.0  # Adaptive DJSCC 训练范围
ADAPTIVE_MAX_SNR = 20.0
EPOCHS = 15  # 增加 Epoch 以确保 Adaptive 模型收敛


# ----------------- 编码器和解码器类 (调整为 32x32 结构 + Adaptive 支持) -----------------
class Encode(tf.keras.Model):
    def __init__(self, c, is_adaptive=False):
        super().__init__()
        self.is_adaptive = is_adaptive

        # 图像处理层
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
            self.snr_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same',
                                                   activation=tf.nn.relu)

        # 最终编码层 (将特征降到 C 维)
        self.conv4 = tf.keras.layers.Conv2D(filters=c, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=1)

    def call(self, inputs):
        if self.is_adaptive:
            image, snr_value = inputs  # inputs 是 (image, snr_value)
        else:
            image = inputs  # inputs 只是 image

        # 1. 处理图像
        x = self.conv1(image)
        x = self.conv2(x)
        x = self.conv3(x)

        # 2. Adaptive DJSCC: SNR 融合
        if self.is_adaptive:
            # 扩展 SNR 嵌入以匹配 (B, 8, 8, 128)
            snr_emb = self.snr_dense(snr_value)
            # (B, 128) -> (B, 1, 1, 128) -> (B, 8, 8, 128)
            snr_emb_conv = self.snr_conv(tf.expand_dims(tf.expand_dims(snr_emb, 1), 1))

            # 融合
            x = x + snr_emb_conv

        # 3. 最终编码
        output = self.conv4(x)
        return output


class Decode(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 逆向操作
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


# ----------------- Adaptive/Fixed DJSCC 模型封装 -----------------
class DJSCC_Model(tf.keras.Model):
    # 统一模型，通过 is_adaptive 和 fixed_snr_db 参数控制行为
    def __init__(self, c, is_adaptive=False, fixed_snr_db=None, min_snr_db=0.0, max_snr_db=20.0):
        super().__init__()
        self.is_adaptive = is_adaptive
        self.fixed_snr_db = fixed_snr_db  # 用于 Fixed DJSCC 训练
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        # 编码器需要知道是否为 Adaptive
        self.encoder = Encode(c, is_adaptive=is_adaptive)
        self.decoder = Decode()

    def call(self, inputs, training=False):
        # inputs 结构：(image, target_image) 用于 Fixed DJSCC
        # inputs 结构：((image, snr_db_tensor), target_image) 用于 Adaptive DJSCC

        if self.is_adaptive:
            # Adaptive 模型从 inputs 中解包 image 和 snr_db_tensor
            image, snr_db_tensor = inputs[0], inputs[1]
        else:
            # Fixed 模型直接从 inputs 中获取 image (这里 snr_db_tensor 是 None 或占位符)
            image = inputs[0]
            snr_db_tensor = tf.constant(self.fixed_snr_db or TEST_SNR_DB, dtype=tf.float32)

        # 1. 确定信道中使用的 SNR
        if training and self.is_adaptive:
            # Adaptive 训练: 随机 SNR
            snr_db = tf.random.uniform(
                shape=(tf.shape(image)[0], 1),
                minval=self.min_snr_db,
                maxval=self.max_snr_db,
                dtype=tf.float32
            )
            # 编码器需要这个随机 SNR
            encoder_input = (image, snr_db)
        elif training:
            # Fixed 训练: 固定 SNR
            snr_db = tf.fill(dims=(tf.shape(image)[0], 1), value=tf.cast(self.fixed_snr_db, tf.float32))
            # Fixed 编码器只需要图像
            encoder_input = image
        else:
            # 测试/预测: 使用传入的 snr_db_tensor (或 Fixed 模型中的默认值)
            snr_db = tf.fill(dims=(tf.shape(image)[0], 1), value=tf.cast(snr_db_tensor[0], tf.float32))
            # 编码器需要这个测试 SNR
            encoder_input = (image, snr_db) if self.is_adaptive else image

        # 2. 编码
        encoded_output = self.encoder(encoder_input)

        # 3. 信道噪声
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_stddev = tf.sqrt(1.0 / (2.0 * snr_linear))

        noise_stddev = tf.expand_dims(tf.expand_dims(noise_stddev, 1), 1)

        if training:
            noise = tf.random.normal(shape=tf.shape(encoded_output), mean=0.0, stddev=noise_stddev)
            noisy_output = encoded_output + noise
        else:
            noisy_output = encoded_output

        # 4. 解码
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
        reconstructed_array = np.zeros_like(image_array.squeeze())

    return reconstructed_array


# ----------------- 训练与可视化 (主要执行逻辑) -----------------

def preprocess(x_data):
    """归一化图像。"""
    x_data = tf.cast(x_data, tf.float32) / 255.0
    return x_data


def map_fixed_format(x):
    """ Fixed DJSCC: (Image) -> (Image, Image)"""
    return x, x


def map_adaptive_format(x):
    """ Adaptive DJSCC: (Image) -> ((Image, SNR_Placeholder), Image)"""
    # 占位符，实际 SNR 在模型 call 中随机生成或在测试时指定
    snr_placeholder = tf.constant([0.0], dtype=tf.float32)
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
    adaptive_test_ds = test_data.map(map_adaptive_format).batch(BATCH_SIZE)

    # ------------------ A. 训练 Fixed DJSCC (SNR 10 dB) ------------------
    fixed_model = DJSCC_Model(c=CHANNEL_DIM, is_adaptive=False, fixed_snr_db=FIXED_SNR_DB)
    fixed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss=tf.keras.losses.MeanSquaredError())

    print(f"\n--- 1. 开始训练 Fixed DJSCC (SNR {FIXED_SNR_DB} dB) ---")
    fixed_model.fit(fixed_train_ds, epochs=EPOCHS, validation_data=fixed_test_ds, verbose=1)

    # ------------------ B. 训练 Adaptive DJSCC (SNR 0-20 dB) ------------------
    adaptive_model = DJSCC_Model(c=CHANNEL_DIM, is_adaptive=True,
                                 min_snr_db=ADAPTIVE_MIN_SNR, max_snr_db=ADAPTIVE_MAX_SNR)
    adaptive_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=tf.keras.losses.MeanSquaredError())

    print(f"\n--- 2. 开始训练 Adaptive DJSCC (SNR {ADAPTIVE_MIN_SNR}-{ADAPTIVE_MAX_SNR} dB) ---")
    adaptive_model.fit(adaptive_train_ds, epochs=EPOCHS, validation_data=adaptive_test_ds, verbose=1)

    # ------------------ C. 三重对比可视化 (在 TEST_SNR_DB 下) ------------------
    print(f"\n--- 3. 开始在 SNR = {TEST_SNR_DB} dB 下进行三重对比 ---")

    # 获取测试图像
    test_images_x, _ = next(iter(fixed_test_ds))  # Fixed DS 返回 (X, Y)
    test_images_tensor = test_images_x
    n_images = min(test_images_tensor.shape[0], 6)  # 展示 6 张图像

    # 构建 Adaptive 模型测试输入：((Image, [TEST_SNR_DB]), Image)
    test_snr_tensor = tf.fill(dims=(n_images, 1), value=tf.constant(TEST_SNR_DB, dtype=tf.float32))
    adaptive_test_input = ((test_images_tensor[:n_images], test_snr_tensor), None)

    # 1. Fixed DJSCC 预测
    fixed_reconstructions = fixed_model.predict(test_images_tensor[:n_images])

    # 2. Adaptive DJSCC 预测
    # 必须使用 model.predict(X) 格式，但 X 必须包含 (Image, SNR)
    adaptive_test_images_only = test_images_tensor[:n_images]
    adaptive_reconstructions = adaptive_model.predict((adaptive_test_images_only, test_snr_tensor))

    # 3. Classical JSSC 预测
    classical_reconstructions = []
    for i in range(n_images):
        img_np = test_images_tensor[i].numpy()
        reconstructed = classical_jssc_process(img_np, snr_db=TEST_SNR_DB)
        classical_reconstructions.append(reconstructed)
    classical_reconstructions = np.stack(classical_reconstructions, axis=0)

    # 绘制对比图表 (四行: Original, Adaptive DJSCC, Fixed DJSCC, Classical JSSC)
    plt.figure(figsize=(15, 8))
    test_images_display = test_images_tensor.numpy()[:n_images]

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
        if classical_reconstructions[i].shape == test_images_display[i].shape:
            plt.imshow(classical_reconstructions[i])
        else:
            plt.imshow(classical_reconstructions[i][:32, :32, :])
        if i == 0: ax.set_title("Classical JSSC")
        plt.axis("off")

    plt.suptitle(f"Triple Comparison on CIFAR-10 (32x32) at Test SNR = {TEST_SNR_DB} dB")
    output_filename = f"triple_djscc_comparison_snr_{TEST_SNR_DB}db_32x32.png"
    plt.savefig(output_filename)
    print(f"\n对比结果已保存为文件: {output_filename}")