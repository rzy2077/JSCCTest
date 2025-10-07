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

# 核心修改: 定义 SNR 测试范围
TEST_SNR_RANGE = [20.0, 10.0, 5.0, 0.0, -5.0, -10.0]
# 注意: -20 dB 噪声过大，可能导致 Classical JSSC 总是解码失败，所以这里只到 -10 dB
# 如果需要 -20 dB, 请自行将 -20.0 加入列表，但 Classical JSSC 报错概率极高

ADAPTIVE_MIN_SNR = 0.0  # Adaptive DJSCC 训练范围
ADAPTIVE_MAX_SNR = 20.0
EPOCHS = 15  # 增加 Epoch 以帮助 Adaptive 模型收敛


# ----------------- 编码器和解码器类 (不变) -----------------

class Encoder(tf.keras.Model):
    def __init__(self, c, is_adaptive=False):
        super().__init__()
        self.is_adaptive = is_adaptive
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same',
                                            activation=tf.nn.relu, strides=1)
        if self.is_adaptive:
            self.snr_dense = tf.keras.layers.Dense(units=128, activation=tf.nn.relu)
            self.snr_conv = tf.keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same',
                                                   activation=tf.nn.relu)
        self.conv4 = tf.keras.layers.Conv2D(filters=c, kernel_size=[3, 3], padding='same',
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

        if self.is_adaptive and snr_value is not None:
            snr_emb = self.snr_dense(snr_value)
            snr_emb_conv = self.snr_conv(tf.expand_dims(tf.expand_dims(snr_emb, 1), 1))
            x = x + snr_emb_conv

        output = self.conv4(x)
        return output


class Decoder(tf.keras.Model):
    def __init__(self):
        super().__init__()
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


# ----------------- Fixed DJSCC 模型封装 (不变) -----------------
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


# ----------------- Adaptive DJSCC 模型封装 (不变) -----------------
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
            # 使用 tf.gather 鲁棒地提取 SNR
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


# --- 普通 JSSC 对比代码 (修改了 BER 限制) ---
def classical_jssc_process(image_array, snr_db):
    ber = 0.5 * 10 ** (-snr_db / 10.0)
    # 修改: 提高 BER 限制到 0.49 (接近随机)，以允许在极低 SNR 下解码存活
    ber = max(1e-6, min(ber, 0.49))

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

        if len(reconstructed_array.shape) == 2:
            reconstructed_array = np.stack([reconstructed_array] * 3, axis=-1)

        if reconstructed_array.shape[0] > IMAGE_SIZE[0] or reconstructed_array.shape[1] > IMAGE_SIZE[1]:
            reconstructed_array = reconstructed_array[:IMAGE_SIZE[0], :IMAGE_SIZE[1], :]
        if reconstructed_array.shape[-1] < 3:
            reconstructed_array = np.pad(reconstructed_array,
                                         ((0, 0), (0, 0), (0, 3 - reconstructed_array.shape[-1])),
                                         mode='constant')

    except Exception as e:
        # 解码失败，返回黑色图像
        print(f"Warning: Decompression failed due to severe channel errors (SNR={snr_db}dB, {e}).")
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
    snr_placeholder = tf.constant(0.0, dtype=tf.float32)
    return (x, snr_placeholder), x


if __name__ == "__main__":
    # 1. 加载和预处理 CIFAR-10 数据
    print("加载和预处理 CIFAR-10 数据集 (32x32)...")
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()

    # 训练数据集
    train_data = tf.data.Dataset.from_tensor_slices(x_train).map(preprocess).shuffle(1024)
    test_data = tf.data.Dataset.from_tensor_slices(x_test).map(preprocess)

    fixed_train_ds = train_data.map(map_fixed_format).batch(BATCH_SIZE)
    fixed_test_ds = test_data.map(map_fixed_format).batch(BATCH_SIZE)
    adaptive_train_ds = train_data.map(map_adaptive_format).batch(BATCH_SIZE)
    adaptive_test_ds = test_data.map(map_adaptive_format).batch(BATCH_SIZE)

    # ------------------ A. 训练 Adaptive DJSCC ------------------
    adaptive_model = AdaptiveDJSCC_Model(c=CHANNEL_DIM,
                                         min_snr_db=ADAPTIVE_MIN_SNR, max_snr_db=ADAPTIVE_MAX_SNR)
    adaptive_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss=tf.keras.losses.MeanSquaredError())

    sample_adaptive_x, _ = next(iter(adaptive_train_ds))
    adaptive_model(sample_adaptive_x)

    print(f"\n--- 1. 开始训练 Adaptive DJSCC (SNR {ADAPTIVE_MIN_SNR}-{ADAPTIVE_MAX_SNR} dB) ---")
    adaptive_model.fit(adaptive_train_ds, epochs=EPOCHS, validation_data=adaptive_test_ds, verbose=1)

    # ------------------ B. 训练 Fixed DJSCC ------------------
    fixed_model = FixedDJSCC_Model(c=CHANNEL_DIM, fixed_snr_db=FIXED_SNR_DB)
    fixed_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss=tf.keras.losses.MeanSquaredError())

    sample_image_batch, _ = next(iter(fixed_train_ds))
    fixed_model(sample_image_batch)

    print(f"\n--- 2. 开始训练 Fixed DJSCC (SNR {FIXED_SNR_DB} dB) ---")
    fixed_model.fit(fixed_train_ds, epochs=EPOCHS, validation_data=fixed_test_ds, verbose=1)

    # ------------------ C. 跨 SNR 范围对比可视化 ------------------

    # 1. 获取一张固定的测试图像
    test_images_x, _ = next(iter(fixed_test_ds))
    # 取第一张图作为所有 SNR 的测试图像
    single_test_image_tensor = tf.expand_dims(test_images_x[0], axis=0)  # Shape (1, 32, 32, 3)
    single_test_image_np = test_images_x[0].numpy()

    n_snr = len(TEST_SNR_RANGE)
    results = {
        'Original': single_test_image_np,
        'Adaptive DJSCC': [],
        f'Fixed DJSCC ({FIXED_SNR_DB}dB)': [],
        'Classical JSSC': []
    }

    print(f"\n--- 3. 开始在 SNR 范围 {TEST_SNR_RANGE} dB 下进行多重对比 ---")

    for snr_db in TEST_SNR_RANGE:
        print(f"  > Processing SNR = {snr_db} dB...")

        # 构造 Adaptive/Fixed 模型测试输入 (Batch Size = 1)
        test_snr_tensor_for_predict = tf.constant([[snr_db]], dtype=tf.float32)

        # 1. Fixed DJSCC 预测
        fixed_reconst = fixed_model.predict(single_test_image_tensor, verbose=0)[0]
        results[f'Fixed DJSCC ({FIXED_SNR_DB}dB)'].append(fixed_reconst)

        # 2. Adaptive DJSCC 预测
        adaptive_reconst = adaptive_model.predict((single_test_image_tensor, test_snr_tensor_for_predict), verbose=0)[0]
        results['Adaptive DJSCC'].append(adaptive_reconst)

        # 3. Classical JSSC 预测
        classical_reconst = classical_jssc_process(single_test_image_np, snr_db=snr_db)
        results['Classical JSSC'].append(classical_reconst)

    # 绘制对比图表
    methods = [f'Fixed DJSCC ({FIXED_SNR_DB}dB)', 'Adaptive DJSCC', 'Classical JSSC']
    n_rows = len(methods) + 1  # +1 for Original

    plt.figure(figsize=(3 * n_snr, 3 * n_rows))

    # 第一行: 原始图片
    for j in range(n_snr):
        ax = plt.subplot(n_rows, n_snr, j + 1)
        # 只需要在第一列显示原始图片
        if j == 0:
            plt.imshow(results['Original'])
            ax.set_title(f"Original\n({TEST_SNR_RANGE[j]}dB)", fontsize=10)
            ax.set_ylabel("Original", rotation=90, size='large')
        else:
            plt.imshow(np.ones(IMAGE_SIZE + (3,)), alpha=0)  # 留白
            ax.set_title(f"{TEST_SNR_RANGE[j]}dB", fontsize=10)
        plt.axis("off")

    # 后续行: 三种方法的重建图片
    for i, method_name in enumerate(methods):
        for j in range(n_snr):
            ax = plt.subplot(n_rows, n_snr, (i + 1) * n_snr + j + 1)

            reconst_img = results[method_name][j]
            # 兼容性处理，以防 JSSC 输出尺寸不完全一致
            if reconst_img.shape == IMAGE_SIZE + (3,):
                plt.imshow(reconst_img)
            else:
                plt.imshow(reconst_img[:IMAGE_SIZE[0], :IMAGE_SIZE[1], :])

            if j == 0:
                ax.set_ylabel(method_name.split('(')[0].strip(), rotation=90, size='large')

            plt.axis("off")

    plt.suptitle(f"Cross-SNR Comparison (Fixed DJSCC trained at {FIXED_SNR_DB} dB)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # 调整布局，避免标题重叠

    output_filename = f"cross_snr_comparison_djscc_{TEST_SNR_RANGE[0]}db_to_{TEST_SNR_RANGE[-1]}db.png"
    plt.savefig(output_filename)
    print(f"\n跨 SNR 对比结果已保存为文件: {output_filename}")