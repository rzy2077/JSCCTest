import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import io
from PIL import Image

# 确保在无头模式下运行
matplotlib.use('Agg')

# --- 假定高分辨率图像数据集路径 ---
# 请将此路径修改为您本地包含高分辨率图像的文件夹路径。
# 文件夹结构应为:
# testPath/
# ├── train/
# │   ├── image1.jpg
# │   └── image2.png
# └── test/
#     ├── image3.jpg
#     └── image4.png
HIGH_RES_DATA_DIR = r'C:\Users\rzy19\PyCharmMiscProject\testPath'
IMAGE_SIZE = (128, 128)  # 请根据您的图片尺寸和模型需求调整
BATCH_SIZE = 32


# ----------------- 编码器和解码器类 (用户提供) -----------------
# 针对高分辨率图像调整模型架构
class Encode(tf.keras.Model):
    def __init__(self, c):
        super().__init__()
        # 增加额外的层以处理更大的输入尺寸
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same',
                                            activation=tf.nn.relu, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same',
                                            activation=tf.nn.relu, strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[5, 5], padding='same',
                                            activation=tf.nn.relu, strides=2)  # 新增一层
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=[5, 5], padding='same',
                                            activation=tf.nn.relu, strides=1)
        self.conv5 = tf.keras.layers.Conv2D(filters=c, kernel_size=[5, 5], padding='same',
                                            activation=tf.nn.relu, strides=1)

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
        # 调整以匹配编码器的输出
        self.dconv1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), padding='same',
                                                      activation=tf.nn.relu, strides=1)
        self.dconv2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), padding='same',
                                                      activation=tf.nn.relu, strides=1)
        self.dconv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), padding='same',
                                                      activation=tf.nn.relu, strides=2)
        self.dconv4 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(5, 5), padding='same',
                                                      activation=tf.nn.relu, strides=2)
        self.dconv5 = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), padding='same',
                                                      activation=tf.nn.sigmoid, strides=2)

    def call(self, input):
        x = self.dconv1(input)
        x = self.dconv2(x)
        x = self.dconv3(x)
        x = self.dconv4(x)
        output = self.dconv5(x)
        return output


# ----------------- DJSCC 模型封装 -----------------
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


# --- 普通 JSSC 对比代码 ---
def classical_jssc_process(image_array, snr_db):
    # 将信噪比转换为误码率 (一个简化的模型)
    ber = 0.5 * 10 ** (-snr_db / 10.0)

    # 1. 源编码 (JPEG压缩)
    # 将numpy数组转换为PIL图像对象，确保数据类型正确
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

    # 3. 解码 (将比特流重新组合并解压缩)
    try:
        corrupted_bytes = np.packbits(bit_stream).tobytes()
        corrupted_buf = io.BytesIO(corrupted_bytes)
        reconstructed_image = Image.open(corrupted_buf)

        reconstructed_array = np.array(reconstructed_image).astype('float32') / 255.0
        # 确保输出与输入维度匹配
        if len(reconstructed_array.shape) == 2:
            reconstructed_array = np.stack([reconstructed_array] * 3, axis=-1)
    except:
        print("Warning: Decompression failed due to severe channel errors.")
        reconstructed_array = np.zeros_like(image_array.squeeze())

    return reconstructed_array


# ----------------- 训练与可视化 -----------------
if __name__ == "__main__":
    # 参数设置
    CHANNEL_DIM = 24
    SNR_DB = 10
    EPOCHS = 5

    # 1. 加载和预处理高分辨率数据
    print(f"从目录 '{HIGH_RES_DATA_DIR}' 加载高分辨率图像...")

    # 使用 image_dataset_from_directory 加载数据，自动处理图像尺寸和归一化
    # 标签设置为 None，因为我们只需要图像本身进行自编码训练
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=HIGH_RES_DATA_DIR + '/train',
        labels=None,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        interpolation='bicubic',
        shuffle=True
    ).map(lambda x: x / 255.0)  # 将像素值归一化到 [0, 1]

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=HIGH_RES_DATA_DIR + '/test',
        labels=None,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        interpolation='bicubic',
        shuffle=False
    ).map(lambda x: x / 255.0)  # 将像素值归一化到 [0, 1]

    # 2. 实例化 DJSCC 模型
    djscc_model = DJSCC_Model(c=CHANNEL_DIM, snr_db=SNR_DB)

    # 3. 编译模型
    djscc_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                        loss=tf.keras.losses.MeanSquaredError())

    # 4. 训练模型
    print(f"开始训练 DJSCC 模型, 信噪比 (SNR) 为 {SNR_DB} dB...")
    djscc_model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=test_ds)

    # 5. 可视化结果
    print("开始可视化重建结果...")

    # 从测试数据集中获取一批图像
    test_images = next(iter(test_ds))
    n_images = test_images.shape[0]

    # 使用 DJSCC 模型进行预测
    reconstructed_images = djscc_model.predict(test_images)

    # 使用传统 JSSC 方法进行重建
    print("开始使用传统 JSSC 方法进行重建...")
    classical_reconstructions = []
    for img in test_images:
        reconstructed = classical_jssc_process(img.numpy(), snr_db=SNR_DB)
        classical_reconstructions.append(reconstructed)
    classical_reconstructions = np.stack(classical_reconstructions, axis=0)

    # 绘制对比图表
    plt.figure(figsize=(10, 6))

    for i in range(n_images):
        # 原始图像 (第1行)
        ax = plt.subplot(3, n_images, i + 1)
        plt.imshow(test_images[i])
        plt.title("Original")
        plt.axis("off")

        # DJSCC 重建图像 (第2行)
        ax = plt.subplot(3, n_images, n_images + i + 1)
        plt.imshow(reconstructed_images[i])
        plt.title("DJSCC")
        plt.axis("off")

        # 传统 JSSC 重建图像 (第3行)
        ax = plt.subplot(3, n_images, 2 * n_images + i + 1)
        plt.imshow(classical_reconstructions[i])
        plt.title("Classical JSSC")
        plt.axis("off")

    plt.suptitle(f"DJSCC vs. Classical JSSC at SNR = {SNR_DB} dB")
    output_filename = f"djscc_vs_classical_snr_{SNR_DB}db.png"
    plt.savefig(output_filename)
    print(f"对比结果已保存为文件: {output_filename}")