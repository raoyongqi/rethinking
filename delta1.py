import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 定义输入和卷积核
input_array = np.ones((1, 13, 1), dtype=np.float32)  # 输入形状 (batch_size, length, channels)
normal_kernel = np.array([1, 0, 1], dtype=np.float32)  # 卷积核：1D 核心 (kernel_size, )

# 设置膨胀率
dilation_rate = 2

# 进行膨胀卷积核插值（将零插入卷积核）
def dilate_kernel(kernel, dilation_rate):
    dilated_kernel = []
    for i in range(len(kernel) - 1):
        dilated_kernel.append(kernel[i])
        dilated_kernel.extend([0] * (dilation_rate - 1))  # 插入零
    dilated_kernel.append(kernel[-1])  # 添加最后一个元素（没有插入零）
    return np.array(dilated_kernel)

# 膨胀卷积核
dilated_kernel = dilate_kernel(normal_kernel, dilation_rate)

# 打印膨胀后的卷积核
print("Dilated Kernel:", dilated_kernel)

# 进行1D卷积操作（手动实现）
def apply_convolution_numpy(input_array, kernel, padding='same'):
    input_length = input_array.shape[1]
    kernel_size = kernel.shape[0]
    
    # 'same' padding
    if padding == 'same':
        pad_size = (kernel_size - 1) // 2
        input_array_padded = np.pad(input_array, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
        output_length = input_length
    else:
        input_array_padded = input_array
        output_length = input_length - kernel_size + 1
    
    # 初始化输出
    output = np.zeros((output_length,))
    
    # 执行卷积操作
    for i in range(output_length):
        window = input_array_padded[0, i:i+kernel_size, 0]
        output[i] = np.sum(window * kernel)
    
    return output

# 计算卷积结果
conv_result = apply_convolution_numpy(input_array, dilated_kernel, padding='same')

# 打印卷积结果
print("Manual Convolution Result:", conv_result)

# 使用 TensorFlow 进行卷积（模拟）
input_tensor = tf.convert_to_tensor(input_array)

conv1d_layer = tf.keras.layers.Conv1D(
    filters=1,               # 卷积核的输出通道数
    kernel_size=3,           # 卷积核的大小
    dilation_rate=2,         # 膨胀率
    padding='same',          # 'same' padding，保证输出和输入长度相同
    use_bias=False,          # 不使用偏置
    kernel_initializer=tf.constant_initializer(normal_kernel)  # 使用自定义卷积核
)

# 进行卷积操作
conv_result_tf = conv1d_layer(input_tensor)

# 打印卷积结果
print("TensorFlow Convolution Result:", conv_result_tf.numpy().flatten())
