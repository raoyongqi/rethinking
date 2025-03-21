import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

input_array = np.ones((1, 9, 1), dtype=np.float32)

normal_kernel = np.array([[1], 
                          [0], 
                          [1]], dtype=np.float32)



normal_kernel = normal_kernel.reshape((3, 1, 1))


def apply_convolution(input_array, kernel, dilation_rate=1):
    conv_layer = tf.keras.layers.Conv1D(
        filters=1, 
        kernel_size=3,
        dilation_rate=dilation_rate, 
        padding='same', 
        use_bias=False, 
        kernel_initializer=tf.constant_initializer(kernel)
    )
    output = conv_layer(input_array)
    return output.numpy().squeeze()

# 创建卷积结果
normal_conv_result = apply_convolution(input_array, normal_kernel, dilation_rate=1)
print(normal_conv_result)


import numpy as np

# 输入数据
input_array = np.ones((1, 7, 1), dtype=np.float32)

# 卷积核
normal_kernel = np.array([[1], [0], [1]], dtype=np.float32)

# 确保卷积核大小正确
normal_kernel = normal_kernel.reshape((3, 1))

# 手动实现卷积过程，计算每个位置的卷积
output = []
for i in range(input_array.shape[1]):  # 7个时间步
    if i == 0 or i == input_array.shape[1] - 1:  # 边缘
        output.append(1)
    else:
        # 取出当前3个连续的元素
        window = input_array[0, i - 1:i + 2, 0]  # 当前窗口
        print(window)
        # 手动执行卷积（卷积核和窗口的点乘）
        result = np.sum(window * normal_kernel.reshape(-1))
        output.append(result)

# 转换为NumPy数组
output = np.array(output)

print(output)
