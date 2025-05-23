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

normal_conv_result = apply_convolution(input_array, normal_kernel, dilation_rate=1)

import numpy as np

input_array = np.ones((1, 7, 1), dtype=np.float32)

normal_kernel = np.array([[1], [0], [1]], dtype=np.float32)

normal_kernel = normal_kernel.reshape((3, 1))

output = []
for i in range(input_array.shape[1]):
    if i == 0 or i == input_array.shape[1] - 1:
        output.append(1)
    else:
        window = input_array[0, i - 1:i + 2, 0]
        print(window)

        result = np.sum(window * normal_kernel.reshape(-1))
        output.append(result)

output = np.array(output)

print(output)
