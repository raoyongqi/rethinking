import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

input_array = np.ones((7, 7))
input_array[3, 3] = 0

conv_kernel = np.ones((3, 3))

# 定义膨胀卷积核（3x3，膨胀因子为2）
dilated_kernel = np.array([[1, 0, 1], 
                           [0, 0, 0], 
                           [1, 0, 1]])

def plot_convolution(ax, title, input_array, kernel, dilation_factor=1):
    ax.imshow(input_array, cmap='gray', interpolation='nearest', extent=[0, 7, 0, 7])
    ax.set_title(title)
    
    kernel_size = kernel.shape[0]
    offset = kernel_size // 2
    
    for i in range(input_array.shape[0]):
        for j in range(input_array.shape[1]):
            if i - offset >= 0 and i + offset < input_array.shape[0] and j - offset >= 0 and j + offset < input_array.shape[1]:
                rect = patches.Rectangle((j - offset, i - offset), kernel_size, kernel_size, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

plot_convolution(ax[0], "Normal Convolution (3x3)", input_array, conv_kernel)

plot_convolution(ax[1], "Dilated Convolution (Dilation = 2)", input_array, dilated_kernel)

plt.tight_layout()
plt.show()
