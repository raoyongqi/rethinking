import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
fig, ax = plt.subplots()

left_start_x = 0.1
right_start_x = 0.9
rect_width = 0.1
rect_height = 0.2
gap = 0.15


for i in range(3):
    ax.add_patch(plt.Rectangle((left_start_x + i * (rect_width + gap), 0.7), rect_width, rect_height, edgecolor="black", facecolor="black"))  # 左侧卷积核元素

for i in range(5):
    if i == 1 or i == 3:
        text_x = right_start_x + i * (rect_width + gap)
        text = ax.text(text_x+0.05, 0.8, '0', horizontalalignment='center', verticalalignment='center', fontsize=50, color='red', weight='bold')

    # 其他元素为黑色矩形
    else:
        ax.add_patch(plt.Rectangle((right_start_x + i * (rect_width + gap), 0.7), rect_width, rect_height, edgecolor="black", facecolor="black"))

ax.text(left_start_x + 1.5 * (rect_width + gap), 0.6, r'$\left[1, 0, 1\right]^T$', horizontalalignment='center', verticalalignment='center', fontsize=15, color='black')

ax.text(right_start_x + 2.5 * (rect_width + gap), 0.6, r'$\left[1, 0, 0, 0, 1\right]^T$', horizontalalignment='center', verticalalignment='center', fontsize=15, color='black')

ax.text(left_start_x + 1.5 * (rect_width + gap), 0.2, 'Convolution Kernel', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
ax.text(right_start_x + 2.5 * (rect_width + gap), 0.2, 'Dilated Convolution Kernel', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')

ax.set_axis_off()

ax.set_xlim(0, 2)
ax.set_ylim(0, 1)
plt.savefig('data/fangkuai.png')

plt.show()
