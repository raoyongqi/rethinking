import matplotlib.pyplot as plt
import seaborn as sns
# 设置 Seaborn 风格
sns.set(style="whitegrid")
# 创建一个图形和坐标轴
fig, ax = plt.subplots()

# 计算左侧卷积核和右侧扩张卷积核的矩形的宽度间隔
left_start_x = 0.1
right_start_x = 0.9
rect_width = 0.1
rect_height = 0.2
gap = 0.15  # 矩形之间的间隔

# 左侧卷积核 [1, 0, 1]^T，使用矩形表示（黑色）
for i in range(3):
    ax.add_patch(plt.Rectangle((left_start_x + i * (rect_width + gap), 0.7), rect_width, rect_height, edgecolor="black", facecolor="black"))  # 左侧卷积核元素

# 右侧扩张卷积核 [1, 0, 0, 0, 1]^T，插入的零用“0”字符表示
for i in range(5):
    # 插入的 0（红色），在需要插入0的地方显示
    if i == 1 or i == 3:
        # 设置0的位置，使其水平方向上居中
        text_x = right_start_x + i * (rect_width + gap)
        text = ax.text(text_x+0.05, 0.8, '0', horizontalalignment='center', verticalalignment='center', fontsize=50, color='red', weight='bold')

    # 其他元素为黑色矩形
    else:
        ax.add_patch(plt.Rectangle((right_start_x + i * (rect_width + gap), 0.7), rect_width, rect_height, edgecolor="black", facecolor="black"))

# 添加卷积核文本 [1, 0, 1]^T（在左边）
ax.text(left_start_x + 1.5 * (rect_width + gap), 0.6, r'$\left[1, 0, 1\right]^T$', horizontalalignment='center', verticalalignment='center', fontsize=15, color='black')

# 添加扩张卷积核的文本 [1, 0, 0, 0, 1]^T（在右边）
ax.text(right_start_x + 2.5 * (rect_width + gap), 0.6, r'$\left[1, 0, 0, 0, 1\right]^T$', horizontalalignment='center', verticalalignment='center', fontsize=15, color='black')

# 在图的下方添加英文标签
ax.text(left_start_x + 1.5 * (rect_width + gap), 0.2, 'Convolution Kernel', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')
ax.text(right_start_x + 2.5 * (rect_width + gap), 0.2, 'Dilated Convolution Kernel', horizontalalignment='center', verticalalignment='center', fontsize=12, color='black')

# 去掉坐标轴
ax.set_axis_off()

# 设置图形的范围
ax.set_xlim(0, 2)
ax.set_ylim(0, 1)
plt.savefig('data/fangkuai.png')

# 显示图形
plt.show()
