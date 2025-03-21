import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # 用于创建方块图例

# 文件名列表
files = ['data/batchnorm_trial_1_time.txt', 'data/batchnorm_trial_2_time.txt', 'data/batchnorm_trial_3_time.txt']

# 创建一个空的列表来存储每个文件的第一行数据
batchnorm_times = []

# 读取每个文件的第一行
for file in files:
    with open(file, 'r') as f:
        first_line = f.readline().strip()  # 读取第一行并去掉换行符
        batchnorm_times.append(float(first_line))  # 将第一行转换为浮动数值并添加到列表中

# 无BatchNorm文件列表
without_files = ['data/without_batchnorm_trial_1_time.txt', 'data/without_batchnorm_trial_2_time.txt', 'data/without_batchnorm_trial_3_time.txt']

# 创建一个空的列表来存储每个文件的第一行数据
without_batchnorm_times = []

# 读取每个文件的第一行
for file in without_files:
    with open(file, 'r') as f:
        first_line = f.readline().strip()  # 读取第一行并去掉换行符
        without_batchnorm_times.append(float(first_line))  # 将第一行转换为浮动数值并添加到列表中

# 创建 DataFrame
df = pd.DataFrame({
    'bn': batchnorm_times,
    'without_bn': without_batchnorm_times
})

# 重塑数据为长格式
df_plot = df.melt(var_name='Condition', value_name='Time')
print(df_plot)
df_plot['Condition1']=df_plot['Condition']

# 设置颜色调色板
custom_palette = {'bn': 'red', 'without_bn': 'blue'}

# 绘制横向柱状图
sns.barplot(x='Time', y='Condition', data=df_plot, hue='Condition', palette=["red", "blue"])

# 去掉 y 轴的刻度标签
plt.ylabel('')  # 去掉 y 轴标签
plt.yticks([])  # 去掉 y 轴刻度
bn_handle = mpatches.Patch(color='red', label='bn')   # 创建红色方块
without_bn_handle = mpatches.Patch(color='blue', label='without_bn')  # 创建蓝色方块

# 添加图例
plt.legend(handles=[bn_handle, without_bn_handle], title='Condition')

# 显示图形
plt.title('Comparison of BatchNorm and Without BatchNorm Times')
plt.xlabel('Time')  # 设置 x 轴标签

# 使用 legend() 自动添加图例（可以去掉手动指定）

plt.show()
