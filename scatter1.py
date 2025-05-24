import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取Excel文件
file_path = 'data/climate_soil_tif.xlsx'
data = pd.read_excel(file_path)

# 2. 检查数据
print("数据前五行：")
print(data.head())  # 查看前几行数据
print("\n列名：")
print(data.columns)  # 查看列名
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20
# 检查所需列是否存在
required_columns = ['WC2.1_5M_BIO_4', 'WC2.1_5M_BIO_15']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"列 '{col}' 不存在于数据中")

# 3. 绘制散点直方图（边际图）
# 方法1：使用Seaborn的jointplot（最简单）
plt.figure(figsize=(10, 8))
sns.jointplot(
    x='WC2.1_5M_BIO_1',   # 气温作为X轴
    y='WC2.1_5M_BIO_12',  # 降水作为Y轴
    data=data,
    kind='scatter',       # 散点图
    height=8,             # 图形大小
    ratio=5,              # 主图与边缘图的比例
    marginal_kws=dict(bins=20),  # 边缘直方图的bins数
    joint_kws=dict(alpha=0.5)    # 散点透明度
)

plt.suptitle('')
# 修改后的坐标轴标签设置（在jointplot部分）
plt.xlabel('Temperature (°C)')  # 假设WC2.1_5M_BIO_4是温度，单位为°C
plt.ylabel('Precipitation (mm)')  # 假设WC2.1_5M_BIO_15是降水，单位为mm
plt.savefig('data/scatter1.png')
plt.show()

# # 方法2：使用Matplotlib手动创建（更灵活）
# fig = plt.figure(figsize=(10, 8))
# grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.5)

# # 主散点图
# main_ax = fig.add_subplot(grid[:-1, :-1])
# main_ax.scatter(
#     data['WC2.1_5M_BIO_4'],   # 气温
#     data['WC2.1_5M_BIO_15'],  # 降水
#     alpha=0.5
# )
# main_ax.set_xlabel('Temperature (WC2.1_5M_BIO_4)')
# main_ax.set_ylabel('Precipitation (WC2.1_5M_BIO_15)')

# # X轴直方图（气温）
# x_hist = fig.add_subplot(grid[-1, :-1], sharex=main_ax)
# x_hist.hist(data['WC2.1_5M_BIO_4'], bins=20, color='gray')
# x_hist.set_xlabel('Temperature Distribution')

# # Y轴直方图（降水）
# y_hist = fig.add_subplot(grid[:-1, -1], sharey=main_ax)
# y_hist.hist(
#     data['WC2.1_5M_BIO_15'], 
#     bins=20, 
#     orientation='horizontal', 
#     color='gray'
# )
# y_hist.set_ylabel('Precipitation Distribution')

# plt.suptitle('Precipitation vs Temperature with Marginal Histograms', y=1.02)
# plt.tight_layout()
# plt.show()