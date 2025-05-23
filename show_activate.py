import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.patches as mpatches  # 用于创建方块图例

elu1 = pd.read_csv('data/elu_trial_1_loss_history.csv')
elu2 = pd.read_csv('data/elu_trial_2_loss_history.csv')
elu3 = pd.read_csv('data/elu_trial_3_loss_history.csv')

LeakyReLU1 = pd.read_csv('data/LeakyReLU_trial_1_loss_history.csv')
LeakyReLU2 = pd.read_csv('data/LeakyReLU_trial_2_loss_history.csv')
LeakyReLU3 = pd.read_csv('data/LeakyReLU_trial_3_loss_history.csv')

relu1 = pd.read_csv('data/relu_trial_1_loss_history.csv')
relu2 = pd.read_csv('data/relu_trial_2_loss_history.csv')
relu3 = pd.read_csv('data/relu_trial_3_loss_history.csv')

shifted_relu1 = pd.read_csv('data/shifted_relu_trial_1_loss_history.csv')
shifted_relu2 = pd.read_csv('data/shifted_relu_trial_2_loss_history.csv')
shifted_relu3 = pd.read_csv('data/shifted_relu_trial_3_loss_history.csv')

# 分别合并每个激活函数的数据
elu_df = pd.concat([elu1, elu2, elu3], ignore_index=True)
LeakyReLU_df = pd.concat([LeakyReLU1, LeakyReLU2, LeakyReLU3], ignore_index=True)
relu_df = pd.concat([relu1, relu2, relu3], ignore_index=True)
shifted_relu_df = pd.concat([shifted_relu1, shifted_relu2, shifted_relu3], ignore_index=True)

# 文件路径列表，按激活函数分类
elu_files = [
    "C:/Users/r/Desktop/rethink_resample/data/elu_trial_1_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/elu_trial_2_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/elu_trial_3_training_time.txt"
]

LeakyReLU_files = [
    "C:/Users/r/Desktop/rethink_resample/data/LeakyReLU_trial_1_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/LeakyReLU_trial_2_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/LeakyReLU_trial_3_training_time.txt"
]

relu_files = [
    "C:/Users/r/Desktop/rethink_resample/data/relu_trial_1_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/relu_trial_2_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/relu_trial_3_training_time.txt"
]

shifted_relu_files = [
    "C:/Users/r/Desktop/rethink_resample/data/shifted_relu_trial_1_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/shifted_relu_trial_2_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/shifted_relu_trial_3_training_time.txt"
]

# 函数：读取文件的第一行并转换为浮动类型
def read_first_line(files, activation_function):
    times = []
    for file in files:
        with open(file, 'r') as f:
            first_line = f.readline().strip()
            times.append(float(first_line))
    return pd.DataFrame({
        'Training Time': times,
        'Activation Function': [activation_function] * len(times)
    })

elu_time_df = read_first_line(elu_files, 'ELU')
LeakyReLU_time_df = read_first_line(LeakyReLU_files, 'LeakyReLU')
relu_time_df = read_first_line(relu_files, 'ReLU')
shifted_relu_time_df = read_first_line(shifted_relu_files, 'Shifted ReLU')

all_time_df = pd.concat([elu_time_df, LeakyReLU_time_df, relu_time_df, shifted_relu_time_df], ignore_index=True)
all_time_df_sorted = all_time_df.sort_values(by='Training Time', ascending=True)

mean_time_df = all_time_df.groupby('Activation Function')['Training Time'].mean().reset_index()

mean_time_df_sorted = mean_time_df.sort_values(by='Training Time', ascending=True)

activation_order = mean_time_df_sorted['Activation Function'].tolist()
relu_handle = mpatches.Patch(color='#7A506D', label='ReLU')
leaky_relu_handle = mpatches.Patch(color='#228B22', label=r"Leaky ReLU")
srelu_handle = mpatches.Patch(color='#B74A33', label="Shifted ReLU")
elu_handle = mpatches.Patch(color='#87CEEB', label=r"ELU")

# with plt.style.context('science'):
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 20

legend_properties = {'weight':'bold'}
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 设置颜色调色板
palette = ['#228B22', '#B74A33', '#87CEEB', '#7A506D']


# 创建一个字典，按排序顺序为每个激活函数分配颜色
activation_color_map = {activation: palette[i] for i, activation in enumerate(activation_order)}


# 按照排序顺序绘制折线图
for activation in activation_order:
    if activation == 'ELU':
        sns.lineplot(data=elu_df, x='Epoch', y='Loss', label='ELU', ax=axes[0], color=activation_color_map[activation], linewidth=2)
    elif activation == 'LeakyReLU':
        sns.lineplot(data=LeakyReLU_df, x='Epoch', y='Loss', label='LeakyReLU', ax=axes[0], color=activation_color_map[activation], linewidth=2)
    elif activation == 'ReLU':
        sns.lineplot(data=relu_df, x='Epoch', y='Loss', label='ReLU', ax=axes[0], color=activation_color_map[activation], linewidth=2)
    elif activation == 'Shifted ReLU':
        sns.lineplot(data=shifted_relu_df, x='Epoch', y='Loss', label='Shifted ReLU', ax=axes[0], color=activation_color_map[activation], linewidth=2)

# 设置左侧图标题和标签
axes[0].set_xlabel('Epoch',)
axes[0].set_ylabel('Loss', )
axes[0].set_ylim(10, 30)
axes[0].tick_params(axis='both', )

axes[0].set_xlim(0, 400) 
legend_1 =axes[0].legend()


sns.barplot(x='Training Time', y='Activation Function', data=all_time_df_sorted, ci="sd", palette=palette, ax=axes[1])

axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('')  # 去掉 y 轴标签
axes[1].set_yticks([])  # 去掉 y 轴刻度
axes[1].tick_params(axis='both', )

legend_2 = axes[1].legend(handles=[relu_handle, leaky_relu_handle,srelu_handle,elu_handle],  loc='upper right')

# 保存图像

plt.suptitle('',)

# 显示图形
plt.tight_layout()
plt.savefig("data/combined_plot_with_confidence_intervals.png", dpi=300)

plt.show()
