import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scienceplots

# 读取每个 CSV 文件
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
            first_line = f.readline().strip()  # 读取并去掉换行符
            times.append(float(first_line))  # 转换为浮动类型并加入列表
    # 创建一个 DataFrame，列为训练时间，并添加激活函数标签
    return pd.DataFrame({
        'Training Time': times,
        'Activation Function': [activation_function] * len(times)
    })

# 读取每个激活函数的文件第一行数据并合并为一个 DataFrame
elu_time_df = read_first_line(elu_files, 'ELU')
LeakyReLU_time_df = read_first_line(LeakyReLU_files, 'LeakyReLU')
relu_time_df = read_first_line(relu_files, 'ReLU')
shifted_relu_time_df = read_first_line(shifted_relu_files, 'Shifted ReLU')

# 合并所有 DataFrame
all_time_df = pd.concat([elu_time_df, LeakyReLU_time_df, relu_time_df, shifted_relu_time_df], ignore_index=True)
all_time_df_sorted = all_time_df.sort_values(by='Training Time', ascending=True)

# 对 DataFrame 按照 "Training Time" 列进行升序排序
mean_time_df = all_time_df.groupby('Activation Function')['Training Time'].mean().reset_index()

# 根据均值排序
mean_time_df_sorted = mean_time_df.sort_values(by='Training Time', ascending=True)

# 获取排序后的激活函数顺序
activation_order = mean_time_df_sorted['Activation Function'].tolist()


# 设置图形大小，创建左右图布局
with plt.style.context('science'):  # 使用科学风格
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 设置颜色调色板
    palette = sns.color_palette("Set2", n_colors=4)


    # 创建一个字典，按排序顺序为每个激活函数分配颜色
    activation_color_map = {activation: palette[i] for i, activation in enumerate(activation_order)}


    # 按照排序顺序绘制折线图
    for activation in activation_order:
        if activation == 'ELU':
            sns.lineplot(data=elu_df, x='Epoch', y='Loss', label='ELU', ax=axes[0], color=activation_color_map[activation])
        elif activation == 'LeakyReLU':
            sns.lineplot(data=LeakyReLU_df, x='Epoch', y='Loss', label='LeakyReLU', ax=axes[0], color=activation_color_map[activation])
        elif activation == 'ReLU':
            sns.lineplot(data=relu_df, x='Epoch', y='Loss', label='ReLU', ax=axes[0], color=activation_color_map[activation])
        elif activation == 'Shifted ReLU':
            sns.lineplot(data=shifted_relu_df, x='Epoch', y='Loss', label='Shifted ReLU', ax=axes[0], color=activation_color_map[activation])

    # 设置左侧图标题和标签
    axes[0].set_title('Loss vs Epoch for Different Activation Functions')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # 右侧图：绘制带置信区间的柱状图
    sns.barplot(x='Training Time', y='Activation Function', data=all_time_df_sorted, ci="sd", palette=palette, ax=axes[1])

    # 设置右侧图标题和标签
    axes[1].set_title('Average Training Time with Confidence Interval for Different Activation Functions')
    axes[1].set_xlabel('Training Time (seconds)')
    axes[1].set_ylabel('Activation Function')

    # 保存图像
    plt.savefig("data/combined_plot_with_confidence_intervals.png", dpi=300)

    # 显示图形
    plt.tight_layout()
    plt.show()
