import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches  # 用于创建方块图例

# 读取 CSV 文件
batchnorm1 = pd.read_csv('data/batchnorm_trial_1_history.csv')
batchnorm2 = pd.read_csv('data/batchnorm_trial_2_history.csv')
batchnorm3 = pd.read_csv('data/batchnorm_trial_3_history.csv')
batchnorm_combined = pd.concat([batchnorm1, batchnorm2 ,batchnorm3], axis=0, ignore_index=False)

without_batchnorm1 = pd.read_csv('data/without_batchnorm_trial_1_history.csv')
without_batchnorm2 = pd.read_csv('data/without_batchnorm_trial_2_history.csv')
without_batchnorm3 = pd.read_csv('data/without_batchnorm_trial_3_history.csv')
without_batchnorm_combined = pd.concat([without_batchnorm1, without_batchnorm2 ,without_batchnorm3], axis=0, ignore_index=False)

batchnorm_combined['epoch'] = batchnorm_combined['Unnamed: 0'] + 1
without_batchnorm_combined['epoch'] = without_batchnorm_combined['Unnamed: 0'] + 1

# 读取时间文件并提取第一行数据
files = ['data/batchnorm_trial_1_time.txt', 'data/batchnorm_trial_2_time.txt', 'data/batchnorm_trial_3_time.txt']
batchnorm_times = []
for file in files:
    with open(file, 'r') as f:
        first_line = f.readline().strip()
        batchnorm_times.append(float(first_line))

without_files = ['data/without_batchnorm_trial_1_time.txt', 'data/without_batchnorm_trial_2_time.txt', 'data/without_batchnorm_trial_3_time.txt']
without_batchnorm_times = []
for file in without_files:
    with open(file, 'r') as f:
        first_line = f.readline().strip()
        without_batchnorm_times.append(float(first_line))

# 创建 DataFrame
df = pd.DataFrame({
    'bn': batchnorm_times,
    'without_bn': without_batchnorm_times
})
import scienceplots
bn_handle = mpatches.Patch(color='red', label='BatchNorm')   # 创建红色方块
without_bn_handle = mpatches.Patch(color='blue', label='without_BatchNorm')  # 创建蓝色方块

with plt.style.context('science'):
# 创建一个 1x2 的子图 (左右布局)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [6, 4]})
    # 绘制 Epoch vs Loss 曲线图
    with plt.style.context('science'):
        sns.lineplot(x='epoch', y='loss', data=batchnorm_combined, color='red', ax=axes[0], label='BatchNorm', linewidth=2)
        sns.lineplot(x='epoch', y='loss', data=without_batchnorm_combined, color='blue', ax=axes[0], label='without_BatchNorm', linewidth=2)

        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('Loss', fontsize=14)
        axes[0].set_ylim(10, 30)
        axes[0].tick_params(axis='both', labelsize=14)
        axes[0].set_xlim(0, 500) 

        axes[0].legend(fontsize=20)

        df_plot = df.melt(var_name='Condition', value_name='Time')
        sns.barplot(x='Time', y='Condition', hue='Condition', data=df_plot, ax=axes[1], palette=["red", "blue"], fontsize=14)

        axes[1].set_ylabel('')  # 去掉 y 轴标签
        axes[1].set_yticks([])
        axes[1].tick_params(axis='both', labelsize=14)
        axes[1].legend(handles=[bn_handle, without_bn_handle], fontsize=14)

    plt.suptitle('', fontsize=16)

    plt.tight_layout()
    plt.savefig("data/combined_plot_with_confidence_intervals.png", dpi=300)
    # 显示图形
    plt.show()
