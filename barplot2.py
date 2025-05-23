import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据的函数
def read_loss_and_time_data(loss_files, time_files, activation_function):
    loss_data = []
    time_data = []
    
    # 读取损失数据
    for file in loss_files:
        loss_df = pd.read_csv(file)
        loss_data.append(loss_df[['Epoch', 'Loss']])
    
    all_loss_data = pd.concat(loss_data, ignore_index=True)
    
    for file in time_files:
        with open(file, 'r') as f:
            time = float(f.readline().strip())
            time_data.append(time)
    
    time_data_repeated = time_data * (len(all_loss_data) // len(time_data)) + time_data[:len(all_loss_data) % len(time_data)]
    
    all_loss_data['Activation Function'] = activation_function
    all_loss_data['Training Time'] = time_data_repeated
    
    return all_loss_data

# 定义文件路径
elu_loss_files = [
    "C:/Users/r/Desktop/rethink_resample/data/elu_trial_1_loss_history.csv",
    "C:/Users/r/Desktop/rethink_resample/data/elu_trial_2_loss_history.csv",
    "C:/Users/r/Desktop/rethink_resample/data/elu_trial_3_loss_history.csv"
]

elu_time_files = [
    "C:/Users/r/Desktop/rethink_resample/data/elu_trial_1_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/elu_trial_2_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/elu_trial_3_training_time.txt"
]

LeakyReLU_loss_files = [
    "C:/Users/r/Desktop/rethink_resample/data/LeakyReLU_trial_1_loss_history.csv",
    "C:/Users/r/Desktop/rethink_resample/data/LeakyReLU_trial_2_loss_history.csv",
    "C:/Users/r/Desktop/rethink_resample/data/LeakyReLU_trial_3_loss_history.csv"
]

LeakyReLU_time_files = [
    "C:/Users/r/Desktop/rethink_resample/data/LeakyReLU_trial_1_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/LeakyReLU_trial_2_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/LeakyReLU_trial_3_training_time.txt"
]

relu_loss_files = [
    "C:/Users/r/Desktop/rethink_resample/data/relu_trial_1_loss_history.csv",
    "C:/Users/r/Desktop/rethink_resample/data/relu_trial_2_loss_history.csv",
    "C:/Users/r/Desktop/rethink_resample/data/relu_trial_3_loss_history.csv"
]

relu_time_files = [
    "C:/Users/r/Desktop/rethink_resample/data/relu_trial_1_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/relu_trial_2_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/relu_trial_3_training_time.txt"
]

shifted_relu_loss_files = [
    "C:/Users/r/Desktop/rethink_resample/data/shifted_relu_trial_1_loss_history.csv",
    "C:/Users/r/Desktop/rethink_resample/data/shifted_relu_trial_2_loss_history.csv",
    "C:/Users/r/Desktop/rethink_resample/data/shifted_relu_trial_3_loss_history.csv"
]

shifted_relu_time_files = [
    "C:/Users/r/Desktop/rethink_resample/data/shifted_relu_trial_1_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/shifted_relu_trial_2_training_time.txt",
    "C:/Users/r/Desktop/rethink_resample/data/shifted_relu_trial_3_training_time.txt"
]

# 读取所有激活函数的数据
elu_df = read_loss_and_time_data(elu_loss_files, elu_time_files, 'ELU')
LeakyReLU_df = read_loss_and_time_data(LeakyReLU_loss_files, LeakyReLU_time_files, 'LeakyReLU')
relu_df = read_loss_and_time_data(relu_loss_files, relu_time_files, 'ReLU')
shifted_relu_df = read_loss_and_time_data(shifted_relu_loss_files, shifted_relu_time_files, 'Shifted ReLU')

# 合并所有数据
all_df = pd.concat([elu_df, LeakyReLU_df, relu_df, shifted_relu_df], ignore_index=True)

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# 绘制损失与周期图
sns.lineplot(x='Epoch', y='Loss', hue='Activation Function', data=all_df, ax=ax1)
ax1.set_title('Loss vs. Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(title='Activation Function', fontsize=10)

# 绘制平均训练时间图
avg_time_df = all_df.groupby('Activation Function')['Training Time'].mean().reset_index()

sns.barplot(x='Activation Function', y='Training Time', data=avg_time_df, ci=None, ax=ax2, palette="Set2")
ax2.set_title('Average Training Time by Activation Function')
ax2.set_xlabel('Activation Function')
ax2.set_ylabel('Average Training Time (seconds)')

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()
