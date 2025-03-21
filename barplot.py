import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
elu_df = read_first_line(elu_files, 'ELU')
LeakyReLU_df = read_first_line(LeakyReLU_files, 'LeakyReLU')
relu_df = read_first_line(relu_files, 'ReLU')
shifted_relu_df = read_first_line(shifted_relu_files, 'Shifted ReLU')

# 合并所有 DataFrame
all_df = pd.concat([elu_df, LeakyReLU_df, relu_df, shifted_relu_df], ignore_index=True)

# 对 DataFrame 按照 "Training Time" 列进行升序排序
all_df_sorted = all_df.sort_values(by='Training Time', ascending=True)

# 绘制带置信区间的柱状图
plt.figure(figsize=(8, 6))
sns.barplot(x='Activation Function', y='Training Time', data=all_df_sorted, ci="sd", palette="Set2")

# 设置图表的标题和标签
plt.xlabel('Activation Function')
plt.ylabel('Training Time (seconds)')
plt.title('Average Training Time with Confidence Interval for Different Activation Functions')
plt.tight_layout()

# 显示图表
plt.show()
