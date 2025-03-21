import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import time

# 加载数据
file_path = 'data/merged_all.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 处理列名
feature_columns = [col for col in data.columns if col != 'Pathogen Load']
X = data[feature_columns]
y = data['Pathogen Load']

# 对数据进行划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# 定义添加BatchNormalization的模型
def build_model_with_batchnorm():
    input_x = tf.keras.layers.Input(shape=(144,))
    x = tf.keras.layers.Reshape(target_shape=[144, 1])(input_x)

    num_blocks = 2
    dilation_rates = (1, 2, 4, 8, 16, 32)

    # 添加卷积层和BatchNormalization
    for _ in range(num_blocks):
        for rate in dilation_rates:
            x = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation="elu", dilation_rate=rate, padding='valid')(x)
            x = tf.keras.layers.BatchNormalization()(x)  # 添加 BatchNormalization 层

    # 添加Dropout层
    x = tf.keras.layers.Dropout(0.2)(x)

    # 添加卷积层和Flatten层
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)  # 添加 BatchNormalization 层
    x = tf.keras.layers.Flatten()(x)

    # 输出层
    x = tf.keras.layers.Dense(1)(x)

    # 创建模型
    model = tf.keras.models.Model(inputs=[input_x], outputs=[x])

    # 编译模型
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.0001))
    return model

# 定义不添加BatchNormalization的模型
def build_model_without_batchnorm():
    input_x = tf.keras.layers.Input(shape=(144,))
    x = tf.keras.layers.Reshape(target_shape=[144, 1])(input_x)

    num_blocks = 2
    dilation_rates = (1, 2, 4, 8, 16, 32)

    # 添加卷积层
    for _ in range(num_blocks):
        for rate in dilation_rates:
            x = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation="elu", dilation_rate=rate, padding='valid')(x)

    # 添加Dropout层
    x = tf.keras.layers.Dropout(0.2)(x)

    # 添加卷积层和Flatten层
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=1)(x)
    x = tf.keras.layers.Flatten()(x)

    # 输出层
    x = tf.keras.layers.Dense(1)(x)

    # 创建模型
    model = tf.keras.models.Model(inputs=[input_x], outputs=[x])

    # 编译模型
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.0001))
    return model

# 定义训练和评估过程
def train_and_evaluate(model, X_train, y_train, X_valid, y_valid):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto', restore_best_weights=True)
    start_time = time.time()  # 记录训练开始时间
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=32, callbacks=[es], verbose=0)
    end_time = time.time()  # 记录训练结束时间
    training_time = end_time - start_time  # 计算训练时间
    return history, training_time



# 定义多次训练的函数
def train_multiple_times(model_builder, X_train, y_train, X_valid, y_valid, num_trials=5, model_type='BatchNorm'):

    
    # 多次训练模型
    for trial in range(1, num_trials + 1):
        print(f"\nStarting trial {trial} of {num_trials}...")
        
        # 使用模型构建器函数来创建模型
        model = model_builder()
        
        # 训练并评估模型
        history, training_time = train_and_evaluate(model, X_train, y_train, X_valid, y_valid)
        
        # 将history.history转为pandas DataFrame
        history_df = pd.DataFrame(history.history)  # 直接将history.history转换为DataFrame
        
        # 保存每次训练的loss和其他指标到CSV文件
        history_df.to_csv(f"data/{model_type}_trial_{trial}_history.csv", index=True)  # 不保存索引
        
        # 保存每次训练的loss到文件中
        np.savetxt(f"data/{model_type}_trial_{trial}_time.txt", [training_time])
    

    return True

num_trials = 3 
train_multiple_times(build_model_with_batchnorm, X_train_scaled, y_train, X_valid_scaled, y_valid, num_trials, model_type='batchnorm')

train_multiple_times(build_model_without_batchnorm, X_train_scaled, y_train, X_valid_scaled, y_valid, num_trials, model_type='without_batchnorm')

# 先弄清数据格式，再进行下一步操作W


# # 合并所有试验的损失数据
# history_with_batchnorm_combined = np.concatenate(history_with_batchnorm_multiple)
# history_without_batchnorm_combined = np.concatenate(history_without_batchnorm_multiple)


# # 绘制带置信区间的拟合线
# with plt.style.context('science'):
#     # 创建一个新的图形
#     fig, axs = plt.subplots(1, 2, figsize=(18, 6))

#     # 拟合损失曲线
#     # 绘制每次训练的损失曲线
#     for i in range(num_trials):
#         sns.lineplot(x=np.arange(len(history_with_batchnorm_multiple[i])), 
#                      y=history_with_batchnorm_multiple[i], 
#                      label=f'Trial {i+1} (With BatchNormalization)', 
#                      color='blue', ci=None, ax=axs[0]) 

#     for i in range(num_trials):
#         sns.lineplot(x=np.arange(len(history_without_batchnorm_multiple[i])), 
#                      y=history_without_batchnorm_multiple[i], 
#                      label=f'Trial {i+1} (Without BatchNormalization)', 
#                      color='red', ci=None, ax=axs[0])

#     # 添加标题和标签
#     axs[0].set_title('MSE Loss Curve with and without BatchNormalization', fontsize=16)
#     axs[0].set_xlabel('Epochs', fontsize=14)
#     axs[0].set_ylabel('MSE Loss', fontsize=14)
#     axs[0].legend(fontsize=12)
#     axs[0].tick_params(axis='both', which='major', labelsize=12)
#     axs[0].grid(True)

#     # 绘制训练时间柱状图
#     times_mean = [np.mean(times_with_batchnorm), np.mean(times_without_batchnorm)]
#     times_std = [np.std(times_with_batchnorm), np.std(times_without_batchnorm)]
#     sns.barplot(x=['With BatchNormalization', 'Without BatchNormalization'], y=times_mean, yerr=times_std, ax=axs[1], palette='Blues', orient='h')  # 设置orient='h'横向

#     # 添加标题和标签
#     axs[1].set_title('Training Time with and without BatchNormalization', fontsize=16)
#     axs[1].set_xlabel('Model Type', fontsize=14)
#     axs[1].set_ylabel('Training Time (seconds)', fontsize=14)
#     axs[1].tick_params(axis='both', which='major', labelsize=12)
    
#     # 显示图形
#     plt.tight_layout()
#     plt.savefig("data/combined_plot_with_confidence_intervals.png", dpi=300)
#     plt.show()
