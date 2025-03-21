import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载数据
file_path = 'data/merged_all.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 处理列名
nan_counts = data.isnull().sum()
print("每列的NaN值数量：")
print(nan_counts)

# 找出包含NaN值的列
nan_columns = nan_counts[nan_counts > 0]
print("\n包含NaN值的列：")
print(nan_columns)

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

# 模型创建函数，接受激活函数作为参数
def create_model(activation_function):
    input_x = tf.keras.layers.Input(shape=(144,))
    x = tf.keras.layers.Reshape(target_shape=[144, 1])(input_x)

    num_blocks = 2
    dilation_rates = (1, 2, 4, 8, 16, 32)

    # 添加卷积层
    for _ in range(num_blocks):
        for rate in dilation_rates:
            x = tf.keras.layers.Conv1D(filters=32, kernel_size=2, activation=activation_function, dilation_rate=rate, padding='valid')(x)

    # 添加Dropout层
    x = tf.keras.layers.Dropout(0.2)(x)

    # 添加卷积层和Flatten层
    x = tf.keras.layers.Conv1D(filters=32, kernel_size=1)(x)
    x = tf.keras.layers.Flatten()(x)

    # 输出层
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inputs=[input_x], outputs=[x])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.0001))
    return model

# 训练并记录每个激活函数的MSE
def train_and_record_loss(activation_function):
    model = create_model(activation_function)
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', restore_best_weights=True)
    
    # 训练模型
    history = model.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=1000, batch_size=32, callbacks=[es], verbose=0)
    
    return history

# 训练不同激活函数的模型并记录MSE损失
history_elu = train_and_record_loss("elu")
history_relu = train_and_record_loss("relu")
history_leakyrelu = train_and_record_loss(tf.keras.layers.LeakyReLU(alpha=0.3))  # 使用LeakyReLU激活函数
import scienceplots
with plt.style.context('science'):

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))

    # 绘制每种激活函数的损失曲线
    plt.plot(history_elu.history['loss'], label='ELU Activation', color='#ADD8E6')  # 淡蓝色
    plt.plot(history_relu.history['loss'], label='ReLU Activation', color='#006400')  # 墨绿色
    plt.plot(history_leakyrelu.history['loss'], label='LeakyReLU Activation', color='#800080')  # 紫色

    # 添加标题和标签
    plt.title('MSE Loss Curve with Different Activation Functions',fontsize=24) 
    plt.xlabel('Epochs', fontsize=22)  # 设置x轴标签字体大小
    plt.ylabel('MSE Loss', fontsize=22)  # 设置y轴标签字体大小
    plt.legend(fontsize=22)  # 设置图例字体大小
    plt.grid(True)
    plt.savefig("data/relu.png", dpi=300)


    plt.show()
    # 比较MSE损失
mse_elu = history_elu.history['loss'][-1]
mse_relu = history_relu.history['loss'][-1]
mse_leakyrelu = history_leakyrelu.history['loss'][-1]

print(f"MSE with ELU: {mse_elu}")
print(f"MSE with ReLU: {mse_relu}")
print(f"MSE with LeakyReLU: {mse_leakyrelu}")

# 比较哪种激活函数更好
if mse_elu < mse_relu and mse_elu < mse_leakyrelu:
    print("ELU provides the best performance.")
elif mse_relu < mse_elu and mse_relu < mse_leakyrelu:
    print("ReLU provides the best performance.")
else:
    print("LeakyReLU provides the best performance.")
