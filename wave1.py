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
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=1000, batch_size=32, callbacks=[es], verbose=0)
    return history

# 训练带BatchNormalization层的模型
model_with_batchnorm = build_model_with_batchnorm()
history_with_batchnorm = train_and_evaluate(model_with_batchnorm, X_train_scaled, y_train, X_valid_scaled, y_valid)

# 训练不带BatchNormalization层的模型
model_without_batchnorm = build_model_without_batchnorm()
history_without_batchnorm = train_and_evaluate(model_without_batchnorm, X_train_scaled, y_train, X_valid_scaled, y_valid)
import scienceplots

with plt.style.context('science'):
    # 绘制MSE损失曲线
# 绘制MSE损失曲线
    plt.figure(figsize=(10, 6))

    # 绘制带BatchNormalization的模型损失曲线
    plt.plot(history_with_batchnorm.history['loss'], label='With BatchNormalization', color='blue')

    # 绘制不带BatchNormalization的模型损失曲线
    plt.plot(history_without_batchnorm.history['loss'], label='Without BatchNormalization', color='red')

    # 添加图标
    plt.title('MSE Loss Curve with and without BatchNormalization', fontsize=24)  # 设置标题字体大小
    plt.xlabel('Epochs', fontsize=22)  # 设置x轴标签字体大小
    plt.ylabel('MSE Loss', fontsize=22)  # 设置y轴标签字体大小
    plt.legend(fontsize=22)  # 设置图例字体大小
    plt.grid(True)

    # 保存高分辨率 PNG 图片
    plt.savefig("data/batch1.png", dpi=300)

    # 显示图形
    plt.show()

