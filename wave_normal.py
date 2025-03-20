import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载数据
file_path = 'data/selection.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 处理列名
# 检查每一列的NaN值
nan_counts = data.isnull().sum()  # 返回每列中NaN值的个数
print("每列的NaN值数量：")
print(nan_counts)

# 找出包含NaN值的列
nan_columns = nan_counts[nan_counts > 0]
print("\n包含NaN值的列：")
print(nan_columns)

feature_columns = [col for col in data.columns if col != 'pathogen load']

X = data[feature_columns]
y = data['pathogen load']

# 对数据进行划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)

scaler = StandardScaler()

# 不知道是不是数据标准化导致的梯度爆炸

# 先进行标准化

# 为了定位问题，可能还是需要比较xlsx之间的差异

# 标准化后R方明显提高了，可能暂时不需要定位问题

# 标准化之后不会出现梯度爆炸的情况


X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# 构建模型
input_x = tf.keras.layers.Input(shape=(16,))
x = tf.keras.layers.Reshape(target_shape=[16, 1])(input_x)

num_blocks = 2
dilation_rates = (1, 2, 4)

# 添加卷积层
for _ in range(num_blocks):
    for rate in dilation_rates:
        x = tf.keras.layers.Conv1D(filters=8, kernel_size=2, activation="elu", dilation_rate=rate, padding='valid')(x)

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

# 使用EarlyStopping防止过拟合
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=1, mode='auto', restore_best_weights=True)

# 训练模型
history = model.fit(X_train_scaled, y_train, validation_data=(X_valid_scaled, y_valid), epochs=1000, batch_size=32, callbacks=[es], verbose=0)


# 定义评估函数
def evaluate(y_true, y_pred):
    print(y_true.shape)
    print(y_pred.shape)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rpd = np.std(y_true) / np.sqrt(mse)
    
    return mse, r2, rpd

def evaluate_model(model, X, y):
    pred = model.predict(X)
    print(y.shape)
    print(pred.shape)

    if(tf.is_tensor(pred)):
        pred = pred.numpy()
    if np.any(np.isnan(y.squeeze())):
        print("y 中有 NaN 值")
    else:
        print("y 中没有 NaN 值")

    if np.any(np.isnan(pred.squeeze())):
        print("pred 中有 NaN 值")
    else:
        print("pred 中没有 NaN 值")

    return pred, evaluate(y.squeeze(), pred.squeeze())

def evaluation(X_valid, y_valid, model):
    y_pred, (mse, r2, rpd) = evaluate_model(model, X_valid, y_valid)
    
    print(f"R-squared: {r2}")
    
    if isinstance(y_valid, (pd.DataFrame, pd.Series)):
        y_valid = y_valid.values
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.values

    y_valid = np.array(y_valid).reshape(-1, 1) if y_valid.ndim == 1 else np.array(y_valid)
    y_pred = np.array(y_pred).reshape(-1, 1) if y_pred.ndim == 1 else np.array(y_pred)

    # 保存实际值与预测值的对比
    results_df = pd.DataFrame({
        'Actual': y_valid[:, 0],
        'Predicted': y_pred[:, 0]
    })
    results_df.to_csv("wavenet2_actual_vs_predicted.csv", index=False)

    # 保存训练历史
    train_df = pd.DataFrame(history.history)
    train_df.to_csv("wavenet2_training_history.csv", index=False)

    return y_pred, mse, r2, rpd


# 进行模型评估
evaluation(X_valid_scaled, y_valid, model)
