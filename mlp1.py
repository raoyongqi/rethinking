import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import BatchNormalization, concatenate

# 读取CSV文件
data = pd.read_excel('data/merged_all.xlsx')  # 替换为你的CSV文件路径

# 假设'pathogen load'是目标列，其他列作为输入
X = data.drop(columns=['Pathogen Load'])  # 输入特征
y = data['Pathogen Load']  # 输出目标

# 检查数据中是否有NaN或Inf值
print(np.any(np.isnan(X)), np.any(np.isinf(X)))
print(np.any(np.isnan(y)), np.any(np.isinf(y)))

# 如果有无效数据，可以进行填充或删除
# 示例：将NaN值填充为列的均值
X = X.fillna(X.mean())
y = y.fillna(y.mean())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义新的神经网络模型
def model():
    # 初始化输入层（13个特征）
    inputs = layers.Input(shape=(X_train.shape[1],))
    
    # 批归一化
    inputs_normalize = BatchNormalization()(inputs)
    
    NN1_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    NN2_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    NN3_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    NN4_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    NN5_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    NN6_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    
    # 合并层
    merge1 = concatenate([NN1_Dense, NN2_Dense]) 
    NN_merge1 = layers.Dense(128, activation='relu')(merge1)
    
    merge2 = concatenate([NN3_Dense, NN4_Dense])
    NN_merge2 = layers.Dense(128, activation='relu')(merge2)
    
    merge3 = concatenate([NN5_Dense, NN6_Dense])
    NN_merge3 = layers.Dense(128, activation='relu')(merge3)
    
    # 全部合并
    merge_all = concatenate([NN_merge1, NN_merge2, NN_merge3])
    
    # 输出层
    output = layers.Dense(1)(merge_all)
    
    return models.Model(inputs=inputs, outputs=output)

# 初始化并编译模型
model = model()

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=1000, batch_size=10)

# 进行预测
y_pred = model(X_test)

# 将y_pred从TensorFlow张量转换为NumPy数组
y_pred = y_pred.numpy()

# 计算R²（决定系数）
r2 = r2_score(y_test, y_pred)
print("R²:", r2)
