import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import BatchNormalization, concatenate

data = pd.read_excel('data/climate_soil_tif.xlsx')  # 替换为你的CSV文件路径

X = data.drop(columns=['RATIO'])  # 输入特征

y = data['RATIO']  # 输出目标


X = X.fillna(X.mean())
y = y.fillna(y.mean())

print(f"Excel 文件有 {data.shape[1]} 列")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def model():

    inputs = layers.Input(shape=(X_train.shape[1],))
    
    inputs_normalize = BatchNormalization()(inputs)
    
    NN1_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    
    NN2_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    
    NN3_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    
    NN4_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    
    NN5_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    
    NN6_Dense = layers.Dense(128, activation='relu')(inputs_normalize)
    
    merge1 = concatenate([NN1_Dense, NN2_Dense]) 
    NN_merge1 = layers.Dense(128, activation='relu')(merge1)
    
    merge2 = concatenate([NN3_Dense, NN4_Dense])
    NN_merge2 = layers.Dense(128, activation='relu')(merge2)
    
    merge3 = concatenate([NN5_Dense, NN6_Dense])
    NN_merge3 = layers.Dense(128, activation='relu')(merge3)
    
    merge_all = concatenate([NN_merge1, NN_merge2, NN_merge3])
    
    output = layers.Dense(1)(merge_all)
    
    return models.Model(inputs=inputs, outputs=output)

model = model()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=10)

y_pred = model(X_test)

y_pred = y_pred.numpy()

r2 = r2_score(y_test, y_pred)


print("R²:", r2)
