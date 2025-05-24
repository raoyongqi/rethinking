import optuna
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import BatchNormalization, concatenate
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})  # 设置全局字体大小为14

# 读取数据
data = pd.read_excel('data/merged_all.xlsx')  # 替换为你的文件路径

# 假设'Pathogen Load'是目标列，其他列作为输入
X = data.drop(columns=['Pathogen Load'])  # 输入特征
y = data['Pathogen Load']  # 输出目标

# 检查并处理数据中的NaN值
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义目标函数
def objective(trial):
    # 定义超参数搜索空间
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    num_layers = trial.suggest_int('num_layers', 3, 6)  # 隐藏层数量
    num_neurons = trial.suggest_categorical('num_neurons', [64, 128, 256])  # 每层神经元数量
    batch_size = trial.suggest_categorical('batch_size', [10, 32, 64])  # 批量大小
    epochs = trial.suggest_int('epochs', 100, 1000)  # 训练轮数

    # 构建模型
    inputs = layers.Input(shape=(X_train.shape[1],))
    inputs_normalize = BatchNormalization()(inputs)
    
    dense_layers = []
    for _ in range(num_layers):
        if not dense_layers:
            dense = layers.Dense(num_neurons, activation='relu')(inputs_normalize)
        else:
            dense = layers.Dense(num_neurons, activation='relu')(dense_layers[-1])
        dense_layers.append(dense)
    
    # 合并层
    merge_all = concatenate(dense_layers)
    
    # 输出层
    output = layers.Dense(1)(merge_all)
    
    model = models.Model(inputs=inputs, outputs=output)
    
    # 编译模型
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # 训练模型
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # 验证模型
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()
    r2 = r2_score(y_test, y_pred)
    
    return r2

study = optuna.create_study(direction='maximize')  
study.optimize(objective, n_trials=30, timeout=300)

print("Best trial:")
trial = study.best_trial
print("  Value: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

import optuna.visualization as vis

# fig_importances = vis.plot_param_importances(study)
# fig_importances.update_layout(
#     font=dict(
#         family="Arial",  # 设置字体
#         size=20,        # 全局字体大小
#     ),
#     width=1200,         # 调整宽度（可选）
#     height=800          # 调整高度（可选）
# )
# fig_importances.update_layout(font=dict(size=25))
# fig_importances.write_image("data/param_importances.png", scale=2)  # scale提高分辨率

# fig_importances.show()

fig_plot_parallel_coordinate = vis.plot_parallel_coordinate(study)
# fig_plot_parallel_coordinate.update_layout(
#     font=dict(
#         family="Arial",  # 设置字体
#         size=20,        # 全局字体大小
#     ),
#     width=1200,         
#     height=800,
#         xaxis=dict(
#         tickangle=0 
#     ),
#         margin=dict(  
#         b=25
#     )
# )

import json

# 获取当前的 layout 配置
layout_config = fig_plot_parallel_coordinate.layout

# 转换为可序列化的字典
layout_dict = layout_config.to_plotly_json()

# 保存到 JSON 文件
with open('parallel_coords_layout.json', 'w') as f:
    json.dump(layout_dict, f, indent=4)  # indent=4 让文件更易读
# fig_plot_parallel_coordinate.update_layout(font=dict(size=24))
# fig_plot_parallel_coordinate.show()


# fig_plot_contour = vis.plot_contour(study)
# fig_plot_contour.update_layout(font=dict(size=25))
# fig_plot_contour.show()

# vis.plot_intermediate_values(study).show()

# vis.plot_edf(study).show()

# vis.plot_edf(study).show()