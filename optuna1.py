import optuna
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import BatchNormalization, concatenate
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time

# 创建optuna结果文件夹
os.makedirs('optuna', exist_ok=True)

plt.rcParams.update({'font.size': 20})  # 设置全局字体大小

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

# 定义回调类用于保存每次试验结果
class TrialCallback:
    def __init__(self):
        self.trial_count = 0
        self.start_time = time.time()
        
    def __call__(self, study, trial):
        self.trial_count += 1
        
        # 获取当前时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建本次试验的DataFrame
        trial_data = {
            'trial_number': trial.number,
            'value': trial.value,
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration_seconds': time.time() - self.start_time,
            **trial.params
        }
        trial_df = pd.DataFrame([trial_data])
        
        # 保存到CSV (追加模式)
        filename = f"optuna/optuna_results_{timestamp}.csv"
        header = not os.path.exists(filename)  # 只在文件不存在时写入表头
        trial_df.to_csv(filename, mode='a', header=header, index=False)
        
        # 每10次试验打印进度
        if self.trial_count % 10 == 0:
            print(f"已完成 {self.trial_count} 次试验，最新结果已保存到 {filename}")

# 定义目标函数
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    num_layers = trial.suggest_int('num_layers', 3, 6)
    num_neurons = trial.suggest_categorical('num_neurons', [64, 128, 256])
    batch_size = trial.suggest_categorical('batch_size', [10, 32, 64])
    epochs = trial.suggest_int('epochs', 100, 1000)

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

# 创建研究并添加回调
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=3000, timeout=300, callbacks=[TrialCallback()])

# 最终汇总保存
print("\n最佳试验结果:")
trial = study.best_trial
print(f"  R2分数: {trial.value:.4f}")
print("  最佳参数:")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 保存完整试验数据
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trials_df = study.trials_dataframe()
trials_df.to_csv(f"optuna/final_summary_{timestamp}.csv", index=False)
print(f"\n所有试验结果已保存到: optuna/final_summary_{timestamp}.csv")