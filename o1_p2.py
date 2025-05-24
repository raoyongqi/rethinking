import optuna
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.layers import BatchNormalization, concatenate
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'Arial',
    'axes.labelsize': 16,
    'axes.titlesize': 18
})

# 读取数据
data = pd.read_excel('data/merged_all.xlsx')
X = data.drop(columns=['Pathogen Load']).fillna(data.mean())
y = data['Pathogen Load'].fillna(data['Pathogen Load'].mean())
X = X.fillna(X.mean())
y = y.fillna(y.mean())
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

# 运行3次独立研究并收集参数重要性
all_importances = []
for run in tqdm(range(3), desc="Optimizing"):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, timeout=600)
    
    # 计算本次运行的参数重要性
    importances = optuna.importance.get_param_importances(study)
    importance_df = pd.DataFrame({
        'parameter': list(importances.keys()),
        'importance': list(importances.values()),
        'run': run
    })
    all_importances.append(importance_df)

# 合并所有重要性结果
combined_importances = pd.concat(all_importances)

# 保存结果到JSON
with open('param_importances.json', 'w') as f:
    json.dump({
        'runs': combined_importances.to_dict('records'),
        'mean_importance': combined_importances.groupby('parameter')['importance'].mean().to_dict()
    }, f, indent=4)

print("参数重要性已保存为 param_importances.json")

# 准备绘图数据
plot_data = combined_importances.groupby('parameter').agg(
    mean_importance=('importance', 'mean'),
    std_importance=('importance', 'std'),
    count=('importance', 'count')
).reset_index().sort_values('mean_importance', ascending=True)

# 创建带误差线的横向柱状图
plt.figure(figsize=(12, 8))
sns.barplot(
    data=plot_data,
    x='mean_importance',
    y='parameter',
    xerr=plot_data['std_importance'],
    orient='h',
    palette='viridis',
    alpha=0.7,
    edgecolor='black',
    linewidth=1
)

# 美化图表
plt.title('Hyperparameter Importance Across 3 Optimization Runs\n(Mean ± Standard Deviation)', pad=20)
plt.xlabel('Relative Importance')
plt.ylabel('Hyperparameter')
plt.xlim(0, 1)
plt.grid(axis='x', linestyle='--', alpha=0.3)

# 添加数值标签
for i, row in plot_data.iterrows():
    plt.text(
        row['mean_importance'] + 0.02, i,
        f"{row['mean_importance']:.2f} ± {row['std_importance']:.2f}",
        va='center',
        fontsize=12
    )

# 调整布局并保存
plt.tight_layout()
plt.savefig('data/parameter_importance_plot.png', dpi=300, bbox_inches='tight')
plt.show()