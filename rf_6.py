import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import json
import pickle

# 加载数据
file_path = 'data/selection.csv'
data = pd.read_csv(file_path)
feature_columns = [col for col in data.columns if col != 'pathogen load']
X = data[feature_columns]
y = data['pathogen load']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)

# 定义参数搜索空间
param_grid = {
    'n_estimators': list(range(2, 40, 4)),
    'min_samples_split': list(range(2, 20, 2)),
}

fixed_params = {
    'random_state': 42,
    'n_jobs': -1
}

def greedy_parameter_search(X_train, y_train, X_valid, y_valid, param_grid, fixed_params):
    best_params = fixed_params.copy()
    best_mse = float('inf')
    history = []
    
    # 按照参数重要性排序
    param_order = ['n_estimators', 'min_samples_split']
    
    for param in param_order:
        current_best_value = None
        current_best_mse = best_mse
        
        print(f"\nOptimizing parameter: {param}")
        
        for value in param_grid[param]:
            test_params = best_params.copy()
            test_params[param] = value
            
            model = ExtraTreesRegressor(**test_params)
            model.fit(X_train, y_train)
            preds = model.predict(X_valid)
            mse = mean_squared_error(y_valid, preds)
            
            print(f"  {param}={value}: MSE = {mse:.4f}")
            
            if mse < current_best_mse:
                current_best_mse = mse
                current_best_value = value
        
        if current_best_value is not None:
            best_params[param] = current_best_value
            best_mse = current_best_mse
            print(f"Best {param} = {current_best_value}, MSE = {best_mse:.4f}")
            
            history.append({
                'param': param,
                'best_value': current_best_value,
                'best_mse': best_mse
            })
    
    # 训练最终模型
    final_model = ExtraTreesRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    return best_params, final_model, history

def plot_heatmap(param_grid, best_params, filename="data/heatmap_final.png"):
    # 生成所有参数组合
    params = list(itertools.product(param_grid['n_estimators'], param_grid['min_samples_split']))
    
    # 计算每个组合的MSE
    mse_values = []
    for n_est, min_split in params:
        model = ExtraTreesRegressor(
            n_estimators=n_est,
            min_samples_split=min_split,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        mse = mean_squared_error(y_valid, preds)
        mse_values.append(mse)
    
    # 转换为矩阵形式
    mse_matrix = np.array(mse_values).reshape(
        len(param_grid['n_estimators']), 
        len(param_grid['min_samples_split'])
    )
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        mse_matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu_r",
        xticklabels=param_grid['min_samples_split'],
        yticklabels=param_grid['n_estimators'],
        cbar_kws={'label': 'MSE'}
    )
    
    # 标记最佳参数位置
    best_n_idx = param_grid['n_estimators'].index(best_params['n_estimators'])
    best_min_idx = param_grid['min_samples_split'].index(best_params['min_samples_split'])
    
    ax.add_patch(plt.Rectangle(
        (best_min_idx, best_n_idx), 1, 1,
        fill=False, edgecolor='red', lw=3
    ))
    ax.text(
        best_min_idx + 0.5, best_n_idx + 0.5, '*',
        ha='center', va='center', color='red', fontsize=20
    )
    
    plt.xlabel('min_samples_split')
    plt.ylabel('n_estimators')
    plt.title('ExtraTrees Hyperparameter Tuning\n(Red star marks greedy search result)')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def save_results(params, model, filename_prefix="data/et_results"):
    with open(f"{filename_prefix}_params.json", 'w') as f:
        json.dump(params, f, indent=4)
    with open(f"{filename_prefix}_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"Results saved to {filename_prefix}_* files")

def load_results(filename_prefix="data/et_results"):
    try:
        with open(f"{filename_prefix}_params.json", 'r') as f:
            params = json.load(f)
        with open(f"{filename_prefix}_model.pkl", 'rb') as f:
            model = pickle.load(f)
        print("Loaded previous results")
        return params, model
    except FileNotFoundError:
        print("No previous results found")
        return None, None

# 执行贪心搜索
best_params, best_model, search_history = load_results()

if best_params is None:
    print("Starting greedy parameter search...")
    best_params, best_model, search_history = greedy_parameter_search(
        X_train, y_train, X_valid, y_valid, param_grid, fixed_params)
    save_results(best_params, best_model)

# 绘制热力图
plot_heatmap(param_grid, best_params)

# 评估最终模型
final_preds = best_model.predict(X_valid)
final_mse = mean_squared_error(y_valid, final_preds)
print(f"\nFinal Model MSE on Validation Set: {final_mse:.4f}")
print("Final Parameters:")
print(json.dumps(best_params, indent=4))