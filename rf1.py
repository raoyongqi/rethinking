import itertools
from sklearn.ensemble import ExtraTreesRegressor  # 改为导入ExtraTrees
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# 定义超参数网格（针对ExtraTrees调整）
hyperparams = {
    'n_estimators': list(range(20, 48, 6)),
    'min_samples_split': list(range(6, 12, 1)),

}

fixed_hyperparams = {
    'random_state': 42,
    'n_jobs': -1  # 使用所有CPU核心
}

def save_results(all_mses, best_hyperparams, best_model, filename_prefix="data/et_results"):  # 修改文件名前缀
    np.save(f"{filename_prefix}_mses.npy", np.array(all_mses))
    with open(f"{filename_prefix}_best_params.json", 'w') as f:
        json.dump(best_hyperparams, f, indent=4)
    with open(f"{filename_prefix}_best_model.pkl", 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Results saved to {filename_prefix}_*.npy/.json/.pkl")

def load_results(filename_prefix="data/et_results"):  # 修改文件名前缀
    try:
        all_mses = np.load(f"{filename_prefix}_mses.npy").tolist()
        with open(f"{filename_prefix}_best_params.json", 'r') as f:
            best_hyperparams = json.load(f)
        with open(f"{filename_prefix}_best_model.pkl", 'rb') as f:
            best_model = pickle.load(f)
        print("Loaded previous results successfully")
        return all_mses, best_hyperparams, best_model
    except FileNotFoundError:
        print("No previous results found")
        return None, None, None

def plot_heatmap(hyperparams, all_mses, best_hyperparams, filename="data/et_heatmap.png"):
    # 准备数据（选择前两个参数进行可视化）
    param1 = list(hyperparams.keys())[0]
    param2 = list(hyperparams.keys())[1]
    values1 = hyperparams[param1]
    values2 = hyperparams[param2]
    
    # 计算每个参数组合的数量
    n_params = len(hyperparams)
    other_params_fixed = {k: best_hyperparams[k] for k in hyperparams if k not in [param1, param2]}
    
    # 提取相关MSE值
    mse_values = []
    for v1 in values1:
        row = []
        for v2 in values2:
            current_params = {param1: v1, param2: v2, **other_params_fixed}
            # 找到匹配的参数组合
            for i, params in enumerate(itertools.product(*hyperparams.values())):
                param_dict = dict(zip(hyperparams.keys(), params))
                if all(param_dict[k] == current_params[k] for k in current_params):
                    row.append(all_mses[i])
                    break
        mse_values.append(row)
    
    mse_matrix = np.array(mse_values)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 20
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(mse_matrix, 
                    annot=True, 
                    fmt=".1f",
                    cmap="YlGnBu_r",
                    xticklabels=values2,
                    yticklabels=values1,
                    cbar_kws={'label': 'MSE'})
    
    # 标记最佳参数
    best_idx1 = values1.index(best_hyperparams[param1])
    best_idx2 = values2.index(best_hyperparams[param2])
    ax.add_patch(plt.Rectangle((best_idx2, best_idx1), 1, 1, fill=False, edgecolor='red', lw=3))
    ax.text(best_idx2+0.3, best_idx1+0.3, '*', 
           ha='center', va='center', color='red', fontsize=20)
    
    ax.set_xlabel(param2, )
    ax.set_ylabel(param1, )
    ax.set_title('ExtraTrees Hyperparameter Tuning', fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def holdout_grid_search(clf, X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams={}):
    all_mses = []
    best_estimator = None
    best_hyperparams = {}
    best_score = float('inf')
    
    param_combinations = list(itertools.product(*hyperparams.values()))
    total = len(param_combinations)

    for i, params in enumerate(param_combinations, 1):
        param_dict = dict(zip(hyperparams.keys(), params))
        estimator = clf(**param_dict, **fixed_hyperparams)
        estimator.fit(X_train, y_train)
        preds = estimator.predict(X_valid)
        mse = mean_squared_error(y_valid, preds)
        all_mses.append(mse)

        print(f'[{i}/{total}] {param_dict}')
        print(f'Val MSE: {mse:.4f}\n')

        if mse < best_score:
            best_score = mse
            best_estimator = estimator
            best_hyperparams = param_dict

    best_hyperparams.update(fixed_hyperparams)
    return all_mses, best_estimator, best_hyperparams

def extra_trees_grid_search(X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams={}):


    all_mses, best_hyperparams, best_et = load_results()
    
    if all_mses is None:
        et = ExtraTreesRegressor
        all_mses, best_et, best_hyperparams = holdout_grid_search(
            et, X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams)
        
        print(f"Best hyperparameters:\n{best_hyperparams}")
        save_results(all_mses, best_hyperparams, best_et)
    
    return all_mses, best_et, best_hyperparams

all_mses, best_et, best_hyperparams = extra_trees_grid_search(
    X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams=fixed_hyperparams)

# 绘制热力图
plot_heatmap(hyperparams, all_mses, best_hyperparams)