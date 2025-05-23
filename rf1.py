

import itertools
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

import scienceplots
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


file_path = 'data/selection.csv'
data = pd.read_csv(file_path)



feature_columns = [col for col in data.columns if col != 'pathogen load']


X = data[feature_columns]


y = data['pathogen load']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=1/9, random_state=42)

def holdout_grid_search(clf, X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams={}):

    all_mses = []

    best_estimator = None

    best_hyperparams = {}
    
    best_score = 1000

    lists = hyperparams.values()
    
    param_combinations = list(itertools.product(*lists))

    total_param_combinations = len(param_combinations)

    for i, params in enumerate(param_combinations, 1):

        param_dict = {}
        for param_index, param_name in enumerate(hyperparams):
            param_dict[param_name] = params[param_index]
            
        estimator = clf(**param_dict, **fixed_hyperparams)

        estimator.fit(X_train, y_train)
        
        preds = estimator.predict(X_valid)
        
        estimator_score = mean_squared_error(y_valid, preds)

        all_mses.append(estimator_score)

        print(f'[{i}/{total_param_combinations}] {param_dict}')
        print(f'Val MSE: {estimator_score}\n')

        if estimator_score < best_score:
            best_score = estimator_score
            best_estimator = estimator
            best_hyperparams = param_dict

    best_hyperparams.update(fixed_hyperparams)
    
    
    return all_mses, best_estimator, best_hyperparams


hyperparams = {
    'max_depth': [10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 700, 1000],
    'n_estimators': [50, 100, 150, 200, 300, 400, 500]
}

fixed_hyperparams = {
    'random_state': 42
}


def random_forest_grid_search(X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams={}):

    rf = RandomForestRegressor

    all_mses, best_rf, best_hyperparams = holdout_grid_search(rf, X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams)

    print(f"Best hyperparameters:\n{best_hyperparams}")
        
    best_hyperparams.update(fixed_hyperparams)
    
    return all_mses, best_rf, best_hyperparams

num_comps = np.arange(1, len(data.columns))



all_mses, best_rf, best_hyperparams = random_forest_grid_search(X_train, y_train, X_valid, y_valid, hyperparams, fixed_hyperparams=fixed_hyperparams)


def plot_rf_metric(scores, objective, yLabel, filename="data/rf_metric.png"):
    font_size = 30
    plt.rcParams.update({
        'font.size': font_size,
        'font.family': 'Arial'
    })
    fig, ax = plt.subplots(figsize=(12, 6))  # 设置宽高比为 2:1
    
    num_configs = np.arange(1, len(scores) + 1)
    ax.plot(num_configs, scores, '-o', color='gray', alpha=0.8, linewidth=4)

    idx = np.argmin(scores) if objective == 'min' else np.argmax(scores)
    ax.plot(num_configs[idx], scores[idx], 'P', color='red', ms=10)

    ax.set_xlabel("Configuration number", fontsize=22)  # 增大 X 轴标签字体
    ax.set_ylabel(yLabel, fontsize=22)  # 增大 Y 轴标签字体

    ax.tick_params(axis='both', labelsize=20)  # 增大坐标轴刻度字体

    plt.tight_layout()  # 防止标签被裁剪
    plt.savefig(filename, dpi=300)  # 保存高分辨率图片
    
    plt.show()
        

plot_rf_metric(all_mses, 'min', 'MSE')