import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              HistGradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor)
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings

warnings.filterwarnings("ignore")

# 1. 读取数据
file_path = 'data/selection.csv'
selection = pd.read_csv(file_path)

X = selection.drop(columns='pathogen load')
y = selection['pathogen load']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 定义模型和参数
models_params = {
    "RandomForestRegressor": (RandomForestRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    }),
    "GradientBoostingRegressor": (GradientBoostingRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),
    "LGBMRegressor": (LGBMRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [-1, 10, 20]
    }),
    "HistGradientBoostingRegressor": (HistGradientBoostingRegressor(random_state=42), {
        'max_iter': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [None, 10, 20]
    }),
    "BaggingRegressor": (BaggingRegressor(random_state=42), {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.75, 1.0]
    }),
    "XGBRegressor": (XGBRegressor(objective="reg:squarederror", random_state=42), {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }),
    "ExtraTreesRegressor": (ExtraTreesRegressor(random_state=42), {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None]
    }),
    "GaussianProcessRegressor": (GaussianProcessRegressor(), {
        'alpha': [1e-10, 1e-5, 1e-2]
    }),
    "DecisionTreeRegressor": (DecisionTreeRegressor(random_state=42), {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }),
    "ExtraTreeRegressor": (ExtraTreeRegressor(random_state=42), {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    })
}

# 3. 训练、超参数搜索和评估
results = []

for model_name, (model, param_grid) in models_params.items():
    print(f"Training {model_name} without GridSearch...")
    
    # No GridSearch training
    model.fit(X_train, y_train)
    y_pred_no_grid = model.predict(X_test)
    mse_no_grid = mean_squared_error(y_test, y_pred_no_grid)
    r2_no_grid = r2_score(y_test, y_pred_no_grid)
    
    results.append({
        'Model': f"{model_name} (No GridSearch)",
        'MSE': round(mse_no_grid, 2),
        'R2': round(r2_no_grid, 2)
    })
    
    print(f"Training {model_name} with GridSearch...")
    
    # GridSearch training
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    y_pred_grid = best_model.predict(X_test)
    mse_grid = mean_squared_error(y_test, y_pred_grid)
    r2_grid = r2_score(y_test, y_pred_grid)
    
    results.append({
        'Model': f"{model_name} (With GridSearch)",
        'MSE': round(mse_grid, 2),
        'R2': round(r2_grid, 2)
    })
    
    print(f"Finished {model_name}. Best Params: {grid_search.best_params_}")

# 4. 结果排序
results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)

# 5. 保存结果到 Excel
output_file = 'model_comparison_results.xlsx'
results_df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")

print("\n--- Model Evaluation Completed! ---")
print(results_df)
