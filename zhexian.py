import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

random_state = 42

np.random.seed(42)
file_path = "data/selection.csv"

selection = pd.read_csv(file_path)

X = selection.drop(columns='pathogen load')
y = selection['pathogen load']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

ntree_values = np.arange(10, 201, 20)
mtry_values = np.arange(3, 11,3)

rmse_dict = {}
r2_dict = {}

# 输出开始训练的提示信息
print("Starting training...")

for mtry in mtry_values:
    rmse_list = []
    r2_list = []
    for ntree in ntree_values:
        
        # 输出当前的 mtry 和 ntree 值
        print(f"Training with mtry={mtry}, ntree={ntree}...")
        
        rf = RandomForestRegressor(n_estimators=ntree, max_features=mtry, random_state=42)
        
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)
        
        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)
    
    rmse_dict[mtry] = rmse_list
    r2_dict[mtry] = r2_list

# 输出训练完成的提示信息
print("Training completed.")



import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
import seaborn as sns
font_size = 30

plt.rcParams.update({
    'font.size': font_size,
    'font.family': 'Arial'
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
colors = sns.color_palette("deep", len(mtry_values))

# RMSE 曲线
for idx, mtry in enumerate(mtry_values):
    spline = UnivariateSpline(ntree_values, rmse_dict[mtry], s=1)  # s=1 控制平滑度
    
    x_smooth = np.linspace(min(ntree_values), max(ntree_values), 300)
    y_smooth = spline(x_smooth)
    ax1.plot(x_smooth, y_smooth, label=f'mtry={mtry}', linewidth=4, color=colors[idx])

# ax1.set_title(''})
ax1.set_xlabel('ntree', fontdict={'size': 18,})
ax1.set_ylabel('RMSE', fontdict={'size': 18, })
ax1.legend(title='mtry', fontsize=16, title_fontsize=16, loc='best')
ax1.tick_params(axis='both', which='major', labelsize=16)

# R² 曲线
for idx, mtry in enumerate(mtry_values):
    spline = UnivariateSpline(ntree_values, r2_dict[mtry], s=1)
    x_smooth = np.linspace(min(ntree_values), max(ntree_values), 300)
    y_smooth = spline(x_smooth)
    ax2.plot(x_smooth, y_smooth, label=f'mtry={mtry}', linewidth=4, color=colors[idx])

ax2.set_title('')
ax2.set_xlabel('ntree', fontdict={'size': 18})
ax2.set_ylabel('R-Squared', fontdict={'size': 18})
ax2.legend(title='mtry', fontsize=16, title_fontsize=16, loc='best')
ax2.tick_params(axis='both', which='major', labelsize=16)

plt.tight_layout()
plt.savefig("data/treemtry_smooth.png", dpi=300)
plt.show()

rmse_df = pd.DataFrame(rmse_dict, index=ntree_values)
rmse_df.index.name = 'ntree'
rmse_df.to_csv("data/rmse_results.csv")

r2_df = pd.DataFrame(r2_dict, index=ntree_values)
r2_df.index.name = 'ntree'
r2_df.to_csv("data/r2_results.csv")

# rmse_df = pd.DataFrame(rmse_dict, index=ntree_values)
# rmse_df.index.name = 'ntree'
# rmse_df = rmse_df.reset_index().melt(id_vars=['ntree'], var_name='mtry', value_name='MSE')

# r2_df = pd.DataFrame(r2_dict, index=ntree_values)
# r2_df.index.name = 'ntree'
# r2_df = r2_df.reset_index().melt(id_vars=['ntree'], var_name='mtry', value_name='R²')

# sorted_rmse_df = rmse_df.sort_values(by='MSE', ascending=True)
# sorted_r2_df = r2_df.sort_values(by='R²', ascending=False)


# sorted_rmse_df.to_latex("sorted_rmse_results.tex", index=False)
# sorted_r2_df.to_latex("sorted_r2_results.tex", index=False)

print("训练结果已保存为 sorted_rmse_results.csv 和 sorted_r2_results.csv")
