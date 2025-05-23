import lazypredict 
from lazypredict import Supervised 
from lazypredict.Supervised import LazyRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd





data = pd.read_csv("data/selection.csv")
data = data.rename(columns={
    'hand_500m_china_03_08': 'hand',
    'hwsd_soil_clm_res_dom_mu': 'dom_mu',
    'hwsd_soil_clm_res_awt_soc': 'awt_soc'
})

# 如果某个列存在，则重命名
if 'hwsd_soil_clm_res_pct_clay' in data.columns:
    data = data.rename(columns={'hwsd_soil_clm_res_pct_clay': 'pct_clay'})


print("ALL AVAILABLE REGRESSION MODELS:")

for i in range(42):
    print(i+1, lazypredict.Supervised.REGRESSORS[i][0])
reg = LazyRegressor(verbose=0,
                     ignore_warnings=True,
                     custom_metric=None,
                     random_state=12,
                     regressors='all',
                    )

X = data.drop(columns=['pathogen load'])
y = data['pathogen load']


# 分割数据集
X_train,X_valid, y_train, y_valid= train_test_split(X, y, test_size=0.2, random_state=42)
models,predictions = reg.fit(X_train, X_valid, y_train, y_valid)
print(models)


import scienceplots
idx = [i for i in range(41)]

# with plt.style.context('science'):
plt.rcParams['font.family'] = 'Arial'         # 设置字体为 Arial
# plt.rcParams['font.weight'] = 'bold'          # 设置默认字体为加粗
# plt.rcParams['axes.labelweight'] = 'bold'     # 坐标轴标签加粗
# plt.rcParams['axes.titleweight'] = 'bold'     # 标题加粗
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

fig, ax = plt.subplots(figsize=(12, 6))  # 设置宽高比 2:1
ax.set_facecolor('white')

ax.plot(idx, models["RMSE"][:41], label="RMSE", marker='o',linewidth=4)

ax.annotate(models.index[0], 
            (1, models["RMSE"][0]), 
            xytext=(3, 3),                fontsize=26,  # 设置字体大小
            arrowprops=dict(arrowstyle="simple", color="red"))
ax.legend(fontsize=20)
print(models.index[40])
ax.annotate(models.index[40], 
            (40, models["RMSE"][40]), 
            xytext=(35, 5),                fontsize=20,  # 设置字体大小
            arrowprops=dict(arrowstyle="simple", color="red"))

ax.set_xlabel("Model Index", fontsize=20)  # 增大 X 轴字体
ax.set_ylabel("Metrics", fontsize=20)  # 增大 Y 轴字体
ax.tick_params(axis='both', labelsize=20)  # 增大坐标轴刻度字体

plt.tight_layout()
plt.savefig("data/lazy1.png", dpi=300)  # 保存高分辨率 PNG 图片
plt.show()
