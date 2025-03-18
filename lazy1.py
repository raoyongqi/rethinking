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
with plt.style.context('science'):
    plt.figure(facecolor='white')

    plt.plot(idx, models["RMSE"][:41]  , label = "RMSE" , marker = 'o' )


    plt.annotate(models.index[0] , 
                (1,models["RMSE"][0]) , 
                xytext  =(3,3),
                arrowprops = dict(
                                arrowstyle = "simple",
                    color = "red"
                                ))
    print(models.index[40] )
    plt.annotate(models.index[40] , 
                (40 , models["RMSE"][40]) ,
                xytext  =(35,5),
                arrowprops = dict(
                                arrowstyle = "simple",
                                    color = "red"

                                ))

    plt.gca().set_facecolor('white')
    plt.tight_layout()

    plt.xlabel("Model Index")
    plt.ylabel("Metrics")
    plt.legend()
    plt.savefig("data/lazy1.png", dpi=300)  # 保存为 PNG 格式，设置较高的 DPI
    plt.show()  # 显示图表


idx = [i for i in range(41)]
with plt.style.context('science'):
    idx = [i for i in range(42)]
    plt.plot(idx, models["Time Taken"] , label = "RMSE" ,marker = "o" )
    plt.xlabel("Model Index")
    plt.ylabel("Time Taken")
    plt.title("Comparison of 42 Different Regressors")
    plt.legend()
    plt.savefig("data/lazy2.png", dpi=300)  # 保存为 PNG 格式，设置较高的 DPI
    plt.show()  # 显示图表
