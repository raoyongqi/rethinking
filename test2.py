import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import rasterio
import os
import matplotlib.pyplot as plt

# 读取 CSV 文件
def read_csv(csv_file):
    data = pd.read_csv(csv_file)
    return data

# 读取 TIF 文件
def read_tif(tif_file):
    with rasterio.open(tif_file) as src:
        tif_data = src.read(1)  # 读取栅格数据
        transform = src.transform  # 获取坐标转换
        return tif_data, transform

# 读取多个 TIF 文件
def read_multiple_tifs(tif_files):
    tif_data_list = []
    transform = None
    for tif_file in tif_files:
        tif_data, transform = read_tif(tif_file)
        tif_data_list.append(tif_data)
    return tif_data_list, transform

# 读取多个 TIF 文件
tif_files = ['new/bio_13.tif', 'new/dom_mu.tif']  # 你可以添加多个 TIF 文件
tif_data_list, transform = read_multiple_tifs(tif_files)

# 检查每个 TIF 数据中的无穷大和 NaN
for i, tif_data in enumerate(tif_data_list):
    print(f"TIF {i+1}:")
    print(f"Contains Inf: {np.any(np.isinf(tif_data))}")
    print(f"Contains NaN: {np.any(np.isnan(tif_data))}")

# 读取训练数据 CSV
train_df = pd.read_csv("data/selection.csv")
train_df = train_df.rename(columns={
    'hand_500m_china_03_08': 'hand',
    'hwsd_soil_clm_res_dom_mu': 'dom_mu',
    'hwsd_soil_clm_res_awt_soc': 'awt_soc'
})

# 如果某个列存在，则重命名
if 'hwsd_soil_clm_res_pct_clay' in train_df.columns:
    train_df = train_df.rename(columns={'hwsd_soil_clm_res_pct_clay': 'pct_clay'})

# 3. 分离特征变量和目标变量
train_X = train_df[['dom_mu', 'bio_13']]
train_y = train_df['pathogen load']  # 目标变量

# 训练随机森林模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_X, train_y)

# 将多个 TIF 数据合并为一个特征矩阵
rows, cols = tif_data_list[0].shape
X = np.stack([tif_data.flatten() for tif_data in tif_data_list], axis=1)  # 使用多个 TIF 文件作为输入特征

# 检查 X 中是否存在无穷大或 NaN
print(X.shape)
print(np.any(np.isinf(X)))  # 是否有无穷大
print(np.any(np.isnan(X)))  # 是否有 NaN

# 进行预测
y_pred = rf.predict(X)

# 将预测结果 reshape 为 2D 数据
y_pred_2d = y_pred.reshape((rows, cols))

# 保存预测结果到 TIF 文件
output_tif = 'data/predicted_output.tif'

with rasterio.open(output_tif, 'w', driver='GTiff', count=1, dtype='float32', 
                   width=cols, height=rows, crs='+proj=latlong', transform=transform) as dst:
    dst.write(y_pred_2d, 1)

# 获取 TIF 文件的最小值和最大值
def get_tif_min_max(tif_file):
    with rasterio.open(tif_file) as src:
        tif_data = src.read(1)       
        max_value = np.nanmax(tif_data)  # 最大值
        min_value = np.nanmin(tif_data)  # 最小值
        
        return min_value, max_value

# 打印预测结果的最小值和最大值
min_val, max_val = get_tif_min_max(output_tif)

print(f"Minimum value: {min_val}")
print(f"Maximum value: {max_val}")

# 显示预测结果图像
with rasterio.open(output_tif) as src:
    pred_data = src.read(1)
    plt.imshow(pred_data, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # 显示颜色条
    plt.title('Predicted 2D Data (from TIF)')  # 设置图标题
    plt.show()
