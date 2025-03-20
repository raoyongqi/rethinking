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

import matplotlib.pyplot as plt
flip_tifs = {'new/dom_mu.tif', 'new/awt_soc.tif','new/pct_clay.tif', 'new/s_sand.tif', 'new/t_sand.tif'}

def read_tif(tif_file):
    with rasterio.open(tif_file) as src:
        tif_data = src.read(1)  # 读取栅格数据
        transform = src.transform  # 获取坐标变换信息

        # **仅在文件名属于 flip_tifs 时翻转**
        if tif_file in flip_tifs:
            tif_data = np.flipud(tif_data)  # 进行上下翻转
            print(f"🔄 TIF 文件 {tif_file} 进行了上下翻转")

        return tif_data, transform

# 读取多个 TIF 并绘制
def read_multiple_tifs(tif_files):
    tif_data_list = []
    shapes = []  
    transform = None

    for tif_file in tif_files:
        tif_data, transform = read_tif(tif_file)
        tif_data_list.append(tif_data)
        shapes.append(tif_data.shape)
        
        # 打印 TIF 形状
        print(f"TIF 文件: {tif_file}, 形状: {tif_data.shape}")



    return tif_data_list, transform

def get_coordinates(rows, cols, transform):
    lon, lat = np.meshgrid(
        np.arange(0, cols) * transform[0] + transform[2],
        np.arange(0, rows) * transform[4] + transform[5]
    )
    return lon.flatten(), lat.flatten()

train_df = pd.read_csv("data/selection.csv")
train_df = train_df.rename(columns={
    'hand_500m_china_03_08': 'hand',
    'hwsd_soil_clm_res_dom_mu': 'dom_mu',
    'hwsd_soil_clm_res_awt_soc': 'awt_soc'
})

if 'hwsd_soil_clm_res_pct_clay' in train_df.columns:
    train_df = train_df.rename(columns={'hwsd_soil_clm_res_pct_clay': 'pct_clay'})

tif_files = ['new/' + col + '.tif' for col in train_df.columns[2:-1]]
tif_data_list, transform = read_multiple_tifs(tif_files)


rows, cols = tif_data_list[0].shape


lon, lat = get_coordinates(rows, cols, transform)


X = np.stack([tif_data.flatten() for tif_data in tif_data_list], axis=1) 


coordinates = np.stack([lon, lat], axis=1)
X = np.concatenate([coordinates,X ], axis=1)

X_df = pd.DataFrame(X, columns=train_df.columns[:-1])

train_X =  train_df.drop(columns=['pathogen load'])
train_y = train_df['pathogen load']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_X, train_y)


y_pred = rf.predict(X_df)

y_pred_2d = y_pred.reshape((rows, cols))

output_tif = 'data/predicted_output.tif'

with rasterio.open(output_tif, 'w', driver='GTiff', count=1, dtype='float32', 
                   width=cols, height=rows, crs='+proj=latlong', transform=transform) as dst:
    dst.write(y_pred_2d, 1)

def get_tif_min_max(tif_file):
    with rasterio.open(tif_file) as src:
        tif_data = src.read(1)       
        max_value = np.nanmax(tif_data)
        min_value = np.nanmin(tif_data)
        
        return min_value, max_value

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
