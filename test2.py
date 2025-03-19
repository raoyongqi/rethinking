import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin

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

# 读取 NC 文件
def read_nc(nc_file):
    # 打开 NC 文件
    nc_data = Dataset(nc_file)
    
    # 读取特定的变量 "AWT_SOC" 和 "DOM_MU"
    awt_soc_data = nc_data.variables['AWT_SOC'][:]
    dom_mu_data = nc_data.variables['DOM_MU'][:]
    
    # 关闭文件
    nc_data.close()
    
    return awt_soc_data, dom_mu_data

# 读取数据
tif_data, transform = read_tif('new/bio_13.tif')
# 检查 TIF 数据中的无穷大和 NaN
print(np.any(np.isinf(tif_data)))  # 是否有无穷大
print(np.any(np.isnan(tif_data)))  # 是否有 NaN

awt_soc, dom_mu = read_nc('HWSD_1247/HWSD_1247/data/HWSD_SOIL_CLM_RES.nc4')
# 检查 NC 数据中的无穷大和 NaN
print(np.any(np.isinf(awt_soc)))  # 是否有无穷大
print(np.any(np.isnan(awt_soc)))  # 是否有 NaN
print(np.any(np.isinf(dom_mu)))  # 是否有无穷大
print(np.any(np.isnan(dom_mu)))  # 是否有 NaN

# 假设 target_shape 已经定义
target_shape = awt_soc[0].shape

def resample_raster(tif_data, transform, target_shape):
    with rasterio.open(tif_data) as src:
        
        
        target_width, target_height = target_shape
        
        resampled_data = src.read(
            out_shape=(src.count, target_height, target_width),
            resampling=Resampling.bilinear  # 这里使用 bilinear 插值，其他方法包括 nearest, cubic 等
        )
        
        transform_resampled = src.transform  # 保留原来的坐标转换

        return resampled_data[0], transform_resampled


resampled_tif, transform_resampled = resample_raster('new/bio_13.tif', transform, awt_soc[0].shape)



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

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_X, train_y)

X = np.stack([resampled_tif.flatten(), awt_soc[0].flatten()], axis=1)  # 输入特征
rows, cols = awt_soc[0].shape

print(X.shape)
print(np.any(np.isinf(X)))  # 是否有无穷大
print(np.any(np.isnan(X)))  # 是否有 NaN

y_pred = rf.predict(X)

y_pred_2d = y_pred.reshape((rows, cols))

output_tif = 'data/predicted_output.tif'

with rasterio.open(output_tif, 'w', driver='GTiff', count=1, dtype='float32', 
                   width=cols, height=rows, crs='+proj=latlong', transform=transform_resampled) as dst:
    dst.write(y_pred_2d, 1)


def get_tif_min_max(tif_file):
    with rasterio.open(tif_file) as src:

        tif_data = src.read(1)       
        max_value = np.nanmax(tif_data)  # 最大值
        min_value = np.nanmin(tif_data)  # 最小值
        
        return min_value, max_value



tif_file = 'data/predicted_output.tif'
min_val, max_val = get_tif_min_max(tif_file)

print(f"Minimum value: {min_val}")
print(f"Maximum value: {max_val}")


with rasterio.open(output_tif) as src:
    pred_data = src.read(1)
    plt.imshow(pred_data, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # 显示颜色条
    plt.title('Predicted 2D Data (from TIF)')  # 设置图标题
    plt.show()
