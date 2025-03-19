import rasterio
import numpy as np
from scipy.ndimage import zoom
from netCDF4 import Dataset

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
tif_data, transform = read_tif('new/bio_1.tif')
awt_soc, dom_mu = read_nc('HWSD_1247/HWSD_1247/data/HWSD_SOIL_CLM_RES.nc4')

# 打印形状
print("TIF 数据形状:", tif_data.shape)
print("AWT_SOC 数据形状:", awt_soc[0].shape)

# 使用 scipy.ndimage.zoom 对 TIF 数据进行重采样
# 假设 awt_soc[0] 的形状是目标形状
target_shape = awt_soc[0].shape

# 计算重采样比例
zoom_factor = (target_shape[0] / tif_data.shape[0], target_shape[1] / tif_data.shape[1])

# 进行重采样
resampled_tif = zoom(tif_data, zoom_factor)

