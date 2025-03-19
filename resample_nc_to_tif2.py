import numpy as np
import rasterio
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import subprocess

# 1. 读取 NetCDF 文件中的特定变量
def read_nc(nc_file, key):
    # 打开 NC 文件
    nc_data = Dataset(nc_file)
    
    # 读取特定的变量
    data = {}

    data[key] = nc_data.variables[key][:]

    lat_nc = nc_data.variables['lat'][:]  # 假设存在 lat 和 lon 变量
    lon_nc = nc_data.variables['lon'][:]
    
    # 关闭文件
    nc_data.close()
    
    return data, lat_nc, lon_nc

# 2. 读取目标 TIFF 文件
def read_tif(tif_file):
    with rasterio.open(tif_file) as src:
        # 获取目标栅格的空间参考信息和形状
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
        tif_data = src.read(1)  # 读取第一波段数据（假设它是一个单波段的 TIFF）

        # 获取 TIFF 文件的经纬度范围
        lon_tif, lat_tif = np.meshgrid(np.linspace(src.bounds[0], src.bounds[2], width), 
                                       np.linspace(src.bounds[1], src.bounds[3], height))
        
        # 获取分辨率（像素大小）
        dx = transform[0]  # X方向分辨率
        dy = -transform[4]  # Y方向分辨率（负数表示方向向下）        
    return lon_tif, lat_tif, transform, width, height, crs, dx, dy

# 3. 将数据保存为 TIFF 文件
def save_to_tif(data, transform, crs, width, height, output_file):
    # 确保 dtype 为 float32
    data = data.astype(np.float32)
    with rasterio.open(output_file, 'w', driver='GTiff', height=height, width=width, count=1, dtype='float32', crs=crs, transform=transform) as dst:
        dst.write(data, 1)

# 4. 绘制保存的 TIFF 文件
def plot_tif(tif_file):
    with rasterio.open(tif_file) as src:
        data = src.read(1)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(data, cmap='viridis', interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.title(f'TIFF: {tif_file}')
    plt.show()

# 5. 执行重采样过程
def resample_nc_to_tif(nc_file, tif_file, keys):
    # 读取 NetCDF 数据
    data, lat_nc, lon_nc = read_nc(nc_file, keys)
    
    # 读取 TIFF 文件，获取其空间信息
    lon_tif, lat_tif, transform, width, height, crs, dx, dy = read_tif(tif_file)
    print(dx, dy)
    
    # 获取沙土数据
    sand_data = data[keys]  # 获取 T_SAND 或 S_SAND 数据
    
    sand_tif = f'data/{keys}.tif'
    
    # 使用 rasterio 将 NetCDF 数据保存为 TIFF
    save_to_tif(sand_data, transform, crs, width, height, sand_tif)
    
    # 使用 gdalwarp 执行重采样
    sand_resampled_file = f'new/{keys.lower()}.tif'
    subprocess.run([  # 对沙土数据进行重采样
        'gdalwarp', 
        '-s_srs', 'EPSG:4326', 
        '-t_srs', 'EPSG:4326',  
        '-r', 'bilinear',  
        '-ot', 'Float32',  # 指定输出数据类型为 Float32
        sand_tif, 
        sand_resampled_file
    ])

    # 打开重采样后的文件并将大于10000000 的值转为 NaN
    def replace_large_values_with_nan(file_path, threshold=10000000):
        with rasterio.open(file_path, 'r+') as src:
            data = src.read(1)  # 读取数据
            max_value = np.nanmax(data)
            print(max_value)
            # 将大于 threshold 的值替换为 NaN
            data[data > threshold] = np.nan
            src.write(data, 1)  # 写回数据

    # 对重采样后的文件应用转换
    replace_large_values_with_nan(sand_resampled_file)

    # 绘制保存的 TIFF 文件
    plot_tif(sand_resampled_file)

    print(f"{keys} 保存为: {sand_resampled_file}")

# 调用函数进行重采样
nc_file_t_sand = 'HWSD_1247/HWSD_1247/data/T_SAND.nc4'
nc_file_s_sand = 'HWSD_1247/HWSD_1247/data/S_SAND.nc4'

tif_file = 'new/bio_13.tif'

# 对 T_SAND.nc4 文件进行重采样
resample_nc_to_tif(nc_file_t_sand, tif_file, "T_SAND")

# 对 S_SAND.nc4 文件进行重采样
resample_nc_to_tif(nc_file_s_sand, tif_file, "S_SAND")
