import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

# 打开数据集
dataset = xr.open_dataset('HWSD_1247/HWSD_1247/data/HWSD_SOIL_CLM_RES.nc4')

# 获取经纬度范围
lat_range = dataset['lat'].values
lon_range = dataset['lon'].values

# 创建地图并设置投影
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([lon_range.min(), lon_range.max(), lat_range.min(), lat_range.max()])

# 添加海岸线和网格线
ax.coastlines()
ax.gridlines()

# 在地图上绘制数据（例如，等值线）
data = dataset['DOM_MU'][0, :, :]  # 替换为实际变量名
contour = ax.contourf(lon_range, lat_range, data, transform=ccrs.PlateCarree())

# 显示图例
plt.colorbar(contour)

plt.show()
max_value = data.max().values

# 打印最大值
print(f"DOM_MU 的最大值是: {max_value}")