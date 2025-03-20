import rasterio
import matplotlib.pyplot as plt

# 指定你的.tif文件路径
tiff_file = 'new/hand.tif'

# 读取.tif文件
with rasterio.open(tiff_file) as src:
    # 读取第一波段数据
    band = src.read(1)

# 创建一个图形，显示.tif数据
plt.figure(figsize=(8, 6))
plt.imshow(band, cmap='viridis')  # 使用 'viridis' 颜色映射
plt.colorbar(label='Value')  # 显示颜色条，表示数值
plt.title('Raster Data Visualization')  # 图标题
plt.xlabel('Column Index')
plt.ylabel('Row Index')
plt.show()
