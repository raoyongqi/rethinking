import rasterio
from rasterio.enums import Resampling
import subprocess

# 读取 TIFF 文件的分辨率
def get_resolution(tif_file):
    with rasterio.open(tif_file) as src:
        transform = src.transform
        dx = transform[0]  # X方向分辨率
        dy = -transform[4]  # Y方向分辨率
    return dx, dy

# 重采样 TIFF 文件到目标分辨率
def resample_tif(input_tif, target_tif, target_dx, target_dy):
    with rasterio.open(input_tif) as src:
        # 获取输入文件的其他信息
        transform = src.transform
        crs = src.crs
        width = src.width
        height = src.height
        
        # 计算新的分辨率
        new_transform = transform * transform.scale((transform[0] / target_dx), (transform[4] / target_dy))
        
        # 执行重采样
        with rasterio.open(target_tif, 'w', driver='GTiff', height=height, width=width, count=1, dtype=src.dtypes[0], crs=crs, transform=new_transform) as dst:
            for i in range(1, src.count + 1):  # 假设处理单波段图像
                data = src.read(i)
                dst.write(data, i)
    print(f"重采样后的文件已保存为: {target_tif}")

# 比较分辨率并进行重采样
def compare_and_resample(tif1, tif2):
    # 获取两个 TIFF 文件的分辨率
    dx1, dy1 = get_resolution(tif1)
    dx2, dy2 = get_resolution(tif2)

    print(f"tif1 分辨率: dx={dx1}, dy={dy1}")
    print(f"tif2 分辨率: dx={dx2}, dy={dy2}")

    # 如果分辨率不同，执行重采样
    if dx1 != dx2 or dy1 != dy2:
        print("分辨率不同，进行重采样")
        # 将 tif1 重采样到 tif2 的分辨率
        resample_tif(tif1, 'new/hand.tif', dx2, dy2)
    else:
        print("两个 TIFF 文件的分辨率相同，无需重采样。")

# 示例文件
tif1 = 'hand/hand_500m_china_03_08.tif'
tif2 = 'new/bio_13.tif'

# 比较并进行重采样
compare_and_resample(tif1, tif2)
