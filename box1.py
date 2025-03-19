import rasterio
from rasterio.mask import mask
import os
import geopandas as gpd
from shapely.geometry import box

def get_tif_bounds(tif_file):
    with rasterio.open(tif_file) as src:
        transform = src.transform
        width = src.width
        height = src.height
        
        # 计算图像的四个角坐标 (左上, 右上, 左下, 右下)
        left = transform[2]
        top = transform[5]
        right = left + transform[0] * width
        bottom = top + transform[4] * height
        
        return (left, bottom, right, top)

import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box

def crop_tif(tif_file, bounds, output_file):
    # 创建边界框，作为 GeoDataFrame
    minx, miny, maxx, maxy = bounds
    bbox = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")

    with rasterio.open(tif_file) as src:
        # 将边界框重新投影到与 TIFF 文件相同的 CRS
        gdf = gdf.to_crs(src.crs)
        
        # 使用边界框掩膜裁剪 TIFF 文件
        out_image, out_transform = mask(src, gdf.geometry, crop=True)

        output_folder = os.path.dirname(output_file)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 检查输出图像的形状
        if len(out_image.shape) == 3:  # (波段数, 高度, 宽度)
            # 只有一个波段
            out_image = out_image[0, :, :]  # 提取第一个波段的二维数据
            # 写入文件
            with rasterio.open(output_file, 'w', driver='GTiff',
                               count=1, dtype=out_image.dtype,
                               crs=src.crs, transform=out_transform,
                               width=out_image.shape[1], height=out_image.shape[0]) as dst:
                dst.write(out_image, 1)

        elif len(out_image.shape) == 4:  # (波段数, 通道数, 高度, 宽度)
            # 处理多波段或单通道的图像
            # 写入文件
            with rasterio.open(output_file, 'w', driver='GTiff',
                               count=src.count, dtype=out_image.dtype,
                               crs=src.crs, transform=out_transform,
                               width=out_image.shape[3], height=out_image.shape[2]) as dst:
                for i in range(1, src.count + 1):
                    dst.write(out_image[i-1, 0, :, :], i)  # 写入每个波段



def crop_all_tifs_in_folder(input_folder, output_folder, bounds):
    # Iterate over all TIFF files in the folder
    for tif_filename in os.listdir(input_folder):
        if tif_filename.endswith(".tif"):
            input_tif = os.path.join(input_folder, tif_filename)
            output_tif = os.path.join(output_folder, tif_filename)
            crop_tif(input_tif, bounds, output_tif)

# 获取要裁剪的 TIFF 文件的经纬度范围
tif_file = 'hand/hand_500m_china_03_08.tif'
bounds = get_tif_bounds(tif_file)
input_folder = 'new'  # 替换为您的输入文件夹路径
output_folder = 'china'  # 替换为您的输出文件夹路径

# 对输入文件夹中的所有 TIFF 文件进行裁剪
crop_all_tifs_in_folder(input_folder, output_folder, bounds)
