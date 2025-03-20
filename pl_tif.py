import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeatur
import os
from matplotlib.colors import LinearSegmentedColormap


# 配置字体
config = {
    "font.family": 'Arial',
    'font.size': 24,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
    'font.weight': 'bold'
}

mpl.rcParams.update(config)

mpl.rcParams['axes.unicode_minus'] = False

# 读取 SHP 作为底图
bound_file = gpd.read_file('GS(2020)4619/GS(2020)4619.shp')

import geopandas as gpd

map_china_path = r'GS(2020)4619\\GS(2020)4619_4326_map.shp'

map_china_gdf = gpd.read_file(map_china_path)

def point_plot(pre_title: str, tif_file: str):

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([73.2, 135, 17, 55], crs=ccrs.PlateCarree()) 


    ax.add_feature(cfeatur.OCEAN.with_scale('10m'), zorder=2)
    ax.add_feature(cfeatur.LAKES.with_scale('10m'), zorder=2)


    bound_file.to_crs("EPSG:4326").plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)


    if map_china_gdf is not None:
        map_china_gdf.to_crs("EPSG:4326").plot(ax=ax, color='none', edgecolor='blue', linewidth=1)


    with rasterio.open(tif_file) as src:
        data = src.read(1)
        no_data_value = src.nodata

        # 处理无效值
        if no_data_value is not None:
            data = np.where(data == no_data_value, np.nan, data)

        # 获取原始的投影信息
        crs = src.crs
        transform = src.transform

        # 如果 TIFF 的投影不是 PlateCarree，则需要转换
        if crs.to_string() != 'EPSG:4326':
            # 这里我们使用 cartopy 的 crs 转换
            from rasterio.warp import reproject, Resampling
            from rasterio.enums import Resampling
            from cartopy.crs import PlateCarree

            # 创建新的目标 CRS（WGS84）
            target_crs = 'EPSG:4326'

            # 重新投影到目标 CRS
            data_reprojected = np.empty_like(data)
            reproject(
                data,
                data_reprojected,
                src_transform=src.transform,
                src_crs=crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )
            data = data_reprojected

        # 重新计算 extent 确保投影一致
        bounds = [transform * (0, 0), transform * (src.width, src.height)]
        extent = [bounds[0][0], bounds[1][0], bounds[1][1], bounds[0][1]]



        # 颜色映射
        clist = ['blue', 'limegreen', 'orange', 'red']

        newcmap = LinearSegmentedColormap.from_list('mycolmap', clist, N=1024)

        im = ax.imshow(data, cmap=newcmap, interpolation='none', extent=extent, transform=ccrs.PlateCarree(), alpha=1)
        
        cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.07, ax.get_position().width, 0.02]) # Set the length, width, height and position of the color bar
        
        plt.colorbar(im, cax=cax, orientation = 'horizontal')

    ax2 = fig.add_axes([0.805, 0.224, 0.1, 0.15], projection=ccrs.PlateCarree())
    ax2.set_extent([104.5, 124, 0, 26], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeatur.OCEAN.with_scale('10m'), zorder=2)

    ax2.imshow(data, cmap=newcmap, interpolation='none', extent=extent, transform=ccrs.PlateCarree(), alpha=1)
    bound_file.to_crs("EPSG:4326").plot(ax=ax2, color='none', edgecolor='black', linewidth=0.8)
    if map_china_gdf is not None:
        map_china_gdf.to_crs("EPSG:4326").plot(ax=ax2, color='none', edgecolor='blue', linewidth=0.8)

    ax2.gridlines(draw_labels=False, linestyle='--', lw=0.3)

    output_file = f'{pre_title}1.png'

    plt.savefig(output_file, dpi=2800, bbox_inches='tight')

    plt.show()

tif_file_path = 'data/cropped_predicted_output.tif'

point_plot('data/China_TIFF', tif_file_path)
