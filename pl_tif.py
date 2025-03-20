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
    ax.set_extent([73.2, 135, 17, 55], crs=ccrs.PlateCarree())  # Set extent (latitude, longitude bounds)

    ax.add_feature(cfeatur.OCEAN.with_scale('10m'), zorder=2)
    ax.add_feature(cfeatur.LAKES.with_scale('10m'), zorder=2)

    bound_file.to_crs("EPSG:4326").plot(ax=ax, color='none', edgecolor='black', linewidth=0.8)

    if map_china_gdf is not None:
        map_china_gdf.to_crs("EPSG:4326").plot(ax=ax, color='none', edgecolor='gray', linewidth=1)

    with rasterio.open(tif_file) as src:
        data = src.read(1)
        no_data_value = src.nodata

        # Handle no-data values
        if no_data_value is not None:
            data = np.where(data == no_data_value, np.nan, data)

        # Get the original projection info
        crs = src.crs
        transform = src.transform



        bounds = [transform * (0, 0), transform * (src.width, src.height)]
        extent = [bounds[0][0], bounds[1][0], bounds[1][1], bounds[0][1]]

        # Define color map
        clist = ['blue', 'limegreen', 'orange', 'red']
        newcmap = LinearSegmentedColormap.from_list('mycolmap', clist, N=1024)

        # Plot the raster data with a custom colormap
        im = ax.imshow(data, cmap=newcmap, interpolation='none', extent=extent, transform=ccrs.PlateCarree(), alpha=1)
        
        # Create and position the color bar
        cax = fig.add_axes([ax.get_position().x0, ax.get_position().y0 - 0.07, ax.get_position().width, 0.02])
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')

        # Add label under the color bar
        cax.set_xlabel('Pathogen Load', fontsize=24, fontweight='bold', labelpad=15)

    # Add gridlines and labels for latitude and longitude
    ax.gridlines(draw_labels=True, linestyle='-', lw=0)  # `-` ensures there is no line drawn
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}°E'))  # Format longitude labels
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}°N'))  # Format latitude labels

    # Inset map
    ax2 = fig.add_axes([0.805, 0.224, 0.1, 0.15], projection=ccrs.PlateCarree())
    ax2.set_extent([104.5, 124, 0, 26], crs=ccrs.PlateCarree())
    ax2.add_feature(cfeatur.OCEAN.with_scale('10m'), zorder=2)
    ax2.imshow(data, cmap=newcmap, interpolation='none', extent=extent, transform=ccrs.PlateCarree(), alpha=1)
    bound_file.to_crs("EPSG:4326").plot(ax=ax2, color='none', edgecolor='black', linewidth=0.8)
    if map_china_gdf is not None:
        map_china_gdf.to_crs("EPSG:4326").plot(ax=ax2, color='none', edgecolor='gray', linewidth=0.8)

    ax2.gridlines(draw_labels=False, linestyle='--', lw=0.3)

    output_file = f'{pre_title}1.png'
    plt.savefig(output_file, dpi=2800, bbox_inches='tight')
    plt.show()


tif_file_path = 'data/cropped_predicted_output.tif'

point_plot('data/China_TIFF', tif_file_path)
