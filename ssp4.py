import numpy as np
import rasterio
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
import rasterio
import os
import matplotlib.pyplot as plt

# 读取 CSV 文件并进行列名重命名
train_df = pd.read_csv("data/selection.csv")
train_df = train_df.rename(columns={
    'hand_500m_china_03_08': 'hand',
    'hwsd_soil_clm_res_dom_mu': 'dom_mu',
    'hwsd_soil_clm_res_awt_soc': 'awt_soc'
})

# 处理 bio 列
if 'hwsd_soil_clm_res_pct_clay' in train_df.columns:
    train_df = train_df.rename(columns={'hwsd_soil_clm_res_pct_clay': 'pct_clay'})

cols = train_df.columns

new_order = [col for col in cols if 'bio' not in col and col != 'tmax'] + \
            [col for col in cols if 'bio' in col or col == 'tmax']

# 重新排列 DataFrame 的列
train_df = train_df[new_order]
bio_columns = train_df.filter(like='bio').columns
bio_columns_numeric = bio_columns.str.replace('bio_', '').astype(int)

columns_to_keep = [col for col in train_df.columns if 'bio' not in col and 'tmax' not in col and 'pathogen load' not in col]


tif_files = ['new/' + col + '.tif' for col in columns_to_keep[2:]]

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

def get_coordinates(rows, cols, transform):
    lon, lat = np.meshgrid(
        np.arange(0, cols) * transform[0] + transform[2],
        np.arange(0, rows) * transform[4] + transform[5]
    )
    return lon.flatten(), lat.flatten()

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


tif_data_list, transform = read_multiple_tifs(tif_files)


rows, cols = tif_data_list[0].shape


lon, lat = get_coordinates(rows, cols, transform)

X = np.stack([tif_data.flatten() for tif_data in tif_data_list], axis=1) 

coordinates = np.stack([lon, lat], axis=1)

tif_file = 'CMIP6/ACCESS-CM2/ssp585/2021-2040/wc2.1_5m_bioc_ACCESS-CM2_ssp585_2021-2040.tif'


flip_tifs = {'new/dom_mu.tif', 'new/awt_soc.tif','new/pct_clay.tif', 'new/s_sand.tif', 'new/t_sand.tif'}

def read_tif(tif_file):
    with rasterio.open(tif_file) as src:
        tif_data = src.read(1)  # 读取栅格数据
        transform = src.transform  # 获取坐标变换信息

        # **仅在文件名属于 flip_tifs 时翻转**
        if tif_file in flip_tifs:
            tif_data = np.flipud(tif_data)  # 进行上下翻转
            print(f"TIF 文件 {tif_file} 进行了上下翻转")

        return tif_data, transform
    


with rasterio.open(tif_file) as src:
    # 获取波段数量
    num_bands = src.count
    print(f"该 TIF 文件包含 {num_bands} 个波段。")

    band_data_list = []

    for bio_num in bio_columns_numeric:
        band_index = bio_num  # 假设 bio_columns_numeric 中的数字对应波段的索引
        if band_index <= num_bands:  # 确保索引在有效范围内
            band_data = src.read(band_index)
            band_data_list.append(band_data)
            print(f"提取波段 {band_index} 数据，形状: {band_data.shape}")

    stacked_bands = np.stack([tif_data.flatten() for tif_data in band_data_list], axis=-1)  

    print(f"堆叠后的数据形状: {stacked_bands.shape}")
import rasterio
import numpy as np

def flatten_and_stack_bands(tif_file):
    # 打开 TIF 文件
    with rasterio.open(tif_file) as src:
        # 获取波段数量
        num_bands = src.count
        print(f"该 TIF 文件包含 {num_bands} 个波段。")

        # 初始化一个列表，用于存储每个波段展平后的数据
        flattened_bands = []

        # 遍历每个波段，进行展平
        for band_index in range(1, num_bands + 1):
            band_data = src.read(band_index)
            flattened_band_data = band_data.flatten()  # 将二维数据展平成一维
            flattened_bands.append(flattened_band_data)

        # 将所有展平后的波段数据堆叠到一起，按列堆叠
        X = np.stack(flattened_bands, axis=1)  # 每一列是一个波段的展平数据

        # 打印堆叠后的数据形状
        print(f"堆叠后的数据形状: {X.shape}")

        summed_band = np.sum(X, axis=1)

        summed_band_2d = summed_band.reshape(-1, 1)

        print(f"相加后的数据形状: {summed_band_2d.shape}")

        return summed_band_2d

tif_file = 'CMIP6/ACCESS-CM2/ssp585/2021-2040/wc2.1_5m_tmax_ACCESS-CM2_ssp585_2021-2040.tif'
tmax = flatten_and_stack_bands(tif_file)

X = np.concatenate([coordinates,X,stacked_bands,tmax], axis=1)
X_df = pd.DataFrame(X, columns=[col for col in train_df.columns if 'pathogen load' not in col])

train_X =  train_df.drop(columns=['pathogen load'])
train_y = train_df['pathogen load']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(train_X, train_y)


y_pred = rf.predict(X_df)

y_pred_2d = y_pred.reshape((rows, cols))

output_tif = 'data/predicted_rf_585.tif'

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



import os
import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
import platform
from shapely.geometry import Polygon, MultiPolygon


base_path = os.getcwd()


geojson_file_path = 'data/中华人民共和国分市.json'  # 请确保这个文件路径正确
gdf_geojson = gpd.read_file(geojson_file_path)

import pandas as pd


df = pd.read_csv('data/climate_soil_loc.csv')
import pandas as pd

cities_data = df[['City','Province','District']]

# Get the unique cities from the "City" column
unique_cities = df['City'].unique()

filtered_gdf = gdf_geojson[gdf_geojson['name'].isin(unique_cities)]


import geopandas as gpd
import matplotlib.pyplot as plt

# 检查并清理无效几何
filtered_gdf = filtered_gdf[filtered_gdf.is_valid]


merged_df = pd.merge(cities_data, filtered_gdf, left_on='City', right_on='name', how='inner')


provinces_to_include = ['西藏自治区', '新疆维吾尔自治区', '甘肃省', '青海省', '四川省', '内蒙古自治区']

filtered_merged_df = merged_df[merged_df['Province'].isin(provinces_to_include)]


province_gdf =  gpd.read_file('data/中华人民共和国.json')

excluded_provinces = ['西藏自治区', '新疆维吾尔自治区', '甘肃省', '青海省', '四川省', '内蒙古自治区']

filtered_province_gdf = province_gdf[~province_gdf['name'].isin(excluded_provinces)]

merged_gdf = gpd.GeoDataFrame( pd.concat([filtered_merged_df, filtered_province_gdf], ignore_index=True))

if platform.system() == "Windows":
    tiff_folder ='data'
else:
    tiff_folder ='data'

tiff_output_folder ='data'


# os.makedirs(geojson_output_folder, exist_ok=True)
# os.makedirs(tiff_output_folder, exist_ok=True)





if not os.path.isdir(tiff_folder):

    print(f"Folder does not exist: {tiff_folder}")

    raise



tiff_file ='predicted_rf_585.tif'


tiff_path = os.path.join(tiff_folder, tiff_file)

tiff_output_path = os.path.join(tiff_output_folder, f'cropped_{tiff_file}')

with rasterio.open(tiff_path) as src:

    image_data = src.read(1).astype(np.float32)

    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=image_data.shape[0],
            width=image_data.shape[1],
            count=1,
            dtype="float32",
            crs=src.crs,
            transform=src.transform,
            nodata=np.nan,
        ) as dataset:
            dataset.write(image_data, 1)

            out_image, out_transform = mask(dataset, merged_gdf.geometry, crop=True, nodata=np.nan)

            # 更新元数据
            out_meta = dataset.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": "float32",  
                "nodata": 0  
            })

            out_image = np.where(np.isnan(out_image), 0, out_image) 

            with rasterio.open(tiff_output_path, "w", **out_meta) as dest:
                dest.write(out_image[0], 1) 

with rasterio.open(tiff_output_path) as cropped_src:
    for idx, row in merged_gdf.iterrows():
        geom = row['geometry']
        if geom.is_empty:
            continue

        polygons = [geom] if geom.geom_type == 'Polygon' else list(geom.geoms)

        all_pixel_values = []

        for poly in polygons:
            coords = np.array(poly.exterior.coords)

            pixel_coords = [cropped_src.index(x, y) for x, y in coords]

            for row, col in pixel_coords:
                if 0 <= row < cropped_src.height and 0 <= col < cropped_src.width:
                    pixel_value = cropped_src.read(1)[row, col]
                    if not np.isnan(pixel_value):
                        all_pixel_values.append(pixel_value)

        if all_pixel_values:
            avg_value = np.nanmean(all_pixel_values) 
            merged_gdf.at[idx, 'value'] = avg_value
        else:
            merged_gdf.at[idx, 'value'] = np.nan  

# target_lon, target_lat = 104.873239, 31.789814

# distances = np.sqrt((lon - target_lon) ** 2 + (lat - target_lat) ** 2)

# nearest_index = np.argmin(distances)

# # 获取对应的 X 数据
# nearest_X = X[nearest_index]

# # 输出结果
# print(f"最近点的经纬度: ({lon[nearest_index]}, {lat[nearest_index]})")
# tif_files = ['new/awt_soc.tif','new/dom_mu.tif', 'new/s_sand.tif', 'new/t_sand.tif', 'new/hand.tif', 
#              'new/srad.tif', 'new/bio_15.tif','new/bio_13.tif',  'new/bio_18.tif', 'new/bio_19.tif', 'new/bio_3.tif', 
#              'new/bio_6.tif', 'new/bio_8.tif', 'new/wind.tif']
# df = pd.DataFrame([nearest_X], columns=['lat','lon']+[os.path.basename(tif).replace('new/', '').replace('.tif', '') for tif in tif_files])

# # 打印结果
# print(df)



# first_row = train_df.iloc[0, :]  # 选取第一行
# print(first_row)