import os
import h5py
import pandas as pd
import numpy as np

import rasterio

folder_path = 'HWSD_1247/HWSD_1247/data'  

import numpy as np
import rasterio
from geopy.distance import geodesic
from scipy.spatial import KDTree

excel_file = 'data/merge.xlsx'  # 替换为你的文件路径
coordinates = pd.read_excel(excel_file)

merged_df = coordinates.set_index(['lat', 'lon'])

files_and_keys = [
    {"filename": "AWC_CLASS.nc4", "keys": ["lat", "lon", "AWC_CLASS"]},
    {"filename": "AWT_S_SOC.nc4", "keys": ["lat", "lon", "SUM_s_c_1"]},
    {"filename": "AWT_T_SOC.nc4", "keys": ["lat", "lon", "SUM_t_c_12"]},
    {"filename": "HWSD_SOIL_CLM_RES.nc4", "keys": ["nlevsoi", "AWT_SOC", "BULK_DEN", "DOM_MU", "DOM_SOC", "PCT_CLAY", "PCT_SAND", "PH", "REF_BULK", "areaupsc", "landfrac", "landmask", "lon", "lat"]},
    {"filename": "ISSOIL.nc4", "keys": ["lat", "lon", "ISSOIL"]},
    {"filename": "MU_GLOBAL.nc4", "keys": ["lat", "lon", "MU_GLOBAL"]},
    {"filename": "REF_DEPTH.nc4", "keys": ["lat", "lon", "REF_DEPTH"]},
    {"filename": "ROOTS.nc4", "keys": ["lat", "lon", "ROOTS"]},
    {"filename": "S_BULK_DEN.nc4", "keys": ["lat", "lon", "S_BULK_DEN"]},
    {"filename": "S_C.nc4", "keys": ["lat", "lon", "s_c"]},
    {"filename": "S_CEC_CLAY.nc4", "keys": ["lat", "lon", "S_CEC_CLAY"]},
    {"filename": "S_CLAY.nc4", "keys": ["lat", "lon", "S_CLAY"]},
    {"filename": "S_GRAVEL.nc4", "keys": ["lat", "lon", "S_GRAVEL"]},
    {"filename": "S_OC.nc4", "keys": ["lat", "lon", "S_OC"]},
    {"filename": "S_PH_H2O.nc4", "keys": ["lat", "lon", "S_PH_H2O"]},
    {"filename": "S_REF_BULK.nc4", "keys": ["lat", "lon", "S_REF_BULK"]},
    {"filename": "S_SAND.nc4", "keys": ["lat", "lon", "S_SAND"]},
    {"filename": "S_SILT.nc4", "keys": ["lat", "lon", "S_SILT"]},
    {"filename": "T_BULK_DEN.nc4", "keys": ["lat", "lon", "T_BULK_DEN"]},
    {"filename": "T_C.nc4", "keys": ["lat", "lon", "t_c"]},
    {"filename": "T_CEC_CLAY.nc4", "keys": ["lat", "lon", "T_CEC_CLAY"]},
    {"filename": "T_CLAY.nc4", "keys": ["lat", "lon", "T_CLAY"]},
    {"filename": "T_GRAVEL.nc4", "keys": ["lat", "lon", "T_GRAVEL"]},
    {"filename": "T_OC.nc4", "keys": ["lat", "lon", "T_OC"]},
    {"filename": "T_PH_H2O.nc4", "keys": ["lat", "lon", "T_PH_H2O"]},
    {"filename": "T_REF_BULK.nc4", "keys": ["lat", "lon", "T_REF_BULK"]},
    {"filename": "T_SAND.nc4", "keys": ["lat", "lon", "T_SAND"]},
    {"filename": "T_SILT.nc4", "keys": ["lat", "lon", "T_SILT"]},
]

total_files = len(files_and_keys)
total_coordinates = len(coordinates)

for file_idx, file_info in enumerate(files_and_keys):
    filename = file_info["filename"]
    keys = file_info["keys"]

    file_path = os.path.join(folder_path, filename)
    file_basename = filename.replace('.nc4', '')  # 去掉文件后缀

    with h5py.File(file_path, 'r') as f:
        temp_data = []

        print(f"\nProcessing {file_path}...")
        if filename == "HWSD_SOIL_CLM_RES.nc4":
            
            keys = keys[:-2] + [keys[-2], keys[-1]]
        
            lat = f[keys[-1]][:]  # 纬度
            lon = f[keys[-2]][:]  # 经度
            
            lat = np.array(lat)
            lon = np.array(lon)
            all_data=[]
            for _, row in coordinates.iterrows():
                lat_val, lon_val = row['lat'], row['lon']
                if lat_val < lat.min() or lat_val > lat.max() or lon_val < lon.min() or lon_val > lon.max():
                    raise ValueError(f"Error: Coordinates ({lat_val}, {lon_val}) are out of bounds. "
                                    f"Valid range: lat ({lat.min()} to {lat.max()}), lon ({lon.min()} to {lon.max()})")



                
                lat_idx = np.argmin(np.abs(lat - lat_val))
                lon_idx = np.argmin(np.abs(lon - lon_val))
                
                extracted_data = row.to_dict()
                temp_data = {"lon": lon_val, "lat": lat_val}
                for key in keys[:-2]:
                    
                    if key == "nlevsoi":
                        continue
                    dataset = f[key]
                    data = dataset[0, lat_idx, lon_idx] if dataset.ndim == 3 else dataset[lat_idx, lon_idx]
                    temp_data[f"{file_basename}_{key}"] = data
                all_data.append(temp_data)

            temp_df = pd.DataFrame(all_data).set_index(['lat', 'lon'])

            if 'Pathogen Load' in temp_df.columns:
                temp_df = temp_df.drop(columns=['Pathogen Load'])

            temp_df = temp_df.loc[~temp_df.index.duplicated()]  # 删除重复的索引

            merged_df = merged_df.merge(temp_df, left_index=True, right_index=True, how='left')
        else:

            lat = f[keys[0]][:]  # 获取纬度数据
            lon = f[keys[1]][:]  # 获取经度数据
    
            lat = np.array(lat)
            lon = np.array(lon)

            for coord_idx, (_, row) in enumerate(coordinates.iterrows()):
                lat_val = row['lat']
                lon_val = row['lon']

                if lat_val < lat.min() or lat_val > lat.max() or lon_val < lon.min() or lon_val > lon.max():
                    raise ValueError(f"Error: Coordinates ({lat_val}, {lon_val}) are out of bounds. "
                                    f"Valid range: lat ({lat.min()} to {lat.max()}), lon ({lon.min()} to {lon.max()})")

                lat_idx = np.argmin(np.abs(lat - lat_val))
                lon_idx = np.argmin(np.abs(lon - lon_val))

                for key in keys[2:]:  # 从第三个键开始提取

                    dataset = f[key]
                    if dataset.ndim == 3:  # 如果是3维数据，取第一层
                        data = dataset[0, lat_idx, lon_idx]
                    else:  # 2维数据，直接索引
                        data = dataset[lat_idx, lon_idx]

                    temp_data.append({"lon": lon_val, "lat": lat_val, f"{file_basename}_{key}": data})

                                    
            temp_df = pd.DataFrame(temp_data).set_index(['lat', 'lon'])
            print(f"Rows: {temp_df.shape[0]}, Columns: {temp_df.shape[1]}")
            print(f"Rows: {merged_df.shape[0]}, Columns: {merged_df.shape[1]}")


            if 'Pathogen Load' in temp_df.columns:
                
                temp_df = temp_df.drop(columns=['Pathogen Load'])
            temp_df = temp_df.loc[~temp_df.index.duplicated()]  # 删除重复的索引

            merged_df = merged_df.merge(temp_df, left_index=True, right_index=True, how='left')


            print(f"Rows: {merged_df.shape[0]}, Columns: {merged_df.shape[1]}")

            print(f"\nFinished processing {filename} ({file_idx + 1}/{total_files})")



file_path = 'data/merge.xlsx'
tif_folder = 'new'

df = pd.read_excel(file_path )

result_df = df.copy()



import rasterio
import numpy as np
from geopy.distance import geodesic
from tqdm import tqdm  # 引入 tqdm



def extract_tiff_data(tiff_file, df, lat_range=5.0, lon_range=5.0):
    tiff_column = []
    
    with rasterio.open(tiff_file) as src:
        band = src.read(1)  # Read the first band
        transform = src.transform  # Get the affine transform parameters

        # Loop through each row in the DataFrame
        for index, row in df.iterrows():
            lat = row['lat']
            lon = row['lon']
            
            try:
                # Get the row and column indices corresponding to the latitude and longitude
                row_idx, col_idx = src.index(lon, lat)
                value = band[row_idx, col_idx]
                
                # If the value is not NaN, use it directly and append to tiff_column
                if not np.isnan(value):
                    tiff_column.append(value)
                else:
                    raise ValueError(f"{tiff_file} location ({lat}, {lon}) has NaN value, searching for the nearest valid data.")
            except Exception as e:
                print(f"Cannot get the row/column index for coordinates ({lat}, {lon}), error: {e}")
                
                # Only create KDTree and valid_pixels when we need to find the nearest valid pixel
                valid_pixels = []
                for i in range(band.shape[0]):
                    for j in range(band.shape[1]):
                        value = band[i, j]
                        if not np.isnan(value):  # Check if it's valid data
                            lon_pixel, lat_pixel = transform * (j, i)  # Get the longitude and latitude of the pixel
                            valid_pixels.append((lat_pixel, lon_pixel, value))
                
                # Only create KDTree if there are valid pixels to query
                valid_coords = [(valid_lat, valid_lon) for valid_lat, valid_lon, valid_value in valid_pixels]
                valid_values = [valid_value for valid_lat, valid_lon, valid_value in valid_pixels]
                kdtree = KDTree(valid_coords)

                # Look for the nearest valid pixel within the given lat/lon range
                closest_value = None
                closest_distance = float('inf')

                # Only search valid pixels within the given lat/lon range
                nearby_coords = [(valid_lat, valid_lon) for valid_lat, valid_lon, valid_value in valid_pixels
                                 if abs(valid_lat - lat) <= lat_range and abs(valid_lon - lon) <= lon_range]

                # If we have valid nearby pixels, query the KDTree for the closest one
                if nearby_coords:
                    _, idx = kdtree.query((lat, lon))
                    closest_value = valid_values[idx]
                    closest_distance = geodesic((lat, lon), (valid_coords[idx][0], valid_coords[idx][1])).meters
                    print(f"Using the closest valid data value: {closest_value} (distance: {closest_distance:.2f} meters)")
                    tiff_column.append(closest_value)
                else:
                    raise ValueError(f"No valid data found within range for location ({lat}, {lon}).")
    
    return tiff_column
excluded_files = ['pct_clay.tif', 'dom_mu.tif', 'awt_soc.tif', 's_sand.tif', 't_sand.tif']
tiff_files = ['new/hand.tif'] + [os.path.join(tif_folder, f) for f in os.listdir(tif_folder) 
             if f.endswith('.tif') and f not in excluded_files]

# 遍历所有的.tif文件并提取数据
for tiff_file in tiff_files:
    # 提取tiff文件对应的数据
    tiff_column = extract_tiff_data(tiff_file, df)
    
    # 将数据添加到结果DataFrame中
    result_df[tiff_file] = tiff_column


output_file = 'merged_data1.xlsx'

result_df = result_df.set_index(['lat', 'lon'])


if 'Pathogen Load' in result_df.columns:
    
    result_df = result_df.drop(columns=['Pathogen Load'])


result_df = result_df.loc[~result_df.index.duplicated()]
# merged_df = pd.read_excel(file_path)
# merged_df = merged_df.set_index(['lat', 'lon'])
merged_df = merged_df.merge(result_df, left_index=True, right_index=True, how='left')
merged_df = merged_df.rename(columns=lambda x: x.replace('.tif', '') if '.tif' in x else x)
merged_df = merged_df.reset_index()
merged_df.to_excel(output_file, index=False)


print(f"数据已成功合并并保存为 Excel 格式：{output_file}")
