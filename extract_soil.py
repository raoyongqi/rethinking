import os
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm


folder_path = 'HWSD_1247/HWSD_1247/data'

excel_file = 'data/merge.xlsx'
coordinates = pd.read_excel(excel_file)

all_extracted_data = []

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

# 获取总的文件数量和坐标数量
total_files = len(files_and_keys)
total_coordinates = len(coordinates)

# 遍历文件和相应的键
for file_idx, file_info in enumerate(files_and_keys):
    filename = file_info["filename"]
    keys = file_info["keys"]

    # 设置每个文件的路径
    file_path = os.path.join(folder_path, filename)

    # 打开 NetCDF 文件
    with h5py.File(file_path, 'r') as f:

        print(f"\nProcessing {file_path}...")

        # 获取 NetCDF 中的经纬度数据
        lat = f[keys[0]][:]  # 获取纬度数据
        lon = f[keys[1]][:]  # 获取经度数据

        # 确保 lat 和 lon 是 numpy 数组
        lat = np.array(lat)
        lon = np.array(lon)

        # 对于 AWT_S_SOC 文件，使用切片 [0, :, :]
        if filename == "AWT_S_SOC.nc4":
            data_slice = f[keys[2]][0, :, :]
        else:
            data_slice = f[keys[2]][:]

        # 遍历原始 CSV 文件中的每一行数据
        for coord_idx, (_, row) in enumerate(coordinates.iterrows()):
            lat_val = row['lat']
            lon_val = row['lon']

            # 确保目标经纬度值在数据的有效范围内
            if lat_val < lat.min() or lat_val > lat.max() or lon_val < lon.min() or lon_val > lon.max():
                print(f"Warning: Coordinates {lat_val}, {lon_val} are out of bounds.")
                raise  # 跳过这个坐标点，处理下一个坐标点

            # 找到与给定经纬度最接近的索引
            lat_idx = np.argmin(np.abs(lat - lat_val))
            lon_idx = np.argmin(np.abs(lon - lon_val))

            # 提取相应的数据
            extracted_data = row.to_dict()  # 保留原始 CSV 行数据
            extracted_data[keys[2]] = data_slice[lat_idx, lon_idx]  # 提取数据并存储

            # 将提取的数据添加到列表中
            all_extracted_data.append(extracted_data)

        # 打印文件处理完成的提示
        print(f"\nFinished processing {filename} ({file_idx + 1}/{total_files})")

# 将合并的数据保存为 Excel 文件
output_file = 'merged_data.xlsx'  # 设置输出文件路径
merged_df = pd.DataFrame(all_extracted_data)
merged_df.to_excel(output_file, index=False)

print(f"数据已成功合并并保存为 Excel 格式：{output_file}")
