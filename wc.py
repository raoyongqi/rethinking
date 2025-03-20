import os
import pandas as pd
import rasterio

# CSV 文件路径
file_path = 'data/merge.xlsx'  # 替换为你的Excel文件路径

# TIFF 文件夹路径
tif_folder = 'wc2.1_5m'  # 替换为包含TIFF文件的文件夹路径

# 读取CSV文件
df = pd.read_excel(file_path )

# 创建一个空的 DataFrame 来存储结果
result_df = df.copy()

# 遍历TIFF文件夹中的所有.tif文件
for tiff_file in os.listdir(tif_folder):
    if tiff_file.endswith('.tif'):
        tiff_path = os.path.join(tif_folder, tiff_file)
        
        # 打开TIFF文件
        with rasterio.open(tiff_path) as src:
            # 获取TIFF文件的值（假设是单波段图像）
            band = src.read(1)
            
            # 为每个TIFF文件创建一个新列
            tiff_column = []
            for index, row in df.iterrows():
                lat = row['lat']
                lon = row['lon']
                
                # 将经纬度转换为行列号
                row_idx, col_idx = src.index(lon, lat)
                
                # 获取对应的栅格值
                value = band[row_idx, col_idx]
                
                tiff_column.append(value)
            
            # 将当前 TIFF 文件的栅格值添加到 DataFrame 中，列名为 TIFF 文件名
            result_df[tiff_file] = tiff_column

# 输出整理后的 DataFrame

# 如果需要保存到CSV文件
result_df.to_csv('output.csv', index=False)
