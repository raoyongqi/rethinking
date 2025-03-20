import numpy as np
import rasterio
import pandas as pd

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

tif_data_list, transform = read_multiple_tifs(tif_files)


rows, cols = tif_data_list[0].shape


lon, lat = get_coordinates(rows, cols, transform)


# tif_file = 'CMIP6/ACCESS-CM2/ssp126/2021-2040/wc2.1_5m_bioc_ACCESS-CM2_ssp126_2021-2040.tif'


# flip_tifs = {'new/dom_mu.tif', 'new/awt_soc.tif','new/pct_clay.tif', 'new/s_sand.tif', 'new/t_sand.tif'}

# def read_tif(tif_file):
#     with rasterio.open(tif_file) as src:
#         tif_data = src.read(1)  # 读取栅格数据
#         transform = src.transform  # 获取坐标变换信息

#         # **仅在文件名属于 flip_tifs 时翻转**
#         if tif_file in flip_tifs:
#             tif_data = np.flipud(tif_data)  # 进行上下翻转
#             print(f"TIF 文件 {tif_file} 进行了上下翻转")

#         return tif_data, transform
    


# with rasterio.open(tif_file) as src:
#     # 获取波段数量
#     num_bands = src.count
#     print(f"该 TIF 文件包含 {num_bands} 个波段。")

#     band_data_list = []

#     for bio_num in bio_columns_numeric:
#         band_index = bio_num  # 假设 bio_columns_numeric 中的数字对应波段的索引
#         if band_index <= num_bands:  # 确保索引在有效范围内
#             band_data = src.read(band_index)
#             band_data_list.append(band_data)
#             print(f"提取波段 {band_index} 数据，形状: {band_data.shape}")

#     stacked_bands = np.stack([tif_data.flatten() for tif_data in band_data_list], axis=-1)  

#     print(f"堆叠后的数据形状: {stacked_bands.shape}")
# def flatten_and_stack_bands(tif_file):
#     # 打开 TIF 文件
#     with rasterio.open(tif_file) as src:
#         # 获取波段数量
#         num_bands = src.count
#         print(f"该 TIF 文件包含 {num_bands} 个波段。")

#         # 初始化一个列表，用于存储每个波段展平后的数据
#         flattened_bands = []

#         # 遍历每个波段，进行展平
#         for band_index in range(1, num_bands + 1):
#             band_data = src.read(band_index)
#             flattened_band_data = band_data.flatten()  # 将二维数据展平成一维
#             flattened_bands.append(flattened_band_data)

#         # 将所有展平后的波段数据堆叠到一起
#         X = np.stack(flattened_bands, axis=1)  # 每一列是一个波段的展平数据

#         # 打印堆叠后的数据形状
#         print(f"堆叠后的数据形状: {X.shape}")

#         # 将所有波段的展平数据相加成一个波段
#         summed_band = np.sum(X, axis=1)  # 对每行进行求和，相当于将每个像素位置的所有波段数据相加

#         # 打印相加后的数据形状
#         print(f"相加后的数据形状: {summed_band.shape}")

#         return summed_band

# tif_file = 'CMIP6/ACCESS-CM2/ssp126/2021-2040/wc2.1_5m_tmax_ACCESS-CM2_ssp126_2021-2040.tif'
# tmax = flatten_and_stack_bands(tif_file)

# X = np.concatenate([tmax,stacked_bands], axis=1)

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