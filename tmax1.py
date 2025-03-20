import numpy as np
import rasterio
from rasterio.enums import Resampling
import pandas as pd
train_df = pd.read_csv("data/selection.csv")
train_df = train_df.rename(columns={
    'hand_500m_china_03_08': 'hand',
    'hwsd_soil_clm_res_dom_mu': 'dom_mu',
    'hwsd_soil_clm_res_awt_soc': 'awt_soc'
})

if 'tmax' in train_df.columns:
    print("train_df contains 'tmax' column")
else:
    print("train_df does not contain 'tmax' column")

if 'tmin' in train_df.columns:
    print("train_df contains 'tmin' column")
else:
    print("train_df does not contain 'tmin' column")
if 'prec' in train_df.columns:
    print("train_df contains 'prec' column")
else:
    print("train_df does not contain 'prec' column")

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
            band_data = src.read(band_index, resampling=Resampling.nearest)
            flattened_band_data = band_data.flatten()  # 将二维数据展平成一维
            flattened_bands.append(flattened_band_data)

        # 将所有展平后的波段数据堆叠到一起
        X = np.stack(flattened_bands, axis=1)  # 每一列是一个波段的展平数据

        # 打印堆叠后的数据形状
        print(f"堆叠后的数据形状: {X.shape}")

        # 将所有波段的展平数据相加成一个波段
        summed_band = np.sum(X, axis=1)  # 对每行进行求和，相当于将每个像素位置的所有波段数据相加

        # 打印相加后的数据形状
        print(f"相加后的数据形状: {summed_band.shape}")

        return summed_band

tif_file = 'CMIP6/ACCESS-CM2/ssp126/2021-2040/wc2.1_5m_tmax_ACCESS-CM2_ssp126_2021-2040.tif'
tmax = flatten_and_stack_bands(tif_file)
