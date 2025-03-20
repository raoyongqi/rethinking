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
print(new_order)