import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import numpy as np

np.random.seed(42)

# 1. 读取Excel文件
file_path = 'data/merged_data1.xlsx'  # 替换为你的文件路径
data = pd.read_excel(file_path)

# 2. 数据预处理
data.columns = data.columns.str.lower()

data['hwsd_soil_clm_res_pct_clay'] = data['hwsd_soil_clm_res_pct_clay'].where(data['hwsd_soil_clm_res_pct_clay'] <= 100, other=pd.NA)
data['hwsd_soil_clm_res_pct_clay'] = data['hwsd_soil_clm_res_pct_clay'].fillna(data['hwsd_soil_clm_res_pct_clay'].mean())

data.columns = [col.replace('_resampled', '') if '_resampled' in col else col for col in data.columns]
data.columns = [col.replace('wc2.1_5m_', '') if col.startswith('wc2.1_5m_') else col for col in data.columns]

new_columns = []
for col in data.columns:
    if '_' in col:
        parts = col.split('_')
        if len(parts) > 1 and parts[0] == parts[-1]:
            new_columns.append('_'.join(parts[:1]))
        elif len(parts) > 2 and parts[1] == parts[-1]:
            new_columns.append('_'.join(parts[:2]))
        elif len(parts) > 3 and parts[2] == parts[-1]:
            new_columns.append('_'.join(parts[:2]))
        else:
            new_columns.append(col)
    else:
        new_columns.append(col)

data.columns = new_columns

# 合并特征

data['MAP'] = data.filter(like='prec_').sum(axis=1)
data['WIND'] = data.filter(like='wind_').mean(axis=1)
data['MAX_MAT'] = data.filter(like='tmax_').mean(axis=1)
data['MIN_MAT'] = data.filter(like='tmin_').mean(axis=1)
data['AVG_MAT'] = data.filter(like='tavg_').mean(axis=1)
data['SRAD'] = data.filter(like='srad_').mean(axis=1)
data['VAPR'] = data.filter(like='vapr_').mean(axis=1)

# 额外特征处理
data['TSEA'] = data['bio_4']
data['PSEA'] = data['bio_15']

# 删除冗余列
data = data.drop(columns=data.filter(like='prec_').columns)
data = data.drop(columns=data.filter(like='srad_').columns)
data = data.drop(columns=data.filter(like='tmax_').columns)
data = data.drop(columns=data.filter(like='tmin_').columns)
data = data.drop(columns=data.filter(like='tavg_').columns)
data = data.drop(columns=data.filter(like='vapr_').columns)
data = data.drop(columns=data.filter(like='wind_').columns)

# 特征和目标变量
feature_columns = [col for col in data.columns if col != 'pathogen load']
X = data[feature_columns]
y = data['pathogen load']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 选择模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 使用 RFE 进行特征选择
rfe = RFE(estimator=rf, n_features_to_select=16)
rfe.fit(X_train, y_train)

# 选出的特征
selected_features = [feature for feature, support in zip(feature_columns, rfe.support_) if support]

# 选择前 16 个特征的数据框
df = data[selected_features]

# 正确的选择列方式
sedf = data[selected_features + ['pathogen load']]

sedf.to_csv('data/selection.csv', index=False)

range_df = pd.DataFrame({
    'Variable': df.columns,               # 获取所有列名
    'Mean': df.mean().round(2).astype(str),           # 计算每列的平均值并四舍五入到两位小数，转换为字符串
    'Std Dev': df.std().round(2).astype(str),         # 计算每列的标准差并四舍五入到两位小数，转换为字符串
    'Range': [f"{df[col].min():.2f} - {df[col].max():.2f}" for col in df.columns],  # 将 Min 和 Max 合并为 Range 列，四舍五入到两位小数，转换为字符串
})

#"lat" 是 "latitude"

simplified_names = {
    'srad': 'Solar Radiation',
    's_sand': 'Soil Sand',
    'lon': 'Longitude',
    'VAPR': 'Vapor Pressure',
    'wind': 'Wind Speed',
    'lat': 'Latitude',
    'ELEV': 'Elevation',
    'MAX_MAT': 'Maximum Temperature',
    'AVG_MAT': 'Average Temperature',
    'MIN_MAT': 'Minimum Temperature',
    'TSEA': 'Temperature Seasonality',
    'PSEA': 'Precipitation Seasonality',
    'MAT': 'MeanAnnual Temperature',
    'MAP': 'MeanAnnual Precipitation',
    't_sand': 'Topsoil Sand',    
    'T_BULK_DEN': 'Topsoil Bulk Density',
    'T_REF_BULK': 'Topsoil ReferenceBulk Density',
    'S_CLAY': 'Soil Clay',
    'S_REF_BULK': 'Soil Reference Bulk Density',
    'T_GRAVEL': 'Topsoil Gravel',
    'ratio': 'Ratio',
   'bio_1': 'Annual Mean Temperature',
    'bio_2': 'Mean Diurnal Range （Mean of monthly （max temp - min temp））',
    'bio_3': 'Isothermality （bio_2/bio_7） （×100）',
    'bio_4': 'Temperature Seasonality （standard deviation ×100）',
    'bio_5': 'Max Temperature of Warmest Month',
    'bio_6': 'Min Temperature of Coldest Month',
    'bio_7': 'Temperature Annual Range （bio_5-bio_6）',
    'bio_8': 'Mean Temperature of Wettest Quarter',
    'bio_9': 'Mean Temperature of Driest Quarter',
    'bio_10': 'Mean Temperature of Warmest Quarter',
    'bio_11': 'Mean Temperature of Coldest Quarter',
    'bio_12': 'Annual Precipitation',
    'bio_13': 'Precipitation of Wettest Month',
    'bio_14': 'Precipitation of Driest Month',
    'bio_15': 'Precipitation Seasonality （Coefficient of Variation）',
    'bio_16': 'Precipitation of Wettest Quarter',
    'bio_17': 'Precipitation of Driest Quarter',
    'bio_18': 'Precipitation of Warmest Quarter',
    'bio_19': 'Precipitation of Coldest Quarter',
    'elev': 'Elevation',
    "prec_01": "Precipitation Jan",
    "prec_02": "Precipitation Feb",
    "prec_03": "Precipitation Mar",
    "prec_04": "Precipitation Apr",
    "prec_05": "Precipitation May",
    "prec_06": "Precipitation Jun",
    "prec_07": "Precipitation Jul",
    "prec_08": "Precipitation Aug",
    "prec_09": "Precipitation Sep",
    "prec_10": "Precipitation Oct",
    "prec_11": "Precipitation Nov",
    "prec_12": "Precipitation Dec",
    "srad_01": "SolarRadiation Jan",
    "srad_02": "SolarRadiation Feb",
    "srad_03": "SolarRadiation Mar",
    "srad_04": "SolarRadiation Apr",
    "srad_05": "SolarRadiation May",
    "srad_06": "SolarRadiation Jun",
    "srad_07": "SolarRadiation Jul",
    "srad_08": "SolarRadiation Aug",
    "srad_09": "SolarRadiation Sep",
    "srad_10": "SolarRadiation Oct",
    "srad_11": "SolarRadiation Nov",
    "srad_12": "SolarRadiation Dec",
    "tavg_01": "TemperatureAvg Jan",
    "tavg_02": "TemperatureAvg Feb",
    "tavg_03": "TemperatureAvg Mar",
    "tavg_04": "TemperatureAvg Apr",
    "tavg_05": "TemperatureAvg May",
    "tavg_06": "TemperatureAvg Jun",
    "tavg_07": "TemperatureAvg Jul",
    "tavg_08": "TemperatureAvg Aug",
    "tavg_09": "TemperatureAvg Sep",
    "tavg_10": "TemperatureAvg Oct",
    "tavg_11": "TemperatureAvg Nov",
    "tavg_12": "TemperatureAvg Dec",
    "tmax_01": "MaxTemperature Jan",
    "tmax_02": "MaxTemperature Feb",
    "tmax_03": "MaxTemperature Mar",
    "tmax_04": "MaxTemperature Apr",
    "tmax_05": "MaxTemperature May",
    "tmax_06": "MaxTemperature Jun",
    "tmax_07": "MaxTemperature Jul",
    "tmax_08": "MaxTemperature Aug",
    "tmax_09": "MaxTemperature Sep",
    "tmax_10": "MaxTemperature Oct",
    "tmax_11": "MaxTemperature Nov",
    "tmax_12": "MaxTemperature Dec",
    "tmin_01": "Min Temperature Jan",
    "tmin_02": "Min Temperature Feb",
    "tmin_03": "Min Temperature Mar",
    "tmin_04": "Min Temperature Apr",
    "tmin_05": "Min Temperature May",
    "tmin_06": "Min Temperature Jun",
    "tmin_07": "Min Temperature Jul",
    "tmin_08": "Min Temperature Aug",
    "tmin_09": "Min Temperature Sep",
    "tmin_10": "Min Temperature Oct",
    "tmin_11": "Min Temperature Nov",
    "tmin_12": "Min Temperature Dec",
    "vapr_01": "Vapor Pressure Jan",
    "vapr_02": "Vapor Pressure Feb",
    "vapr_03": "Vapor Pressure Mar",
    "vapr_04": "Vapor Pressure Apr",
    "vapr_05": "Vapor Pressure May",
    "vapr_06": "Vapor Pressure Jun",
    "vapr_07": "Vapor Pressure Jul",
    "vapr_08": "Vapor Pressure Aug",
    "vapr_09": "Vapor Pressure Sep",
    "vapr_10": "Vapor Pressure Oct",
    "vapr_11": "Vapor Pressure Nov",
    "vapr_12": "Vapor Pressure Dec",
    "wind_01": "Wind Speed Jan",
    "wind_02": "Wind Speed Feb",
    "wind_03": "Wind Speed Mar",
    "wind_04": "Wind Speed Apr",
    "wind_05": "Wind Speed May",
    "wind_06": "Wind Speed Jun",
    "wind_07": "Wind Speed Jul",
    "wind_08": "Wind Speed Aug",
    "wind_09": "Wind Speed Sep",
    "wind_10": "Wind Speed Oct",
    "wind_11": "Wind Speed Nov",
    "wind_12": "Wind Speed Dec",
    "hwsd_soil_clm_res_awt_soc": "Area-weighted soil organic carbon content",
    "hwsd_soil_clm_res_dom_mu": "Dominant mapping unit ID from HWSD （MU_GLOBAL above）",
    "hwsd_soil_clm_res_pct_clay": "Soil clay fraction by percent weight",
    "s_bulk": "Soil Bulk",
    "s_cec": "Soil CEC",
    "s_clay": "Soil Clay",
    "s_c": "Soil C",
    "s_gravel": "Soil Gravel",
    "s_oc": "Soil Organic Carbon",
    "s_ph": "Soil pH",
    "s_ref": "Soil Reference",
    "s_sand": "Soil Sand",
    "s_silt": "Soil Silt",
    "t_bulk": "Topsoil Bulk",
    "t_cec": "Topsoil CEC",
    "t_clay": "Topsoil Clay",
    "t_c": "Topsoil C",
    "t_gravel": "Topsoil Gravel",
    "t_oc": "Topsoil Organic Carbon",
    "t_ph": "Topsoil pH",
    "t_ref": "Topsoil Reference",
    "t_sand": "Topsoil Sand",
    "t_silt": "Topsoil Silt",
    "t_silt": "Topsoil Silt",
    "hand": "Height Above the Nearest Drainage",

}

units = {
    'lat': '°',
    'lon': '°',
    'hwsd soil clm res awt soc': 'kg C m-2',
    'hwsd soil clm res dom mu': 'numerical ID',
    'hwsd soil clm res pct clay': '%',
    's sand': '%',
    't sand': '%',
    'hand': 'm',
    'srad': 'kJ m-2 day-1',
    'bio 11': '°C',
    'bio 13': 'mm',
    'bio 15': 'mm',
    'bio 18': 'mm',
    'bio 3': '%',
    'bio 8': '°C',
    'wind': 'm/s'
}

# 假设第一列包含变量名称，第二列是数据

range_df['Variable'] = range_df['Variable'].replace('hand_500m_china_03_08', 'hand')


# 添加 Simplified Name 列
range_df  = range_df[['Variable', 'Mean', 'Std Dev', 'Range']]
range_df['Simplified Name'] = range_df['Variable'].map(simplified_names)

range_df['Variable'] = range_df['Variable'].map(lambda x: x.replace('_', r' '))

range_df['Variable'] = range_df['Variable'].map(lambda x: f"{x} （{units.get(x, 'N/A')}）")

range_df  = range_df[['Variable','Simplified Name', 'Mean', 'Std Dev', 'Range']]

range_df.to_excel('data/range_df_output.xlsx', index=False)