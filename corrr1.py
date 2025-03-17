import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import re
from openpyxl import Workbook
from openpyxl.styles import Font

# 文件路径
file_path = "data/selection.csv"  # 替换为你的文件路径

# 读取数据
selection = pd.read_csv(file_path)
selection = selection.drop(columns=['pathogen load'])

selection.columns=selection.columns.str.replace('hwsd_soil_clm_res_awt_soc','awt_soc')
selection.columns=selection.columns.str.replace('hwsd_soil_clm_res_pct_clay','pct_clay')
selection.columns=selection.columns.str.replace('hwsd_soil_clm_res_dom_mu','dom_mu')
selection.columns=selection.columns.str.replace('hand_500m_china_03_08', 'hand')

selection.columns = selection.columns.str.replace('_', ' ')

# 计算 Spearman 相关系数矩阵和 p 值矩阵
corr_matrix = selection.corr(method='spearman')  # Spearman 相关系数矩阵
p_matrix = selection.corr(method=lambda x, y: spearmanr(x, y)[1])  # p 值矩阵

# 格式化相关系数矩阵
formatted_matrix = corr_matrix.copy()
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix.columns)):
        # 保留两位小数并添加显著性标记
        value = f"{corr_matrix.iloc[i, j]:.2f}"
        
        # 检查相关系数是否为负数，若为负数则加粗
        if corr_matrix.iloc[i, j] < 0:
            value = f"{value}"
        
        # 添加显著性标记
        if p_matrix.iloc[i, j] < 0.05:
            value += "*"
        
        formatted_matrix.iloc[i, j] = value

# 创建掩码，只显示下三角部分
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=0)  # k=0 包括对角线

# 提取下三角部分的相关系数（不包括对角线）
lower_triangle_corr = formatted_matrix.where(~mask)

heatmap_data = lower_triangle_corr.dropna(how='all', axis=0).dropna(how='all', axis=1)
formatted_df = heatmap_data.fillna("").astype(str)

output_file = "data/formatted_correlation_matrix.xlsx"  # 选择输出文件路径



with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    formatted_df.to_excel(writer, index=False, sheet_name='Correlation Matrix')

    # 获取工作簿和工作表
    workbook = writer.book
    worksheet = workbook['Correlation Matrix']

    # 设置字体格式
    for row in worksheet.iter_rows(min_row=2, min_col=2, max_row=worksheet.max_row, max_col=worksheet.max_column):
        for cell in row:
            if '-' in cell.value:  # 识别加粗标记
                cell.font = Font(bold=True)  # 设置字体加粗

    workbook.save(output_file)

print(f"Formatted correlation matrix saved to {output_file}")
