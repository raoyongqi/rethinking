import pandas as pd

# 读取 Excel 文件
df = pd.read_excel("data/ruqinfeiruqin.xlsx", sheet_name=0)

# 去重
df = df.drop_duplicates(subset=["ID", "Species"], keep="first")

# 重命名列
df = df.rename(columns={
    "D0": "Disease_0",
    "D1": "Disease_1",
    "D2": "Disease_2",
    "D3": "Disease_3",
    "D4": "Disease_4",
    "D5": "Disease_5"
})

# 填充缺失值
df = df.fillna(0)

# 计算 Severity
def calculate_severity(row):
    total_disease = (row["Disease_0"] + row["Disease_1"] + row["Disease_2"] +
                     row["Disease_3"] + row["Disease_4"] + row["Disease_5"])
    if total_disease == 0:
        return 0
    return (row["Disease_0"] * 0 + row["Disease_1"] * 1 + row["Disease_2"] * 2 +
            row["Disease_3"] * 3 + row["Disease_4"] * 4 + row["Disease_5"] * 5) / (6 * total_disease)

df["Severity"] = df.apply(calculate_severity, axis=1)

# 计算 PL
df["PL"] = df["Severity"] * df["Biomass"]

# 计算 Biomass 和 PL 的聚合
biomass_result = df.groupby("ID")["Biomass"].sum().reset_index()
pl_result = df.groupby("ID")["PL"].sum().reset_index()

# 合并两个数据框
result = pd.merge(biomass_result, pl_result, on="ID")
# 定义转换函数


import re

def convert_hn_to_HN1_v3(hn_code):

    return re.sub(r"hn-S(\d+)", r"HN1-na\1", hn_code)

result['Pathogen Load'] = result["PL"] / result["Biomass"] * 100

def remove_na1_v2(input_str):
    
    return re.sub(r"-na\d+", "", input_str)



result["Prefix"] = result["ID"].str.extract(r'(^[^-]+)')
print(result)
result["ID"] = result["ID"].apply(remove_na1_v2)



output_df = pd.read_csv("data/output.csv")

output_df['Site'] = output_df['Site'].apply(convert_hn_to_HN1_v3)


merged_df = pd.merge(result, output_df, left_on="ID", right_on="Site")
print(merged_df)

selected_columns = merged_df[["lon", "lat", 'Pathogen Load']]


selected_columns.to_excel("data/merge.xlsx", index=False)
