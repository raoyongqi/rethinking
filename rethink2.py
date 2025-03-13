import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 读取Excel文件
file_path = 'merged_data1.xlsx'  # 替换为你文件的路径
df = pd.read_excel(file_path)

# 查看数据的前几行，确保加载正确
print(df.head())

# 将所有列中 'Pathogen Load' 以外的列作为特征
X = df.drop(columns=['Pathogen Load'])  # 这里删除了 'Pathogen Load' 列
y = df['Pathogen Load']  # 'Pathogen Load' 作为目标列

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测结果
y_pred = model.predict(X_test)

# 计算 R² 得分
r2 = r2_score(y_test, y_pred)

# 打印 R² 得分
print(f'R² Score: {r2:.4f}')
