import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from boruta import BorutaPy

# 读取CSV文件
file_path = 'output.csv'  # 替换为你文件的路径
df = pd.read_csv(file_path)

# 查看数据的前几行，确保加载正确
print(df.head())

# 将所有列中 'Pathogen Load' 以外的列作为特征
X = df.drop(columns=['Pathogen Load'])  # 这里删除了 'Pathogen Load' 列
y = df['Pathogen Load']  # 'Pathogen Load' 作为目标列

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 使用 Boruta 进行特征选择，并显示进度
boruta = BorutaPy(rf, n_estimators='auto', random_state=42, verbose=2)
boruta.fit(X_train.values, y_train.values)

# 打印被选中的重要特征
print("Selected Features:", X.columns[boruta.support_])

# 仅保留被选中的特征
X_train_selected = boruta.transform(X_train.values)
X_test_selected = boruta.transform(X_test.values)

# 创建新的随机森林回归模型并训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# 预测结果
y_pred = model.predict(X_test_selected)

# 计算 R² 得分
r2 = r2_score(y_test, y_pred)

# 打印 R² 得分
print(f'R² Score: {r2:.4f}')
