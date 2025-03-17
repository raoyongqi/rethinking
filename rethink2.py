import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFE

# 读取Excel文件
file_path = 'merged_data1.xlsx'  # 替换为你文件的路径
df = pd.read_excel(file_path)
df['lat'].fillna(method='ffill', inplace=True)  # Forward fill for missing latitudes

# 查看数据的前几行，确保加载正确

# 将所有列中 'Pathogen Load' 以外的列作为特征
X = df.drop(columns=['Pathogen Load'])  # 这里删除了 'Pathogen Load' 列
y = df['Pathogen Load']  # 'Pathogen Load' 作为目标列

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 使用 RFE 进行特征选择
rfe = RFE(rf, n_features_to_select=16)  # 选择最重要的 10 个特征
rfe.fit(X_train, y_train)

# 打印被选中的重要特征
selected_features = X.columns[rfe.support_]
print("Selected Features:", selected_features)

# 仅保留被选中的特征
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 创建新的随机森林回归模型并训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# 预测结果
y_pred = model.predict(X_test_selected)

# 计算 R² 得分
r2 = r2_score(y_test, y_pred)

# 打印 R² 得分
print(f'R² Score: {r2:.4f}')

# 获取特征的排名
ranking = rfe.ranking_
print("Feature Ranking:", ranking)

# 获取特征的系数（重要性得分）
coef = rfe.estimator_.coef_
print("Feature Coefficients:", coef)
