import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

random_state = 42  # 你可以设置任何整数值
# 假设 X 和 y 是你的特征和目标变量数据
# 这里我们随机生成一些数据来作为示例
np.random.seed(42)
file_path = "data/selection.csv"  # 替换为你的文件路径
selection = pd.read_csv(file_path)

# 处理自变量和因变量
X = selection.drop(columns='RATIO')
y = selection['RATIO']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)

# 设置不同的 ntree 和 mtry 值
ntree_values = np.arange(10, 201, 20)  # ntree 从 10 到 100，间隔为 10
mtry_values = np.arange(3, 11)  # mtry 从 3 到 10

# 初始化存储 RMSE 和 R² 的字典
rmse_dict = {}
r2_dict = {}

# 在不同的 ntree 和 mtry 参数下训练模型并计算 RMSE 和 R²
for mtry in mtry_values:
    rmse_list = []
    r2_list = []
    for ntree in ntree_values:
        print(mtry, ntree)
        # 初始化随机森林模型
        rf = RandomForestRegressor(n_estimators=ntree, max_features=mtry, random_state=42)
        
        # 训练模型
        rf.fit(X_train, y_train)
        
        # 预测测试集
        y_pred = rf.predict(X_test)
        
        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rmse_list.append(rmse)
        
        # 计算 R²
        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)
    
    # 存储每个 mtry 对应的 RMSE 和 R² 值
    rmse_dict[mtry] = rmse_list
    r2_dict[mtry] = r2_list
# 创建折线图
import matplotlib.pyplot as plt

# 创建折线图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 设置字体大小
font = {'size': 14}

# 绘制 RMSE 的折线图
for mtry in mtry_values:
    ax1.plot(ntree_values, rmse_dict[mtry], label=f'mtry={mtry}', linewidth=2)  # 加粗线条
ax1.set_title('RMSE vs ntree and mtry', fontdict={'size': 16, 'weight': 'bold'})  # 增大标题字体并加粗
ax1.set_xlabel('ntree', fontdict={'size': 14, 'weight': 'bold'})  # 增大X轴标签字体并加粗
ax1.set_ylabel('RMSE', fontdict={'size': 14, 'weight': 'bold'})  # 增大Y轴标签字体并加粗
ax1.legend(title='mtry', fontsize=12, title_fontsize=14, loc='best')  # 设置图例字体大小和标题

# 绘制 R² 的折线图
for mtry in mtry_values:
    ax2.plot(ntree_values, r2_dict[mtry], label=f'mtry={mtry}', linewidth=2)  # 加粗线条
ax2.set_title('R² vs ntree and mtry', fontdict={'size': 16, 'weight': 'bold'})  # 增大标题字体并加粗
ax2.set_xlabel('ntree', fontdict={'size': 14, 'weight': 'bold'})  # 增大X轴标签字体并加粗
ax2.set_ylabel('R²', fontdict={'size': 14, 'weight': 'bold'})  # 增大Y轴标签字体并加粗
ax2.legend(title='mtry', fontsize=12, title_fontsize=14, loc='best')  # 设置图例字体大小和标题

# 调整布局
plt.tight_layout()
plt.savefig("ntreemtry.png", dpi=300)  # 增加分辨率保存为高质量图片

# 显示图表
plt.show()


rmse_df = pd.DataFrame(rmse_dict, index=ntree_values)
rmse_df.index.name = 'ntree'
rmse_df.to_csv("rmse_results.csv")

# 将 R² 结果转换为 DataFrame 并保存
r2_df = pd.DataFrame(r2_dict, index=ntree_values)
r2_df.index.name = 'ntree'
r2_df.to_csv("r2_results.csv")

# 将 RMSE 结果转换为长格式 DataFrame
rmse_df = pd.DataFrame(rmse_dict, index=ntree_values)
rmse_df.index.name = 'ntree'
rmse_df = rmse_df.reset_index().melt(id_vars=['ntree'], var_name='mtry', value_name='MSE')

# 将 R² 结果转换为长格式 DataFrame
r2_df = pd.DataFrame(r2_dict, index=ntree_values)
r2_df.index.name = 'ntree'
r2_df = r2_df.reset_index().melt(id_vars=['ntree'], var_name='mtry', value_name='R²')

# 排序 MSE 值（由低到高）和 R² 值（由高到低）
sorted_rmse_df = rmse_df.sort_values(by='MSE', ascending=True)
sorted_r2_df = r2_df.sort_values(by='R²', ascending=False)


# 将排序后的结果保存为 LaTeX 表格
sorted_rmse_df.to_latex("sorted_rmse_results.tex", index=False)
sorted_r2_df.to_latex("sorted_r2_results.tex", index=False)

print("训练结果已保存为 sorted_rmse_results.csv 和 sorted_r2_results.csv")
