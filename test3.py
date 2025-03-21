import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches  # 用于创建方块图例

# 示例数据
data = pd.DataFrame({
    'Category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'Time': [10, 12, 14, 9, 8, 7, 5, 6, 4],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male']
})

# 绘制柱状图
sns.barplot(x='Category', y='Time', data=data, hue='Gender', palette=["red", "blue"])

# 手动创建图例方块
male_handle = mpatches.Patch(color='red', label='Male')   # 创建红色方块
female_handle = mpatches.Patch(color='blue', label='Female')  # 创建蓝色方块

# 添加图例
plt.legend(handles=[male_handle, female_handle], title='Gender')

# 显示图形
plt.title('Category vs Time with Hue for Gender')
plt.show()
