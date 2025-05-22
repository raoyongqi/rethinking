import numpy as np
import pandas as pd

def rank_data(data):
    """
    对数据进行排名，处理相同值的情况
    """
    n = data.shape[0]
    ranks = np.zeros_like(data, dtype=float)
    for col in range(data.shape[1]):
        sorted_indices = np.argsort(data[:, col])
        sorted_data = data[sorted_indices, col]
        unique_values, unique_indices = np.unique(sorted_data, return_index=True)
        ranks[sorted_indices, col] = np.searchsorted(unique_values, sorted_data) + 1
    return ranks

def spearman_correlation_matrix(data):

    ranks = rank_data(data)
    
    n = ranks.shape[0]
    mean_ranks = np.mean(ranks, axis=0)
    std_ranks = np.std(ranks, axis=0, ddof=1)
    
    corr_matrix = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                cov = np.mean((ranks[:, i] - mean_ranks[i]) * (ranks[:, j] - mean_ranks[j]))
                corr_matrix[i, j] = cov / (std_ranks[i] * std_ranks[j])
    
    return corr_matrix

# 示例数据
data_df = pd.DataFrame({
    'A': [1, 4, 7, 10],
    'B': [2, 5, 8, 11],
    'C': [3, 6, 9, 12]
})

# 将DataFrame转换为NumPy数组
data_np = data_df.to_numpy()
import scipy.stats as stats

corr_matrix = stats.spearmanr(data_np)

# 使用Pandas计算Spearman相关系数矩阵
corr_matrix_corr = data_df.corr(method='spearman')

print("NumPy计算的Spearman相关系数矩阵:")
print(corr_matrix)

print("\nPandas计算的Spearman相关系数矩阵:")
print(corr_matrix_corr)