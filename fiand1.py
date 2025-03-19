import os
import rasterio
import numpy as np


def process_tif_files(folder_path, newfolder_path, keyword, output_filename, calc_type='mean'):
    """
    处理指定文件夹中的 TIFF 文件，计算包含关键词的文件的均值或总和，并保存为新的 TIFF 文件。
    
    :param folder_path: 文件夹路径
    :param keyword: 文件名中包含的关键词（如 'wind' 或 'tavg'）
    :param output_filename: 输出的 TIFF 文件名
    :param calc_type: 'mean' 或 'sum'，指定计算均值或总和
    """
    # 获取当前文件夹中的所有文件
    tif_files = []
    for file in os.listdir(folder_path):
        # 只查找当前文件夹中的 .tif 文件，并且文件名中包含指定的关键词
        if file.lower().endswith('.tif') and keyword.lower() in file.lower():
            tif_files.append(os.path.join(folder_path, file))

    # 如果找到文件
    if tif_files:
        print(f"找到以下包含 '{keyword}' 的 TIFF 文件：")
        for tif_file in tif_files:
            print(tif_file)

        # 打开第一个文件获取数据维度
        with rasterio.open(tif_files[0]) as src:
            profile = src.profile
            data_sum = src.read(1).astype(np.float64)  # 使用 float64 以防止溢出
            count = 1  # 计数器，用于计算均值

        # 循环加载其他文件并累加数据
        for tif_file in tif_files[1:]:
            with rasterio.open(tif_file) as src:
                data = src.read(1).astype(np.float64)  # 读取当前文件，并使用 float64
                data_sum += data  # 累加数据
                count += 1  # 计数增加

        # 如果选择计算均值
        if calc_type == 'mean':
            data_result = data_sum / count
        else:
            # 如果选择计算总和
            data_result = data_sum

        # 检查是否包含无穷大值
        if np.any(np.isinf(data_result)):
            print(f"警告：处理后的数据中包含无穷大值，在文件 '{output_filename}' 中。")
            # 你可以选择将无穷大值替换为 NaN 或其他数值
            data_result = np.nan_to_num(data_result, nan=np.nan, posinf=np.nan, neginf=np.nan)

        # 保存结果到新的 TIFF 文件
        output_file = os.path.join(newfolder_path, output_filename)
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data_result.astype(np.float64), 1)  # 保存为新的文件，使用 float64
        print(f"计算 {calc_type} 后的 TIFF 文件已保存为: {output_file}")

    else:
        print(f"没有找到包含 '{keyword}' 的 TIFF 文件。")
# 设置文件夹路径
folder_path = r'wc2.1_5m'  # 替换为你的文件夹路径
newfolder_path = r'new'  # 替换为你的文件夹路径

# 调用函数处理包含 'wind' 的 TIFF 文件，计算均值
process_tif_files(folder_path, newfolder_path, 'wind', 'wind.tif', calc_type='mean')

# 调用函数处理包含 'tavg' 的 TIFF 文件，计算均值
process_tif_files(folder_path, newfolder_path, 'tavg', 'mat.tif', calc_type='mean')

# 调用函数处理包含 'tmin' 的 TIFF 文件，计算均值
process_tif_files(folder_path, newfolder_path, 'tmin', 'tmin.tif', calc_type='mean')

# 调用函数处理包含 'tmax' 的 TIFF 文件，计算均值
process_tif_files(folder_path, newfolder_path, 'tmax', 'tmax.tif', calc_type='mean')

# 调用函数处理包含 'prec' 的 TIFF 文件，计算总和
process_tif_files(folder_path, newfolder_path, 'prec', 'map.tif', calc_type='sum')

# 调用函数处理包含 'srad' 的 TIFF 文件，计算均值
process_tif_files(folder_path, newfolder_path, 'srad', 'srad.tif', calc_type='mean')

# import os
# import shutil

# def move_tif_files(src_folder, dest_folder, keywords,prefix_to_remove='wc2.1_5m_'):
#     """
#     将指定文件夹中包含关键词的 TIFF 文件移动到另一个文件夹。
    
#     :param src_folder: 源文件夹路径
#     :param dest_folder: 目标文件夹路径
#     :param keywords: 要匹配的关键词列表
#     """
#     # 确保目标文件夹存在
#     if not os.path.exists(dest_folder):
#         os.makedirs(dest_folder)
    
#     # 获取源文件夹中的所有文件
#     found_files = False  # 标记是否找到符合条件的文件
#     for file in os.listdir(src_folder):
#         print(f"检查文件: {file}")  # 打印出每个文件名以供调试
#         # 检查文件是否是 .tif 文件，并且文件名中包含指定的关键词之一
#         if file.lower().endswith('.tif') and any(keyword.lower() in file.lower() for keyword in keywords):
#             found_files = True
#             print(f"找到符合条件的文件: {file}")
#             src_path = os.path.join(src_folder, file)
#             if file.lower().startswith(prefix_to_remove.lower()):
#                 file = file[len(prefix_to_remove):]
#             dest_path = os.path.join(dest_folder, file)
            
#             # 移动文件
#             shutil.copy(src_path, dest_path)
#             print(f"已将文件 {file} 移动到 {dest_folder}")
    
#     if not found_files:
#         print("未找到符合条件的文件")

# # 如果该脚本作为主程序运行
# if __name__ == "__main__":
#     # 设置文件夹路径
#     folder_path = r'wc2.1_5m'  # 替换为你的源文件夹路径
#     newfolder_path = r'new'  # 替换为你的目标文件夹路径

#     # 关键词列表
#     keywords = ['bio', 'elev']

#     # 调用函数
#     move_tif_files(folder_path, newfolder_path, keywords)
