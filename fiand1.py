import os
import rasterio
import numpy as np

def process_tif_files(folder_path, keyword, output_filename):
    """
    处理指定文件夹中的 TIFF 文件，计算包含关键词的文件的均值，并保存为新的 TIFF 文件。
    
    :param folder_path: 文件夹路径
    :param keyword: 文件名中包含的关键词（如 'wind' 或 'tavg'）
    :param output_filename: 输出的 TIFF 文件名
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
            data_sum = src.read(1).astype(np.float32)  # 使用浮点数以防止溢出
            count = 1  # 计数器，用于计算均值

        # 循环加载其他文件并累加数据
        for tif_file in tif_files[1:]:
            with rasterio.open(tif_file) as src:
                data = src.read(1).astype(np.float32)  # 读取当前文件
                data_sum += data  # 累加数据
                count += 1  # 计数增加

        # 计算均值
        data_mean = data_sum / count

        # 保存均值结果到新的 TIFF 文件
        output_file = os.path.join(newfolder_path, output_filename)
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(data_mean.astype(np.float32), 1)  # 保存为新的文件
        print(f"计算均值后的 TIFF 文件已保存为: {output_file}")

    else:
        print(f"没有找到包含 '{keyword}' 的 TIFF 文件。")

# 设置文件夹路径
folder_path = r'wc2.1_5m'  # 替换为你的文件夹路径
newfolder_path = r'new'  # 替换为你的文件夹路径

# 调用函数处理包含 'wind' 的 TIFF 文件
process_tif_files(folder_path, 'wind', 'wind.tif')

# 调用函数处理包含 'tavg' 的 TIFF 文件
process_tif_files(folder_path, 'tavg', 'tavg.tif')


# 调用函数处理包含 'wind' 的 TIFF 文件
process_tif_files(folder_path, 'wind', 'wind.tif')

# 调用函数处理包含 'tavg' 的 TIFF 文件
process_tif_files(folder_path, 'tavg', 'tavg.tif')

