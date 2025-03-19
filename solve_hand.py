import os
import subprocess
import rasterio
from rasterio.enums import Resampling

# 使用 GDAL Warp 进行重采样
def gdalwarp_resample(input_tif, target_tif, target_dx, target_dy, output_folder, target_width, target_height):
    # 读取原始 TIFF 文件的宽度和高度
    with rasterio.open(input_tif) as src:
        original_width = src.width
        original_height = src.height

    # 构建 gdalwarp 命令行，用于重采样
    print(str(target_dx))
    resample_cmd = [
        'gdalwarp', 
        '-tr', str(target_dx), str(target_dy),  # 设置目标分辨率
        '-r', 'bilinear',  # 设置重采样方法（你可以选择 'nearest', 'bilinear', 'cubic' 等）
        input_tif, target_tif  # 输入文件和输出文件
    ]
    
    # 执行 gdalwarp 命令进行分辨率重采样
    subprocess.run(resample_cmd, check=True)
    print(f"重采样后的文件已保存为: {target_tif}")
    
    # 获取目标 TIFF 的形状
    resampled_width, resampled_height = get_shape(target_tif)

    # 打印原始文件形状
    with rasterio.open(input_tif) as src:
        print(f"原始文件形状: {src.width} x {src.height}")

    print(f"重采样后的文件形状: {resampled_width} x {resampled_height}")
    
    # 如果重采样后的文件尺寸与目标尺寸不一致，进行调整
    if resampled_width != target_width or resampled_height != target_height:
        print("重采样后的文件形状与目标形状不匹配，使用 gdal_translate 调整图像大小")
        final_tif = os.path.join(output_folder, 'final_resampled_hand.tif')
        translate_cmd = [
            'gdal_translate',
            '-outsize', str(target_width), str(target_height),  # 设置最终输出文件的宽度和高度
            target_tif, final_tif  # 输入重采样后的文件和输出文件
        ]
        # 执行 gdal_translate 命令调整图像大小
        subprocess.run(translate_cmd, check=True)
        print(f"最终输出文件已保存为: {final_tif}")
        return final_tif  # 返回调整大小后的文件
    else:
        print("重采样后的文件形状与目标文件形状相同，无需进一步调整。")
        return target_tif  # 返回重采样后的文件

# 获取 TIFF 文件的分辨率
def get_resolution(tif_file):
    with rasterio.open(tif_file) as src:
        transform = src.transform
        dx = transform[0]  # X方向分辨率
        dy = -transform[4]  # Y方向分辨率
    return dx, dy

# 获取 TIFF 文件的形状
def get_shape(tif_file):
    with rasterio.open(tif_file) as src:
        return src.width, src.height

# 比较分辨率并进行重采样
def compare_and_resample(tif1, tif2, output_folder):
    # 获取两个 TIFF 文件的分辨率
    dx1, dy1 = get_resolution(tif1)
    dx2, dy2 = get_resolution(tif2)

    print(f"tif1 分辨率: dx={dx1}, dy={dy1}")
    print(f"tif2 分辨率: dx={dx2}, dy={dy2}")

    # 获取目标文件的宽度和高度
    with rasterio.open(tif2) as src:
        target_width = src.width
        target_height = src.height

    # 如果分辨率不同，执行重采样
    if dx1 != dx2 or dy1 != dy2:
        print("分辨率不同，进行重采样")
        # 将 tif1 重采样到 tif2 的分辨率
        target_tif = os.path.join(output_folder, 'resampled_hand.tif')
        
        final_resampled_file = gdalwarp_resample(tif1, target_tif, dx2, dy2, output_folder, target_width, target_height)
        
        # 获取目标 TIFF 的形状
        original_shape = get_shape(tif2)
        resampled_shape = get_shape(final_resampled_file)

        print(f"原始文件形状: {original_shape}")
        print(f"重采样文件形状: {resampled_shape}")

        # 如果形状不同，抛出异常
        if original_shape != resampled_shape:
            raise ValueError(f"重采样后的文件形状 {resampled_shape} 与原始文件形状 {original_shape} 不相同")
    else:
        print("两个 TIFF 文件的分辨率相同，无需重采样。")

# 示例文件
tif1 = 'hand/hand100_4326.tif'
tif2 = 'new/bio_13.tif'
output_folder = 'new'

# 比较并进行重采样
compare_and_resample(tif1, tif2, output_folder)
