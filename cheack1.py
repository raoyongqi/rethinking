import rasterio

def check_tif_coordinates(tif_file):
    with rasterio.open(tif_file) as src:
        transform = src.transform  # 获取坐标变换信息
        crs = src.crs  # 获取坐标参考系统
    return transform, crs

# 读取两个 TIF 文件的坐标信息
bio_1_transform, bio_1_crs = check_tif_coordinates('new/bio_1.tif')
hand_transform, hand_crs = check_tif_coordinates('new/hand.tif')

# 打印坐标变换信息和坐标参考系统
print(f"bio_1.tif Transform: {bio_1_transform}")
print(f"bio_1.tif CRS: {bio_1_crs}")
print(f"hand.tif Transform: {hand_transform}")
print(f"hand.tif CRS: {hand_crs}")

# 检查坐标系是否相同
if bio_1_crs == hand_crs:
    print("两个 TIF 文件使用相同的坐标参考系统。")
else:
    print("两个 TIF 文件使用不同的坐标参考系统。")
