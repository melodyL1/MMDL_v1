import os
import numpy as np
import rasterio
from osgeo import gdal, gdalconst


def setnodata(band, v):
    nodata_value = band.GetNoDataValue()
    # 将a_band中的数据读取为数组
    a_array = band.ReadAsArray()
    # 将数组中所有为0的值替换为无数据值
    a_array = np.where(a_array == v, nodata_value, a_array)
    # 将修改后的数组写回到a_band
    band.WriteArray(a_array)

def NoDataCount(dataset):
    band = dataset.GetRasterBand(1)
    image_data = band.ReadAsArray()

    # 获取nodata像元值
    nodata_value = band.GetNoDataValue()

    # 统计nodata像元值的数量
    nodata_count = np.count_nonzero(image_data == nodata_value)

    # 输出nodata像元值的数量
    print("The nodata pixels:",nodata_value)
    print("The count of nodata pixels is:", nodata_count)
    return nodata_value, nodata_count


# 定义文件路径和文件名
# folder_path = r"Y:\lipengao\sar\4"
# mask_file_path = r"Q:\s2\s2\4qvyunpinjie\muniheiyanjiuqvyanmo.tif"
# # 云文件路径
# q_file_path = r"Y:\lipengao\sar\4\quality_scene_classification"
# q_suffix = "quality_scene_classification.tif"
# # 输出文件夹
# out_path_0 = r"Y:\lipengao\sar\5"

def S3(folder_path, out_path_0, mask_file_path):
    print(folder_path)
    print(out_path_0)
    print(mask_file_path)

def S2(folder_path, out_path_0, mask_file_path):
    q_file_path = folder_path + "\quality_scene_classification"
    q_suffix = "quality_scene_classification.tif"
    # 遍历指定路径下的所有文件夹和文件
    for root, dirs, files in os.walk(folder_path):
        # 遍历文件夹
        for directory in dirs:
            if directory.startswith("B"):  # 如果文件夹名以字母"B"开头
                folder_path = os.path.join(root, directory)  # 获取文件夹的完整路径

                # 遍历文件夹内的所有文件
                for file in os.listdir(folder_path):
                    if file.endswith(".tif"):  # 如果文件是.tif格式图像,获取图像文件的完整路径
                        file_path = os.path.join(folder_path, file)
                        name = os.path.basename(file_path)
                        print(file_path)
                        b_file = gdal.Open(file_path, gdalconst.GA_Update)
                        b_nodata, b_nodata_count = NoDataCount(b_file)
                        # 将mask_file的所有像元值替换为b_nodata值，同时将Nodata值替换为b_nodata # 设置b_nodata值
                        b_nodata_array = np.array([[b_nodata]])
                        # 改写yanjiuqvyanmo
                        mask_file = gdal.Open(mask_file_path, gdalconst.GA_Update)
                        # 获取maske_file的行数和列数
                        mask_cols = mask_file.RasterXSize
                        mask_rows = mask_file.RasterYSize
                        # 获取maske_file的Nodata值
                        mask_nodata = mask_file.GetRasterBand(1).GetNoDataValue()
                        array = mask_file.ReadAsArray()
                        array[:] = b_nodata
                        mask_file.GetRasterBand(1).WriteArray(array, 0, 0)
                        mask_file.FlushCache()
                        mask_file.GetRasterBand(1).SetNoDataValue(b_nodata)
                        NoDataCount(mask_file)
                        mask_array = mask_file.GetRasterBand(1).ReadAsArray()
                        # 读取云掩膜文件
                        q_name = name[:60] + q_suffix
                        q_file = gdal.Open(os.path.join(q_file_path, q_name))
                        # 获取q_file和b_file的行数和列数
                        q_cols = q_file.RasterXSize
                        q_rows = q_file.RasterYSize
                        b_cols = b_file.RasterXSize
                        b_rows = b_file.RasterYSize

                        # 定义要记录满足条件的像元位置的列表
                        pixel_positions = []

                        # # 遍历q_file的像元值
                        # for row in range(q_rows):
                        #     for col in range(q_cols):
                        #         q_pixel_value = q_file.GetRasterBand(1).ReadAsArray(col, row, 1, 1)[0, 0]
                        #
                        #         # 判断像元值是否满足条件
                        #         if q_pixel_value in [2, 3, 9, 10]:
                        #             # 记录满足条件的像元位置
                        #             pixel_positions.append((row, col))
                        #
                        #             # 将b_file相应位置的像元值改为b_nodata
                        #             b_file.GetRasterBand(1).WriteArray(b_nodata_array, col, row)
                        #             NoDataCount(b_file)

                        # 读取q_file和b_file的数据为NumPy数组
                        q_data = q_file.GetRasterBand(1).ReadAsArray()
                        b_data = b_file.GetRasterBand(1).ReadAsArray()

                        # 定义条件
                        conditions = np.isin(q_data, [2, 3, 9, 10])
                        # 使用np.where函数选择满足条件的像素位置
                        row_indices, col_indices = np.where(conditions)
                        # 记录满足条件的像素位置
                        pixel_positions = list(zip(row_indices, col_indices))
                        # 将b_file相应位置的像元值改为b_nodata
                        b_data[row_indices, col_indices] = 0

                        # 将修改后的b_data写入b_file
                        b_file.GetRasterBand(1).WriteArray(b_data)
                        NoDataCount(b_file)

                        # 获取重叠区域
                        b_band = b_file.GetRasterBand(1)
                        b_data = b_band.ReadAsArray()
                        mask_band = mask_file.GetRasterBand(1)
                        mask_data = mask_band.ReadAsArray()
                        driver = gdal.GetDriverByName('GTiff')
                        out_path = out_path_0
                        out_path = os.path.join(out_path, name[-6:-4])
                        out_path_2 = os.path.join(out_path, name)
                        # 检查目录是否存在
                        if not os.path.exists(out_path):
                            # 目录不存在，创建它
                            os.makedirs(out_path)
                        a_dataset = driver.Create(out_path_2, mask_file.RasterXSize, mask_file.RasterYSize, 1,
                                                  gdal.GDT_Float32)
                        # 将 mask_file 数据复制到 a.tif
                        a_band = a_dataset.GetRasterBand(1)
                        a_band.WriteArray(mask_data)

                        # 获取 b_file 与 maske_file 的重叠部分
                        b_cols = b_file.RasterXSize
                        b_rows = b_file.RasterYSize
                        mask_cols = mask_file.RasterXSize
                        mask_rows = mask_file.RasterYSize

                        # 获取b_file和mask_file的坐标转换器
                        b_transform = b_file.GetGeoTransform()
                        mask_transform = mask_file.GetGeoTransform()

                        # 确定相交部分的范围
                        b_transform0 = float(round(b_transform[0]))
                        b_transform1 = float(round(b_transform[1]))
                        b_transform2 = float(round(b_transform[2]))
                        b_transform3 = float(round(b_transform[3]))
                        b_transform4 = float(round(b_transform[4]))
                        b_transform5 = float(round(b_transform[5]))
                        mask_transform0 = float(round(mask_transform[0]))
                        mask_transform1 = float(round(mask_transform[1]))
                        mask_transform2 = float(round(mask_transform[2]))
                        mask_transform3 = float(round(mask_transform[3]))
                        mask_transform4 = float(round(mask_transform[4]))
                        mask_transform5 = float(round(mask_transform[5]))
                        zero = int(0)

                        b_x = b_transform0 + (b_transform1 * b_cols)
                        b_y = b_transform3 - (b_transform1 * b_rows)

                        mask_x = mask_transform0 + (mask_transform1 * mask_cols)
                        mask_y = mask_transform3 - (mask_transform1 * mask_rows)

                        # 情况一
                        if mask_transform0 > b_transform0 and mask_transform3 < b_transform3 and b_x > mask_x and b_y < mask_y:
                            x_size = mask_file.RasterXSize
                            y_size = mask_file.RasterYSize
                            x_offset = (mask_transform0 - b_transform0) / mask_transform1
                            y_offset = (b_transform3 - mask_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=int(x_offset), yoff=int(y_offset),
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=zero, yoff=zero)

                        elif mask_transform0 > b_transform0 and mask_transform3 < b_transform3 and b_x < mask_x and b_y < mask_y:
                            x_size = mask_file.RasterXSize - ((mask_x - b_x) / mask_transform1)
                            y_size = mask_file.RasterYSize
                            x_offset = (mask_transform0 - b_transform0) / mask_transform1
                            y_offset = (b_transform3 - mask_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=int(x_offset), yoff=int(y_offset),
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=zero, yoff=zero)

                        elif mask_transform0 > b_transform0 and mask_transform3 < b_transform3 and b_x > mask_x and b_y > mask_y:
                            x_size = mask_file.RasterXSize
                            y_size = mask_file.RasterYSize - ((b_y - mask_y) / mask_transform1)
                            x_offset = (mask_transform0 - b_transform0) / mask_transform1
                            y_offset = (b_transform3 - mask_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=int(x_offset), yoff=int(y_offset),
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=zero, yoff=zero)

                        elif mask_transform0 > b_transform0 and mask_transform3 < b_transform3 and b_x < mask_x and b_y > mask_y:
                            x_size = mask_file.RasterXSize - ((mask_x - b_x) / mask_transform1)
                            y_size = mask_file.RasterYSize - ((b_y - mask_y) / mask_transform1)
                            x_offset = (mask_transform0 - b_transform0) / mask_transform1
                            y_offset = (b_transform3 - mask_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=int(x_offset), yoff=int(y_offset),
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=zero, yoff=zero)

                        # 情况二
                        elif mask_transform0 > b_transform0 and mask_transform3 > b_transform3 and b_x > mask_x and b_y < mask_y:
                            x_size = mask_file.RasterXSize
                            y_size = mask_file.RasterYSize - ((mask_transform3 - b_transform3) / mask_transform1)
                            x_offset = (mask_transform0 - b_transform0) / mask_transform1
                            y_offset = (mask_transform3 - b_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=int(x_offset), yoff=zero,
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=zero, yoff=int(y_offset))

                        elif mask_transform0 > b_transform0 and mask_transform3 > b_transform3 and b_x < mask_x and b_y < mask_y:
                            x_size = mask_file.RasterXSize - ((mask_x - b_x) / mask_transform1)
                            y_size = mask_file.RasterYSize - ((mask_transform3 - b_transform3) / mask_transform1)
                            x_offset = (mask_transform0 - b_transform0) / mask_transform1
                            y_offset = (mask_transform3 - b_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=int(x_offset), yoff=zero,
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=zero, yoff=int(y_offset))

                        elif mask_transform0 > b_transform0 and mask_transform3 > b_transform3 and b_x > mask_x and b_y > mask_y:
                            x_size = mask_file.RasterXSize
                            y_size = b_file.RasterYSize
                            x_offset = (mask_transform0 - b_transform0) / mask_transform1
                            y_offset = (mask_transform3 - b_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=int(x_offset), yoff=zero,
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=zero, yoff=int(y_offset))

                        elif mask_transform0 > b_transform0 and mask_transform3 > b_transform3 and b_x < mask_x and b_y > mask_y:
                            x_size = mask_file.RasterXSize - ((mask_x - b_x) / mask_transform1)
                            y_size = b_file.RasterYSize
                            x_offset = (mask_transform0 - b_transform0) / mask_transform1
                            y_offset = (mask_transform3 - b_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=int(x_offset), yoff=zero,
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=zero, yoff=int(y_offset))

                        # 情况三
                        elif mask_transform0 < b_transform0 and mask_transform3 < b_transform3 and b_x > mask_x and b_y < mask_y:
                            x_size = mask_file.RasterXSize - ((b_transform0 - mask_transform0) / mask_transform1)
                            y_size = mask_file.RasterYSize
                            x_offset = (b_transform0 - mask_transform0) / mask_transform1
                            y_offset = (b_transform3 - mask_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=zero, yoff=int(y_offset),
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=int(x_offset), yoff=zero)

                        elif mask_transform0 < b_transform0 and mask_transform3 < b_transform3 and b_x < mask_x and b_y < mask_y:
                            x_size = b_file.RasterXSize
                            y_size = mask_file.RasterYSize
                            x_offset = (b_transform0 - mask_transform0) / mask_transform1
                            y_offset = (b_transform3 - mask_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=zero, yoff=int(y_offset),
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=int(x_offset), yoff=zero)

                        elif mask_transform0 < b_transform0 and mask_transform3 < b_transform3 and b_x > mask_x and b_y > mask_y:
                            x_size = mask_file.RasterXSize - ((b_transform0 - mask_transform0) / mask_transform1)
                            y_size = mask_file.RasterYSize - ((b_y - mask_y) / mask_transform1)
                            x_offset = (b_transform0 - mask_transform0) / mask_transform1
                            y_offset = (b_transform3 - mask_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=zero, yoff=int(y_offset),
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=int(x_offset), yoff=zero)

                        elif mask_transform0 < b_transform0 and mask_transform3 < b_transform3 and b_x < mask_x and b_y > mask_y:
                            x_size = b_file.RasterXSize
                            y_size = mask_file.RasterYSize - ((b_y - mask_y) / mask_transform1)
                            x_offset = (b_transform0 - mask_transform0) / mask_transform1
                            y_offset = (b_transform3 - mask_transform3) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=zero, yoff=int(y_offset),
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=int(x_offset), yoff=zero)

                        # 情况四
                        elif mask_transform0 < b_transform0 and mask_transform3 > b_transform3 and b_x > mask_x and b_y < mask_y:
                            x_size = mask_file.RasterXSize - ((b_transform0 - mask_transform0) / mask_transform1)
                            y_size = mask_file.RasterYSize - ((mask_transform3 - b_transform3) / mask_transform1)
                            x_offset = (b_transform0 - mask_transform0) / mask_transform1
                            y_offset = (mask_transform3 - b_transform0) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=zero, yoff=zero,
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=int(x_offset), yoff=int(y_offset))

                        elif mask_transform0 < b_transform0 and mask_transform3 > b_transform3 and b_x < mask_x and b_y < mask_y:
                            x_size = b_file.RasterXSize
                            y_size = b_file.RasterYSize - ((mask_y - b_y) / mask_transform1)
                            x_offset = (b_transform0 - mask_transform0) / mask_transform1
                            y_offset = (mask_transform3 - b_transform0) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=zero, yoff=zero,
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=int(x_offset), yoff=int(y_offset))

                        elif mask_transform0 < b_transform0 and mask_transform3 > b_transform3 and b_x > mask_x and b_y > mask_y:
                            x_size = mask_file.RasterXSize - ((b_transform0 - mask_transform0) / mask_transform1)
                            y_size = b_file.RasterYSize
                            x_offset = (b_transform0 - mask_transform0) / mask_transform1
                            y_offset = (mask_transform3 - b_transform0) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=zero, yoff=zero,
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=int(x_offset), yoff=int(y_offset))

                        elif mask_transform0 < b_transform0 and mask_transform3 > b_transform3 and b_x < mask_x and b_y > mask_y:
                            x_size = b_file.RasterXSize
                            y_size = b_file.RasterYSize
                            x_offset = (b_transform0 - mask_transform0) / mask_transform1
                            y_offset = (mask_transform3 - b_transform0) / mask_transform1
                            b_data = b_band.ReadAsArray(xoff=zero, yoff=zero,
                                                        win_xsize=int(x_size), win_ysize=int(y_size))
                            a_band.WriteArray(b_data, xoff=int(x_offset), yoff=int(y_offset))
                        else:
                            print("数据没有相交，无法裁剪")

                        # 设置仿射变换和投影信息
                        a_dataset.SetGeoTransform(mask_file.GetGeoTransform())
                        a_dataset.SetProjection(mask_file.GetProjection())
                        a_dataset.GetRasterBand(1).SetNoDataValue(b_nodata)
                        NoDataCount(a_dataset)
                        setnodata(a_band, 0)
                        # 关闭数据集
                        b_file = None
                        mask_file = None
                        a_file = None
