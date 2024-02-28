import os
import shutil
import scipy.io as sio
import yaml
import numpy as np
import random
import argparse
from os.path import join as pjoin
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from osgeo import gdal


def setDir():
    filepath = 'dataset/split_dataset'
    if not os.path.isdir(filepath):
        os.makedirs(filepath)


def random_unison(a, b, c, rstate=None):
    assert len(a) == len(b) & len(a) == len(c)
    p = np.random.RandomState(seed=rstate).permutation(len(a))
    return a[p], b[p], c[p]


# load data , PCA (optional) and Normalization
def loadData(cfg):
    data_path = 'dataset/'
    dataset = cfg['data']["dataset"]
    num_components = cfg['data']['num_components']
    if dataset == 'xaingfu':
        data = sio.loadmat(os.path.join(data_path, '.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, '.mat'))['data']
    elif dataset == 'munihei':
        data = sio.loadmat(os.path.join(data_path, '.mat'))['data']
        labels = sio.loadmat(os.path.join(data_path, '.mat'))['data']
    else:
        print("NO DATASET")
        exit()
    print("load {} original image successfully".format(dataset))

    shapeor = data.shape
    data = data.reshape(-1, data.shape[-1])

    # PCA or not
    if num_components is not None:
        data = PCA(n_components=num_components).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = num_components

    # Normalization
    data = StandardScaler().fit_transform(data)
    data = data.reshape(shapeor)
    num_class = len(np.unique(labels))-1
    return data, labels, num_class


def padWithZeros(X, margin):
    """
    :param X: input, shape:[H,W,C]
    :param margin: padding
    :return: new data, shape:[H+2*margin, W+2*margin, C]
    """
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def get_patch(cfg, data, x, y):
    """
    get one patch according it's position
    """
    windowSize = cfg['data']['PPsize']
    margin = int((windowSize - 1) / 2)
    x += margin
    y += margin
    zeroPaddeddata = padWithZeros(data, margin=margin)
    patch = zeroPaddeddata[x - margin:x + margin + 1, y - margin:y + margin + 1]
    return patch

# 复制填充
def Pcreat_PP_Replication_Padding(cfg, X, y):
    windowSize = cfg['data']['PPsize']
    removeZeroLabels = cfg['data']["remove_zeros"]
    margin = int((windowSize - 1) / 2)

    # 使用边界复制填充
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='edge')

    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchesLocations = []

    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchesLocations.append([r - margin, c - margin])
            patchIndex = patchIndex + 1

    # 移除标签为零的块
    patchesLocations = np.asarray(patchesLocations)
    if removeZeroLabels:
        patchesData = patchesData[(patchesLabels > 0) & (patchesLabels <= 14), :, :, :]
        patchesLocations = patchesLocations[(patchesLabels > 0) & (patchesLabels <= 14)]
        patchesLabels = patchesLabels[(patchesLabels > 0) & (patchesLabels <= 14)]
        patchesLabels -= 1

    return patchesData, patchesLabels.astype("int"), patchesLocations

def create_PP_Mirror_Padding(cfg, X, y):
    windowSize = cfg['data']['PPsize']
    removeZeroLabels = cfg['data']["remove_zeros"]
    margin = int((windowSize - 1) / 2)

    # 使用镜像填充
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='reflect')

    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchesLocations = []

    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchesLocations.append([r - margin, c - margin])
            patchIndex = patchIndex + 1

    # 移除标签为零的块
    patchesLocations = np.asarray(patchesLocations)
    if removeZeroLabels:
        patchesData = patchesData[(patchesLabels > 0) & (patchesLabels <= 14), :, :, :]
        patchesLocations = patchesLocations[(patchesLabels > 0) & (patchesLabels <= 14)]
        patchesLabels = patchesLabels[(patchesLabels > 0) & (patchesLabels <= 14)]
        patchesLabels -= 1

    return patchesData, patchesLabels.astype("int"), patchesLocations


def create_PP(cfg, X, y):
    windowSize = cfg['data']['PPsize']
    removeZeroLabels = cfg['data']["remove_zeros"]
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)

    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchesLocations = []

    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchesLocations.append([r-margin, c-margin])
            patchIndex = patchIndex + 1
    # remove unlabeled patches
    patchesLocations = np.asarray(patchesLocations)
    if removeZeroLabels:
        patchesData = patchesData[(patchesLabels > 0) & (patchesLabels <= 14), :, :, :]
        patchesLocations = patchesLocations[(patchesLabels > 0) & (patchesLabels <= 14)]
        patchesLabels = patchesLabels[(patchesLabels > 0) & (patchesLabels <= 14)]
        patchesLabels -= 1
    return patchesData, patchesLabels.astype("int"), patchesLocations

def normalizations(data):
    min_value = np.min(data[data > 0])
    max_value = np.max(data[data > 0])
    data[data > 0] = (data[data > 0] - min_value) / (max_value - min_value)
    return data

# splitting dataset
def split_data(pixels, labels, indexes, percent, rand_state=69):
        pixels_number = np.unique(labels, return_counts=1)[1]
        train_set_size = [int(np.ceil(a*percent)) for a in pixels_number]
        tr_size = int(sum(train_set_size))
        te_size = int(sum(pixels_number)) - int(sum(train_set_size))
        sizetr = np.array([tr_size]+list(pixels.shape)[1:])
        sizete = np.array([te_size]+list(pixels.shape)[1:])
        tr_index = []
        te_index = []
        train_x = np.empty((sizetr))
        train_y = np.empty((tr_size), dtype=int)
        test_x = np.empty((sizete))
        test_y = np.empty((te_size),dtype=int)
        trcont = 0
        tecont = 0
        for cl in np.unique(labels):
            pixels_cl = pixels[labels == cl]
            labels_cl = labels[labels == cl]
            indexes_cl = indexes[labels == cl]
            pixels_cl, labels_cl, indexes_cl = random_unison(pixels_cl, labels_cl, indexes_cl, rstate=rand_state)
            for cont, (a, b, c) in enumerate(zip(pixels_cl, labels_cl, indexes_cl)):
                if cont < train_set_size[cl]:
                    train_x[trcont, :, :, :] = a
                    train_y[trcont] = b
                    tr_index.append(c)
                    trcont += 1
                else:
                    test_x[tecont, :, :, :] = a
                    test_y[tecont] = b
                    te_index.append(c)
                    tecont += 1
        tr_index = np.asarray(tr_index)
        te_index = np.asarray(te_index)
        train_x, train_y, tr_index = random_unison(train_x, train_y, tr_index, rstate=rand_state)
        return train_x, test_x, train_y, test_y, tr_index, te_index
def create_dataset():
    with open("configs/config.yml") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # load data
    # data, labels, num_class = loadData(cfg)
    import torch
    device = torch.device("cpu")
    data_path = 'dataset/'
    dataset = cfg['data']["dataset"]
    tif = cfg['data']['data']
    y_train_tif = cfg['data']['y_train_tif']
    y_test_tif = cfg['data']['y_test_tif']
    y = cfg['data']['y']
    manually = cfg['test']['manually']

    num_components = cfg['data']['num_components']
    band_components = cfg['data']['band_components']
    ndvi_components = cfg['data']['ndvi_components']
    sar_components = cfg['data']['sar_components']
    dsm_components = cfg['data']['dsm_components']
    normalization = cfg['data']['normalization']
    normalization_band_dsm = cfg['data']['normalization_band_dsm']
    normalization_sar = cfg['data']['normalization_sar']
    normalization_sar_band = cfg['data']['normalization_sar_band']
    PCAA = cfg['data']['PCAA']
    # 读取data
    datasets = gdal.Open(os.path.join(data_path, tif), gdal.GA_ReadOnly)
    # 获取波段数和图像尺寸
    num_bands = datasets.RasterCount
    width = datasets.RasterXSize
    height = datasets.RasterYSize
    # 创建一个空的 NumPy 数组来存储数据
    data = np.empty((height, width, num_bands), dtype=np.float32)
    # 逐个读取每个波段的数据并存储到数组中
    for band_idx in range(num_bands):
        band = datasets.GetRasterBand(band_idx + 1)  # 波段索引从1开始
        data[:, :, band_idx] = band.ReadAsArray()
    print("data shape is {} ".format(data.shape))

    # 归一化
    if normalization_sar == 1:
        data1 = data[:, :, :ndvi_components]
        data2 = data[:, :, ndvi_components:ndvi_components + sar_components]
        data3 = data[:, :, ndvi_components + sar_components:ndvi_components + sar_components + sar_components]
        data4 = data[:, :, ndvi_components + sar_components + sar_components:]
        data1 = data1
        data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
        data3 = (data3 - np.min(data3)) / (np.max(data3) - np.min(data3))
        data4 = data4
        data_combined = np.concatenate((data1, data2, data3, data4), axis=-1)
        data = data_combined
        print(data1.shape)
        print(data2.shape)
        print(data3.shape)
        print(data4.shape)
    if normalization_band_dsm == 1:
        data = data
        data1 = data[:, :, :ndvi_components]
        data2 = data[:, :, ndvi_components:ndvi_components + band_components]
        data3 = data[:, :, ndvi_components + band_components:]
        data1 = data1
        data21 = normalizations(data2[:, :, 0])
        data22 = normalizations(data2[:, :, 1])
        data23 = normalizations(data2[:, :, 2])
        data24 = normalizations(data2[:, :, 3])
        data25 = normalizations(data2[:, :, 4])
        data2 = np.dstack((data21, data22, data23, data24, data25))
        # data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
        data31 = normalizations(data3[:, :, 0])
        data32 = normalizations(data3[:, :, 1])
        data3 = np.dstack((data31, data32))
        data_combined = np.concatenate((data1, data2, data3), axis=-1)
        # data_combined = np.dstack((data1, data2, data3))
        data = data_combined
    if normalization_sar_band == 1:
        data1 = data[:, :, :ndvi_components]
        data2 = data[:, :, ndvi_components:ndvi_components + sar_components]
        data3 = data[:, :, ndvi_components + sar_components:ndvi_components + sar_components + sar_components]
        data4 = data[:, :, ndvi_components + sar_components + sar_components:]
        data1 = data1
        data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
        data3 = (data3 - np.min(data3)) / (np.max(data3) - np.min(data3))
        data4 = data4

        data41 = normalizations(data4[:, :, 0])
        data42 = normalizations(data4[:, :, 1])
        data43 = normalizations(data4[:, :, 2])
        data44 = normalizations(data4[:, :, 3])
        data45 = normalizations(data4[:, :, 4])
        data46 = normalizations(data4[:, :, 5])
        data47 = normalizations(data4[:, :, 6])
        data48 = normalizations(data4[:, :, 7])
        data49 = normalizations(data4[:, :, 8])
        data410 = normalizations(data4[:, :, 9])
        data411 = normalizations(data4[:, :, 10])

        data4 = np.dstack((data41, data42, data43, data44, data45, data46, data47, data48, data49, data410, data411))

        data_combined = np.concatenate((data1, data2, data3, data4), axis=-1)
        data = data_combined
        print(data1.shape)
        print(data2.shape)
        print(data3.shape)
        print(data4.shape)
    if normalization == 1:
        data1 = data
        data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))
        data = data1

    # PCA or not
    if PCAA > 0:
        shapeor = data.shape
        data = data.reshape(-1, data.shape[-1])
        data = PCA(n_components=PCAA).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = PCAA
        # Normalization
        data = StandardScaler().fit_transform(data)
        data = data.reshape(shapeor)
        print("PCA_data.shape:", data.shape)

    y_gdal = gdal.Open(os.path.join(data_path, y), gdal.GA_ReadOnly)
    num_bands = y_gdal.RasterCount
    width = y_gdal.RasterXSize
    height = y_gdal.RasterYSize
    y_data = np.empty((height, width, num_bands), dtype=np.float32)
    for band_idx in range(num_bands):
        band = y_gdal.GetRasterBand(band_idx + 1)
        y_data[:, :, band_idx] = band.ReadAsArray()
    print("y_data shape is {} ".format(y_data.shape))
    y_data = y_data.astype('uint8')

    num_class = len(np.unique(y_data)) - 1

    if manually == 0:
        # 随机采样
        # # create patches
        x_train, y_train, train_index = create_PP(cfg, data, y_data)

        # splitting dataset

        x_train, x_test, y_train, y_test, train_index, test_index = split_data(x_train, y_train, train_index,
                                                                               cfg['data']["tr_percent"],
                                                                               cfg['data']["rand_state"])

        x_val, x_test, y_val, y_test, val_index, new_test_index = split_data(x_test, y_test, test_index,
                                                                             cfg['data']["val_percent"],
                                                                             cfg['data']["rand_state"])
        # positions of testSet
        test_positions = np.zeros((y_data.shape[0], y_data.shape[1]))
        for pos in test_index:
            test_positions[pos[0]][pos[1]] = 1
    else:

        yset_train = gdal.Open(os.path.join(data_path, y_train_tif), gdal.GA_ReadOnly)
        num_bands = yset_train.RasterCount
        width = yset_train.RasterXSize
        height = yset_train.RasterYSize
        y_train_data = np.empty((height, width, num_bands), dtype=np.float32)
        for band_idx in range(num_bands):
            band = yset_train.GetRasterBand(band_idx + 1)
            y_train_data[:, :, band_idx] = band.ReadAsArray()
        print("y_train shape is {} ".format(y_train_data.shape))
        y_train_data = y_train_data.astype('uint8')

        yset_test = gdal.Open(os.path.join(data_path, y_test_tif), gdal.GA_ReadOnly)
        num_bands = yset_test.RasterCount
        width = yset_test.RasterXSize
        height = yset_test.RasterYSize
        y_test_data = np.empty((height, width, num_bands), dtype=np.float32)
        for band_idx in range(num_bands):
            band = yset_test.GetRasterBand(band_idx + 1)
            y_test_data[:, :, band_idx] = band.ReadAsArray()
        print("y_test shape is {} ".format(y_test_data.shape))
        y_test_data = y_test_data.astype('uint8')
        # 手动采样
        x_train, y_train, train_index = create_PP(cfg, data, y_train_data)
        x_test, y_test, test_index = create_PP(cfg, data, y_test_data)
        x_val, x_test, y_val, y_test, val_index, new_test_index = split_data(x_test, y_test, test_index,
                                                                             cfg['data']["val_percent"],
                                                                             cfg['data']["rand_state"])
        # positions of testSet
        test_positions = np.zeros(y_test_data.shape)

        for pos in test_index:
            test_positions[pos[0]][pos[1]] = 1

    # show the shape of each dataset
    print("x_train shape:", x_train.shape)
    print("x_val shape:", x_val.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)
    print("data max:", data.max())
    print("data min:", data.min())

    setDir()
    fix_data_path = 'dataset/split_dataset/'

    # save each dataset and testSet position；
    np.save(pjoin(fix_data_path + "testSet_position.npy"), test_positions)
    np.save(pjoin(fix_data_path + "x_train.npy"), x_train)
    np.save(pjoin(fix_data_path + "x_val.npy"), x_val)
    np.save(pjoin(fix_data_path + "x_test.npy"), x_test)
    np.save(pjoin(fix_data_path + "y_train.npy"), y_train)
    np.save(pjoin(fix_data_path + "y_val.npy"), y_val)
    np.save(pjoin(fix_data_path + "y_test.npy"), y_test)
    print("creat dataset over!")


if __name__ == '__main__':
    with open("configs/config.yml") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    # load data
    # data, labels, num_class = loadData(cfg)
    import torch
    device = torch.device("cpu")
    data_path = 'dataset/'
    dataset = cfg['data']["dataset"]
    tif = cfg['data']['data']
    y_train_tif = cfg['data']['y_train_tif']
    y_test_tif = cfg['data']['y_test_tif']
    y = cfg['data']['y']
    manually = cfg['test']['manually']

    num_components = cfg['data']['num_components']
    band_components = cfg['data']['band_components']
    ndvi_components = cfg['data']['ndvi_components']
    sar_components = cfg['data']['sar_components']
    dsm_components = cfg['data']['dsm_components']
    normalization = cfg['data']['normalization']
    normalization_band_dsm = cfg['data']['normalization_band_dsm']
    normalization_sar = cfg['data']['normalization_sar']
    normalization_sar_band = cfg['data']['normalization_sar_band']
    PCAA = cfg['data']['PCAA']
# 读取data
    datasets = gdal.Open(os.path.join(data_path, tif), gdal.GA_ReadOnly)
    # 获取波段数和图像尺寸
    num_bands = datasets.RasterCount
    width = datasets.RasterXSize
    height = datasets.RasterYSize
    # 创建一个空的 NumPy 数组来存储数据
    data = np.empty((height, width, num_bands), dtype=np.float32)
    # 逐个读取每个波段的数据并存储到数组中
    for band_idx in range(num_bands):
        band = datasets.GetRasterBand(band_idx + 1)  # 波段索引从1开始
        data[:, :, band_idx] = band.ReadAsArray()
    print("data shape is {} ".format(data.shape))

    # 归一化
    if normalization_sar == 1:
        data1 = data[:, :, :ndvi_components]
        data2 = data[:, :, ndvi_components:ndvi_components+sar_components]
        data3 = data[:, :, ndvi_components+sar_components:ndvi_components+sar_components+sar_components]
        data4 = data[:, :, ndvi_components + sar_components + sar_components:]
        data1 = data1
        data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
        data3 = (data3 - np.min(data3)) / (np.max(data3) - np.min(data3))
        data4 = data4
        data_combined = np.concatenate((data1, data2, data3, data4), axis=-1)
        data = data_combined
        print(data1.shape)
        print(data2.shape)
        print(data3.shape)
        print(data4.shape)
    if normalization_band_dsm == 1:
        data = data
        data1 = data[:, :, :ndvi_components]
        data2 = data[:, :, ndvi_components:ndvi_components+band_components]
        data3 = data[:, :, ndvi_components+band_components:]
        data1 = data1
        data21 = normalizations(data2[:, :, 0])
        data22 = normalizations(data2[:, :, 1])
        data23 = normalizations(data2[:, :, 2])
        data24 = normalizations(data2[:, :, 3])
        data25 = normalizations(data2[:, :, 4])
        data2 = np.dstack((data21, data22, data23, data24, data25))
        #data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
        data31 = normalizations(data3[:, :, 0])
        data32 = normalizations(data3[:, :, 1])
        data3 = np.dstack((data31, data32))
        data_combined = np.concatenate((data1, data2, data3), axis=-1)
        #data_combined = np.dstack((data1, data2, data3))
        data = data_combined
    if normalization_sar_band == 1:
        data1 = data[:, :, :ndvi_components]
        data2 = data[:, :, ndvi_components:ndvi_components + sar_components]
        data3 = data[:, :, ndvi_components + sar_components:ndvi_components + sar_components + sar_components]
        data4 = data[:, :, ndvi_components + sar_components + sar_components:]
        data1 = data1
        data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2))
        data3 = (data3 - np.min(data3)) / (np.max(data3) - np.min(data3))
        data4 = data4

        data41 = normalizations(data4[:, :, 0])
        data42 = normalizations(data4[:, :, 1])
        data43 = normalizations(data4[:, :, 2])
        data44 = normalizations(data4[:, :, 3])
        data45 = normalizations(data4[:, :, 4])
        data46 = normalizations(data4[:, :, 5])
        data47 = normalizations(data4[:, :, 6])
        data48 = normalizations(data4[:, :, 7])
        data49 = normalizations(data4[:, :, 8])
        data410 = normalizations(data4[:, :, 9])
        data411 = normalizations(data4[:, :, 10])

        data4 = np.dstack((data41, data42, data43, data44, data45, data46, data47, data48, data49, data410, data411))

        data_combined = np.concatenate((data1, data2, data3, data4), axis=-1)
        data = data_combined
        print(data1.shape)
        print(data2.shape)
        print(data3.shape)
        print(data4.shape)
    if normalization == 1:
        data1 = data
        data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))
        data = data1

    #PCA or not
    if PCAA > 0:
        shapeor = data.shape
        data = data.reshape(-1, data.shape[-1])
        data = PCA(n_components=PCAA).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = PCAA
        # Normalization
        data = StandardScaler().fit_transform(data)
        data = data.reshape(shapeor)
        print("PCA_data.shape:",data.shape)

    y_gdal = gdal.Open(os.path.join(data_path, y), gdal.GA_ReadOnly)
    num_bands = y_gdal.RasterCount
    width = y_gdal.RasterXSize
    height = y_gdal.RasterYSize
    y_data = np.empty((height, width, num_bands), dtype=np.float32)
    for band_idx in range(num_bands):
        band = y_gdal.GetRasterBand(band_idx + 1)
        y_data[:, :, band_idx] = band.ReadAsArray()
    print("y_data shape is {} ".format(y_data.shape))
    y_data = y_data.astype('uint8')







    num_class = len(np.unique(y_data)) - 1

    if manually == 0:
        # 随机采样
        # # create patches
        x_train, y_train, train_index = create_PP(cfg, data, y_data)

        # splitting dataset

        x_train, x_test, y_train, y_test, train_index, test_index = split_data(x_train, y_train, train_index,
                                                                             cfg['data']["tr_percent"],
                                                                             cfg['data']["rand_state"])

        x_val, x_test, y_val, y_test, val_index, new_test_index = split_data(x_test, y_test, test_index,
                                                                             cfg['data']["val_percent"],
                                                                             cfg['data']["rand_state"])
        # positions of testSet
        test_positions = np.zeros((y_data.shape[0],y_data.shape[1]))
        for pos in test_index:
            test_positions[pos[0]][pos[1]] = 1
    else:

        yset_train = gdal.Open(os.path.join(data_path, y_train_tif), gdal.GA_ReadOnly)
        num_bands = yset_train.RasterCount
        width = yset_train.RasterXSize
        height = yset_train.RasterYSize
        y_train_data = np.empty((height, width, num_bands), dtype=np.float32)
        for band_idx in range(num_bands):
            band = yset_train.GetRasterBand(band_idx + 1)
            y_train_data[:, :, band_idx] = band.ReadAsArray()
        print("y_train shape is {} ".format(y_train_data.shape))
        y_train_data = y_train_data.astype('uint8')

        yset_test = gdal.Open(os.path.join(data_path, y_test_tif), gdal.GA_ReadOnly)
        num_bands = yset_test.RasterCount
        width = yset_test.RasterXSize
        height = yset_test.RasterYSize
        y_test_data = np.empty((height, width, num_bands), dtype=np.float32)
        for band_idx in range(num_bands):
            band = yset_test.GetRasterBand(band_idx + 1)
            y_test_data[:, :, band_idx] = band.ReadAsArray()
        print("y_test shape is {} ".format(y_test_data.shape))
        y_test_data = y_test_data.astype('uint8')
        # 手动采样
        x_train, y_train, train_index = create_PP(cfg, data, y_train_data)
        x_test,  y_test,  test_index = create_PP(cfg, data, y_test_data)
        x_val, x_test, y_val, y_test, val_index, new_test_index = split_data(x_test, y_test, test_index,
                                                                             cfg['data']["val_percent"],
                                                                             cfg['data']["rand_state"])
        #positions of testSet
        test_positions = np.zeros(y_test_data.shape)

        for pos in test_index:
            test_positions[pos[0]][pos[1]] = 1



    # show the shape of each dataset
    print("x_train shape:", x_train.shape)
    print("x_val shape:", x_val.shape)
    print("x_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)
    print("data max:", data.max())
    print("data min:", data.min())

    setDir()
    fix_data_path = 'dataset/split_dataset/'

    # save each dataset and testSet position；
    np.save(pjoin(fix_data_path+"testSet_position.npy"), test_positions)
    np.save(pjoin(fix_data_path+"x_train.npy"), x_train)
    np.save(pjoin(fix_data_path+"x_val.npy"), x_val)
    np.save(pjoin(fix_data_path+"x_test.npy"), x_test)
    np.save(pjoin(fix_data_path + "y_train.npy"), y_train)
    np.save(pjoin(fix_data_path + "y_val.npy"), y_val)
    np.save(pjoin(fix_data_path + "y_test.npy"), y_test)
    print("creat dataset over!")

