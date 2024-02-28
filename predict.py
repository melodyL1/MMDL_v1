import os
import torch
import argparse
import numpy as np
import yaml
import time
import auxil
from TPPI.utils import convert_state_dict
from TPPI.models import get_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from osgeo import gdal, osr


def predict_patches(data, model, cfg, device, logdir):
    transfer_data_start = time.time()
    started = cfg["data"]["started"]
    end = cfg["data"]["end"]
    if cfg["model"] == 'sklearn':
        data2 = data[:,started:end,:,:]
        sklearndata = data2.permute(0, 2, 3, 1)
        sklearndata = sklearndata.numpy()
        sklearndata = sklearndata.reshape((sklearndata.shape[0], -1))

        # data2 = data[:, started:end, 2, 2]
        # sklearndata = data2.numpy()
    else:
        data = data.to(device)
    transfer_data_end = time.time()
    transfer_time = transfer_data_end - transfer_data_start
    predicted = []
    bs = cfg["prediction"]["batch_size"]
    num_components = cfg["data"]["num_components"]
    tsp = time.time()
    if cfg["model"] == 'HybridSN_multi':
        loaded_weights = np.loadtxt(os.path.join(logdir,"weights.csv"), delimiter=',')
    if cfg["model"] == 'sklearn':
        import joblib
        loaded_model = joblib.load(logdir+'/model.pkl')
        Probability = loaded_model.predict_proba(sklearndata)
        print('sklearn')
    else:
        with torch.no_grad():
            for i in range(0, data.shape[0], bs):
                end_index = i + bs
                batch_data = data[i:end_index]
                batch_data = batch_data[:, started:end, :, :]

                if cfg["model"] == 'HybridSN_multi':
                    outputs,q1,q2,q3,q4,q5 = model(batch_data,loaded_weights)
                elif cfg["model"] != 'HybridSN_multi':
                    outputs = model(batch_data)
                [predicted.append(a) for a in outputs.cpu().numpy()]
        Probability = np.array(predicted)
    tep = time.time()
    prediction_time = tep - tsp
    return prediction_time, transfer_time, Probability

def normalizations(data):
    min_value = np.min(data[data > 0])
    max_value = np.max(data[data > 0])
    data[data > 0] = (data[data > 0] - min_value) / (max_value - min_value)
    return data

from sklearn.decomposition import PCA
def applyPCA(X, numComponents=15):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca

def timeCost_TPPP2(cfg, logdir, input_file):
    name = cfg["data"]["dataset"]
    datasetname = str(name)
    modelname= str(cfg['model'])
    pca_components=str(cfg["data"]["num_components"])
    device = auxil.get_device()
    # 设置保存路径
    savepath = './Result/' + datasetname +"_"+ modelname +"_PPsize"+ str(cfg['data']['PPsize'])+"_epochs"+str(cfg['train']['epochs'])+"_PCA"+pca_components+'/'
    try:
        os.makedirs(savepath)
    except FileExistsError:
        pass

    # Setup image
    from osgeo import gdal
    data_path = 'dataset/'
    tif = input_file
    ytif = cfg['prediction']['y']

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


    datasets = gdal.Open(os.path.join(data_path, tif), gdal.GA_ReadOnly)
    # 获取波段数和图像尺寸
    num_bands = datasets.RasterCount
    width = datasets.RasterXSize
    height = datasets.RasterYSize
    # 创建一个空的 NumPy 数组来存储数据
    img = np.empty((height, width, num_bands), dtype=np.float32)
    # 逐个读取每个波段的数据并存储到数组中
    for band_idx in range(num_bands):
        band = datasets.GetRasterBand(band_idx + 1)  # 波段索引从1开始
        img[:, :, band_idx] = band.ReadAsArray()
    print("img shape is {} ".format(img.shape))

    #归一化
    data = img
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
        data2 = data[:, :, ndvi_components:ndvi_components+sar_components]
        data3 = data[:, :, ndvi_components+sar_components:ndvi_components+sar_components+sar_components]
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
    img = data

    #PCA or not
    if PCAA > 0:
        shapeor = img.shape
        data = img.reshape(-1, img.shape[-1])
        data = PCA(n_components=PCAA).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = PCAA
        # Normalization
        data = StandardScaler().fit_transform(data)
        img = data.reshape(shapeor)
        print("PCA_data.shape:",data.shape)

    yset = gdal.Open(os.path.join(data_path, ytif), gdal.GA_ReadOnly)
    # 获取波段数和图像尺寸
    num_bands = yset.RasterCount
    width = yset.RasterXSize
    height = yset.RasterYSize
    # 创建一个空的 NumPy 数组来存储数据
    y = np.empty((height, width, num_bands), dtype=np.float32)
    # 逐个读取每个波段的数据并存储到数组中
    for band_idx in range(num_bands):
        band = yset.GetRasterBand(band_idx + 1)  # 波段索引从1开始
        y[:, :, band_idx] = band.ReadAsArray()
    print("gt shape is {} ".format(y.shape))
    gt = y.astype('uint8')


    # image processing
    time_pre_start = time.time()
    # StandardScaler
    #sahnchu
    # img = img[:, :, 5, np.newaxis]
    shapeor = img.shape
    img = img.reshape(-1, img.shape[-1])
    img = StandardScaler().fit_transform(img)
    img = img.reshape(shapeor)
    # create patch
    data = auxil.creat_PP(cfg["data"]["PPsize"], img)
    # NHWC -> NCHW
    data = data.transpose(0, 3, 1, 2)
    data = torch.from_numpy(data).float()
    time_pre_end = time.time()
    time_pre_processing = time_pre_end - time_pre_start
    print("creat patch {} data over!", data.shape)

    # setup model:
    model = get_model(cfg['model'], cfg['data']['dataset'])
    if cfg["model"] != 'sklearn':
        state = convert_state_dict(
            torch.load(os.path.join(logdir, cfg["train"]["best_model_path"]))[
                "model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(device)

    # transfer model to GPU
    model.to(device)

    # predicting
    print("predicting...")
    pt, tt, outputs = predict_patches(data, model, cfg, device, logdir)
    print(outputs.shape)

    # get result and reshape
    outputs = np.array(outputs)
    pred = np.argmax(outputs, axis=1)
    pred = np.reshape(pred, (img.shape[0], img.shape[1]))

    # show predicted result
    pred += 1
    from PIL import Image
    # 将NumPy数组转换为PIL图像
    image = Image.fromarray(np.uint8(pred))
    # 设置保存路径
    # save_path = r'Y:\lipengao\daima\TPPI\Result\picture\output.tif'
    # # 保存图像为.tif文件
    # image.save(save_path)

    auxil.decode_segmap(pred)
    geotransform = yset.GetGeoTransform()
    projection = yset.GetProjection()
    output_path = savepath + datasetname + "_" + modelname + "_predictions_All.tif"

    # Create a new raster dataset
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(
        output_path,
        pred.shape[1],  # Width
        pred.shape[0],  # Height
        1,  # Number of bands
        gdal.GDT_Byte  # Data type (change if needed)
    )

    # Apply the geotransform and projection information
    output_dataset.SetGeoTransform(geotransform)
    output_dataset.SetProjection(projection)

    # Write the predicted result to the raster band
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(pred.astype(np.uint8))  # Adjust data type if needed
    output_band.FlushCache()

    # Close the dataset to save changes
    output_dataset = None

def timeCost_TPPP(cfg, logdir):
    name = cfg["data"]["dataset"]
    datasetname = str(name)
    modelname= str(cfg['model'])
    pca_components=str(cfg["data"]["num_components"])
    device = auxil.get_device()
    # 设置保存路径
    savepath = './Result/' + datasetname +"_"+ modelname +"_PPsize"+ str(cfg['data']['PPsize'])+"_epochs"+str(cfg['train']['epochs'])+"_PCA"+pca_components+'/'
    try:
        os.makedirs(savepath)
    except FileExistsError:
        pass

    # Setup image
    from osgeo import gdal
    data_path = 'dataset/'
    tif = cfg['data']['data']
    ytif = cfg['prediction']['y']

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


    datasets = gdal.Open(os.path.join(data_path, tif), gdal.GA_ReadOnly)
    # 获取波段数和图像尺寸
    num_bands = datasets.RasterCount
    width = datasets.RasterXSize
    height = datasets.RasterYSize
    # 创建一个空的 NumPy 数组来存储数据
    img = np.empty((height, width, num_bands), dtype=np.float32)
    # 逐个读取每个波段的数据并存储到数组中
    for band_idx in range(num_bands):
        band = datasets.GetRasterBand(band_idx + 1)  # 波段索引从1开始
        img[:, :, band_idx] = band.ReadAsArray()
    print("img shape is {} ".format(img.shape))

    #归一化
    data = img
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
        data2 = data[:, :, ndvi_components:ndvi_components+sar_components]
        data3 = data[:, :, ndvi_components+sar_components:ndvi_components+sar_components+sar_components]
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
    img = data

    #PCA or not
    if PCAA > 0:
        shapeor = img.shape
        data = img.reshape(-1, img.shape[-1])
        data = PCA(n_components=PCAA).fit_transform(data)
        shapeor = np.array(shapeor)
        shapeor[-1] = PCAA
        # Normalization
        data = StandardScaler().fit_transform(data)
        img = data.reshape(shapeor)
        print("PCA_data.shape:",data.shape)

    yset = gdal.Open(os.path.join(data_path, ytif), gdal.GA_ReadOnly)
    # 获取波段数和图像尺寸
    num_bands = yset.RasterCount
    width = yset.RasterXSize
    height = yset.RasterYSize
    # 创建一个空的 NumPy 数组来存储数据
    y = np.empty((height, width, num_bands), dtype=np.float32)
    # 逐个读取每个波段的数据并存储到数组中
    for band_idx in range(num_bands):
        band = yset.GetRasterBand(band_idx + 1)  # 波段索引从1开始
        y[:, :, band_idx] = band.ReadAsArray()
    print("gt shape is {} ".format(y.shape))
    gt = y.astype('uint8')


    # image processing
    time_pre_start = time.time()
    # StandardScaler
    #sahnchu
    # img = img[:, :, 5, np.newaxis]
    shapeor = img.shape
    img = img.reshape(-1, img.shape[-1])
    img = StandardScaler().fit_transform(img)
    img = img.reshape(shapeor)
    # create patch
    data = auxil.creat_PP(cfg["data"]["PPsize"], img)
    # NHWC -> NCHW
    data = data.transpose(0, 3, 1, 2)
    data = torch.from_numpy(data).float()
    time_pre_end = time.time()
    time_pre_processing = time_pre_end - time_pre_start
    print("creat patch {} data over!", data.shape)

    # setup model:
    model = get_model(cfg['model'], cfg['data']['dataset'])
    if cfg["model"] != 'sklearn':
        state = convert_state_dict(
            torch.load(os.path.join(logdir, cfg["train"]["best_model_path"]))[
                "model_state"])
        model.load_state_dict(state)
        model.eval()
        model.to(device)

    # transfer model to GPU
    model.to(device)

    # predicting
    print("predicting...")
    pt, tt, outputs = predict_patches(data, model, cfg, device, logdir)
    print(outputs.shape)

    # get result and reshape
    outputs = np.array(outputs)
    pred = np.argmax(outputs, axis=1)
    pred = np.reshape(pred, (gt.shape[0], gt.shape[1]))

    # show predicted result
    pred += 1
    from PIL import Image
    # 将NumPy数组转换为PIL图像
    image = Image.fromarray(np.uint8(pred))
    # 设置保存路径
    # save_path = r'Y:\lipengao\daima\TPPI\Result\picture\output.tif'
    # # 保存图像为.tif文件
    # image.save(save_path)

    auxil.decode_segmap(pred)
    geotransform = yset.GetGeoTransform()
    projection = yset.GetProjection()
    output_path = savepath + datasetname + "_" + modelname + "_predictions_All.tif"

    # Create a new raster dataset
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(
        output_path,
        pred.shape[1],  # Width
        pred.shape[0],  # Height
        1,  # Number of bands
        gdal.GDT_Byte  # Data type (change if needed)
    )

    # Apply the geotransform and projection information
    output_dataset.SetGeoTransform(geotransform)
    output_dataset.SetProjection(projection)

    # Write the predicted result to the raster band
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(pred.astype(np.uint8))  # Adjust data type if needed
    output_band.FlushCache()

    # Close the dataset to save changes
    output_dataset = None



def RunPredicttif():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    name = cfg["data"]["dataset"]
    datasetname = str(name)
    modelname= str(cfg['model'])
    pca_components=str(cfg["data"]["num_components"])
    device = auxil.get_device()
    logdir = './Result/' + datasetname + "_" + modelname + "_PPsize" + str(cfg['data']['PPsize']) + "_epochs" + str(
        cfg['train']['epochs']) + "_PCA" + pca_components + '/'+str(cfg["run_ID"])
    # logdir = os.path.join("runs", cfg["model"], str(cfg["run_ID"]))
    timeCost_TPPP(cfg, logdir)

def RunPredicttif2(input_file):
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    name = cfg["data"]["dataset"]
    datasetname = str(name)
    modelname= str(cfg['model'])
    pca_components=str(cfg["data"]["num_components"])
    device = auxil.get_device()
    logdir = './Result/' + datasetname + "_" + modelname + "_PPsize" + str(cfg['data']['PPsize']) + "_epochs" + str(
        cfg['train']['epochs']) + "_PCA" + pca_components + '/'+str(cfg["run_ID"])
    # logdir = os.path.join("runs", cfg["model"], str(cfg["run_ID"]))
    timeCost_TPPP2(cfg, logdir, input_file)


def predict():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    name = cfg["data"]["dataset"]
    datasetname = str(name)
    modelname = str(cfg['model'])
    pca_components = str(cfg["data"]["num_components"])
    device = auxil.get_device()
    logdir = './Result/' + datasetname + "_" + modelname + "_PPsize" + str(cfg['data']['PPsize']) + "_epochs" + str(
        cfg['train']['epochs']) + "_PCA" + pca_components + '/' + str(cfg["run_ID"])
    # logdir = os.path.join("runs", cfg["model"], str(cfg["run_ID"]))
    timeCost_TPPP(cfg, logdir)
if __name__ == "__main__":
    yytif = 'D:/ruanjian/MultiModalDeepLearningApp/example/RunPredict/RunPredict.tif'
    RunPredicttif2(yytif)