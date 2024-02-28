import os
import yaml
import shutil
import random
import argparse
from tensorboardX import SummaryWriter
import torch
import torch.nn.parallel
import numpy as np
import time
from torch.autograd.variable import Variable
from TPPI.models import get_model
from TPPI.optimizers import get_optimizer
from TPPI.schedulers import get_scheduler
from TPPI.loaders.Dataloader_train import get_trainLoader
from TPPI.utils import get_logger
import auxil
from TPPP_predict import RunPredict
from predict import RunPredicttif
import warnings
import sys
import subprocess
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
# action参数可以设置为ignore，一位一次也不喜爱你是，once表示为只显示一次
warnings.filterwarnings(action='ignore')


def train(cfg, train_loader, val_loader, model, loss_fn, optimizer, device, tr_writer, val_writer, logdir, logger, xx, yy):
    # tr_writer：训练时 TensorBoard 的写入器；logdir: TensorBoard 的日志文件目录。
    # logger: 日志记录器。
    # Setup lr_scheduler
    scheduler = get_scheduler(optimizer, cfg["train"]["lr_schedule"])
    best_err1 = 100
    # TODO 1
    if cfg["model"] == 'sklearn':
        from sklearn.ensemble import RandomForestClassifier
        import joblib

        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(xx, yy)
        joblib.dump(rf_classifier, logdir+'/model.pkl')
        print('Sklearn Finish')

    else:
        save_epoch = []
        start_epoch = 0
        continue_path = os.path.join(logdir, "continue_model.pkl")#表示一个模型的保存路径，在训练过程中可以从该模型继续训练。
        if os.path.isfile(continue_path):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(continue_path)#从检查点加载模型和优化器
            )
            checkpoint = torch.load(continue_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    continue_path, checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(continue_path))

        best_acc = -1
        epoch = start_epoch
        flag = True
        num_components = cfg["data"]["num_components"]
        classs = cfg["data"]["classs"]
        normalization_band_dsm = cfg['data']['normalization_band_dsm']
        normalization_sar_band = cfg['data']['normalization_sar_band']
        normalization_sar = cfg['data']['normalization_sar']
        started = cfg["data"]["started"]
        end = cfg["data"]["end"]
        all_targets0 = []
        all_pred11 = []
        all_pred22 = []
        all_pred33 = []
        all_pred44 = []
        all_pred55 = []
        weights = np.ones((5, classs))
        while epoch <= cfg["train"]["epochs"] and flag:
            model.train()
            train_accs = np.ones((len(train_loader))) * -1000.0
            train_losses = np.ones((len(train_loader))) * -1000.0
            # TODO 2
            if cfg["model"] == 'HybridSN_multi':
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                    #：将输入和目标张量转换为autograd.Variable类型。
                    inputs = inputs[:, started:end, :, :]  #num_components:0波段到多少波段为工作波段
                    weights = weights
                    outputs, out1, out2, out3, out4, out5= model(inputs,weights)
                    # TODO
                    targets0 = targets.cpu().detach().numpy()
                    all_targets0.append(targets0)

                    out11 = out1.cpu().detach().numpy()
                    pred11 = np.argmax(out11, axis=1)
                    all_pred11.append(pred11)

                    out22 = out2.cpu().detach().numpy()
                    pred22 = np.argmax(out22, axis=1)
                    all_pred22.append(pred22)

                    out33 = out3.cpu().detach().numpy()
                    pred33 = np.argmax(out33, axis=1)
                    all_pred33.append(pred33)

                    out44 = out4.cpu().detach().numpy()
                    pred44 = np.argmax(out44, axis=1)
                    all_pred44.append(pred44)

                    out55 = out5.cpu().detach().numpy()
                    pred55 = np.argmax(out55, axis=1)
                    all_pred55.append(pred55)
                    # TODO
                    loss = loss_fn(outputs, targets)
                    train_losses[batch_idx] = loss.item()#记录该batch的训练损失。
                    train_accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()#记录该batch的训练准确率
                    optimizer.zero_grad()#清空优化器的梯度缓存。
                    loss.backward()#进行反向传播计算梯度。
                    optimizer.step()#根据梯度更新模型参数。

                # TODO
                all_targets01 = np.concatenate(all_targets0, axis=0)
                all_targets01 = all_targets01.reshape(-1, 1)
                all_targets02 = all_targets01.reshape(-1, )

                all_pre11 = np.concatenate(all_pred11, axis=0)
                classification1, confusion1, result1 = auxil.reports(all_pre11, all_targets01)
                result11 = result1[3:3 + classs]
                # f0_per_class11 = fbeta_score(all_targets02, all_pre11, beta=1, average=None)
                # result11 = f0_per_class11.tolist()
                print(result11)

                all_pre22 = np.concatenate(all_pred22, axis=0)
                classification2, confusion2, result2 = auxil.reports(all_pre22, all_targets01)
                result22 = result2[3:3+classs]
                # f0_per_class22 = fbeta_score(all_targets02, all_pre22, beta=1, average=None)
                # result22 = f0_per_class22.tolist()
                print(result22)

                all_pre33 = np.concatenate(all_pred33, axis=0)
                classification3, confusion3, result3 = auxil.reports(all_pre33, all_targets01)
                result33 = result3[3:3+classs]
                # f0_per_class33 = fbeta_score(all_targets02, all_pre33, beta=1, average=None)
                # result33 = f0_per_class33.tolist()
                print(result33)

                all_pre44 = np.concatenate(all_pred44, axis=0)
                classification4, confusion4, result4 = auxil.reports(all_pre44, all_targets01)
                result44 = result4[3:3+classs]
                # f0_per_class44 = fbeta_score(all_targets02, all_pre44, beta=1, average=None)
                # result44 = f0_per_class44.tolist()
                print(result44)

                all_pre55 = np.concatenate(all_pred55, axis=0)
                classification5, confusion5, result5 = auxil.reports(all_pre55, all_targets01)
                result55 = result5[3:3 + classs]
                # f0_per_class44 = fbeta_score(all_targets02, all_pre44, beta=1, average=None)
                # result44 = f0_per_class44.tolist()
                print(result55)

                result11 = np.array(result11)
                result22 = np.array(result22)
                result33 = np.array(result33)
                result44 = np.array(result44)
                result55 = np.array(result55)
                if normalization_sar or normalization_sar_band == 1:
                    # wei1 = result11 / (result11 + result22 + result33 + result44 + result55)
                    # wei2 = result22 / (result11 + result22 + result33 + result44 + result55)
                    # wei3 = result33 / (result11 + result22 + result33 + result44 + result55)
                    # wei4 = result44 / (result11 + result22 + result33 + result44 + result55)
                    # wei5 = result55 / (result11 + result22 + result33 + result44 + result55)

                    wei1 = result11 / (result22 + result33 + result44 + result55)
                    wei2 = result22 / (result22 + result33 + result44 + result55)
                    wei3 = result33 / (result22 + result33 + result44 + result55)
                    wei4 = result44 / (result22 + result33 + result44 + result55)
                    wei5 = result55 / (result22 + result33 + result44 + result55)

                if normalization_band_dsm == 1:
                    wei1 = (result11 / (result11 + result22 + result33 + result44)) #* (result1[2] * 0.01)
                    wei2 = (result22 / (result11 + result22 + result33 + result44)) #* (result2[2] * 0.01)
                    wei3 = (result33 / (result11 + result22 + result33 + result44)) #* (result1[2] * 0.01)
                    wei4 = (result44 / (result11 + result22 + result33 + result44))
                    wei5 = wei1
                weights = np.vstack([wei1, wei2, wei3, wei4, wei5])
                # TODO
                print(weights)

                scheduler.step()#更新学习率调度程序。
                train_loss = np.average(train_losses)#计算所有batch的平均训练损失
                train_acc = np.average(train_accs)#计算所有batch的平均训练准确率。
                fmt_str = "Iter [{:d}/{:d}]  \nTrain_loss: {:f}  Train_acc: {:f}"
                print_str = fmt_str.format(
                    epoch + 1,
                    cfg["train"]["epochs"],
                    train_loss,
                    train_acc,
                )
                tr_writer.add_scalar("loss", train_loss, epoch+1)#使用tensorboard记录训练损失。
                print(print_str)
                logger.info(print_str)

                state = {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }#状态保存在state字典中
                # save to the continue path
                torch.save(state, continue_path)


                epoch += 1
                if (epoch + 1) % cfg["train"]["val_interval"] == 0 or (epoch + 1) == cfg["train"]["epochs"]:
                    model.eval()
                    val_accs = np.ones((len(val_loader))) * -1000.0
                    val_losses = np.ones((len(val_loader))) * -1000.0
                    with torch.no_grad():
                        for batch_idy, (inputs, targets) in enumerate(val_loader):
                            inputs, targets = inputs.to(device), targets.to(device)
                            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                            inputs = inputs[:, started:end, :, :]
                            outputs, q1, q2, q3, q4, q5 = model(inputs,weights)
                            # 对于SSNet_AEAE_IN方法，使用如下两行
                            # if outputs.shape[0] < cfg["train"]["batch_size"]:
                            #     outputs = outputs.unsqueeze(0)
                            val_losses[batch_idy] = loss_fn(outputs, targets).item()
                            val_accs[batch_idy] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
                        val_loss = np.average(val_losses)
                        val_acc = np.average(val_accs)

                    fmt_str = "Val_loss: {:f}  Val_acc: {:f}"
                    print_str = fmt_str.format(
                        val_loss,
                        val_acc,
                    )
                    val_writer.add_scalar("loss", val_loss, epoch)
                    print(print_str)
                    logger.info(print_str)

                    if val_acc > best_acc:
                        save_epoch.append(epoch)
                        best_acc = val_acc
                        state = {
                            'epoch': epoch + 1,
                            'best_acc': best_acc,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                        }
                        torch.save(state, os.path.join(logdir, "best_model.pth.tar"))
                        np.savetxt(os.path.join(logdir,"weights.csv"), weights, delimiter=',')

                if epoch == cfg["train"]["epochs"]:

                    flag = False
                    break
            # TODO 3
            elif cfg["model"] != 'HybridSN_multi':
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                    # ：将输入和目标张量转换为autograd.Variable类型。
                    inputs = inputs[:, started:end, :, :]   #started:前面几个波段不要   end:end-started就是波段数
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    train_losses[batch_idx] = loss.item()  # 记录该batch的训练损失。
                    train_accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()  # 记录该batch的训练准确率
                    optimizer.zero_grad()  # 清空优化器的梯度缓存。
                    loss.backward()  # 进行反向传播计算梯度。
                    optimizer.step()  # 根据梯度更新模型参数。
                print(weights)

                scheduler.step()  # 更新学习率调度程序。
                train_loss = np.average(train_losses)  # 计算所有batch的平均训练损失
                train_acc = np.average(train_accs)  # 计算所有batch的平均训练准确率。
                fmt_str = "Iter [{:d}/{:d}]  \nTrain_loss: {:f}  Train_acc: {:f}"
                print_str = fmt_str.format(
                    epoch + 1,
                    cfg["train"]["epochs"],
                    train_loss,
                    train_acc,
                )
                tr_writer.add_scalar("loss", train_loss, epoch + 1)  # 使用tensorboard记录训练损失。
                print(print_str)
                logger.info(print_str)

                state = {
                    'epoch': epoch + 1,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                }  # 状态保存在state字典中
                # save to the continue path
                torch.save(state, continue_path)

                epoch += 1
                if (epoch + 1) % cfg["train"]["val_interval"] == 0 or (epoch + 1) == cfg["train"]["epochs"]:
                    model.eval()
                    val_accs = np.ones((len(val_loader))) * -1000.0
                    val_losses = np.ones((len(val_loader))) * -1000.0
                    with torch.no_grad():
                        for batch_idy, (inputs, targets) in enumerate(val_loader):
                            inputs, targets = inputs.to(device), targets.to(device)
                            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                            inputs = inputs[:, started:end, :, :]
                            outputs = model(inputs)
                            # 对于SSNet_AEAE_IN方法，使用如下两行
                            # if outputs.shape[0] < cfg["train"]["batch_size"]:
                            #     outputs = outputs.unsqueeze(0)
                            val_losses[batch_idy] = loss_fn(outputs, targets).item()
                            val_accs[batch_idy] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
                        val_loss = np.average(val_losses)
                        val_acc = np.average(val_accs)

                    fmt_str = "Val_loss: {:f}  Val_acc: {:f}"
                    print_str = fmt_str.format(
                        val_loss,
                        val_acc,
                    )
                    val_writer.add_scalar("loss", val_loss, epoch)
                    print(print_str)
                    logger.info(print_str)

                    if val_acc > best_acc:
                        save_epoch.append(epoch)
                        best_acc = val_acc
                        state = {
                            'epoch': epoch + 1,
                            'best_acc': best_acc,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                        }
                        torch.save(state, os.path.join(logdir, "best_model.pth.tar"))

                if epoch == cfg["train"]["epochs"]:
                    flag = False
                    break

        print(save_epoch)

# def Train():
#     parser = argparse.ArgumentParser(description='HSIC model Training')
#     parser.add_argument(
#         "--config",
#         nargs="?",
#         type=str,
#         default="configs/config.yml",
#         help="Configuration file to use",
#     )
#     args = parser.parse_args()
#     state = {k: v for k, v in args._get_kwargs()}
#     with open(args.config) as fp:
#         cfg = yaml.load(fp, Loader=yaml.FullLoader)
#
#     name = cfg["data"]["dataset"]
#     seeda = cfg["data"]["seed"]
#     datasetname = str(name)
#     modelname = str(cfg['model'])
#     pca_components = str(cfg["data"]["num_components"])
#     device = auxil.get_device()
#     logdir = './Result/' + datasetname + "_" + modelname + "_PPsize" + str(cfg['data']['PPsize']) + "_epochs" + str(
#         cfg['train']['epochs']) + "_PCA" + pca_components + '/' + str(cfg["run_ID"])
#     # logdir = os.path.join("runs", cfg["model"], str(cfg["run_ID"]))
#     logs_file = os.path.join(logdir, "output.txt")
#     import shutil
#     #if os.path.exists(logdir):
#         # shutil.rmtree(logdir)
#         #os.removedirs(logdir)
#     if not os.path.exists(logdir):
#         os.makedirs(logdir)
#     # #TODO logger
#     # sys.stdout = open(logs_file, "w")
#     # sys.stderr = open(logs_file, "w")
#
#     tr_writer = SummaryWriter(log_dir=os.path.join(logdir + "/train/"))
#     val_writer = SummaryWriter(log_dir=os.path.join(logdir + "/val/"))
#     print("RUNDIR: {}".format(logdir))
#     shutil.copy(args.config, logdir)
#
#     logger = get_logger(logdir)
#     logger.info("Let begin!")
#
#     # Setup seeds
#     torch.manual_seed(cfg.get("seed", seeda))
#     torch.cuda.manual_seed(cfg.get("seed", seeda))
#     np.random.seed(cfg.get("seed", seeda))
#     random.seed(cfg.get("seed", seeda))
#
#     # Setup device
#     device = auxil.get_device()
#
#     # Setup Dataloader
#     train_loader, val_loader, num_classes, n_bands, xx, yy = get_trainLoader(cfg, logdir)
#
#     # Setup Model
#     model = get_model(cfg['model'], cfg['data']['dataset'])
#     model = model.to(device)
#
#     PPsize = cfg['data']['PPsize']
#
#     # from torchsummary import summary
#     # summary(model, (n_bands,PPsize,PPsize))
#
#     print("model load successfully")
#
#     # Setup optimizer, lr_scheduler and loss function
#     optimizer_cls = get_optimizer(cfg)  # sgd, adam, RMSprop
#     optimizer_params = {k: v for k, v in cfg["train"]["optimizer"].items() if k != "name"}
#     if cfg["train"]["optimizer"]["name"] == "sgd":
#         optimizer_params.pop("betas", None)
#         optimizer_params.pop("eps", None)
#         optimizer_params.pop("alpha", None)
#         print("optimizer = sgd")
#     elif cfg["train"]["optimizer"]["name"] == "adam":
#         optimizer_params.pop("momentum", None)
#         optimizer_params.pop("alpha", None)
#         print("optimizer = adam")
#     elif cfg["train"]["optimizer"]["name"] == "RMSprop":
#         optimizer_params.pop("momentum", None)
#         optimizer_params.pop("betas", None)
#         print("optimizer = RMSprop")
#
#     optimizer = optimizer_cls(model.parameters(), **optimizer_params)
#     logger.info("Using optimizer {}".format(optimizer))
#
#     # Setup loss function
#     loss_fn = torch.nn.CrossEntropyLoss()
#
#     # training model
#     start_time = time.time()
#
#     train(cfg, train_loader, val_loader, model, loss_fn, optimizer, device, tr_writer, val_writer, logdir, logger, xx,
#           yy)
#
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print("训练时间(秒)：", elapsed_time)
#
#     # training over!
#     print("Training is over!")
#     logger.info("Training is over!")
#
#     # RunPredict
#     start_time = time.time()
#     RunPredict()
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     print("预测时间(秒)：", elapsed_time)
#     RunPredicttif()
#     sys.stdout.close()
#     sys.stderr.close()
#     sys.stdout = sys.__stdout__
#     sys.stderr = sys.__stderr__
#     run_txt = './Result/' + datasetname + "_" + modelname + "_PPsize" + str(cfg['data']['PPsize']) + "_epochs" + str(
#         cfg['train'][
#             'epochs']) + "_PCA" + pca_components + "/" + "classification_report_" + datasetname + "_" + modelname + "dataset.txt"
#     subprocess.run(["notepad.exe", run_txt])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HSIC model Training')
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/config.yml",
        help="Configuration file to use",
    )
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    name = cfg["data"]["dataset"]
    seeda = cfg["data"]["seed"]
    datasetname = str(name)
    modelname= str(cfg['model'])
    pca_components=str(cfg["data"]["num_components"])
    device = auxil.get_device()
    logdir = './Result/' + datasetname + "_" + modelname + "_PPsize" + str(cfg['data']['PPsize']) + "_epochs" + str(
        cfg['train']['epochs']) + "_PCA" + pca_components + '/'+str(cfg["run_ID"])
    # logdir = os.path.join("runs", cfg["model"], str(cfg["run_ID"]))
    logs_file = os.path.join(logdir,"output.txt")
    if os.path.exists(logdir):
        import shutil
        shutil.rmtree(logdir)
        # os.removedirs(logdir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    # #TODO logger
    # sys.stdout = open(logs_file, "w")
    # sys.stderr = open(logs_file, "w")

    tr_writer = SummaryWriter(log_dir=os.path.join(logdir+"/train/"))
    val_writer = SummaryWriter(log_dir=os.path.join(logdir+"/val/"))
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)


    logger = get_logger(logdir)
    logger.info("Let begin!")

    # Setup seeds
    torch.manual_seed(cfg.get("seed", seeda))
    torch.cuda.manual_seed(cfg.get("seed", seeda))
    np.random.seed(cfg.get("seed", seeda))
    random.seed(cfg.get("seed", seeda))

    # Setup device
    device = auxil.get_device()

    # Setup Dataloader
    train_loader, val_loader, num_classes, n_bands, xx, yy = get_trainLoader(cfg,logdir)

    # Setup Model
    model = get_model(cfg['model'], cfg['data']['dataset'])
    model = model.to(device)

    PPsize=cfg['data']['PPsize']

    # from torchsummary import summary
    # summary(model, (n_bands,PPsize,PPsize))


    print("model load successfully")

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)  #sgd, adam, RMSprop
    optimizer_params = {k: v for k, v in cfg["train"]["optimizer"].items() if k != "name"}
    if cfg["train"]["optimizer"]["name"] == "sgd":
        optimizer_params.pop("betas", None)
        optimizer_params.pop("eps", None)
        optimizer_params.pop("alpha", None)
        print("optimizer = sgd")
    elif cfg["train"]["optimizer"]["name"] == "adam":
        optimizer_params.pop("momentum", None)
        optimizer_params.pop("alpha", None)
        print("optimizer = adam")
    elif cfg["train"]["optimizer"]["name"] == "RMSprop":
        optimizer_params.pop("momentum", None)
        optimizer_params.pop("betas", None)
        print("optimizer = RMSprop")

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    # Setup lr_scheduler
    scheduler = get_scheduler(optimizer, cfg["train"]["lr_schedule"])
    best_err1 = 100

    # Setup loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # training model
    start_time = time.time()

    train(cfg, train_loader, val_loader, model, loss_fn, optimizer, device, tr_writer, val_writer, logdir, logger, xx, yy)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("训练时间(秒)：", elapsed_time)

    # training over!
    print("Training is over!")
    logger.info("Training is over!")

    #RunPredict
    start_time = time.time()
    RunPredict()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("预测时间(秒)：", elapsed_time)
    RunPredicttif()
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    run_txt = './Result/' + datasetname + "_" + modelname + "_PPsize" + str(cfg['data']['PPsize']) + "_epochs" + str(
        cfg['train']['epochs']) + "_PCA" + pca_components + "/" + "classification_report_" + datasetname + "_" + modelname + "dataset.txt"
    subprocess.run(["notepad.exe", run_txt])
