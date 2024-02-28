import yaml
import os
import auxil
import argparse
import matplotlib.pyplot as plt
import re

parser = argparse.ArgumentParser(description='HSIC model Training')
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
# logdir = './Result/' + datasetname + "_" + modelname + "_PPsize" + str(cfg['data']['PPsize']) + "_epochs" + str(
#     cfg['train']['epochs']) + "_PCA" + pca_components + '/' + str(cfg["run_ID"])
logdir = './Result/' + "matplotlibtxt/xiangfu_dier/"
max = 0.5




# 指定包含txt文件的目录
directory_path = logdir

# 获取目录下所有txt文件
txt_files = [file for file in os.listdir(directory_path) if file.endswith('.txt')]

# 提取每个文件的Val_loss并绘制折线图
for i, txt_file in enumerate(txt_files):
    file_path = os.path.join(directory_path, txt_file)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    iteration_losses = []
    for line in lines:
        if 'Val_loss' in line:
            val_loss_match = re.search(r'Val_loss: (\d+\.\d+)', line)
            if val_loss_match:
                val_loss = float(val_loss_match.group(1))
                if val_loss>max:
                    val_loss = max
                iteration_losses.append(val_loss)

    # 选择不同的颜色和标签
    color = plt.cm.viridis(i / len(txt_files))
    label = os.path.splitext(txt_file)[0]

    # 绘制折线图
    iterations = range(1, len(iteration_losses) + 1)
    plt.plot(iterations, iteration_losses, marker='o', linestyle='-',
             color=color, label=label, linewidth=1,markersize=3)

# 添加图例
plt.legend()

# 设置图形标题和标签
plt.title('Validation Loss over Iterations for Different Files')
plt.xlabel('Iterations')
plt.ylabel('Validation Loss')
plt.grid(True)

# 保存图形为SVG格式
plt.savefig(os.path.join(logdir,'validation_loss_plots.svg'), format='svg')

# 显示图形
plt.show()
