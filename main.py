import os
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QMessageBox, QApplication, QMainWindow, QMenu, QAction, QVBoxLayout, QWidget, QDockWidget, QLabel, QFileDialog, QListWidget, QTextEdit, QDialog, QLabel, QLineEdit, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtGui
from osgeo import gdal
import numpy as np
import subprocess
import sys
# from train import Train
from create_yml import update_yaml_file
from create_dataset import create_dataset
from S2 import *
from predict import RunPredicttif2

#预测
class PredictionDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("模型预测")
        self.setGeometry(200, 200, 400, 150)

        # 创建参数输入框和选择按钮
        self.input_file_edit = QLineEdit(self)

        input_file_button = QPushButton("选择文件", self)
        input_file_button.clicked.connect(self.select_file)

        start_button = QPushButton("开始处理", self)
        start_button.clicked.connect(self.start_prediction)

        # 创建布局
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("输入预测文件：", self))
        layout.addWidget(self.input_file_edit)
        layout.addWidget(input_file_button)
        layout.addWidget(start_button)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择文件', filter='TIFF Files (*.tif *.tiff)')
        if file_path:
            self.input_file_edit.setText(file_path)

    def start_prediction(self):
        input_file = self.input_file_edit.text()

        # 执行预测函数
        print(input_file)
        RunPredicttif2(input_file)

        # 显示处理完成对话框
        self.show_completion_message()

    def show_completion_message(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("预测")
        msg_box.setText("预测完成")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setMinimumSize(400, 200)
        msg_box.exec_()

# 在主窗口类中添加打开预测对话框的方法
def open_prediction_dialog(self):
    prediction_dialog = PredictionDialog()
    prediction_dialog.exec_()



#S2数据预处理
class Sentinel2PreprocessingDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sentinel-2 数据预处理")
        self.setGeometry(200, 200, 400, 200)

        # 创建参数输入框和选择按钮
        self.input_folder_edit = QLineEdit(self)
        self.output_folder_edit = QLineEdit(self)
        self.mask_file_edit = QLineEdit(self)

        input_folder_button = QPushButton("选择文件夹", self)
        input_folder_button.clicked.connect(lambda: self.select_folder(self.input_folder_edit))

        output_folder_button = QPushButton("选择文件夹", self)
        output_folder_button.clicked.connect(lambda: self.select_folder(self.output_folder_edit))

        mask_file_button = QPushButton("选择文件", self)
        mask_file_button.clicked.connect(lambda: self.select_file(self.mask_file_edit))

        start_button = QPushButton("开始处理", self)
        start_button.clicked.connect(self.start_preprocessing)

        # 创建布局
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("输入数据文件夹：", self))
        layout.addWidget(self.input_folder_edit)
        layout.addWidget(input_folder_button)
        layout.addWidget(QLabel("输出数据文件夹：", self))
        layout.addWidget(self.output_folder_edit)
        layout.addWidget(output_folder_button)
        layout.addWidget(QLabel("掩膜文件：", self))
        layout.addWidget(self.mask_file_edit)
        layout.addWidget(mask_file_button)
        layout.addWidget(start_button)

    def select_folder(self, line_edit):
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder_path:
            line_edit.setText(folder_path)

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择文件', filter='TIFF Files (*.tif *.tiff)')
        if file_path:
            line_edit.setText(file_path)

    def start_preprocessing(self):
        input_folder = self.input_folder_edit.text()
        output_folder = self.output_folder_edit.text()
        mask_file_path = self.mask_file_edit.text()

        # 执行Sentinel-2数据预处理函数
        S2(input_folder, output_folder, mask_file_path)

        # 显示处理完成对话框
        self.show_completion_message()

    def show_completion_message(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("完成")
        msg_box.setText("处理完成")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setMinimumSize(400, 200)
        msg_box.exec_()

#训练数据生成
class GenerateDatasetDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("生成训练数据集")
        self.setGeometry(200, 200, 400, 300)

        # 创建参数输入框和选择按钮
        # self.model_edit = QLineEdit(self)
        self.dataset_edit = QLineEdit(self)
        self.data_edit = QLineEdit(self)
        self.y_edit = QLineEdit(self)
        self.started_edit = QLineEdit(self)
        self.end_edit = QLineEdit(self)
        self.modal_channel_edit = QLineEdit(self)

        self.validation_percent_edit = QLineEdit(self)
        self.train_percent_edit = QLineEdit(self)

        # 创建下拉列表按钮
        self.model_combobox = QComboBox(self)
        self.model_combobox.addItems(
            ["RF", "1D CNN", "2D CNN", "3D CNN", "VGG 16", "HybridSN", "ViT", "SSTN", "SSFTTnet", "TransHSI", "HRN"])

        # 创建选择文件按钮
        self.data_button = QPushButton("选择文件", self)
        self.y_button = QPushButton("选择文件", self)

        self.data_button.clicked.connect(lambda: self.select_file(self.data_edit))
        self.y_button.clicked.connect(lambda: self.select_file(self.y_edit))

        # 创建开始生成按钮
        start_button = QPushButton("开始生成", self)
        start_button.clicked.connect(self.start_generation)

        # 创建布局
        # layout = QVBoxLayout(self)
        # layout.addWidget(QLabel("Model：", self))
        # layout.addWidget(self.model_edit)
        # layout.addWidget(QLabel("Project name：", self))
        # layout.addWidget(self.dataset_edit)
        # layout.addWidget(QLabel("Data：", self))
        # layout.addWidget(self.data_edit)
        # layout.addWidget(self.data_button)
        # layout.addWidget(QLabel("Y：", self))
        # layout.addWidget(self.y_edit)
        # layout.addWidget(self.y_button)
        # layout.addWidget(QLabel("Started：", self))
        # layout.addWidget(self.started_edit)
        # layout.addWidget(QLabel("End：", self))
        # layout.addWidget(self.end_edit)
        # layout.addWidget(QLabel("Modal Channel：", self))
        # layout.addWidget(self.modal_channel_edit)
        # layout.addWidget(self.modal_channel_button)
        # layout.addWidget(QLabel("Validation Percent：", self))
        # layout.addWidget(self.validation_percent_edit)
        # layout.addWidget(QLabel("Train Percent：", self))
        # layout.addWidget(self.train_percent_edit)
        # layout.addWidget(start_button)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Model：", self))
        layout.addWidget(self.model_combobox)
        layout.addWidget(QLabel("Project name：", self))
        layout.addWidget(self.dataset_edit)
        layout.addWidget(QLabel("Data：", self))
        layout.addWidget(self.data_edit)
        layout.addWidget(self.data_button)
        layout.addWidget(QLabel("Y：", self))
        layout.addWidget(self.y_edit)
        layout.addWidget(self.y_button)
        layout.addWidget(QLabel("Started：", self))
        layout.addWidget(self.started_edit)
        layout.addWidget(QLabel("End：", self))
        layout.addWidget(self.end_edit)
        layout.addWidget(QLabel("Modal Channel：", self))
        layout.addWidget(self.modal_channel_edit)
        layout.addWidget(start_button)

        layout.addWidget(QLabel("Train Percent：", self))
        layout.addWidget(self.train_percent_edit)
        layout.addWidget(QLabel("Validation Percent：", self))
        layout.addWidget(self.validation_percent_edit)
        layout.addWidget(start_button)

    def select_file(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择文件')
        if file_path:
            line_edit.setText(file_path)

    def start_generation(self):
        # 获取输入框中的参数
        model = self.model_combobox.currentText()
        print(model)
        dataset = self.dataset_edit.text()
        data = self.data_edit.text()
        y = self.y_edit.text()
        started = self.started_edit.text()
        end = self.end_edit.text()
        modal_channel = self.modal_channel_edit.text()
        validation_percent = self.validation_percent_edit.text()
        train_percent = self.train_percent_edit.text()

        modal_values = modal_channel.strip('[]').split(',')  # 去除方括号并分割字符串
        m1, m2, m3, m4 = modal_values  # 将分割后的值分别赋给 m1、m2、m3、m4

        # 调用生成数据集函数
        update_yaml_file(model, dataset, data, y, started,
                         end, m1, m2, m3, m4,train_percent ,validation_percent)
        create_dataset()
        self.show_completion_message()

    def show_completion_message(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("完成")
        msg_box.setText("处理完成")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setMinimumSize(400, 200)  # 设置最小尺寸
        msg_box.setMaximumSize(800, 400)  # 设置最大尺寸
        msg_box.exec_()

# 训练
class TrainingDialog(QDialog):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        self.setWindowTitle("模型训练")
        self.setGeometry(200, 200, 400, 150)

        # 创建参数输入框和选择按钮
        self.algorithm_edit = QLineEdit(self)
        start_training_button = QPushButton("开始训练", self)
        start_training_button.clicked.connect(self.start_training)

        # 创建布局
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Project name：", self))
        layout.addWidget(self.algorithm_edit)
        layout.addWidget(start_training_button)

    def start_training(self):
        # 获取输入框中的参数
        algorithm = self.algorithm_edit.text()
        # 显示信息
        self.main_window.show_info(f'模型：{algorithm} 已经开始训练')
        import subprocess
        subprocess.Popen(["python", "train.py"])
        self.show_completion_message()

    def show_completion_message(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("训练")
        msg_box.setText("训练已开始")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.setMinimumSize(400, 200)  # 设置最小尺寸
        msg_box.setMaximumSize(800, 400)  # 设置最大尺寸
        msg_box.exec_()




class MultiModalDeepLearningApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multi-Modal Deep Learning")
        self.setGeometry(100, 100, 800, 600)

        # 添加菜单栏
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')
        open_action = QAction('打开', self)
        open_action.triggered.connect(self.open_folder)
        settings_action = QAction('设置', self)
        file_menu.addAction(open_action)
        file_menu.addAction(settings_action)

        # 预处理菜单
        preprocessing_menu = menubar.addMenu('预处理')
        #sentinel1_action = QAction('sentinel-1预处理', self)
        sentinel2_action = QAction('sentinel-2预处理', self)
        sentinel2_action.triggered.connect(self.open_sentinel2_preprocessing_dialog)
        #uav_action = QAction('UAV预处理', self)
        generate_dataset_action = QAction('生成训练数据集', self)
        generate_dataset_action.triggered.connect(self.open_generate_dataset_dialog)
        #preprocessing_menu.addAction(sentinel1_action)
        preprocessing_menu.addAction(sentinel2_action)
        #preprocessing_menu.addAction(uav_action)
        preprocessing_menu.addAction(generate_dataset_action)

        # 训练与预测菜单
        train_predict_menu = menubar.addMenu('训练与预测')
        train_action = QAction('训练', self)
        train_action.triggered.connect(self.open_training_dialog)
        predict_action = QAction('预测', self)
        predict_action.triggered.connect(self.open_prediction_dialog)
        train_predict_menu.addAction(train_action)
        train_predict_menu.addAction(predict_action)


        # 帮助菜单
        help_menu = menubar.addMenu('帮助')
        about_mmda_action = QAction('关于MMDA', self)
        language_action = QAction('语言', self)
        help_menu.addAction(about_mmda_action)
        help_menu.addAction(language_action)

        # 添加图层显示小窗口
        dock_widget = QDockWidget('图层显示', self)
        self.layer_list_widget = QListWidget(dock_widget)
        dock_widget.setWidget(self.layer_list_widget)
        self.addDockWidget(1, dock_widget)  # 将小窗口停靠在左侧

        # 设置主窗口布局
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建用于显示图片的区域
        self.image_label = QLabel(self)
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.image_label)

        # 创建用于显示提示信息的文本框
        self.info_text_edit = QTextEdit(self)
        self.info_text_edit.setReadOnly(True)  # 设置为只读
        layout.addWidget(self.info_text_edit)

        # 连接信号与槽
        self.layer_list_widget.itemSelectionChanged.connect(self.handle_item_selection)

        # 初始化文件夹路径为None
        self.folder_path = None

    def open_generate_dataset_dialog(self):
        generate_dataset_dialog = GenerateDatasetDialog()
        generate_dataset_dialog.exec_()

    def open_folder(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹', os.path.expanduser('~'))
        if self.folder_path:
            # 获取文件夹下所有.tif格式的文件
            tif_files = [f for f in os.listdir(self.folder_path) if f.lower().endswith('.jpg')]

            # 在'图层显示'窗口中显示所有.tif图层
            self.layer_list_widget.clear()
            self.layer_list_widget.addItems(tif_files)

            # 显示提示信息
            self.show_info(f'已打开文件夹：{self.folder_path}')

    def handle_item_selection(self):
        selected_items = self.layer_list_widget.selectedItems()
        if selected_items:
            selected_file = selected_items[0].text()
            if self.folder_path:
                file_path = os.path.join(self.folder_path, selected_file)
                data = self.read_tif(file_path)
                print(data.shape)
                self.display_rgb_image(data)

    def read_tif(self, data_path):
        datasets = gdal.Open(data_path, gdal.GA_ReadOnly)
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
        return data

    def display_rgb_image(self, data):
        # 假设数据的前三个波段是RGB通道
        rgb_image = data[:, :, :3]

        # 将数据压缩到0-255范围
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        # 将数据显示在窗口布局中
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # 在 QLabel 中显示图像
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignCenter)

    def show_info(self, message):
        # 在文本框中显示提示信息
        self.info_text_edit.append(message)

    def open_training_dialog(self):
        training_dialog = TrainingDialog(self)
        training_dialog.exec_()

    def open_sentinel2_preprocessing_dialog(self):
        preprocessing_dialog = Sentinel2PreprocessingDialog()
        preprocessing_dialog.exec_()

    def open_prediction_dialog(self):
        prediction_dialog = PredictionDialog()
        prediction_dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultiModalDeepLearningApp()
    window.show()
    sys.exit(app.exec_())
