import sys
import cv2
import numpy as np
import os
import re
import shutil
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                            QPushButton, QFileDialog, QLabel, QCheckBox,
                            QMessageBox, QFrame, QLineEdit, QGroupBox,
                            QGridLayout, QScrollArea, QSizePolicy, QDialog, QListWidget,
                            QProgressBar, QComboBox, QSlider, QRadioButton, QButtonGroup,
                            QInputDialog)
from PyQt5.QtGui import (QImage, QPixmap, QFont, QMouseEvent)
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import traceback
from glob import glob
import json
import time

# 从拆分的文件中导入
from diff_ui_components import ZoomableLabel, ThresholdRangeWidget
from diff_processing import DetectionThread
from diff_defect_viewer import DefectViewer

# --- 新增: 导入相机模块 ---
HIK_SDK_AVAILABLE = False # 在 try 块之前初始化
try:
    import hik_camera
    import diff_camera_dialog
    HIK_SDK_AVAILABLE = True # 仅在成功导入后设为 True
except ImportError as e:
    print(f"警告: 无法加载相机模块: {e}")
    # HIK_SDK_AVAILABLE 保持 False
except Exception as e:
    print(f"加载相机模块时发生未知错误: {e}")
    # HIK_SDK_AVAILABLE 保持 False
# --- 结束 ---


class PcbDefectDetector(QWidget):
    def __init__(self):
        super().__init__()
        global HIK_SDK_AVAILABLE
        self.template_paths = []
        self.image_path = None
        self.folder = None
        self.image_paths = []
        self.output_image = None
        self.threshold_ranges = []
        self.current_output_dir = ""
        self.detection_thread = None

        # --- 相机状态变量 ---
        self.camera_manager = None
        # 读取全局 HIK_SDK_AVAILABLE
        if HIK_SDK_AVAILABLE:
            try:
                self.camera_manager = hik_camera.CameraManager()
            except Exception as e:
                print(f"初始化 CameraManager 失败: {e}")
                # *** 修复: 使用 global 关键字修改全局变量 ***
                HIK_SDK_AVAILABLE = False

        self.camera_dialog = None
        self.camera_thread = None
        self.current_cam_object = None
        self.hik_device_list = None
        self.capture_target = None
        # --- 结束 ---

        # ( ... YOLO 加载 ... )
        try:
            appdata_dir = os.getenv('APPDATA')
            if appdata_dir:
                ultralytics_dir = os.path.join(appdata_dir, 'Ultralytics')
                os.makedirs(ultralytics_dir, exist_ok=True)
            self.yolo_model = YOLO('best.pt') if os.path.exists('best.pt') else None
            if self.yolo_model is None:
                self.create_default_ultralytics_settings()
        except Exception as e:
            print(f"YOLO 模型加载错误: {str(e)}")
            self.yolo_model = None
            try:
                self.create_default_ultralytics_settings()
            except Exception as e_settings:
                print(f"创建默认YOLO设置失败: {e_settings}")

        self.initUI()
        self.load_settings()

    def initUI(self):
        self.setWindowTitle('PCB 异物检测工具')
        self.setGeometry(100, 100, 1600, 1000)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        image_layout = QHBoxLayout()
        image_layout.setSpacing(20)

        template_group = QGroupBox("标准模板")
        result_group = QGroupBox("检测结果")

        self.template_label = ZoomableLabel()
        self.result_label = ZoomableLabel()

        self.btn_load_template = QPushButton('加载单模板图')
        self.btn_clear_template = QPushButton('清除模板图')
        self.btn_load_template_folder = QPushButton('加载模板文件夹')
        self.btn_load_image = QPushButton('加载检测图')
        self.btn_clear_image = QPushButton('清除检测图')
        self.btn_load_folder = QPushButton('加载检测文件夹')

        # --- 相机按钮 ---
        self.btn_cam_template = QPushButton('相机取图')
        self.btn_cam_image = QPushButton('相机取图')
        if not HIK_SDK_AVAILABLE:
            self.btn_cam_template.setEnabled(False)
            self.btn_cam_template.setToolTip("海康SDK未找到")
            self.btn_cam_image.setEnabled(False)
            self.btn_cam_image.setToolTip("海康SDK未找到")
        # --- 结束 ---

        self.btn_load_template.clicked.connect(self.load_template)
        self.btn_load_template_folder.clicked.connect(self.load_template_folder)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_folder.clicked.connect(self.load_image_folder)
        self.btn_clear_template.clicked.connect(self.clear_template)
        self.btn_clear_image.clicked.connect(self.clear_image)

        # --- 连接相机按钮 ---
        self.btn_cam_template.clicked.connect(lambda: self.toggle_camera('template'))
        self.btn_cam_image.clicked.connect(lambda: self.toggle_camera('image'))
        # --- 结束 ---

        template_layout = QVBoxLayout(template_group)
        result_layout = QVBoxLayout(result_group)

        template_layout.addWidget(self.template_label, stretch=4)

        template_btn_layout = QHBoxLayout()
        template_btn_layout.addWidget(self.btn_load_template)
        template_btn_layout.addWidget(self.btn_load_template_folder)
        template_btn_layout.addWidget(self.btn_cam_template) # --- 新增 ---
        template_btn_layout.addWidget(self.btn_clear_template)
        template_layout.addLayout(template_btn_layout, stretch=1)

        result_layout.addWidget(self.result_label, stretch=4)

        result_btn_layout = QHBoxLayout()
        result_btn_layout.addWidget(self.btn_load_image)
        result_btn_layout.addWidget(self.btn_load_folder)
        result_btn_layout.addWidget(self.btn_cam_image) # --- 新增 ---
        result_btn_layout.addWidget(self.btn_clear_image)
        result_layout.addLayout(result_btn_layout, stretch=1)

        image_layout.addWidget(template_group, stretch=1)
        image_layout.addWidget(result_group, stretch=1)
        main_layout.addLayout(image_layout, stretch=3)

        # ( ... 参数组UI, 阈值, 底部按钮 ... 保持不变 )
        param_group = QGroupBox("检测参数")
        param_layout = QGridLayout(param_group)

        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("组合方式:"))
        self.radio_and = QRadioButton("与")
        self.radio_or = QRadioButton("或")
        self.radio_and.setChecked(True)
        combo_layout.addWidget(self.radio_and)
        combo_layout.addWidget(self.radio_or)
        self.combo_group = QButtonGroup()
        self.combo_group.addButton(self.radio_and)
        self.combo_group.addButton(self.radio_or)
        param_layout.addLayout(combo_layout, 0, 0, 1, 2)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("就绪")
        param_layout.addWidget(QLabel("进度:"), 0, 2)
        param_layout.addWidget(self.progress_bar, 0, 3)
        param_layout.addWidget(self.progress_label, 0, 4)

        threshold_group = QGroupBox("阈值条件设置")
        threshold_layout = QVBoxLayout(threshold_group)

        btn_layout = QHBoxLayout()
        self.btn_add_binary = QPushButton("添加预处理二值化阈值")
        self.btn_add_binary.setStyleSheet("background-color: #99ccff;")
        self.btn_add_binary.clicked.connect(lambda: self.add_threshold_range("二值化阈值"))

        self.btn_add_area = QPushButton("添加面积阈值过滤")
        self.btn_add_area.setStyleSheet("background-color: #99ccff;")
        self.btn_add_area.clicked.connect(lambda: self.add_threshold_range("面积阈值"))

        self.btn_add_ratio = QPushButton("添加比例阈值过滤")
        self.btn_add_ratio.setStyleSheet("background-color: #99ccff;")
        self.btn_add_ratio.clicked.connect(lambda: self.add_threshold_range("比例阈值"))

        self.btn_add_gray = QPushButton("添加预处理灰度差阈值")
        self.btn_add_gray.setStyleSheet("background-color: #99ccff;")
        self.btn_add_gray.clicked.connect(lambda: self.add_threshold_range("灰度差阈值"))

        self.btn_add_rgb = QPushButton("添加RGB值阈值过滤")
        self.btn_add_rgb.setStyleSheet("background-color: #99ccff;")
        self.btn_add_rgb.clicked.connect(lambda: self.add_threshold_range("RGB值阈值"))

        self.btn_add_rgb_binary = QPushButton("添加预处理RGB二值化")
        self.btn_add_rgb_binary.setStyleSheet("background-color: #99ccff;")
        self.btn_add_rgb_binary.clicked.connect(lambda: self.add_threshold_range("RGB二值化"))
        self.btn_add_brightness = QPushButton("添加亮度阈值过滤")
        self.btn_add_brightness.setStyleSheet("background-color: #99ccff;")
        self.btn_add_brightness.clicked.connect(lambda: self.add_threshold_range("区域平均亮度阈值"))

        btn_layout.addWidget(self.btn_add_binary)
        btn_layout.addWidget(self.btn_add_gray)
        btn_layout.addWidget(self.btn_add_rgb_binary)
        btn_layout.addWidget(self.btn_add_area)
        btn_layout.addWidget(self.btn_add_ratio)
        btn_layout.addWidget(self.btn_add_brightness)
        btn_layout.addWidget(self.btn_add_rgb)

        threshold_layout.addLayout(btn_layout)

        self.threshold_scroll = QScrollArea()
        self.threshold_scroll.setWidgetResizable(True)
        self.threshold_container = QWidget()
        self.threshold_container_layout = QVBoxLayout(self.threshold_container)
        self.threshold_container_layout.setSpacing(5)
        self.threshold_scroll.setWidget(self.threshold_container)

        threshold_layout.addWidget(self.threshold_scroll)
        param_layout.addWidget(threshold_group, 1, 0, 1, 5)
        main_layout.addWidget(param_group, stretch=1)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        self.cb_use_ai = QCheckBox('启用AI过滤')
        self.cb_debug = QCheckBox('保存中间结果图')
        self.cb_auto_delete = QCheckBox('自动删除原图')
        self.cb_monitor_folder = QCheckBox('持续监控文件夹')
        self.cb_monitor_folder.setEnabled(False)

        btn_layout.addWidget(self.cb_auto_delete)

        if not self.yolo_model:
            self.cb_use_ai.setEnabled(False)
            self.cb_use_ai.setText("启用AI (模型未加载)")

        btn_layout.addWidget(self.cb_use_ai)
        btn_layout.addWidget(self.cb_debug)
        btn_layout.addWidget(self.cb_monitor_folder)
        btn_layout.addStretch(1)

        self.btn_detect = QPushButton('🔍 开始检测')
        self.btn_detect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_detect.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton('⏹ 终止检测')
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)

        self.btn_view_defects = QPushButton('🔍 查看缺陷')
        self.btn_view_defects.clicked.connect(self.view_defects)

        btn_layout.addWidget(self.btn_detect)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_view_defects)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

        self.add_threshold_range("二值化阈值")
        self.add_threshold_range("面积阈值")
        self.add_threshold_range("比例阈值")

    # --- 相机控制方法 (与上一版相同) ---

    def toggle_camera(self, target):
        if not HIK_SDK_AVAILABLE or self.camera_manager is None:
            QMessageBox.critical(self, "错误", "相机SDK未正确加载，无法使用相机功能。")
            return

        btn = self.btn_cam_template if target == 'template' else self.btn_cam_image

        if self.camera_thread is not None and self.camera_thread.isRunning():
            if self.capture_target == target:
                print("发送捕获信号...")
                self.camera_thread.capture_and_stop()
                btn.setText("采集中...")
                btn.setEnabled(False)
            else:
                QMessageBox.warning(self, "提示", "请先点击另一路相机的“点击取图”按钮。")
            return

        if self.camera_thread is not None:
             QMessageBox.warning(self, "错误", "相机线程已在运行，请先停止。")
             return

        try:
            devices, self.hik_device_list = self.camera_manager.list_devices()
            if not devices or not self.hik_device_list:
                QMessageBox.warning(self, "未找到相机", "未枚举到任何海康威视相机设备。")
                return
        except Exception as e:
            QMessageBox.critical(self, "枚举失败", f"枚举相机设备时出错: {e}")
            return

        item, ok = QInputDialog.getItem(self, "选择相机", "可用设备:", devices, 0, False)

        if ok and item:
            try:
                device_index = devices.index(item)
                self.current_cam_object = self.camera_manager.connect(self.hik_device_list, device_index)

                if self.current_cam_object is None:
                    QMessageBox.critical(self, "连接失败", "无法连接到所选相机。")
                    return

                self.capture_target = target

                self.camera_dialog = diff_camera_dialog.CameraFeedDialog(self)
                self.camera_dialog.dialog_closed_signal.connect(self.stop_camera_feed)

                self.camera_thread = hik_camera.CameraThread(self.current_cam_object)
                self.camera_thread.new_frame_signal.connect(self.camera_dialog.update_image)
                self.camera_thread.frame_captured_signal.connect(self.on_frame_captured)
                self.camera_thread.finished.connect(self.on_camera_thread_finished)
                self.camera_thread.error_signal.connect(self.on_camera_error)

                self.camera_thread.start()
                self.camera_dialog.show()

                btn.setText("点击取图")
                other_btn = self.btn_cam_image if target == 'template' else self.btn_cam_template
                other_btn.setEnabled(False)

            except Exception as e:
                QMessageBox.critical(self, "启动失败", f"启动相机时出错: {e}\n\n请检查相机是否被占用，或查看控制台输出。")
                traceback.print_exc()
                self.on_camera_thread_finished()

    def stop_camera_feed(self):
        if self.camera_thread:
            print("正在停止相机线程...")
            self.camera_thread.stop()

    def on_camera_error(self, message):
        QMessageBox.critical(self, "相机错误", message)
        self.stop_camera_feed()

    def on_frame_captured(self, save_path, width, height):
        print(f"帧已捕获! 保存至: {save_path}, 尺寸: {width}x{height}")
        try:
            pixmap = QPixmap(save_path)
            if pixmap.isNull():
                raise Exception(f"加载保存的 PNG 文件失败: {save_path}")

            if self.capture_target == 'template':
                self.template_paths = [save_path]
                self.template_label.setPixmap(pixmap)
                self.btn_load_template.setText(f"已加载: {os.path.basename(save_path)}")
                self.btn_load_template_folder.setText("加载模板文件夹")

            else:
                self.image_paths = [save_path]
                self.image_path = save_path
                self.folder = None
                self.result_label.setPixmap(pixmap)
                self.btn_load_image.setText(f"已加载: {os.path.basename(save_path)}")
                self.btn_load_folder.setText("加载检测文件夹")
                self.cb_monitor_folder.setEnabled(False)
                self.cb_monitor_folder.setChecked(False)

        except Exception as e:
            QMessageBox.critical(self, "捕获失败", f"处理捕获的帧时出错: {e}")
            traceback.print_exc()

    def on_camera_thread_finished(self):
        print("相机线程已结束，正在清理...")

        if self.camera_dialog:
            try:
                self.camera_dialog.dialog_closed_signal.disconnect(self.stop_camera_feed)
            except TypeError:
                pass
            self.camera_dialog.close()
            self.camera_dialog = None

        if self.current_cam_object and self.camera_manager:
            self.camera_manager.disconnect(self.current_cam_object)
            self.current_cam_object = None

        self.camera_thread = None

        self.btn_cam_template.setText("相机取图")
        self.btn_cam_image.setText("相机取图")
        self.btn_cam_template.setEnabled(HIK_SDK_AVAILABLE)
        self.btn_cam_image.setEnabled(HIK_SDK_AVAILABLE)

        self.capture_target = None
        print("相机清理完毕")

    # --- 结束相机控制方法 ---

    def clear_template(self):
        self.template_paths = []
        self.template_label.clear()
        self.btn_load_template.setText('加载单模板图')
        self.btn_load_template_folder.setText('加载模板文件夹')

    def clear_image(self):
        self.image_path = None
        self.image_paths = []
        self.folder = None
        self.result_label.clear()
        self.btn_load_image.setText('加载检测图')
        self.btn_load_folder.setText('加载检测文件夹')
        self.progress_label.setText("就绪")
        self.cb_monitor_folder.setEnabled(False)
        self.cb_monitor_folder.setChecked(False)

    def create_default_ultralytics_settings(self):
        appdata_dir = os.getenv('APPDATA')
        if not appdata_dir:
            print("无法获取 APPDATA 目录，跳过创建 Ultralytics 默认设置。")
            return

        settings_path = os.path.join(appdata_dir, 'Ultralytics', 'settings.json')
        default_settings = {
            "settings_version": "0.0.6",
            "openvino_msg": True,
            "runs_dir": os.path.join(os.getcwd(), "runs"),
            "weights_dir": os.path.join(os.getcwd(), "weights"),
            "datasets_dir": os.path.join(os.getcwd(), "datasets")
        }
        try:
            os.makedirs(os.path.dirname(settings_path), exist_ok=True)
            with open(settings_path, 'w') as f:
                json.dump(default_settings, f, indent=2)
            print(f"已创建新的 Ultralytics 设置文件: {settings_path}")
        except Exception as e:
            print(f"无法创建 Ultralytics 设置文件: {str(e)}")

    def save_settings(self):
        settings = {
            'threshold_conditions': [],
            'use_ai': self.cb_use_ai.isChecked(),
            'combo_method': 'and' if self.radio_and.isChecked() else 'or',
            'debug': self.cb_debug.isChecked(),
            'monitor_folder': self.cb_monitor_folder.isChecked(),
            'auto_delete': self.cb_auto_delete.isChecked(),
        }
        for widget in self.threshold_ranges:
            condition = {'type': widget.threshold_type}
            if widget.threshold_type == "二值化阈值":
                condition['low'] = widget.low_input.text()
                condition['high'] = widget.high_input.text()
            elif widget.threshold_type == "面积阈值":
                condition['min'] = widget.min_input.text()
                condition['max'] = widget.max_input.text()
            elif widget.threshold_type == "比例阈值":
                condition['min_ratio'] = widget.min_ratio_input.text()
                condition['max_ratio'] = widget.max_ratio_input.text()
            elif widget.threshold_type == "灰度差阈值":
                condition['threshold'] = widget.threshold_input.text()
            elif widget.threshold_type == "RGB值阈值":
                condition['r_min'] = widget.r_min.text()
                condition['r_max'] = widget.r_max.text()
                condition['g_min'] = widget.g_min.text()
                condition['g_max'] = widget.g_max.text()
                condition['b_min'] = widget.b_min.text()
                condition['b_max'] = widget.b_max.text()
                condition['ratio'] = widget.ratio_input.text()
            elif widget.threshold_type == "RGB二值化":
                condition['r_min'] = widget.r_min.text()
                condition['r_max'] = widget.r_max.text()
                condition['g_min'] = widget.g_min.text()
                condition['g_max'] = widget.g_max.text()
                condition['b_min'] = widget.b_min.text()
                condition['b_max'] = widget.b_max.text()
            elif widget.threshold_type == "区域平均亮度阈值":
                condition['min_bright'] = widget.min_bright_input.text()
                condition['max_bright'] = widget.max_bright_input.text()
            settings['threshold_conditions'].append(condition)

        os.makedirs('config', exist_ok=True)
        with open('config/settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)

    def load_settings(self):
        try:
            if not os.path.exists('config/settings.json'):
                # 如果文件不存在，尝试创建默认设置
                print("未找到设置文件，正在创建默认设置...")
                self.create_default_ultralytics_settings() # 尝试修复 YOLO 设置
                return False # 首次运行，不加载旧设置
                
            with open('config/settings.json', 'r', encoding='utf-8') as f:
                settings = json.load(f)

            for widget in self.threshold_ranges[:]:
                self.remove_threshold_range(widget)

            for condition in settings.get('threshold_conditions', []):
                if 'type' in condition:
                    self.add_threshold_range(condition['type'])
                else:
                    print(f"跳过损坏的阈值条件: {condition}")


            for i, condition in enumerate(settings.get('threshold_conditions', [])):
                if i >= len(self.threshold_ranges): break
                widget = self.threshold_ranges[i]

                if condition['type'] == "二值化阈值":
                    widget.low_input.setText(condition.get('low', '100'))
                    widget.high_input.setText(condition.get('high', '255'))
                elif condition['type'] == "面积阈值":
                    widget.min_input.setText(condition.get('min', '100'))
                    widget.max_input.setText(condition.get('max', '1000'))
                elif condition['type'] == "比例阈值":
                    widget.min_ratio_input.setText(condition.get('min_ratio', '0.5'))
                    widget.max_ratio_input.setText(condition.get('max_ratio', '5.0'))
                elif condition['type'] == "灰度差阈值":
                    widget.threshold_input.setText(condition.get('threshold', '30'))
                elif condition['type'] == "RGB值阈值":
                    widget.r_min.setText(condition.get('r_min', '0'))
                    widget.r_max.setText(condition.get('r_max', '255'))
                    widget.g_min.setText(condition.get('g_min', '0'))
                    widget.g_max.setText(condition.get('g_max', '255'))
                    widget.b_min.setText(condition.get('b_min', '0'))
                    widget.b_max.setText(condition.get('b_max', '255'))
                    widget.ratio_input.setText(condition.get('ratio', '0.8'))
                elif condition['type'] == "RGB二值化":
                    widget.r_min.setText(condition.get('r_min', '0'))
                    widget.r_max.setText(condition.get('r_max', '255'))
                    widget.g_min.setText(condition.get('g_min', '0'))
                    widget.g_max.setText(condition.get('g_max', '255'))
                    widget.b_min.setText(condition.get('b_min', '0'))
                    widget.b_max.setText(condition.get('b_max', '255'))
                elif condition['type'] == "区域平均亮度阈值":
                    widget.min_bright_input.setText(condition.get('min_bright', '0'))
                    widget.max_bright_input.setText(condition.get('max_bright', '255'))

            self.cb_use_ai.setChecked(settings.get('use_ai', False))
            if settings.get('combo_method', 'and') == 'and':
                self.radio_and.setChecked(True)
            else:
                self.radio_or.setChecked(True)
            self.cb_debug.setChecked(settings.get('debug', False))
            self.cb_monitor_folder.setChecked(settings.get('monitor_folder', False))
            self.cb_auto_delete.setChecked(settings.get('auto_delete', False))

            return True
        except json.JSONDecodeError as e:
             print(f"加载设置文件 'config/settings.json' 失败: JSON 解析错误 - {e}")
             # 可以选择删除损坏的文件或提示用户
             try:
                 os.remove('config/settings.json')
                 print("已删除损坏的设置文件。下次启动将使用默认设置。")
             except OSError:
                 pass
             return False
        except Exception as e:
            print(f"加载设置失败: {str(e)}")
            traceback.print_exc()
            return False

    def closeEvent(self, event):
        self.stop_camera_feed()
        self.stop_detection()
        self.save_settings()
        if hasattr(self, 'camera_manager') and self.camera_manager:
            del self.camera_manager
        event.accept()

    def add_threshold_range(self, threshold_type):
        range_widget = ThresholdRangeWidget(threshold_type)
        range_widget.remove_btn.clicked.connect(lambda: self.remove_threshold_range(range_widget))
        self.threshold_container_layout.addWidget(range_widget)
        self.threshold_ranges.append(range_widget)
        return range_widget

    def remove_threshold_range(self, widget):
        if widget in self.threshold_ranges:
            self.threshold_ranges.remove(widget)
            widget.deleteLater()

    def load_template(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模板图", "", "图片文件 (*.png *.jpg *.jpeg)")
        if path:
            self.template_paths = [path]
            self.load_and_display_image(path, self.template_label)
            self.btn_load_template.setText(f"已加载: {os.path.basename(path)}")
            self.btn_load_template_folder.setText("加载模板文件夹")

    def load_template_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择模板图文件夹", "")
        if folder:
            template_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                template_paths.extend(glob(os.path.join(folder, ext)))
            if not template_paths:
                QMessageBox.warning(self, "警告", "文件夹中没有找到图片文件！")
                return
            self.template_paths = template_paths
            self.load_and_display_image(template_paths[0], self.template_label)
            self.btn_load_template_folder.setText(f"已加载: {len(template_paths)}张模板图")
            self.btn_load_template.setText("加载单模板图")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择检测图", "", "图片文件 (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            self.image_paths = [path]
            self.folder = None
            self.load_and_display_image(path, self.result_label)
            self.btn_load_image.setText(f"已加载: {os.path.basename(path)}")
            self.btn_load_folder.setText("加载检测文件夹")
            self.progress_label.setText(f"已选择1张图片")
            self.cb_monitor_folder.setEnabled(False)
            self.cb_monitor_folder.setChecked(False)

    def load_image_folder(self):
        self.folder = QFileDialog.getExistingDirectory(self, "选择检测图片文件夹", "")
        if self.folder:
            self.image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                self.image_paths.extend(glob(os.path.join(self.folder, ext)))

            if not self.image_paths:
                print("文件夹中没有找到图片文件，将仅监控新文件。")
                self.image_path = None
                self.result_label.clear()
            else:
                self.image_path = self.image_paths[0]
                self.load_and_display_image(self.image_path, self.result_label)

            self.btn_load_folder.setText(f"已加载: {len(self.image_paths)}张图片")
            self.progress_label.setText(f"已选择{len(self.image_paths)}张图片")
            self.btn_load_image.setText("加载检测图")
            self.cb_monitor_folder.setEnabled(True)

    def reflash_image_folder(self):
        if self.folder:
            self.image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                self.image_paths.extend(glob(os.path.join(self.folder, ext)))

            if not self.image_paths and self.image_path is None:
                 self.result_label.clear()
            elif self.image_paths and self.image_path is None:
                self.image_path = self.image_paths[0]
                self.load_and_display_image(self.image_path, self.result_label)

            self.btn_load_folder.setText(f"已加载: {len(self.image_paths)}张图片")
            self.progress_label.setText(f"已选择{len(self.image_paths)}张图片")

            return self.image_paths
        return []

    def load_and_display_image(self, img_path, label):
        try:
            pixmap = QPixmap(img_path)
            if pixmap.isNull():
                raise ValueError("QPixmap 无法加载图像文件")
            label.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败:\n{str(e)}")

    def start_detection(self):
        if hasattr(self, 'detection_thread') and self.detection_thread and self.detection_thread.isRunning():
            QMessageBox.warning(self, "警告", "检测正在进行中，请等待完成或终止！")
            return

        if not self.template_paths:
            QMessageBox.warning(self, "警告", "请先加载模板图！")
            return

        monitoring_mode = self.cb_monitor_folder.isChecked() and (self.folder is not None)

        if not monitoring_mode and not self.image_paths:
             QMessageBox.warning(self, "警告", "请加载检测图片或选择要监控的文件夹！")
             return

        current_image_paths = self.image_paths
        if monitoring_mode:
            current_image_paths = self.reflash_image_folder()

        try:
            threshold_conditions = self.get_threshold_conditions()
            if not threshold_conditions:
                QMessageBox.warning(self, "警告", "请添加至少一个阈值条件！")
                return

            self.current_output_dir = "output"
            os.makedirs(self.current_output_dir, exist_ok=True)

            use_ai = self.cb_use_ai.isChecked()
            combo_method = "and" if self.radio_and.isChecked() else "or"
            debug = self.cb_debug.isChecked()

            self.detection_thread = DetectionThread(
                self,
                current_image_paths,
                self.current_output_dir,
                threshold_conditions,
                use_ai,
                combo_method,
                debug,
                monitoring_mode=monitoring_mode
            )

            self.detection_thread.progress_signal.connect(self.update_progress)
            self.detection_thread.finished_signal.connect(self.detection_finished)
            self.detection_thread.intermediate_result_signal.connect(self.handle_intermediate_result)

            self.btn_detect.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_view_defects.setEnabled(False)
            self.cb_monitor_folder.setEnabled(False)

            if monitoring_mode:
                self.progress_bar.setMaximum(0)
                self.progress_label.setText("开始监控...")
            else:
                # 确保路径列表不为空
                max_val = len(current_image_paths) if current_image_paths else 100
                self.progress_bar.setMaximum(max_val)
                self.progress_bar.setValue(0)
                self.progress_label.setText("开始检测...")

            self.detection_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测失败:\n{str(e)}\n{traceback.format_exc()}")

    def stop_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.progress_label.setText("正在终止检测...")
            self.btn_stop.setEnabled(False)

    def update_progress(self, value, message):
        if self.cb_monitor_folder.isChecked():
            self.progress_bar.setMaximum(0)
        else:
            max_val = len(self.image_paths) if self.image_paths else 100
            self.progress_bar.setMaximum(max_val)
            self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def detection_finished(self, success, message):
        self.progress_label.setText(message)
        self.btn_detect.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_view_defects.setEnabled(True)
        if self.folder:
            self.cb_monitor_folder.setEnabled(True)

        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100 if success else 0)

        if success and not self.cb_monitor_folder.isChecked() and self.cb_auto_delete.isChecked():
            try:
                # 使用 self.image_paths (最后一次处理的文件列表)
                paths_to_delete = self.image_paths if self.image_paths else []
                for img_path in paths_to_delete:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                if paths_to_delete:
                    print(f"✅ {len(paths_to_delete)} 个原图已自动删除。")
            except Exception as e:
                print(f"❌ 删除原图失败: {e}")

        if success:
            if not self.cb_monitor_folder.isChecked():
                QMessageBox.information(self, "完成", "检测完成！")
        else:
            QMessageBox.warning(self, "警告", "检测未完成或失败！")

    def handle_intermediate_result(self, name, image):
        pass

    def get_threshold_conditions(self):
        conditions = {
            "二值化阈值": [], "RGB二值化": [], "面积阈值": [], "比例阈值": [],
            "灰度差阈值": [], "RGB值阈值": [], "区域平均亮度阈值": []
        }
        for widget in self.threshold_ranges:
            try:
                if widget.threshold_type == "二值化阈值":
                    low, high = sorted([int(widget.low_input.text()), int(widget.high_input.text())])
                    conditions["二值化阈值"].append((low, high))
                elif widget.threshold_type == "RGB二值化":
                    r_min, r_max = sorted([int(widget.r_min.text()), int(widget.r_max.text())])
                    g_min, g_max = sorted([int(widget.g_min.text()), int(widget.g_max.text())])
                    b_min, b_max = sorted([int(widget.b_min.text()), int(widget.b_max.text())])
                    conditions["RGB二值化"].append((r_min, r_max, g_min, g_max, b_min, b_max))
                elif widget.threshold_type == "面积阈值":
                    min_val, max_val = sorted([int(widget.min_input.text()), int(widget.max_input.text())])
                    conditions["面积阈值"].append((min_val, max_val))
                elif widget.threshold_type == "比例阈值":
                    min_ratio, max_ratio = sorted([float(widget.min_ratio_input.text()), float(widget.max_ratio_input.text())])
                    conditions["比例阈值"].append((min_ratio, max_ratio))
                elif widget.threshold_type == "灰度差阈值":
                    threshold = int(widget.threshold_input.text())
                    conditions["灰度差阈值"].append(threshold)
                elif widget.threshold_type == "RGB值阈值":
                    r_min, r_max = sorted([int(widget.r_min.text()), int(widget.r_max.text())])
                    g_min, g_max = sorted([int(widget.g_min.text()), int(widget.g_max.text())])
                    b_min, b_max = sorted([int(widget.b_min.text()), int(widget.b_max.text())])
                    ratio = float(widget.ratio_input.text())
                    conditions["RGB值阈值"].append((r_min, r_max, g_min, g_max, b_min, b_max, ratio))
                elif widget.threshold_type == "区域平均亮度阈值":
                    min_bright, max_bright = sorted([int(widget.min_bright_input.text()), int(widget.max_bright_input.text())])
                    conditions["区域平均亮度阈值"].append((min_bright, max_bright))
            except ValueError:
                QMessageBox.warning(self, "输入错误", f"阈值 '{widget.threshold_type}' 的输入值无效，请检查。")
                return None # 返回 None 表示验证失败
        return conditions

    def view_defects(self):
        output_dir = "output"
        if hasattr(self, 'current_output_dir'):
            output_dir = self.current_output_dir
        viewer = DefectViewer(parent=self, output_dir=output_dir)
        viewer.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    ex = PcbDefectDetector()
    ex.showMaximized()
    sys.exit(app.exec_())