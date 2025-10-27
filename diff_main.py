# 文件名: diff_main.py
# 描述: (已修改) 实现相机直连预览，默认首个相机，修复自动触发。

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
from PyQt5.QtCore import Qt, pyqtSlot # --- 新增 pyqtSlot ---
from ultralytics import YOLO
import traceback
from glob import glob
import json
import time
import subprocess # 用于启动Mark工具

# 从拆分的文件中导入
from diff_ui_components import ZoomableLabel, ThresholdRangeWidget
from diff_processing import DetectionThread
from diff_defect_viewer import DefectViewer

# --- 导入相机模块 ---
HIK_SDK_AVAILABLE = False # 在 try 块之前初始化
try:
    import hik_camera
    # import diff_camera_dialog # --- 不再需要 ---
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
        self.current_output_dir = os.path.join(os.getcwd(), "output")
        self.detection_thread = None

        # --- 相机状态变量 ---
        self.camera_manager = None
        # 读取全局 HIK_SDK_AVAILABLE
        if HIK_SDK_AVAILABLE:
            try:
                self.camera_manager = hik_camera.CameraManager()
            except Exception as e:
                print(f"初始化 CameraManager 失败: {e}")
                HIK_SDK_AVAILABLE = False

        # --- 修改: 移除 camera_dialog ---
        # self.camera_dialog = None
        self.camera_thread = None
        self.current_cam_object = None
        self.hik_device_list = None
        self.capture_target_label = None # --- 新增: 记录目标Label ---
        self.is_camera_previewing = False # --- 新增: 预览状态标志 ---
        # --- 结束修改 ---

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
        self.setWindowTitle('PCB 异物检测工具 (集成Mark点校正)')
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
        self.btn_cam_template = QPushButton('相机预览') # <-- 修改文本
        self.btn_cam_image = QPushButton('相机预览')    # <-- 修改文本
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

        # --- 连接相机按钮 (目标改为 ZoomableLabel) ---
        self.btn_cam_template.clicked.connect(lambda: self.toggle_camera(self.template_label))
        self.btn_cam_image.clicked.connect(lambda: self.toggle_camera(self.result_label))
        # --- 结束 ---

        template_layout = QVBoxLayout(template_group)
        result_layout = QVBoxLayout(result_group)

        template_layout.addWidget(self.template_label, stretch=4)

        template_btn_layout = QHBoxLayout()
        template_btn_layout.addWidget(self.btn_load_template)
        template_btn_layout.addWidget(self.btn_load_template_folder)
        template_btn_layout.addWidget(self.btn_cam_template)
        
        self.btn_open_marker = QPushButton('设置校正Mark点')
        self.btn_open_marker.setStyleSheet("background-color: #FFC300; color: black;")
        self.btn_open_marker.clicked.connect(self.open_marker_tool)
        template_btn_layout.addWidget(self.btn_open_marker)
        
        template_btn_layout.addWidget(self.btn_clear_template)
        template_layout.addLayout(template_btn_layout, stretch=1)

        result_layout.addWidget(self.result_label, stretch=4)

        result_btn_layout = QHBoxLayout()
        result_btn_layout.addWidget(self.btn_load_image)
        result_btn_layout.addWidget(self.btn_load_folder)
        result_btn_layout.addWidget(self.btn_cam_image)
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

        btn_layout_thresh = QHBoxLayout() # --- 重命名避免冲突 ---
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

        btn_layout_thresh.addWidget(self.btn_add_binary)
        btn_layout_thresh.addWidget(self.btn_add_gray)
        btn_layout_thresh.addWidget(self.btn_add_rgb_binary)
        btn_layout_thresh.addWidget(self.btn_add_area)
        btn_layout_thresh.addWidget(self.btn_add_ratio)
        btn_layout_thresh.addWidget(self.btn_add_brightness)
        btn_layout_thresh.addWidget(self.btn_add_rgb)

        threshold_layout.addLayout(btn_layout_thresh)

        self.threshold_scroll = QScrollArea()
        self.threshold_scroll.setWidgetResizable(True)
        self.threshold_container = QWidget()
        self.threshold_container_layout = QVBoxLayout(self.threshold_container)
        self.threshold_container_layout.setSpacing(5)
        self.threshold_scroll.setWidget(self.threshold_container)

        threshold_layout.addWidget(self.threshold_scroll)
        param_layout.addWidget(threshold_group, 1, 0, 1, 5)
        main_layout.addWidget(param_group, stretch=1)

        btn_layout_bottom = QHBoxLayout() # --- 重命名避免冲突 ---
        btn_layout_bottom.setSpacing(10)

        # --- Mark点校正复选框 ---
        self.cb_use_alignment = QCheckBox('启用Mark点校正')
        self.cb_use_alignment.setStyleSheet("font-weight: bold; color: #FF5733;")
        btn_layout_bottom.addWidget(self.cb_use_alignment)
        # --- 结束 ---

        self.cb_use_ai = QCheckBox('启用AI过滤')
        self.cb_debug = QCheckBox('保存中间结果图')
        self.cb_auto_delete = QCheckBox('自动删除原图')
        self.cb_monitor_folder = QCheckBox('持续监控文件夹')
        self.cb_monitor_folder.setEnabled(False)

        btn_layout_bottom.addWidget(self.cb_auto_delete)

        if not self.yolo_model:
            self.cb_use_ai.setEnabled(False)
            self.cb_use_ai.setText("启用AI (模型未加载)")

        btn_layout_bottom.addWidget(self.cb_use_ai)
        btn_layout_bottom.addWidget(self.cb_debug)
        btn_layout_bottom.addWidget(self.cb_monitor_folder)
        
        self.btn_set_output_dir = QPushButton('设置输出目录')
        self.btn_set_output_dir.clicked.connect(self.set_output_directory)
        btn_layout_bottom.addWidget(self.btn_set_output_dir)
        
        btn_layout_bottom.addStretch(1)

        self.btn_detect = QPushButton('🔍 开始检测')
        self.btn_detect.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_detect.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton('⏹ 终止检测')
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)

        self.btn_view_defects = QPushButton('🔍 查看缺陷')
        self.btn_view_defects.clicked.connect(self.view_defects)

        btn_layout_bottom.addWidget(self.btn_detect)
        btn_layout_bottom.addWidget(self.btn_stop)
        btn_layout_bottom.addWidget(self.btn_view_defects)
        main_layout.addLayout(btn_layout_bottom)

        self.setLayout(main_layout)

        self.add_threshold_range("二值化阈值")
        self.add_threshold_range("面积阈值")
        self.add_threshold_range("比例阈值")

    def open_marker_tool(self):
        try:
            mark_script_path = 'mark.py'
            if not os.path.exists(mark_script_path):
                QMessageBox.critical(self, "错误", f"未找到Mark点工具脚本: {mark_script_path}")
                return

            if not self.template_paths:
                QMessageBox.warning(self, "请先加载模板", "请先在主界面加载一个“标准模板图”，Mark工具将自动使用该图。")
                return
            
            template_to_pass = self.template_paths[0]

            error_log_path = os.path.join(os.getcwd(), "mark_tool_error.log")
            
            command_list = [sys.executable, mark_script_path, template_to_pass]
            
            with open(error_log_path, 'w') as error_log:
                proc = subprocess.Popen(
                    command_list, 
                    stderr=error_log, 
                    stdout=subprocess.DEVNULL
                )

            time.sleep(1) 
            
            if os.path.exists(error_log_path) and os.path.getsize(error_log_path) > 0:
                with open(error_log_path, 'r') as f:
                    error_content = f.read()
                QMessageBox.critical(self, "Mark工具启动失败", 
                    f"Mark点工具(mark.py)启动时遇到错误，界面可能无法弹出。\n\n"
                    f"请检查环境是否缺少库 (如 'Pillow' 或 'imutils')。\n\n"
                    f"错误详情 (已保存到 {error_log_path}):\n{error_content[:500]}...") 
                return 

            QMessageBox.information(self, "提示", 
                f"Mark点工具已启动 (自动加载: {os.path.basename(template_to_pass)})。\n\n"
                "1. 请在工具中依次框选【3个】Mark点。\n"
                "2. 点击【保存Mark点并退出】。\n\n"
                "完成后，请勾选主界面的“启用Mark点校正”复选框。")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开Mark点工具: {e}\n{traceback.format_exc()}")
    
    # --- 修改: 相机控制逻辑 ---
    def toggle_camera(self, target_label: ZoomableLabel):
        """启动或停止相机预览/捕获"""
        if not HIK_SDK_AVAILABLE or self.camera_manager is None:
            QMessageBox.critical(self, "错误", "相机SDK未正确加载。")
            return

        # 确定是哪个按钮被点击
        is_template_target = (target_label == self.template_label)
        btn = self.btn_cam_template if is_template_target else self.btn_cam_image

        # --- 情况1: 当前正在预览，用户点击进行捕获 ---
        if self.is_camera_previewing and self.capture_target_label == target_label:
            if self.camera_thread and self.camera_thread.isRunning():
                print("发送捕获信号...")
                btn.setText("采集中...")
                btn.setEnabled(False)
                self.camera_thread.capture_and_stop() # 线程内部会处理停止和信号发射
            return # 等待 frame_captured_signal

        # --- 情况2: 当前正在预览其他窗口，用户点击当前窗口 ---
        elif self.is_camera_previewing and self.capture_target_label != target_label:
             QMessageBox.warning(self, "提示", "请先停止另一个窗口的相机预览或完成取图。")
             return

        # --- 情况3: 当前没有预览，用户点击启动预览 ---
        elif not self.is_camera_previewing:
            try:
                # 自动选择第一个相机
                devices, self.hik_device_list = self.camera_manager.list_devices()
                if not devices or not self.hik_device_list:
                    QMessageBox.warning(self, "未找到相机", "未枚举到任何海康威视相机设备。")
                    return

                # 连接到第一个设备 (index 0)
                device_index = 0
                self.current_cam_object = self.camera_manager.connect(self.hik_device_list, device_index)

                if self.current_cam_object is None:
                    QMessageBox.critical(self, "连接失败", f"无法连接到第一个相机: {devices[0]}")
                    self.hik_device_list = None # 清理
                    return

                self.capture_target_label = target_label # 记录目标

                # 创建并启动线程
                self.camera_thread = hik_camera.CameraThread(self.current_cam_object)
                self.camera_thread.new_frame_signal.connect(self.update_camera_preview)
                self.camera_thread.frame_captured_signal.connect(self.on_frame_captured)
                self.camera_thread.finished.connect(self.on_camera_thread_finished)
                self.camera_thread.error_signal.connect(self.on_camera_error)

                self.camera_thread.start()
                self.is_camera_previewing = True

                # 更新按钮状态
                btn.setText("点击取图")
                other_btn = self.btn_cam_image if is_template_target else self.btn_cam_template
                other_btn.setEnabled(False) # 禁用另一个相机按钮
                target_label.setText("正在连接相机...") # 提示用户

            except Exception as e:
                QMessageBox.critical(self, "启动失败", f"启动相机时出错: {e}\n{traceback.format_exc()}")
                self.on_camera_thread_finished() # 出错时确保清理

        # --- 情况4: 其他（理论上不应发生，可能是状态错误） ---
        else:
             print("未知的相机状态，请尝试重启程序。")

    @pyqtSlot(np.ndarray)
    def update_camera_preview(self, img_bgr):
        """更新目标Label的图像"""
        if self.capture_target_label and self.is_camera_previewing:
            try:
                h, w, ch = img_bgr.shape
                if h > 0 and w > 0:
                    bytes_per_line = ch * w
                    # 注意 QImage 使用 BGR 数据
                    q_img = QImage(img_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.capture_target_label.setPixmap(pixmap) # 直接更新目标Label
                else:
                    self.capture_target_label.setText("无效帧")

            except Exception as e:
                print(f"更新预览失败: {e}")
                self.capture_target_label.setText("预览错误")
                # 可以考虑在这里停止相机
                # self.stop_camera_feed()

    def stop_camera_feed(self):
        """停止相机预览线程"""
        if self.camera_thread and self.camera_thread.isRunning():
            print("正在停止相机预览...")
            self.camera_thread.stop()
            # finished信号会自动调用 on_camera_thread_finished 进行清理

    def on_camera_error(self, message):
        QMessageBox.critical(self, "相机错误", message)
        self.stop_camera_feed() # 发生错误时停止

    @pyqtSlot(str, int, int) # --- 明确信号参数 ---
    def on_frame_captured(self, save_path, width, height):
        """相机成功捕获一帧并保存后调用"""
        print(f"帧已捕获! 保存至: {save_path}, 尺寸: {width}x{height}")
        # 线程已经停止，状态将在 on_camera_thread_finished 中重置
        try:
            pixmap = QPixmap(save_path)
            if pixmap.isNull():
                raise Exception(f"加载保存的 PNG 文件失败: {save_path}")

            # 根据捕获目标更新UI和状态
            if self.capture_target_label == self.template_label:
                self.template_paths = [save_path]
                self.template_label.setPixmap(pixmap) # 显示最终捕获的图
                self.btn_load_template.setText(f"已加载: {os.path.basename(save_path)}")
                self.btn_load_template_folder.setText("加载模板文件夹")
                # 模板取图后不需要自动检测

            elif self.capture_target_label == self.result_label:
                self.image_paths = [save_path]
                self.image_path = save_path # <--- 关键：设置 image_path
                self.folder = None
                self.result_label.setPixmap(pixmap) # 显示最终捕获的图
                self.btn_load_image.setText(f"已加载: {os.path.basename(save_path)}")
                self.btn_load_folder.setText("加载检测文件夹")
                self.cb_monitor_folder.setEnabled(False)
                self.cb_monitor_folder.setChecked(False)

                # --- 修复: 确认 image_path 已设置后再触发检测 ---
                print("相机取图（检测图）完成，自动触发检测...")
                # 稍微延迟以确保UI更新完成 (可选，但有时有帮助)
                # QTimer.singleShot(100, self.start_detection)
                self.start_detection() # 直接调用
                # --- 结束修复 ---
            else:
                 print("警告: 捕获目标未知!")

        except Exception as e:
            QMessageBox.critical(self, "处理捕获失败", f"处理捕获的帧时出错: {e}")
            traceback.print_exc()
        finally:
             # 确保按钮状态在捕获后重置 (也在 finished 里做)
             self.reset_camera_buttons()


    def on_camera_thread_finished(self):
        """相机线程结束后的清理工作"""
        print("相机线程已结束，正在清理...")

        # --- 修改: 不再需要 camera_dialog ---
        # if self.camera_dialog: ...

        if self.current_cam_object and self.camera_manager:
            self.camera_manager.disconnect(self.current_cam_object)
            self.current_cam_object = None

        self.camera_thread = None
        self.is_camera_previewing = False # 重置预览状态
        self.capture_target_label = None # 清空目标
        self.hik_device_list = None # 清理设备列表

        # 重置按钮状态
        self.reset_camera_buttons()

        print("相机清理完毕")

    def reset_camera_buttons(self):
        """重置两个相机按钮的文本和状态"""
        self.btn_cam_template.setText("相机预览")
        self.btn_cam_image.setText("相机预览")
        self.btn_cam_template.setEnabled(HIK_SDK_AVAILABLE)
        self.btn_cam_image.setEnabled(HIK_SDK_AVAILABLE)

    # --- 结束相机控制修改 ---

    def set_output_directory(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择输出目录",
            self.current_output_dir,
            QFileDialog.ShowDirsOnly
        )
        if path:
            self.current_output_dir = path
            QMessageBox.information(self, "设置成功", f"检测结果将保存到:\n{self.current_output_dir}")

    def clear_template(self):
        self.stop_camera_feed() # 如果正在预览模板区，停止
        self.template_paths = []
        self.template_label.clear()
        self.template_label.setText("模板图像") # 恢复提示
        self.btn_load_template.setText('加载单模板图')
        self.btn_load_template_folder.setText('加载模板文件夹')

    def clear_image(self):
        self.stop_camera_feed() # 如果正在预览结果区，停止
        self.image_path = None
        self.image_paths = []
        self.folder = None
        self.result_label.clear()
        self.result_label.setText("结果图像") # 恢复提示
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
            'output_dir': self.current_output_dir,
            'use_alignment': self.cb_use_alignment.isChecked() # --- 保存Mark点设置 ---
        }
        for widget in self.threshold_ranges:
            condition = {'type': widget.threshold_type}
            if widget.threshold_type == "二值化阈值":
                condition['low'] = widget.low_input.text()
                condition['high'] = widget.high_input.text()
            # ... (其他阈值类型) ...
            elif widget.threshold_type == "面积阈值":
                condition['min'] = widget.min_input.text()
                condition['max'] = widget.max_input.text()
            elif widget.threshold_type == "比例阈值":
                condition['min_ratio'] = widget.min_ratio_input.text()
                condition['max_ratio'] = widget.max_ratio_input.text()
            elif widget.threshold_type == "灰度差阈值":
                condition['threshold'] = widget.threshold_input.text()
            elif widget.threshold_type == "RGB值阈值":
                condition['r_min'] = widget.r_min.text(); condition['r_max'] = widget.r_max.text()
                condition['g_min'] = widget.g_min.text(); condition['g_max'] = widget.g_max.text()
                condition['b_min'] = widget.b_min.text(); condition['b_max'] = widget.b_max.text()
                condition['ratio'] = widget.ratio_input.text()
            elif widget.threshold_type == "RGB二值化":
                condition['r_min'] = widget.r_min.text(); condition['r_max'] = widget.r_max.text()
                condition['g_min'] = widget.g_min.text(); condition['g_max'] = widget.g_max.text()
                condition['b_min'] = widget.b_min.text(); condition['b_max'] = widget.b_max.text()
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
                print("未找到设置文件，正在创建默认设置...")
                self.create_default_ultralytics_settings()
                return False
                
            with open('config/settings.json', 'r', encoding='utf-8') as f:
                settings = json.load(f)

            for widget in self.threshold_ranges[:]:
                self.remove_threshold_range(widget)

            for condition in settings.get('threshold_conditions', []):
                if 'type' in condition:
                    widget = self.add_threshold_range(condition['type']) # --- 修改: 获取新创建的widget ---
                    # --- 在这里设置加载的值 ---
                    if widget: # 确保 widget 创建成功
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
                else:
                    print(f"跳过损坏的阈值条件: {condition}")

            # --- 移除旧的阈值加载循环 ---
            # for i, condition in enumerate(settings.get('threshold_conditions', [])): ...

            self.cb_use_ai.setChecked(settings.get('use_ai', False))
            if settings.get('combo_method', 'and') == 'and':
                self.radio_and.setChecked(True)
            else:
                self.radio_or.setChecked(True)
            self.cb_debug.setChecked(settings.get('debug', False))
            self.cb_monitor_folder.setChecked(settings.get('monitor_folder', False))
            self.cb_auto_delete.setChecked(settings.get('auto_delete', False))
            
            default_output_dir = os.path.join(os.getcwd(), "output")
            self.current_output_dir = settings.get('output_dir', default_output_dir)
            
            self.cb_use_alignment.setChecked(settings.get('use_alignment', False))

            return True
        except json.JSONDecodeError as e:
             print(f"加载设置文件 'config/settings.json' 失败: JSON 解析错误 - {e}")
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
        self.stop_camera_feed() # 确保停止相机
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
        self.stop_camera_feed() # 加载时停止预览
        path, _ = QFileDialog.getOpenFileName(self, "选择模板图", "", "图片文件 (*.png *.jpg *.jpeg)")
        if path:
            self.template_paths = [path]
            self.load_and_display_image(path, self.template_label)
            self.btn_load_template.setText(f"已加载: {os.path.basename(path)}")
            self.btn_load_template_folder.setText("加载模板文件夹")

    def load_template_folder(self):
        self.stop_camera_feed() # 加载时停止预览
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
        self.stop_camera_feed() # 加载时停止预览
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
        self.stop_camera_feed() # 加载时停止预览
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

            # --- 选择文件夹后自动开始监控 ---
            self.cb_monitor_folder.setChecked(True)
            print("检测文件夹已加载，自动开始监控...")
            self.start_detection()
            # --- 结束 ---

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

    def load_and_display_image(self, img_path, label: ZoomableLabel):
        try:
            pixmap = QPixmap(img_path)
            if pixmap.isNull():
                raise ValueError("QPixmap 无法加载图像文件")
            label.setPixmap(pixmap)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败:\n{str(e)}")
            label.setText("加载失败") # 提示用户

    def start_detection(self):
        # --- 新增: 开始检测前确保相机已停止 ---
        if self.is_camera_previewing:
             reply = QMessageBox.question(self, "相机预览中",
                                         "相机正在预览，开始检测将停止预览并使用当前加载的图片/文件夹。是否继续？",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
             if reply == QMessageBox.No:
                 return
             self.stop_camera_feed()
        # --- 结束新增 ---

        if hasattr(self, 'detection_thread') and self.detection_thread and self.detection_thread.isRunning():
            QMessageBox.warning(self, "警告", "检测正在进行中，请等待完成或终止！")
            return

        if not self.template_paths:
            QMessageBox.warning(self, "警告", "请先加载模板图！")
            return

        monitoring_mode = self.cb_monitor_folder.isChecked() and (self.folder is not None)

        # --- 修改: 即使在监控模式下，也要检查 image_paths 或 folder ---
        if not self.image_paths and not self.folder:
             QMessageBox.warning(self, "警告", "请加载检测图片或选择要监控的文件夹！")
             return
        # --- 结束修改 ---

        current_image_paths = self.image_paths
        if monitoring_mode:
            current_image_paths = self.reflash_image_folder() # 刷新列表以包含最新文件

        try:
            threshold_conditions = self.get_threshold_conditions()
            if not threshold_conditions:
                # QMessageBox is shown in get_threshold_conditions
                return

            os.makedirs(self.current_output_dir, exist_ok=True)

            use_ai = self.cb_use_ai.isChecked()
            combo_method = "and" if self.radio_and.isChecked() else "or"
            debug = self.cb_debug.isChecked()
            
            use_alignment = self.cb_use_alignment.isChecked()

            # --- 传递 current_image_paths ---
            self.detection_thread = DetectionThread(
                self,
                current_image_paths, # 使用当前的文件列表
                self.current_output_dir,
                threshold_conditions,
                use_ai,
                combo_method,
                debug,
                monitoring_mode=monitoring_mode,
                use_alignment=use_alignment
            )

            self.detection_thread.progress_signal.connect(self.update_progress)
            self.detection_thread.finished_signal.connect(self.detection_finished)
            self.detection_thread.intermediate_result_signal.connect(self.handle_intermediate_result)

            self.btn_detect.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_view_defects.setEnabled(False)
            self.cb_monitor_folder.setEnabled(False) # 检测期间禁用监控切换

            if monitoring_mode:
                self.progress_bar.setMaximum(0) # 不定进度条
                self.progress_label.setText("开始监控...")
            else:
                max_val = len(current_image_paths) if current_image_paths else 1 # 防止除零
                self.progress_bar.setMaximum(max_val)
                self.progress_bar.setValue(0)
                self.progress_label.setText("开始检测...")

            self.detection_thread.start()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动检测失败:\n{str(e)}\n{traceback.format_exc()}")
            # 出错时恢复按钮状态
            self.btn_detect.setEnabled(True)
            self.btn_stop.setEnabled(False)
            if self.folder: self.cb_monitor_folder.setEnabled(True)


    def stop_detection(self):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.progress_label.setText("正在终止检测...")
            self.btn_stop.setEnabled(False) # 禁用停止按钮，等待线程结束

    def update_progress(self, value, message):
        if self.cb_monitor_folder.isChecked():
            self.progress_bar.setMaximum(0) # 保持不定
            self.progress_bar.setValue(-1) # 某些样式下会动
        else:
            max_val = self.progress_bar.maximum() # 获取当前最大值
            if max_val > 0: # 确保不是不定模式
                self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def detection_finished(self, success, message):
        self.progress_label.setText(message)
        self.btn_detect.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_view_defects.setEnabled(True)
        # 只有在选择了文件夹的情况下才重新启用监控复选框
        if self.folder:
            self.cb_monitor_folder.setEnabled(True)

        # 重置进度条
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100 if success else 0)

        # 自动删除逻辑 (保持不变)
        if success and not self.cb_monitor_folder.isChecked() and self.cb_auto_delete.isChecked():
            try:
                # 使用线程结束时处理的文件列表 (从线程获取可能更安全，但暂时用self.image_paths)
                paths_to_delete = self.detection_thread.processed_files if hasattr(self.detection_thread, 'processed_files') else self.image_paths
                deleted_count = 0
                for img_path in paths_to_delete:
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                            deleted_count += 1
                        except Exception as del_e:
                             print(f"❌ 删除文件 {img_path} 失败: {del_e}")
                if deleted_count > 0:
                    print(f"✅ {deleted_count} 个原图已自动删除。")
            except Exception as e:
                print(f"❌ 删除原图时出错: {e}")

        if success:
            if not self.cb_monitor_folder.isChecked(): # 仅单次运行时提示
                QMessageBox.information(self, "完成", "检测完成！")
        else:
             if message != "监控已停止": # 避免停止监控时弹出警告
                QMessageBox.warning(self, "警告", f"检测未完成或失败！\n{message}")

    def handle_intermediate_result(self, name, image):
        pass # 目前未使用

    def get_threshold_conditions(self):
        conditions = {
            "二值化阈值": [], "RGB二值化": [], "面积阈值": [], "比例阈值": [],
            "灰度差阈值": [], "RGB值阈值": [], "区域平均亮度阈值": []
        }
        valid = True
        for widget in self.threshold_ranges:
            try:
                if widget.threshold_type == "二值化阈值":
                    low, high = sorted([int(widget.low_input.text()), int(widget.high_input.text())])
                    conditions["二值化阈值"].append((low, high))
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
                elif widget.threshold_type == "RGB二值化":
                    r_min, r_max = sorted([int(widget.r_min.text()), int(widget.r_max.text())])
                    g_min, g_max = sorted([int(widget.g_min.text()), int(widget.g_max.text())])
                    b_min, b_max = sorted([int(widget.b_min.text()), int(widget.b_max.text())])
                    conditions["RGB二值化"].append((r_min, r_max, g_min, g_max, b_min, b_max))
                elif widget.threshold_type == "区域平均亮度阈值":
                    min_bright, max_bright = sorted([int(widget.min_bright_input.text()), int(widget.max_bright_input.text())])
                    conditions["区域平均亮度阈值"].append((min_bright, max_bright))
            except ValueError:
                QMessageBox.warning(self, "输入错误", f"阈值 '{widget.threshold_type}' 的输入值无效，请检查。")
                valid = False
                break # 发现一个错误就停止
        return conditions if valid else None # 验证失败返回 None

    def view_defects(self):
        output_dir = self.current_output_dir
        if not output_dir or not os.path.exists(output_dir):
            output_dir = "output" # 备用路径
            
        viewer = DefectViewer(parent=self, output_dir=output_dir)
        viewer.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    ex = PcbDefectDetector()
    ex.showMaximized()
    sys.exit(app.exec_())