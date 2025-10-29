# 文件名: diff_defect_viewer.py
# 描述: (已修改) 允许窗口缩放并添加最大化/最小化按钮，并默认最大化打开。

import os
import re
import cv2
import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout,
                            QPushButton, QFileDialog, QLabel, QMessageBox,
                            QDialog, QListWidget, QGroupBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QColor, QBrush, QFont
from PyQt5.QtCore import Qt, QPoint, QRect # --- 重新导入 QRect ---

# 假设 diff_ui_components.py 在同一个目录下
from diff_ui_components import ZoomableLabel

class DefectViewer(QDialog):
    def __init__(self, parent=None, output_dir=None):
        super().__init__(parent)
        self.setWindowTitle("缺陷查看器 (快捷键: ↑/↓切换, →标记误报)")
        
        # --- 新增: 设置窗口标志，允许最大化和调整大小 ---
        self.setWindowFlags(
            self.windowFlags() | 
            Qt.WindowMaximizeButtonHint | 
            Qt.WindowMinimizeButtonHint
        ) #
        # --- 结束新增 ---
        
        # --- 恢复原始尺寸 (作为初始尺寸) ---
        self.resize(1600, 700)
        self.output_dir = output_dir
        self.defect_files = []
        self.defect_positions = []  # 存储缺陷位置 (x, y, w, h) - 仍然需要用于缩放定位
        self.false_positives = []   # 追踪误报状态 (True/False)
        self.current_index = 0
        self.result_dirs = []
        self.current_result_index = -1
        self.original_img = None
        self.original_img_rgb = None
        # --- 移除 template_img ---
        # self.template_img = None
        self.initUI()
        
        if self.output_dir:
            self.load_result_dirs(self.output_dir)
            
        # --- 新增: 默认最大化打开 ---
        self.showMaximized()
        # --- 结束新增 ---

    def initUI(self):
        layout = QHBoxLayout(self)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        top_layout = QHBoxLayout()
        top_layout.addStretch()
        self.btn_up = QPushButton("↑")
        self.btn_up.setFixedSize(60, 40)
        self.btn_up.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.btn_up.clicked.connect(lambda: self.move_image(0, -30))
        top_layout.addWidget(self.btn_up)
        top_layout.addStretch()
        left_layout.addLayout(top_layout)

        center_layout = QHBoxLayout()
        self.btn_left = QPushButton("←")
        self.btn_left.setFixedSize(40, 60)
        self.btn_left.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.btn_left.clicked.connect(lambda: self.move_image(-30, 0))
        center_layout.addWidget(self.btn_left)

        self.zoom_label = ZoomableLabel()
        center_layout.addWidget(self.zoom_label, stretch=1)

        self.btn_right = QPushButton("→")
        self.btn_right.setFixedSize(40, 60)
        self.btn_right.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.btn_right.clicked.connect(lambda: self.move_image(30, 0))
        center_layout.addWidget(self.btn_right)

        left_layout.addLayout(center_layout, stretch=1)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        self.btn_down = QPushButton("↓")
        self.btn_down.setFixedSize(60, 40)
        self.btn_down.setStyleSheet("font-size: 16pt; font-weight: bold;")
        self.btn_down.clicked.connect(lambda: self.move_image(0, 30))
        bottom_layout.addWidget(self.btn_down)
        bottom_layout.addStretch()
        left_layout.addLayout(bottom_layout)

        layout.addWidget(left_widget, stretch=3)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        path_layout = QHBoxLayout()
        self.btn_select_path = QPushButton("选择结果目录")
        self.btn_select_path.setStyleSheet("background-color: #99ccff;")
        self.btn_select_path.clicked.connect(self.select_result_path)

        self.btn_prev_board = QPushButton("上一单板")
        self.btn_prev_board.setStyleSheet("background-color: #99ccff;")
        self.btn_prev_board.clicked.connect(self.prev_result_dir)

        self.btn_next_board = QPushButton("下一单板")
        self.btn_next_board.setStyleSheet("background-color: #99ccff;")
        self.btn_next_board.clicked.connect(self.next_result_dir)

        path_layout.addWidget(self.btn_select_path)
        path_layout.addWidget(self.btn_prev_board)
        path_layout.addWidget(self.btn_next_board)
        right_layout.addLayout(path_layout)

        self.board_label = QLabel("单板名: 未选择")
        self.board_label.setStyleSheet("""
            font-weight: bold;
            color: blue;
            padding: 5px;
        """)
        self.board_label.setWordWrap(True)
        # --- 恢复原始宽度 ---
        self.board_label.setFixedWidth(360)
        self.board_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_layout.addWidget(self.board_label)

        # --- 恢复单个缺陷显示框 ---
        self.defect_label = QLabel("缺陷区域")
        self.defect_label.setAlignment(Qt.AlignCenter)
        self.defect_label.setFixedSize(360, 360)
        self.defect_label.setStyleSheet("border: 1px solid gray;")
        right_layout.addWidget(self.defect_label)
        # --- 结束恢复 ---

        defect_list_group = QGroupBox("缺陷列表 (按序号排序)")
        defect_list_layout = QVBoxLayout(defect_list_group)
        self.defect_list = QListWidget()
        self.defect_list.setStyleSheet("font-size: 12pt;")
        self.defect_list.currentRowChanged.connect(self.on_list_click)
        defect_list_layout.addWidget(self.defect_list)
        right_layout.addWidget(defect_list_group)

        # 按钮布局
        btn_layout = QHBoxLayout()
        self.btn_prev = QPushButton("上一个")
        self.btn_false_positive = QPushButton("误报")
        self.btn_next = QPushButton("下一个")
        self.btn_generate = QPushButton("生成结果图片")

        self.btn_prev.clicked.connect(self.prev_defect)
        self.btn_false_positive.clicked.connect(self.mark_false_positive)
        self.btn_false_positive.setStyleSheet("background-color: #ff9900;")
        self.btn_next.clicked.connect(self.next_defect)
        self.btn_generate.clicked.connect(self.generate_result_image)
        self.btn_generate.setStyleSheet("background-color: #99ff99;")

        btn_layout.addWidget(self.btn_prev)
        btn_layout.addWidget(self.btn_false_positive)
        btn_layout.addWidget(self.btn_next)
        btn_layout.addWidget(self.btn_generate)

        right_layout.addLayout(btn_layout)

        layout.addWidget(right_widget, stretch=1)

        # --- 移除: 已移动到 __init__ 末尾 ---
        # if self.output_dir:
        #     self.load_result_dirs(self.output_dir)

    def keyPressEvent(self, event):
        """重写键盘事件，添加快捷键支持"""
        key = event.key()
        if key == Qt.Key_Up:
            self.prev_defect()
        elif key == Qt.Key_Down:
            self.next_defect()
        elif key == Qt.Key_Right:
            self.mark_false_positive()
            self.next_defect()
        else:
            super().keyPressEvent(event)

    def move_image(self, dx, dy):
        if self.zoom_label.offset:
            self.zoom_label.offset += QPoint(dx, dy)
            self.zoom_label.update()

    def select_result_path(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择检测结果目录",
            os.path.join(os.getcwd(), "output"),
            QFileDialog.ShowDirsOnly
        )
        if path:
            self.load_result_dirs(path)

    def load_result_dirs(self, parent_dir):
        self.result_dirs = []
        for dir_name in os.listdir(parent_dir):
            dir_path = os.path.join(parent_dir, dir_name)
            if os.path.isdir(dir_path) and dir_name.startswith("结果_"):
                self.result_dirs.append(dir_path)
        # --- 按修改时间排序 ---
        try:
            self.result_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        except OSError:
            print("警告：无法访问某些结果目录的修改时间，排序可能不准确。")

        if not self.result_dirs:
            QMessageBox.warning(self, "警告", f"目录中未找到结果文件夹: {parent_dir}")
            return

        self.current_result_index = 0
        self.load_defect_files(self.result_dirs[self.current_result_index])

    def prev_result_dir(self):
        if not self.result_dirs:
            return
        self.current_result_index -= 1
        if self.current_result_index < 0:
            self.current_result_index = len(self.result_dirs) - 1
        self.load_defect_files(self.result_dirs[self.current_result_index])

    def next_result_dir(self):
        if not self.result_dirs:
            return
        self.current_result_index += 1
        if self.current_result_index >= len(self.result_dirs):
            self.current_result_index = 0
        self.load_defect_files(self.result_dirs[self.current_result_index])

    def load_defect_files(self, output_dir):
        self.output_dir = output_dir
        dir_name = os.path.basename(output_dir)
        display_name = dir_name.replace("结果_", "", 1)

        text = f"""
        <div style='text-align: left;'>
            <b>单板:</b> {display_name}<br>
            ({self.current_result_index+1}/{len(self.result_dirs)})
        </div>
        """
        self.board_label.setText(text)

        # --- 只加载测试图 ---
        image_path = self.find_image_in_directory(output_dir)
        if not image_path:
            QMessageBox.warning(self, "警告", f"目录中未找到检测图像: {output_dir}")
            # --- 清理状态 ---
            self.original_img = None
            self.original_img_rgb = None
            self.zoom_label.clear()
            self.defect_list.clear()
            self.defect_files = []
            self.defect_positions = []
            self.false_positives = []
            self.defect_label.setText("未加载图像")
            return

        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            QMessageBox.critical(self, "错误", f"无法加载图像: {image_path}")
            return
        self.original_img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.display_original_image()

        # --- 移除加载模板图的代码 ---
        # template_path = os.path.join(output_dir, "template_image.jpg") ...
        # self.template_img = None

        defects_dir = os.path.join(output_dir, "defects")
        if not os.path.exists(defects_dir):
            self.defect_list.clear() # 清空旧列表
            # --- 仍然显示原图，但提示无缺陷 ---
            # QMessageBox.warning(self, "警告", f"未找到缺陷目录: {defects_dir}")
            self.defect_files = []
            self.defect_positions = []
            self.false_positives = []
            self.defect_label.setText("无缺陷")
            self.zoom_label.clear_defect() # 清除可能残留的标记
            return

        # **重置所有列表**
        self.defect_files = []
        self.defect_positions = []
        self.false_positives = []
        self.defect_list.clear()

        defect_info = []
        # --- 简化：直接从文件名解析信息 ---
        # 仍然依赖文件名格式 defect_id_center(x,y).png
        for filename in os.listdir(defects_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                match_center = re.search(r'(\d+)_center\((\d+),(\d+)\)', filename)
                if match_center:
                    index = int(match_center.group(1))
                    center_x = int(match_center.group(2))
                    center_y = int(match_center.group(3))
                    # --- 简化：存储中心点，稍后在update_defect_display中计算用于缩放的rect ---
                    # 无法从文件名直接获得 w, h
                    defect_info.append({
                        'index': index,
                        'path': os.path.join(defects_dir, filename),
                        'center': (center_x, center_y)
                    })

        defect_info.sort(key=lambda x: x['index'])

        for info in defect_info:
            self.defect_files.append(info['path'])
            # --- 暂时用中心点填充，后面会更新 ---
            self.defect_positions.append((info['center'][0], info['center'][1], 10, 10)) # 默认10x10
            self.false_positives.append(False)
            self.defect_list.addItem(f"缺陷 {info['index']}")

        if self.defect_files:
            self.current_index = 0
            self.defect_list.setCurrentRow(0) # 会触发 on_list_click -> update_defect_display
        else:
            self.defect_label.clear()
            self.defect_label.setText("无缺陷")
            self.zoom_label.clear_defect()
            # QMessageBox.information(self, "提示", "目录中未找到已保存的缺陷图片") # 减少提示


    def mark_false_positive(self):
        """标记/取消标记当前缺陷为误报"""
        if not self.defect_files or self.current_index >= len(self.defect_files):
            return

        current_status = self.false_positives[self.current_index]
        self.false_positives[self.current_index] = not current_status

        item = self.defect_list.item(self.current_index)
        if self.false_positives[self.current_index]:
            item.setBackground(QColor('lightgreen'))
        else:
            item.setBackground(QBrush())

    def generate_result_image(self):
        """根据当前未标记为误报的缺陷列表，重新生成结果图片"""
        if not self.output_dir or self.original_img is None:
            QMessageBox.warning(self, "警告", "请先加载包含原始图像的结果目录")
            return

        result_img = self.original_img.copy()

        valid_defect_counter = 1
        for i, pos_or_center in enumerate(self.defect_positions): # 现在可能是 (x,y,w,h) 或 (cx,cy,?,?)
            if not self.false_positives[i]:
                 # --- 从 defect_files 解析中心点 ---
                center_x, center_y = -1, -1
                match = re.search(r'center\((\d+),(\d+)\)', os.path.basename(self.defect_files[i]))
                if match:
                    center_x = int(match.group(1))
                    center_y = int(match.group(2))
                # --- 如果解析失败，则跳过 ---
                if center_x == -1: continue

                cv2.drawMarker(result_img, (center_x, center_y), (0, 0, 255),
                               cv2.MARKER_CROSS, 15, 1)
                cv2.putText(result_img, str(valid_defect_counter),
                            (center_x + 15, center_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)
                valid_defect_counter += 1

        result_path = os.path.join(self.output_dir, "updated_result.png")
        try:
            cv2.imwrite(result_path, result_img)

            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(result_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.zoom_label.setPixmap(QPixmap.fromImage(qimg)) # 更新主显示

            QMessageBox.information(self, "成功", f"已生成更新后的结果图片:\n{result_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存图片失败: {str(e)}")

    def find_image_in_directory(self, directory):
        # 查找顺序: tested_image.jpg > 校正后的图像.png > final_result.jpg > 其他 jpg/png
        names_to_check = ["tested_image.jpg", "校正后的图像.png", "final_result.jpg"]
        for name in names_to_check:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path

        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')) and "template" not in filename:
                return os.path.join(directory, filename)
        return None

    def display_original_image(self):
        h, w, ch = self.original_img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(self.original_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.zoom_label.setPixmap(QPixmap.fromImage(qimg))

    # --- 移除 _convert_cv_to_qpixmap ---

    def update_defect_display(self, index):
        if not self.defect_files or index >= len(self.defect_files):
            return

        # --- 从文件名获取中心点用于缩放定位 ---
        center_x, center_y = -1, -1
        match = re.search(r'center\((\d+),(\d+)\)', os.path.basename(self.defect_files[index]))
        if match:
            center_x = int(match.group(1))
            center_y = int(match.group(2))
            # 更新 self.defect_positions 用于缩放 (假设缺陷大小为 50x50 用于定位)
            approx_w, approx_h = 50, 50
            approx_x = max(0, center_x - approx_w // 2)
            approx_y = max(0, center_y - approx_h // 2)
            self.defect_positions[index] = (approx_x, approx_y, approx_w, approx_h)
            self.zoom_label.set_current_defect(approx_x, approx_y, approx_w, approx_h)
        else:
             print(f"无法从文件名解析中心点: {self.defect_files[index]}")
             self.zoom_label.clear_defect() # 清除标记

        # --- 显示缺陷区域图片 ---
        defect_path = self.defect_files[index]
        pixmap = QPixmap(defect_path)
        if not pixmap.isNull():
            self.defect_label.setPixmap(pixmap.scaled(
                self.defect_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        else:
            self.defect_label.setText("无法加载缺陷图片")

        # --- 移除模板区域显示逻辑 ---
        # if self.template_img is None: ...

    def on_list_click(self, row):
        # --- 增加边界检查 ---
        if 0 <= row < len(self.defect_files):
            self.current_index = row
            self.update_defect_display(row)

    def prev_defect(self):
        if self.defect_files and self.current_index > 0:
            self.current_index -= 1
            self.defect_list.setCurrentRow(self.current_index)

    def next_defect(self):
        if self.defect_files and self.current_index < len(self.defect_files) - 1:
            self.current_index += 1
            self.defect_list.setCurrentRow(self.current_index)