import sys
import os
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QVBoxLayout, QHBoxLayout,
                             QFileDialog, QScrollArea, QSizePolicy, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QWheelEvent
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft) # 修改对齐方式，使其从左上角开始而不是居中
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #2D2D30; border: 1px solid #3F3F46;")
        self.setCursor(Qt.CrossCursor)
        # self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # 当 setWidgetResizable(False) 时，这个策略不是必需的

        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.drawing = False
        self.rect_start = QPoint()
        self.rect_end = QPoint()
        self.rectangles = []
        self.current_rect = None
        self.image_pos = QPoint(0, 0)

    def load_image(self, file_path):
        self.original_pixmap = QPixmap(file_path)
        if self.original_pixmap.isNull():
            return

        # 获取滚动区域视口的大小
        scroll_area = self.parent().parent()
        available_size = scroll_area.viewport().size()

        # 计算适合的缩放比例（保持宽高比）
        pixmap_size = self.original_pixmap.size()
        scale_width = (available_size.width() - 2) / pixmap_size.width() # 减去边框
        scale_height = (available_size.height() - 2) / pixmap_size.height() # 减去边框
        self.scale_factor = min(scale_width, scale_height, 1.0) # 初始加载时不放大图片

        # 限制最小缩放比例
        self.scale_factor = max(0.02, self.scale_factor)

        self.update_scaled_pixmap()
        self.rectangles = []
        self.update()

    def set_rectangles(self, rects):
        """设置矩形列表（用于加载mask配置）"""
        self.rectangles = rects
        self.update()

    def undo_last_rectangle(self):
        if self.rectangles:
            self.rectangles.pop()
            self.update()

    def resizeEvent(self, event):
        # FIX: 删除了这里的逻辑。
        # 之前的逻辑会在窗口大小改变时重置缩放比例，这会覆盖用户通过滚轮设置的缩放级别。
        # QScrollArea 会自动处理视口变化，所以我们不需要在这里重新计算缩放。
        super().resizeEvent(event)

    def update_scaled_pixmap(self):
        if self.original_pixmap:
            scaled_size = self.original_pixmap.size() * self.scale_factor
            self.scaled_pixmap = self.original_pixmap.scaled(
                scaled_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.setPixmap(self.scaled_pixmap)
            # FIX: 调整 QLabel 的大小以匹配缩放后的 QPixmap。
            # 这是让 QScrollArea 知道其内容有多大的关键步骤。
            self.resize(self.scaled_pixmap.size())


    def wheelEvent(self, event: QWheelEvent):
        if not self.original_pixmap:
            return

        # 计算缩放因子
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = 1.25 if zoom_in else 0.8
        new_scale = self.scale_factor * zoom_factor

        # 限制缩放范围
        if 0.02 <= new_scale <= 10.0:
            scroll_area = self.parent().parent()
            if not (scroll_area and isinstance(scroll_area, QScrollArea)):
                return
            
            h_bar = scroll_area.horizontalScrollBar()
            v_bar = scroll_area.verticalScrollBar()

            # 记录缩放前的鼠标指向的坐标和滚动条位置
            mouse_pos_before_zoom = event.pos()
            h_val_before_zoom = h_bar.value()
            v_val_before_zoom = v_bar.value()

            # 更新缩放并重绘
            self.scale_factor = new_scale
            self.update_scaled_pixmap()

            # 计算缩放后鼠标应该指向的坐标
            mouse_pos_after_zoom = mouse_pos_before_zoom * zoom_factor

            # 计算新的滚动条位置，以保持鼠标指向的点在屏幕上的位置不变
            h_val_after_zoom = mouse_pos_after_zoom.x() - mouse_pos_before_zoom.x() + h_val_before_zoom
            v_val_after_zoom = mouse_pos_after_zoom.y() - mouse_pos_before_zoom.y() + v_val_before_zoom

            h_bar.setValue(int(h_val_after_zoom))
            v_bar.setValue(int(v_val_after_zoom))


    def get_pixmap_rect(self):
        """获取图片在label中的实际显示区域"""
        if not self.scaled_pixmap:
            return QRect()
        
        # FIX: 由于我们设置了对齐方式为左上角，并且label大小与pixmap大小一致，
        # 所以图片的矩形区域就是 (0, 0, width, height)。
        return self.scaled_pixmap.rect()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.scaled_pixmap:
            self.drawing = True
            self.rect_start = event.pos()
            self.rect_end = event.pos()
            self.current_rect = QRect(self.rect_start, self.rect_end)
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing and self.scaled_pixmap:
            self.rect_end = event.pos()
            self.current_rect = QRect(self.rect_start, self.rect_end).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing and self.scaled_pixmap:
            self.drawing = False
            if self.current_rect.isValid():
                pixmap_rect = self.get_pixmap_rect()
                
                # 确保绘制的矩形在图片内
                clipped_rect = self.current_rect.intersected(pixmap_rect)

                # 转换坐标到原始图片尺寸
                original_size = self.original_pixmap.size()
                scaled_size = self.scaled_pixmap.size()
                
                if scaled_size.width() == 0 or scaled_size.height() == 0:
                    return

                scale_x = original_size.width() / scaled_size.width()
                scale_y = original_size.height() / scaled_size.height()

                rect = QRect(
                    int(clipped_rect.x() * scale_x),
                    int(clipped_rect.y() * scale_y),
                    int(clipped_rect.width() * scale_x),
                    int(clipped_rect.height() * scale_y)
                )

                if rect.isValid():
                    self.rectangles.append(rect)
            self.current_rect = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        if not self.scaled_pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 获取图片的实际显示区域 (现在是整个label)
        pixmap_rect = self.get_pixmap_rect()
        
        # 计算从原始坐标到显示坐标的缩放比例
        original_size = self.original_pixmap.size()
        scaled_size = self.scaled_pixmap.size()

        if original_size.width() == 0 or original_size.height() == 0:
            return

        scale_x = scaled_size.width() / original_size.width()
        scale_y = scaled_size.height() / original_size.height()

        # 绘制所有已保存的矩形
        painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.SolidLine))
        for rect in self.rectangles:
            display_rect = QRect(
                int(rect.x() * scale_x),
                int(rect.y() * scale_y),
                int(rect.width() * scale_x),
                int(rect.height() * scale_y)
            )
            painter.drawRect(display_rect)

        # 绘制当前正在绘制的矩形
        if self.current_rect:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            painter.drawRect(self.current_rect)

class MaskGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片遮罩生成器")
        self.setGeometry(100, 100, 1024, 768) # 增大初始窗口大小

        # 创建主部件和布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 创建图片显示区域
        self.image_label = ImageLabel()

        # 添加滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        # FIX: 将 setWidgetResizable 设置为 False
        # 这是解决滚动条问题的关键。它告诉 QScrollArea 不要调整 image_label 的大小，
        # 而是根据 image_label 的实际大小（我们手动设置）来显示滚动条。
        scroll_area.setWidgetResizable(False)
        scroll_area.setStyleSheet("background-color: #2D2D30; border: none;") # 移除边框
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded) # 按需显示滚动条
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)   # 按需显示滚动条

        # 创建按钮区域
        button_layout = QHBoxLayout()
        self.load_button = QPushButton("加载图片")
        self.load_button.setStyleSheet(
            "QPushButton { background-color: #0078D7; color: white; padding: 8px 16px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #106EBE; }"
            "QPushButton:pressed { background-color: #005A9E; }"
        )
        self.undo_button = QPushButton("回退")
        self.undo_button.setStyleSheet(
            "QPushButton { background-color: #4C4C4C; color: white; padding: 8px 16px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #5C5C5C; }"
            "QPushButton:pressed { background-color: #3C3C3C; }"
        )
        self.clear_button = QPushButton("清除矩形")
        self.clear_button.setStyleSheet(
            "QPushButton { background-color: #D83B01; color: white; padding: 8px 16px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #C23501; }"
            "QPushButton:pressed { background-color: #A52C00; }"
        )
        self.generate_button = QPushButton("生成遮罩")
        self.generate_button.setStyleSheet(
            "QPushButton { background-color: #107C10; color: white; padding: 8px 16px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0E6B0E; }"
            "QPushButton:pressed { background-color: #0B5A0B; }"
        )
        self.load_mask_button = QPushButton("加载Mask配置")
        self.load_mask_button.setStyleSheet(
            "QPushButton { background-color: #68217A; color: white; padding: 8px 16px; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #5C1C6C; }"
            "QPushButton:pressed { background-color: #4A1757; }"
        )

        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.undo_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.load_mask_button)

        # 添加缩放提示标签
        info_label = QLabel("提示: 使用鼠标滚轮缩放图片 | 按住鼠标左键拖动绘制矩形 | 回退按钮可撤销上一步操作")
        info_label.setStyleSheet("color: #A0A0A0; padding: 5px; font-size: 10pt;")
        info_label.setAlignment(Qt.AlignCenter)

        # 添加到主布局
        main_layout.addWidget(scroll_area)
        main_layout.addWidget(info_label)
        main_layout.addLayout(button_layout)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # 连接信号
        self.load_button.clicked.connect(self.load_image)
        self.generate_button.clicked.connect(self.generate_mask)
        self.clear_button.clicked.connect(self.clear_rectangles)
        self.undo_button.clicked.connect(self.undo_last_rectangle)
        self.load_mask_button.clicked.connect(self.load_mask_config)

        # 存储当前图片路径
        self.current_image_path = ""

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tif)"
        )
        if file_path:
            self.current_image_path = file_path
            self.image_label.load_image(file_path)

    def clear_rectangles(self):
        if self.image_label:
            self.image_label.rectangles = []
            self.image_label.update()

    def undo_last_rectangle(self):
        if self.image_label:
            self.image_label.undo_last_rectangle()

    def generate_mask(self):
        if not self.current_image_path or not self.image_label.original_pixmap:
            QMessageBox.warning(self, "警告", "请先加载图片！")
            return

        # 获取原始图片尺寸
        original_size = self.image_label.original_pixmap.size()
        width, height = original_size.width(), original_size.height()

        # 创建全白的QImage
        mask_image = QImage(width, height, QImage.Format_RGB888)
        mask_image.fill(Qt.white)

        # 创建QPainter绘制黑色矩形
        painter = QPainter(mask_image)
        painter.setBrush(Qt.black)
        painter.setPen(Qt.NoPen)

        for rect in self.image_label.rectangles:
            painter.drawRect(rect)
        painter.end()

        # 保存mask.png到原始图片同一目录
        save_dir = os.path.dirname(self.current_image_path)
        base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]

        # 保存遮罩图片
        mask_image_path = os.path.join(save_dir, f"{base_name}_mask.png")
        mask_image.save(mask_image_path, "PNG")

        # 保存JSON配置文件
        json_path = os.path.join(save_dir, f"{base_name}_mask.json")
        self.save_mask_config(json_path)

        # 显示成功消息
        self.statusBar().showMessage(f"遮罩和配置已保存至: {save_dir}", 5000)
        # 显示成功对话框
        QMessageBox.information(
            self,
            "操作成功",
            f"遮罩图片已保存至:\n{mask_image_path}\n\n"
            f"遮罩配置已保存至:\n{json_path}",
            QMessageBox.Ok
        )

    def save_mask_config(self, json_path):
        """保存mask配置到JSON文件"""
        config = {
            "image_path": self.current_image_path,
            "rectangles": []
        }

        for rect in self.image_label.rectangles:
            config["rectangles"].append({
                "x": rect.x(),
                "y": rect.y(),
                "width": rect.width(),
                "height": rect.height()
            })

        with open(json_path, 'w') as f:
            json.dump(config, f, indent=4)

    def load_mask_config(self):
        """加载mask配置文件"""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "请先加载图片！")
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Mask配置文件",
            os.path.dirname(self.current_image_path),
            "JSON文件 (*.json)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                config = json.load(f)

            # 检查配置文件是否匹配当前图片
            if config.get("image_path") != self.current_image_path:
                reply = QMessageBox.question(
                    self, "图片不匹配",
                    "配置文件中的图片路径与当前加载的图片不一致。是否继续加载？",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return

            # 解析矩形数据
            rectangles = []
            for rect_data in config.get("rectangles", []):
                rect = QRect(
                    rect_data["x"],
                    rect_data["y"],
                    rect_data["width"],
                    rect_data["height"]
                )
                rectangles.append(rect)

            # 更新图像标签中的矩形
            self.image_label.set_rectangles(rectangles)

            # 显示成功消息
            self.statusBar().showMessage(f"Mask配置已加载: {file_path}", 5000)
            QMessageBox.information(
                self, "加载成功",
                f"已成功加载 {len(rectangles)} 个矩形框配置",
                QMessageBox.Ok
            )

        except Exception as e:
            QMessageBox.critical(
                self, "加载错误",
                f"加载配置文件时出错:\n{str(e)}",
                QMessageBox.Ok
            )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # 设置深色主题
    dark_palette = app.palette()
    dark_palette.setColor(dark_palette.Window, QColor(30, 30, 30))
    dark_palette.setColor(dark_palette.WindowText, Qt.white)
    dark_palette.setColor(dark_palette.Base, QColor(25, 25, 25))
    dark_palette.setColor(dark_palette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ToolTipBase, Qt.white)
    dark_palette.setColor(dark_palette.ToolTipText, Qt.white)
    dark_palette.setColor(dark_palette.Text, Qt.white)
    dark_palette.setColor(dark_palette.Button, QColor(53, 53, 53))
    dark_palette.setColor(dark_palette.ButtonText, Qt.white)
    dark_palette.setColor(dark_palette.BrightText, Qt.red)
    dark_palette.setColor(dark_palette.Highlight, QColor(142, 45, 197).lighter())
    dark_palette.setColor(dark_palette.HighlightedText, Qt.black)
    app.setPalette(dark_palette)

    # 设置应用样式
    app.setStyleSheet("""
        QMainWindow {
            background-color: #252526;
        }
        QStatusBar {
            background-color: #333333;
            color: white;
        }
        QScrollBar:vertical {
            background: #2D2D30;
            width: 15px;
            margin: 0px 0px 0px 0px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #3F3F46;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #505058;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        QScrollBar:horizontal {
            background: #2D2D30;
            height: 15px;
            margin: 0px 0px 0px 0px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
            background: #3F3F46;
            min-width: 20px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #505058;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
        }
    """)

    window = MaskGeneratorApp()
    window.show()
    sys.exit(app.exec_())