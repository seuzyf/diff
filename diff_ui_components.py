import sys
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QLineEdit, 
                             QPushButton, QSizePolicy)
from PyQt5.QtGui import (QIntValidator, QDoubleValidator, QPixmap, QWheelEvent, 
                         QPen, QPainter, QTransform)
from PyQt5.QtCore import Qt, QPoint, QRect

class ZoomableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        self.setMinimumSize(1, 1)
        self.setMouseTracking(True)
        self.current_defect_rect = None
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
    
    def set_current_defect(self, x, y, w, h):
        self.current_defect_rect = QRect(x, y, w, h)
        self.update()
    
    def clear_defect(self):
        self.current_defect_rect = None
        self.update()
    
    def paintEvent(self, event):
        if not self.scaled_pixmap:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        widget_center = self.rect().center()
        pixmap_rect = self.scaled_pixmap.rect()
        pixmap_rect.moveCenter(widget_center + self.offset)
        painter.drawPixmap(pixmap_rect, self.scaled_pixmap, self.scaled_pixmap.rect())
        
        if self.current_defect_rect:
            x = self.current_defect_rect.x()
            y = self.current_defect_rect.y()
            w = self.current_defect_rect.width()
            h = self.current_defect_rect.height()
            cx = x + w // 2
            cy = y + h // 2
            scaled_cx = cx * self.scale_factor
            scaled_cy = cy * self.scale_factor
            final_x = pixmap_rect.x() + scaled_cx
            final_y = pixmap_rect.y() + scaled_cy
            painter.setPen(QPen(Qt.red, 2))
            
            # --- 修改: 使十字线贯穿整个视图 ---
            view_rect = self.rect() # 获取整个控件的矩形
            # 绘制水平线 (贯穿)
            painter.drawLine(view_rect.left(), final_y, view_rect.right(), final_y)
            # 绘制垂直线 (贯穿)
            painter.drawLine(final_x, view_rect.top(), final_x, view_rect.bottom())
            # --- 结束修改 ---
        
        painter.end()
    
    def wheelEvent(self, event: QWheelEvent):
        if self.underMouse() and self.original_pixmap:
            widget_pos = event.pos()
            widget_center = self.rect().center()
            pixmap_rect = self.scaled_pixmap.rect()
            pixmap_rect.moveCenter(widget_center + self.offset)
            
            if pixmap_rect.width() > 0 and pixmap_rect.height() > 0:
                mouse_in_pixmap_x = (widget_pos.x() - pixmap_rect.x()) / pixmap_rect.width()
                mouse_in_pixmap_y = (widget_pos.y() - pixmap_rect.y()) / pixmap_rect.height()
                
                degrees = event.angleDelta().y() / 8
                steps = degrees / 15
                scale_change = 1.1 ** steps
                
                old_scale = self.scale_factor
                self.scale_factor *= scale_change
                self.scale_factor = max(0.01, min(1.0, self.scale_factor))
                
                if abs(old_scale - self.scale_factor) > 0.001:
                    self.update_display_pixmap()
                    new_pixmap_rect = self.scaled_pixmap.rect()
                    new_pixmap_rect.moveCenter(widget_center + self.offset)
                    
                    desired_x = new_pixmap_rect.x() + mouse_in_pixmap_x * new_pixmap_rect.width()
                    desired_y = new_pixmap_rect.y() + mouse_in_pixmap_y * new_pixmap_rect.height()
                    
                    delta_x = widget_pos.x() - desired_x
                    delta_y = widget_pos.y() - desired_y
                    self.offset += QPoint(delta_x, delta_y)
                    self.update()
            event.accept()
        else:
            super().wheelEvent(event)
    
    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        self.reset_view()
    
    def reset_view(self):
        if self.original_pixmap:
            widget_size = self.size()
            img_size = self.original_pixmap.size()
            width_ratio = widget_size.width() / img_size.width()
            height_ratio = widget_size.height() / img_size.height()
            self.scale_factor = min(width_ratio, height_ratio, 1.0)
            self.offset = QPoint(0, 0)
            self.update_display_pixmap()
    
    def update_display_pixmap(self):
        if self.original_pixmap:
            scaled_size = self.original_pixmap.size() * self.scale_factor
            self.scaled_pixmap = self.original_pixmap.scaled(
                scaled_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.update()
    
    def resizeEvent(self, event):
        if self.original_pixmap:
            self.reset_view()
        super().resizeEvent(event)

class ThresholdRangeWidget(QWidget):
    def __init__(self, threshold_type, parent=None):
        super().__init__(parent)
        self.threshold_type = threshold_type
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)
        
        # 显示阈值类型标签
        type_label = QLabel(f"{threshold_type}:")
        type_label.setStyleSheet("font-weight: bold; color: #336699;")
        self.layout.addWidget(type_label)
        
        # 根据不同类型创建不同的输入控件
        if threshold_type == "二值化阈值":
            self.low_input = QLineEdit("100")
            self.low_input.setFixedWidth(60)
            self.low_input.setValidator(QIntValidator(0, 255))
            self.high_input = QLineEdit("255")
            self.high_input.setFixedWidth(60)
            self.high_input.setValidator(QIntValidator(0, 255))
            self.layout.addWidget(QLabel("低:"))
            self.layout.addWidget(self.low_input)
            self.layout.addWidget(QLabel("高:"))
            self.layout.addWidget(self.high_input)
            
        elif threshold_type == "面积阈值":
            self.min_input = QLineEdit("100")
            self.min_input.setFixedWidth(60)
            self.min_input.setValidator(QIntValidator(0, 100000))
            self.max_input = QLineEdit("1000")
            self.max_input.setFixedWidth(60)
            self.max_input.setValidator(QIntValidator(0, 100000))
            self.layout.addWidget(QLabel("最小:"))
            self.layout.addWidget(self.min_input)
            self.layout.addWidget(QLabel("最大:"))
            self.layout.addWidget(self.max_input)
            
        elif threshold_type == "比例阈值":
            self.min_ratio_input = QLineEdit("0.5")
            self.min_ratio_input.setFixedWidth(60)
            self.min_ratio_input.setValidator(QDoubleValidator(0.01, 10.0, 2))
            self.max_ratio_input = QLineEdit("5.0")
            self.max_ratio_input.setFixedWidth(60)
            self.max_ratio_input.setValidator(QDoubleValidator(0.01, 10.0, 2))
            self.layout.addWidget(QLabel("最小:"))
            self.layout.addWidget(self.min_ratio_input)
            self.layout.addWidget(QLabel("最大:"))
            self.layout.addWidget(self.max_ratio_input)
            
        elif threshold_type == "灰度差阈值":
            self.threshold_input = QLineEdit("30")
            self.threshold_input.setFixedWidth(60)
            self.threshold_input.setValidator(QIntValidator(0, 255))
            self.layout.addWidget(QLabel("阈值:"))
            self.layout.addWidget(self.threshold_input)
            
        elif threshold_type == "RGB值阈值":
            # RGB范围
            self.r_min = QLineEdit("0")
            self.r_max = QLineEdit("255")
            self.g_min = QLineEdit("0")
            self.g_max = QLineEdit("255")
            self.b_min = QLineEdit("0")
            self.b_max = QLineEdit("255")
            # 比例阈值
            self.ratio_input = QLineEdit("0.8")
            
            for w in [self.r_min, self.r_max, self.g_min, self.g_max, self.b_min, self.b_max]:
                w.setFixedWidth(40)
                w.setValidator(QIntValidator(0, 255))
            self.ratio_input.setFixedWidth(40)
            self.ratio_input.setValidator(QDoubleValidator(0.0, 1.0, 2))
            
            rgb_layout = QHBoxLayout()
            rgb_layout.addWidget(QLabel("R:"))
            rgb_layout.addWidget(self.r_min)
            rgb_layout.addWidget(QLabel("-"))
            rgb_layout.addWidget(self.r_max)
            rgb_layout.addWidget(QLabel("G:"))
            rgb_layout.addWidget(self.g_min)
            rgb_layout.addWidget(QLabel("-"))
            rgb_layout.addWidget(self.g_max)
            rgb_layout.addWidget(QLabel("B:"))
            rgb_layout.addWidget(self.b_min)
            rgb_layout.addWidget(QLabel("-"))
            rgb_layout.addWidget(self.b_max)
            rgb_layout.addWidget(QLabel("比例:"))
            rgb_layout.addWidget(self.ratio_input)
            
            self.layout.addLayout(rgb_layout)
        
        elif threshold_type == "RGB二值化":
            # RGB范围
            self.r_min = QLineEdit("0")
            self.r_max = QLineEdit("255")
            self.g_min = QLineEdit("0")
            self.g_max = QLineEdit("255")
            self.b_min = QLineEdit("0")
            self.b_max = QLineEdit("255")
            
            for w in [self.r_min, self.r_max, self.g_min, self.g_max, self.b_min, self.b_max]:
                w.setFixedWidth(40)
                w.setValidator(QIntValidator(0, 255))
            
            rgb_layout = QHBoxLayout()
            rgb_layout.addWidget(QLabel("R:"))
            rgb_layout.addWidget(self.r_min)
            rgb_layout.addWidget(QLabel("-"))
            rgb_layout.addWidget(self.r_max)
            rgb_layout.addWidget(QLabel("G:"))
            rgb_layout.addWidget(self.g_min)
            rgb_layout.addWidget(QLabel("-"))
            rgb_layout.addWidget(self.g_max)
            rgb_layout.addWidget(QLabel("B:"))
            rgb_layout.addWidget(self.b_min)
            rgb_layout.addWidget(QLabel("-"))
            rgb_layout.addWidget(self.b_max)
            
            self.layout.addLayout(rgb_layout)
        elif threshold_type == "区域平均亮度阈值":
            self.min_bright_input = QLineEdit("0")
            self.min_bright_input.setFixedWidth(60)
            self.min_bright_input.setValidator(QIntValidator(0, 255))
            self.max_bright_input = QLineEdit("255")
            self.max_bright_input.setFixedWidth(60)
            self.max_bright_input.setValidator(QIntValidator(0, 255))
            self.layout.addWidget(QLabel("最小亮度:"))
            self.layout.addWidget(self.min_bright_input)
            self.layout.addWidget(QLabel("最大亮度:"))
            self.layout.addWidget(self.max_bright_input)
        
        # 移除按钮
        self.remove_btn = QPushButton("移除")
        self.remove_btn.setFixedWidth(60)
        self.remove_btn.setStyleSheet("background-color: #ff9999;")
        self.layout.addWidget(self.remove_btn)
        self.layout.addStretch()