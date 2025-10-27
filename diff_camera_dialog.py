import sys
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal

class CameraFeedDialog(QDialog):
    """一个简单的对话框，用于显示实时相机画面"""
    
    # 定义一个信号，当窗口被关闭时发出
    dialog_closed_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("实时相机画面 - (点击“点击取图”按钮进行采集)")
        self.setMinimumSize(800, 600)
        
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel("正在连接相机...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000; color: #FFF;")
        self.layout.addWidget(self.image_label)
        
        self.pixmap_size = QPixmap(self.size()).size()

    def update_image(self, q_img: QImage):
        """更新显示的图像"""
        if not q_img.isNull():
            # 缩放图像以适应标签大小
            self.image_label.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.image_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))

    def resizeEvent(self, event):
        """在窗口大小改变时更新pixmap尺寸记录"""
        self.pixmap_size = QPixmap(self.size()).size()
        super().resizeEvent(event)

    def closeEvent(self, event):
        """当用户关闭窗口时（例如按X）"""
        print("相机预览窗口已关闭")
        self.dialog_closed_signal.emit() # 发出信号
        event.accept()