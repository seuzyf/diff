# 文件名: mark.py
# 描述: (已修改) 接收命令行参数自动加载模板，并实现加载时自适应缩放。

import os
import cv2
import numpy as np
import time
import threading
import queue
import json
import sys # --- 新增: 用于读取命令行参数 ---
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import imutils

class ImageAlignmentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mark点配置工具")
        self.root.state('zoomed')

        # 初始化变量
        self.template_path = None
        self.template_image = None
        self.display_image_pil = None
        self.display_image_tk = None
        self.reference_points = []
        self.mark_ids = []
        self.scale_factor = 1.0
        self.current_scale = 1.0
        self.canvas_img_id = None
        self.canvas_x = 0
        self.canvas_y = 0
        self.template_rois = []
        self.template_centers = []
        
        # --- 配置保存路径 ---
        self.config_dir = "config"
        os.makedirs(self.config_dir, exist_ok=True)
        self.json_path = os.path.join(self.config_dir, "marks.json")
        self.roi_paths = [
            os.path.join(self.config_dir, "mark_roi_1.png"),
            os.path.join(self.config_dir, "mark_roi_2.png"),
            os.path.join(self.config_dir, "mark_roi_3.png")
        ]

        # 创建GUI组件
        self.create_widgets()

        # 鼠标事件绑定
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)

        # 状态变量
        self.drawing = False
        self.rect_id = None
        self.start_x = None
        self.start_y = None

        self.status_label = Label(self.root, text="请加载模板图，并依次框选3个Mark点", bd=1, relief=SUNKEN, anchor=W)
        self.status_label.pack(fill=X, padx=10, pady=5)
        
        # --- 修改: 启动时检查参数 ---
        self.root.after(100, self.process_initial_load)
        # --- 结束修改 ---

    def create_widgets(self):
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        control_frame = Frame(main_frame, width=300, padx=10, pady=10)
        control_frame.pack(side=LEFT, fill=Y)
        control_frame.pack_propagate(False)
        
        template_frame = LabelFrame(control_frame, text="1. 模板图片", padx=5, pady=5)
        template_frame.pack(fill=X, pady=5)
        
        self.btn_select_template = Button(template_frame, text="选择模板图片", command=self.select_template)
        self.btn_select_template.pack(side=LEFT, padx=5)
        self.template_label = Label(template_frame, text="未选择", wraplength=200, justify=LEFT)
        self.template_label.pack(side=LEFT, padx=5)
        
        points_frame = LabelFrame(control_frame, text="2. 标记点选择 (在模板上框选3个)", padx=5, pady=5)
        points_frame.pack(fill=X, pady=5)
        
        self.points_status = Label(points_frame, text="已选择 0/3 个标记区域")
        self.points_status.pack(anchor=W)
        
        delete_frame = Frame(points_frame)
        delete_frame.pack(fill=X, pady=5)
        
        self.delete_buttons = []
        for i in range(3):
            btn = Button(delete_frame, text=f"删除标记{i+1}", command=lambda idx=i: self.delete_mark(idx), state=DISABLED)
            btn.pack(side=LEFT, padx=2)
            self.delete_buttons.append(btn)
        
        Button(points_frame, text="刷新显示", command=self.redraw_marks).pack(fill=X, pady=2)
        
        process_frame = LabelFrame(control_frame, text="3. 保存", padx=5, pady=5)
        process_frame.pack(fill=X, pady=5)
        
        self.save_button = Button(process_frame, text="保存Mark点并退出", command=self.save_and_close, bg="#4CAF50", fg="white", height=2)
        self.save_button.pack(fill=X, pady=5)
        
        image_frame = Frame(main_frame)
        image_frame.pack(side=RIGHT, fill=BOTH, expand=True)
        
        h_scroll = Scrollbar(image_frame, orient=HORIZONTAL)
        h_scroll.pack(side=BOTTOM, fill=X)
        v_scroll = Scrollbar(image_frame)
        v_scroll.pack(side=RIGHT, fill=Y)
        
        self.canvas = Canvas(image_frame, width=800, height=600, xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set, bg="gray")
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        h_scroll.config(command=self.canvas.xview)
        v_scroll.config(command=self.canvas.yview)

    # --- 新增: 处理启动加载 ---
    def process_initial_load(self):
        """检查命令行参数或加载现有配置"""
        passed_path = None
        if len(sys.argv) > 1:
            passed_path = sys.argv[1]

        if passed_path and os.path.exists(passed_path):
            self.update_status(f"已自动加载模板: {os.path.basename(passed_path)}")
            self.template_path = passed_path
            self.template_label.config(text=os.path.basename(self.template_path))
            self.btn_select_template.config(state=DISABLED) # 禁用按钮
            self.load_template_image(self.template_path)
            self.display_image(auto_fit=True) # 自适应显示
            self.load_existing_config() # 加载此模板对应的Mark点
        else:
            if passed_path:
                self.update_status(f"错误: 传入路径无效: {passed_path}")
            # 没传路径，或路径无效，尝试加载上次的配置
            self.load_existing_config()
            if self.template_image is not None:
                # 如果从json加载了图片，也自适应显示
                self.display_image(auto_fit=True) 
    # --- 结束新增 ---

    def load_existing_config(self):
        """尝试加载已有的Mark配置"""
        try:
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as f:
                    config = json.load(f)
                
                # 仅当没有传入路径时，才使用JSON中的路径
                if self.template_path is None: 
                    self.template_path = config.get('template_path')
                
                # 如果当前模板路径与配置中的路径匹配，则加载Mark点
                if self.template_path and self.template_path == config.get('template_path'):
                    self.template_centers = config.get('centers', [])
                    self.reference_points = config.get('rects', []) # 加载矩形框
                    rois_loaded = 0
                    self.template_rois = []
                    
                    if not self.template_image: # 确保图像已加载
                        self.load_template_image(self.template_path)

                    for i in range(len(self.template_centers)):
                        roi_path = self.roi_paths[i]
                        if os.path.exists(roi_path):
                            roi = cv2.imread(roi_path)
                            if roi is not None:
                                self.template_rois.append(roi)
                                rois_loaded += 1
                    
                    if len(self.reference_points) == 3 and rois_loaded == 3:
                        self.points_status.config(text=f"已加载 3/3 个标记区域")
                        self.update_status("已加载现有配置。可重新框选或直接退出。")
                        self.update_delete_buttons()
                        self.redraw_marks() # 确保加载后重绘
                    else:
                        # 配置不完整，清空
                        self.reference_points = []
                        self.template_rois = []
                        self.template_centers = []
                else:
                    # 模板不匹配，清空旧标记
                    self.reference_points = []
                    self.template_rois = []
                    self.template_centers = []

        except Exception as e:
            print(f"加载旧配置失败: {e}")
            self.reference_points = []
            self.template_rois = []
            self.template_centers = []


    def save_and_close(self):
        """保存配置到config文件夹并退出"""
        if len(self.reference_points) != 3 or len(self.template_rois) != 3 or len(self.template_centers) != 3:
            messagebox.showerror("错误", "必须选择三个有效的标记区域才能保存！")
            return
        
        if not self.template_path or not self.template_image is not None:
            messagebox.showerror("错误", "模板图片未成功加载！")
            return
            
        try:
            # 1. 保存 ROIs
            for i in range(3):
                cv2.imwrite(self.roi_paths[i], self.template_rois[i])
            
            # 2. 保存 JSON (中心点 和 矩形框)
            config = {
                "template_path": self.template_path,
                "centers": self.template_centers,
                "rects": self.reference_points # 保存矩形框
            }
            with open(self.json_path, 'w') as f:
                json.dump(config, f, indent=4)
                
            messagebox.showinfo("成功", f"Mark点配置已成功保存到 {self.config_dir} 目录。")
            self.root.destroy() # 关闭窗口

        except Exception as e:
            messagebox.showerror("保存失败", f"保存Mark配置时出错: {str(e)}")


    def update_status(self, message):
        """更新状态栏文本"""
        self.status_label.config(text=message)


    def select_template(self):
        path = filedialog.askopenfilename(
            title="选择模板图片",
            filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp"), ("所有文件", "*.*")]
        )
        if path:
            # 清空旧的标记
            self.reference_points = []
            self.mark_ids = []
            self.template_rois = []
            self.template_centers = []
            self.points_status.config(text="已选择 0/3 个标记区域")
            self.update_delete_buttons()

            self.template_path = path
            self.template_label.config(text=os.path.basename(self.template_path))
            self.load_template_image(self.template_path)
            self.display_image(auto_fit=True) # --- 修改: 手动选择也自适应 ---
            self.update_status("请选择三个标记区域")
    
    def load_template_image(self, path):
        self.template_image = cv2.imread(path)
        if self.template_image is None:
            messagebox.showerror("错误", "无法加载模板图片")
            self.template_path = None
            self.template_label.config(text="未选择")
            self.btn_select_template.config(state=NORMAL) # 确保按钮可用

    # --- 修改: 增加 auto_fit 功能 ---
    def display_image(self, auto_fit=False):
        if self.template_image is None: return

        # --- 新增: 自适应缩放逻辑 ---
        if auto_fit:
            # 强制Tkinter更新画布尺寸
            self.canvas.update_idletasks() 
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            
            if canvas_w > 1 and canvas_h > 1: # 确保画布已渲染
                img_h, img_w = self.template_image.shape[:2]
                if img_w == 0 or img_h == 0: return

                scale_w = canvas_w / img_w
                scale_h = canvas_h / img_h
                
                # 取较小的比例并留出一点边距
                self.current_scale = min(scale_w, scale_h) * 0.98 
                self.current_scale = max(0.01, self.current_scale) # 最小缩放
                
                self.canvas_x = 0 # 自动适应时重置视图
                self.canvas_y = 0
            else:
                # 画布未就绪，使用默认值
                self.current_scale = 0.1 
        # --- 结束新增 ---
            
        h, w = self.template_image.shape[:2]
        scaled_w = int(w * self.current_scale)
        scaled_h = int(h * self.current_scale)
        
        # 避免尺寸为0
        if scaled_w <= 0 or scaled_h <= 0:
            return
            
        display_image = cv2.resize(self.template_image, (scaled_w, scaled_h))
        display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        
        self.display_image_pil = Image.fromarray(display_image)
        self.display_image_tk = ImageTk.PhotoImage(self.display_image_pil)
        
        self.canvas.delete("all")
        self.canvas_img_id = self.canvas.create_image(
            self.canvas_x, self.canvas_y, anchor=NW, image=self.display_image_tk)
        
        self.canvas.config(scrollregion=self.canvas.bbox(ALL))
        self.scale_factor = w / scaled_w if scaled_w != 0 else 1.0
        self.redraw_marks()
        
    def redraw_marks(self):
        if not self.scale_factor or self.scale_factor == 0: return
        
        for mark_id in self.mark_ids:
            self.canvas.delete(mark_id[0])
            self.canvas.delete(mark_id[1])
        self.mark_ids = []
        for idx, (x1, y1, x2, y2) in enumerate(self.reference_points):
            # --- 修改: 坐标换算 ---
            # 原始坐标 -> 显示坐标
            sx1, sy1 = x1 / self.scale_factor, y1 / self.scale_factor
            sx2, sy2 = x2 / self.scale_factor, y2 / self.scale_factor
            rect_id = self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="red", width=2)
            text_id = self.canvas.create_text(sx1, sy1 - 15, text=f"Mark{idx+1}", fill="red", font=("Arial", 10, "bold"), anchor=NW)
            self.mark_ids.append((rect_id, text_id))
    
    def update_delete_buttons(self):
        count = len(self.reference_points)
        for i, btn in enumerate(self.delete_buttons):
            btn.config(state=NORMAL if i < count else DISABLED)

    def delete_mark(self, index):
        
        if index < len(self.reference_points):
            self.reference_points.pop(index)
            self.template_rois.pop(index)
            self.template_centers.pop(index)
            self.points_status.config(text=f"已选择 {len(self.reference_points)}/3 个标记区域")
            self.redraw_marks()
            self.update_delete_buttons()
            self.update_status(f"已删除标记 {index+1}")

    def on_click(self, event):

        if len(self.reference_points) >= 3:
            messagebox.showinfo("提示", "已有3个标记，请先删除一个再添加新标记")
            return
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red", width=2)

    def on_drag(self, event):
        
        if self.drawing and self.rect_id:
            end_x = self.canvas.canvasx(event.x)
            end_y = self.canvas.canvasy(event.y)
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, end_x, end_y)

    def on_release(self, event):
        
        if not self.drawing or not self.rect_id: return
        self.drawing = False
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        if abs(end_x - self.start_x) > 5 and abs(end_y - self.start_y) > 5:
            # 转换为原始图像坐标
            x1, y1 = int(min(self.start_x, end_x) * self.scale_factor), int(min(self.start_y, end_y) * self.scale_factor)
            x2, y2 = int(max(self.start_x, end_x) * self.scale_factor), int(max(self.start_y, end_y) * self.scale_factor)
            
            # --- 确保坐标在图像内 ---
            h, w = self.template_image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            # --- 结束 ---

            roi = self.template_image[y1:y2, x1:x2]
            if roi.size > 0:
                self.reference_points.append((x1, y1, x2, y2))
                self.template_rois.append(roi)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                self.template_centers.append((center_x, center_y))
            
            self.points_status.config(text=f"已选择 {len(self.reference_points)}/3 个标记区域")
            self.update_delete_buttons()
            self.redraw_marks() # 使用原始坐标重绘
        else:
            self.canvas.delete(self.rect_id)
        self.rect_id = None
        
    def on_mousewheel(self, event):
        
        scale_factor = 1.1
        if event.num == 5 or event.delta == -120:
            self.current_scale /= scale_factor
            self.current_scale = max(0.01, self.current_scale)
        elif event.num == 4 or event.delta == 120:
            self.current_scale *= scale_factor
            self.current_scale = min(10.0, self.current_scale)
        
        # --- 修改: 滚轮缩放不自动适应 ---
        self.display_image(auto_fit=False)


if __name__ == "__main__":
    root = Tk()
    app = ImageAlignmentApp(root)
    root.mainloop()