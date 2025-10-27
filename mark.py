import os
import cv2
import numpy as np
import time
import threading
import queue
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import imutils

class ImageAlignmentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片位置校正工具")
        self.root.state('zoomed')

        # 初始化变量
        self.template_path = None
        self.template_image = None
        self.display_image_pil = None
        self.display_image_tk = None
        self.reference_points = []
        self.mark_ids = []
        self.input_dir = None
        self.output_dir = None
        self.scale_factor = 1.0
        self.current_scale = 1.0
        self.canvas_img_id = None
        self.canvas_x = 0
        self.canvas_y = 0
        self.template_rois = []
        self.template_centers = []
        self.delete_after_process = BooleanVar(value=False)  # 新增：自动删除标志

        # --- 新增的变量 ---
        self.monitoring = False  # 监控状态标志
        self.monitor_thread = None # 监控线程
        self.processed_files = set() # 存储已处理的文件名
        self.status_queue = queue.Queue() # 用于线程安全的状态更新
        self.error_queue = queue.Queue()  # 新增：错误消息队列

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

        # 启动队列检查循环
        self.check_queue()

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
        
        process_frame = LabelFrame(control_frame, text="3. 监控与处理", padx=5, pady=5)
        process_frame.pack(fill=X, pady=5)
        
        self.btn_select_input = Button(process_frame, text="选择输入文件夹", command=self.select_input_dir)
        self.btn_select_input.pack(fill=X, pady=2)
        self.input_dir_label = Label(process_frame, text="未选择", wraplength=280, justify=LEFT)
        self.input_dir_label.pack(fill=X, pady=2)
        
        self.btn_select_output = Button(process_frame, text="选择输出文件夹", command=self.select_output_dir)
        self.btn_select_output.pack(fill=X, pady=2)
        self.output_dir_label = Label(process_frame, text="未选择", wraplength=280, justify=LEFT)
        self.output_dir_label.pack(fill=X, pady=2)
        
        # 新增：自动删除原图选项
        auto_delete_frame = Frame(process_frame)
        auto_delete_frame.pack(fill=X, pady=5)
        self.chk_auto_delete = Checkbutton(auto_delete_frame, text="处理后删除原图", 
                                          variable=self.delete_after_process,
                                          onvalue=True, offvalue=False)
        self.chk_auto_delete.pack(side=LEFT, padx=5)
        
        # 警告标签
        self.warning_label = Label(process_frame, text="", fg="red", wraplength=280, justify=LEFT)
        self.warning_label.pack(fill=X, pady=2)
        
        # --- 修改的按钮区域 ---
        monitor_control_frame = Frame(process_frame)
        monitor_control_frame.pack(fill=X, pady=5)
        
        self.start_button = Button(monitor_control_frame, text="开始监控", command=self.start_monitoring, bg="#4CAF50", fg="white")
        self.start_button.pack(side=LEFT, expand=True, fill=X, padx=2)
        
        self.stop_button = Button(monitor_control_frame, text="停止监控", command=self.stop_monitoring, bg="#f44336", fg="white", state=DISABLED)
        self.stop_button.pack(side=LEFT, expand=True, fill=X, padx=2)
        
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
        
        self.status_label = Label(self.root, text="准备就绪", bd=1, relief=SUNKEN, anchor=W)
        self.status_label.pack(fill=X, padx=10, pady=5)
    
    def start_monitoring(self):
        """开始监控文件夹"""
        # 验证输入
        if not self.template_image is not None:
            messagebox.showerror("错误", "请先选择模板图片")
            return
        if len(self.reference_points) != 3:
            messagebox.showerror("错误", "请选择三个有效的标记区域")
            return
        if not self.input_dir:
            messagebox.showerror("错误", "请选择待处理的输入文件夹")
            return
        if not self.output_dir:
            messagebox.showerror("错误", "请选择输出文件夹")
            return
        if self.input_dir == self.output_dir:  # 新增：检查路径是否相同
            messagebox.showerror("错误", "输入路径和输出路径不能相同！")
            return

        self.monitoring = True
        self.processed_files.clear() # 每次开始都清空已处理列表
        
        # 禁用按钮
        self.start_button.config(state=DISABLED)
        self.stop_button.config(state=NORMAL)
        self.btn_select_template.config(state=DISABLED)
        self.btn_select_input.config(state=DISABLED)
        self.btn_select_output.config(state=DISABLED)
        self.chk_auto_delete.config(state=DISABLED)  # 禁用自动删除选项
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.update_status("监控已开始...")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1) # 等待线程结束

        # 启用按钮
        self.start_button.config(state=NORMAL)
        self.stop_button.config(state=DISABLED)
        self.btn_select_template.config(state=NORMAL)
        self.btn_select_input.config(state=NORMAL)
        self.btn_select_output.config(state=NORMAL)
        self.chk_auto_delete.config(state=NORMAL)  # 重新启用自动删除选项

        self.update_status("监控已停止。")

    def _monitor_loop(self):
        """在后台线程中运行的监控循环"""
        while self.monitoring:
            try:
                all_files = os.listdir(self.input_dir)
                image_files = {f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))}
                
                # 找出新增的文件
                new_files = image_files - self.processed_files
                
                if new_files:
                    for filename in new_files:
                        if not self.monitoring: break # 如果在处理过程中点击停止，则退出
                        self.status_queue.put(f"检测到新图片: {filename}，正在处理...")
                        time.sleep(3)
                        self._process_single_image(filename)
                        self.processed_files.add(filename) # 加入已处理集合
                        self.status_queue.put(f"处理完成: {filename}")
                
                if not self.monitoring: break

                self.status_queue.put(f"正在监控中... {len(self.processed_files)}个文件已处理。")
                time.sleep(2) # 每2秒扫描一次
            except FileNotFoundError:
                self.error_queue.put(f"输入文件夹 '{self.input_dir}' 不存在")
                self.stop_monitoring() # 自动停止
                break
            except Exception as e:
                self.error_queue.put(f"监控线程出错: {str(e)}")
                time.sleep(5) # 出错后等待更长时间

    def _process_single_image(self, filename):
        """处理单张图片的逻辑"""
        input_path = os.path.join(self.input_dir, filename)
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            image = cv2.imread(input_path)
            if image is None: 
                self.error_queue.put(f"无法读取图片: {filename}")
                return

            # 检查图片大小是否与模板一致
            if image.shape != self.template_image.shape:
                self.error_queue.put(f"图片尺寸不匹配: {filename}\n模板尺寸: {self.template_image.shape[:2]}\n当前图片尺寸: {image.shape[:2]}")
                return

            current_points = []
            for idx, roi in enumerate(self.template_rois):
                center_x, center_y = self.template_centers[idx]
                img_height, img_width = image.shape[:2]
                search_size = min(img_width, img_height) // 10
                
                x1 = max(0, center_x - search_size)
                y1 = max(0, center_y - search_size)
                x2 = min(img_width, center_x + search_size)
                y2 = min(img_height, center_y + search_size)
                
                search_area = image[y1:y2, x1:x2]
                if search_area.size == 0:
                    self.error_queue.put(f"在图片 {filename} 中搜索区域无效")
                    return
                
                result = cv2.matchTemplate(search_area, roi, cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                
                if maxVal < 0.5:
                    self.error_queue.put(f"在图片 {filename} 中标记点匹配度低 ({maxVal:.2f})")
                    return
                
                (startX, startY) = maxLoc
                abs_startX, abs_startY = startX + x1, startY + y1
                centerX, centerY = abs_startX + roi.shape[1] // 2, abs_startY + roi.shape[0] // 2
                current_points.append((centerX, centerY))

            if len(current_points) != 3:
                self.error_queue.put(f"在图片 {filename} 中未能找到所有3个标记点")
                return
            
            template_points = np.float32(self.template_centers)
            current_points = np.float32(current_points)

            M, _ = cv2.estimateAffinePartial2D(current_points, template_points)
            if M is None:
                self.error_queue.put(f"无法为图片 {filename} 计算变换矩阵")
                return

            rows, cols = self.template_image.shape[:2]
            aligned_image = cv2.warpAffine(image, M, (cols, rows), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            
            cv2.imwrite(output_path, aligned_image)
            
            # 新增：如果勾选了自动删除选项，则删除原图
            if self.delete_after_process.get():
                try:
                    os.remove(input_path)
                    self.status_queue.put(f"已删除原图: {filename}")
                except Exception as e:
                    self.error_queue.put(f"删除原图失败: {str(e)}")

        except Exception as e:
            self.error_queue.put(f"处理 {filename} 时出错: {str(e)}")

    def check_queue(self):
        """定期检查队列并更新GUI状态标签"""
        try:
            # 检查状态队列
            message = self.status_queue.get_nowait()
            self.status_label.config(text=message)
        except queue.Empty:
            pass
            
        try:
            # 检查错误队列并弹出对话框
            error_msg = self.error_queue.get_nowait()
            messagebox.showerror("处理错误", error_msg)
        except queue.Empty:
            pass
            
        finally:
            self.root.after(100, self.check_queue)

    def update_status(self, message):
        """将状态消息放入队列"""
        self.status_queue.put(message)


    def select_template(self):
        self.template_path = filedialog.askopenfilename(
            title="选择模板图片",
            filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp"), ("所有文件", "*.*")]
        )
        if self.template_path:
            self.template_label.config(text=os.path.basename(self.template_path))
            self.load_template_image()
            self.reference_points = []
            self.mark_ids = []
            self.template_rois = []
            self.template_centers = []
            self.points_status.config(text="已选择 0/3 个标记区域")
            self.update_status("请选择三个标记区域")
            self.update_delete_buttons()
    
    def load_template_image(self):
        self.template_image = cv2.imread(self.template_path)
        if self.template_image is not None:
            self.current_scale = 0.05
            self.canvas_x = 0
            self.canvas_y = 0
            self.display_image()
        else:
            messagebox.showerror("错误", "无法加载模板图片")

    def display_image(self):
        if self.template_image is None: return
            
        h, w = self.template_image.shape[:2]
        scaled_w = int(w * self.current_scale)
        scaled_h = int(h * self.current_scale)
        
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
        for mark_id in self.mark_ids:
            self.canvas.delete(mark_id[0])
            self.canvas.delete(mark_id[1])
        self.mark_ids = []
        for idx, (x1, y1, x2, y2) in enumerate(self.reference_points):
            sx1, sy1 = x1 * self.current_scale, y1 * self.current_scale
            sx2, sy2 = x2 * self.current_scale, y2 * self.current_scale
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
            x1, y1 = int(min(self.start_x, end_x) * self.scale_factor), int(min(self.start_y, end_y) * self.scale_factor)
            x2, y2 = int(max(self.start_x, end_x) * self.scale_factor), int(max(self.start_y, end_y) * self.scale_factor)
            self.reference_points.append((x1, y1, x2, y2))
            
            roi = self.template_image[y1:y2, x1:x2]
            if roi.size > 0:
                self.template_rois.append(roi)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                self.template_centers.append((center_x, center_y))
            
            self.points_status.config(text=f"已选择 {len(self.reference_points)}/3 个标记区域")
            self.update_delete_buttons()
            self.redraw_marks()
        else:
            self.canvas.delete(self.rect_id)
        self.rect_id = None
        
    def on_mousewheel(self, event):
        
        scale_factor = 1.1
        if event.num == 5 or event.delta == -120:
            self.current_scale /= scale_factor
            self.current_scale = max(0.1, self.current_scale)
        elif event.num == 4 or event.delta == 120:
            self.current_scale *= scale_factor
            self.current_scale = min(10.0, self.current_scale)
        self.display_image()

    def select_input_dir(self):
        input_dir = filedialog.askdirectory(title="选择待处理图片文件夹")
        if input_dir:
            # 新增：检查输入输出路径是否相同
            if self.output_dir and input_dir == self.output_dir:
                self.warning_label.config(text="警告：输入路径和输出路径不能相同！")
                return
            else:
                self.warning_label.config(text="")
            
            self.input_dir = input_dir
            self.input_dir_label.config(text=input_dir)

    def select_output_dir(self):
        output_dir = filedialog.askdirectory(title="选择输出文件夹")
        if output_dir:
            # 新增：检查输入输出路径是否相同
            if self.input_dir and output_dir == self.input_dir:
                self.warning_label.config(text="警告：输出路径和输入路径不能相同！")
                return
            else:
                self.warning_label.config(text="")
            
            self.output_dir = output_dir
            self.output_dir_label.config(text=output_dir)

if __name__ == "__main__":
    root = Tk()
    app = ImageAlignmentApp(root)
    root.mainloop()