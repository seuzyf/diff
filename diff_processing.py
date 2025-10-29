import os
import cv2
import numpy as np
import shutil
import json
import time
import gc
import traceback
import concurrent.futures
from PyQt5.QtCore import QThread, pyqtSignal
from glob import glob
import re # <-- 添加缺失的 import

# --- 新增: API上传所需的库 ---
import requests
import io
import threading # <-- 新增: 用于后台上传
# --- 结束新增 ---


# --- 新增: 转换OpenCV图像到内存字节流的辅助函数 ---
def mat_to_bytes_io(image, ext='.jpg', quality=95):
    """
    Converts an OpenCV image (NumPy array) to an in-memory byte stream (io.BytesIO).
    """
    try:
        # 编码为JPEG格式
        is_success, buffer = cv2.imencode(ext, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not is_success:
            raise Exception("cv2.imencode failed")
        # 创建内存中的字节流
        return io.BytesIO(buffer)
    except Exception as e:
        print(f"Error converting image to bytes: {e}")
        return None
# --- 结束新增 ---


class DetectionThread(QThread):
    progress_signal = pyqtSignal(int, str)  # 进度值, 消息
    finished_signal = pyqtSignal(bool, str)  # 是否成功, 消息
    intermediate_result_signal = pyqtSignal(str, np.ndarray)  # 中间结果名称, 图像数据

    # --- 修改: 重新添加 use_alignment 参数 ---
    def __init__(self, detector, image_paths, output_dir, threshold_conditions,
                 use_ai, combo_method, debug, monitoring_mode=False,
                 use_alignment=False, # <-- 添加缺失的参数
                 parent=None):
        super().__init__(parent)
        self.detector = detector  # This is the main PcbDefectDetector window instance
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.threshold_conditions = threshold_conditions
        self.use_ai = use_ai
        self.combo_method = combo_method
        self.debug = debug
        self._is_running = True

        # --- 新增属性 ---
        self.monitoring_mode = monitoring_mode
        self.monitor_interval = 3 # 监控间隔时间（秒）
        self.processed_files = set() # 用于存储已处理文件的路径

        # --- 新增: Mark点校正相关 ---
        self.use_alignment = use_alignment # <-- 接收参数值
        self.align_template_rois = []
        self.align_template_centers = []
        self.align_template_rects = [] # <-- 新增: Mark点矩形框
        self.template_image_for_alignment = None # <-- 新增: 用于存储模板图
        self.template_gray_for_alignment = None # <-- 新增: 用于存储灰度模板图
        # --- 结束新增 ---

    def run(self):
        try:
            # --- 修改: 如果启用，首先加载Mark点配置和模板图 ---
            if self.use_alignment:
                self.progress_signal.emit(0, "正在加载Mark点校正配置...")
                if not self.load_alignment_config():
                    self.finished_signal.emit(False, "Mark点校正配置加载失败，请检查config/marks.json")
                    return # 直接返回，不继续执行
                
                # --- 新增: 加载模板图以获取尺寸信息 ---
                if not self.detector.template_paths:
                    self.finished_signal.emit(False, "Mark点校正失败: 未加载任何模板图")
                    return
                try:
                    self.template_image_for_alignment = cv2.imread(self.detector.template_paths[0])
                    if self.template_image_for_alignment is None:
                        raise Exception(f"无法加载模板图: {self.detector.template_paths[0]}")
                    # --- 新增: 预先转换为灰度图 ---
                    self.template_gray_for_alignment = cv2.cvtColor(self.template_image_for_alignment, cv2.COLOR_BGR2GRAY)
                    
                except Exception as e:
                    self.finished_signal.emit(False, f"Mark点校正失败: {e}")
                    return
                # --- 结束新增 ---
                
                self.progress_signal.emit(0, "Mark点配置加载成功。")
            # --- 结束修改 ---

            # 1. 预处理模板图（所有模式都需要）
            self.preprocess_templates()
            if not self._is_running: return

            processed_count = 0 # 记录成功处理的数量
            if self.monitoring_mode:
                # --- 监控模式 ---
                self.progress_signal.emit(0, "监控模式启动... 正在处理初始文件...")
                # 立即扫描并处理一次现有文件
                processed_count += self._scan_and_process() # 获取本次扫描处理的数量

                while self._is_running:
                    self.progress_signal.emit(0, f"正在监控... {len(self.processed_files)}个文件已处理")
                    # 使用循环休眠
                    for _ in range(self.monitor_interval * 2):
                        if not self._is_running: break
                        time.sleep(0.5)
                    if not self._is_running: break
                    processed_count += self._scan_and_process() # 累加处理数量

                self.finished_signal.emit(True, "监控已停止") # 监控模式结束不报告总数

            else:
                # --- 单次运行模式 ---
                total = len(self.image_paths)
                for idx, image_path in enumerate(self.image_paths):
                    if not self._is_running:
                        self.progress_signal.emit(idx, "检测已终止")
                        return
                    self.progress_signal.emit(idx + 1, f"处理中: {idx+1}/{total} - {os.path.basename(image_path)}")
                    # 调用处理逻辑，并检查是否成功
                    success = self._process_single_image(image_path)
                    if success:
                        # self.processed_files.add(image_path) # 不再需要，_process_single_image内部处理
                        processed_count += 1
                    # 不论成功失败，进度条都应该前进
                # 完成信号使用成功处理的数量
                self.finished_signal.emit(True, f"完成! 已处理 {processed_count} 张图片")

        except Exception as e:
            traceback.print_exc()
            self.finished_signal.emit(False, f"检测线程出错: {str(e)}")

    # --- 新增: 加载Mark点配置的方法 ---
    def load_alignment_config(self):
        try:
            config_dir = "config"
            json_path = os.path.join(config_dir, "marks.json")
            if not os.path.exists(json_path):
                 self.progress_signal.emit(0, f"错误: 未找到Mark配置文件: {json_path}")
                 return False

            with open(json_path, 'r') as f:
                config = json.load(f)

            self.align_template_centers = config.get('centers')
            self.align_template_rects = config.get('rects') # <-- 新增: 加载矩形框
            
            if not self.align_template_centers or len(self.align_template_centers) != 3:
                raise ValueError("marks.json中'centers'配置无效或数量不为3")
            
            # --- 新增检查 ---
            if not self.align_template_rects or len(self.align_template_rects) != 3:
                raise ValueError("marks.json中'rects'配置无效或数量不为3 (请使用最新mark.py重新保存)")
            # --- 结束新增 ---


            self.align_template_rois = []
            for i in range(1, 4):
                roi_path = os.path.join(config_dir, f"mark_roi_{i}.png")
                roi = cv2.imread(roi_path)
                if roi is None:
                    raise FileNotFoundError(f"未找到Mark ROI图片: {roi_path}")
                self.align_template_rois.append(roi)

            if len(self.align_template_rois) != 3:
                raise ValueError("Mark ROI图片加载不完整 (必须是3个)")

            return True
        except Exception as e:
            self.progress_signal.emit(0, f"加载Mark配置失败: {e}")
            traceback.print_exc()
            return False
    # --- 结束新增 ---

    # --- 替换: 图像校正方法 (移除 ECC, 只保留 Mark点校正) ---
    def align_image(self, image):
        # 检查 Mark ROIs 和模板图是否已加载
        if not self.align_template_rois or len(self.align_template_rois) != 3:
            self.progress_signal.emit(0, "校正失败: Mark ROIs未加载")
            return None
        if self.template_image_for_alignment is None or self.template_gray_for_alignment is None:
            self.progress_signal.emit(0, "校正失败: 基准模板图(或灰度图)未加载")
            return None

        # 检查图片大小是否与模板一致
        if image.shape != self.template_image_for_alignment.shape:
            self.progress_signal.emit(0, f"校正失败: 图片尺寸不匹配\n模板: {self.template_image_for_alignment.shape[:2]}\n图片: {image.shape[:2]}")
            return None

        # === 第 1 步: Mark点粗校正 ===
        current_points = []
        all_match_values = []
        for idx, roi in enumerate(self.align_template_rois):
            center_x, center_y = self.align_template_centers[idx]
            img_height, img_width = image.shape[:2]
            search_size = min(img_width, img_height) // 10 
            
            x1 = max(0, int(center_x - search_size))
            y1 = max(0, int(center_y - search_size))
            x2 = min(img_width, int(center_x + search_size))
            y2 = min(img_height, int(center_y + search_size))
            
            search_area = image[y1:y2, x1:x2]

            if search_area.size == 0 or (roi.shape[0] > search_area.shape[0] or roi.shape[1] > search_area.shape[1]):
                self.progress_signal.emit(0, f"校正失败(Pass 1): Mark {idx+1} 搜索区域无效")
                return None
                
            result = cv2.matchTemplate(search_area, roi, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            all_match_values.append(maxVal)
            
            if maxVal < 0.5:
                self.progress_signal.emit(0, f"校正失败(Pass 1): Mark {idx+1} 匹配度过低 ({maxVal:.2f} < 0.5)")
                return None
            
            (startX, startY) = maxLoc
            abs_startX, abs_startY = startX + x1, startY + y1
            centerX, centerY = abs_startX + roi.shape[1] // 2, abs_startY + roi.shape[0] // 2
            current_points.append((centerX, centerY))

        print(f"Pass 1 (Mark点) 匹配度: {[f'{v:.2f}' for v in all_match_values]}")

        if len(current_points) != 3:
            self.progress_signal.emit(0, "校正失败(Pass 1): 未能找到所有3个Mark点")
            return None
            
        template_points = np.float32(self.align_template_centers)
        current_points = np.float32(current_points)

        M_initial, _ = cv2.estimateAffinePartial2D(current_points, template_points)
        if M_initial is None:
            self.progress_signal.emit(0, "校正失败(Pass 1): 无法计算变换矩阵")
            return None

        rows, cols = self.template_image_for_alignment.shape[:2] 
        aligned_image_pass1 = cv2.warpAffine(image, M_initial, (cols, rows), 
                                             flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, 
                                             borderValue=(0, 0, 0))
        
        if self.debug:
            cv2.imwrite(os.path.join(self.output_dir, "校正图_Pass1_Mark点.png"), aligned_image_pass1)

        # === 第 2 步: ECC 精细校正 (已根据用户请求移除) ===
        
        return aligned_image_pass1 # <-- 直接返回Pass 1的结果
    # --- 结束替换 ---

    # --- [新增] API上传的后台辅助方法 ---
    def _upload_image_in_background(self, image_to_upload, basename):
        """
        在单独的线程中运行，用于上传图像，不阻塞主检测线程。
        """
        try:
            # 此函数在单独的线程中运行。
            print(f"后台上传中: {basename}...")
            
            image_bytes_io = mat_to_bytes_io(image_to_upload) 
            
            if image_bytes_io:
                url = "http://fod-train.hissit.huawei.com:8083/api/ml-predict/detect-online/detect/"
                payload = {}
                # 使用原始basename (无扩展名) + .jpg 作为文件名
                files = [
                    ('image', (f'{basename}.jpg', image_bytes_io, 'image/jpeg'))
                ]
                headers = {}

                # 设置10秒超时
                response = requests.request("POST", url, headers=headers, data=payload, files=files, timeout=10)
                
                if response.status_code == 200:
                    print(f"后台上传成功: {basename}. 响应: {response.text[:100]}...")
                else:
                    print(f"后台上传失败: {basename}. 状态码: {response.status_code}, 响应: {response.text}")
            else:
                print(f"后台上传: 无法转换 {basename} 为字节, 跳过.")

        except requests.exceptions.RequestException as e_req:
            # 处理网络相关的错误 (如连接超时, DNS错误)
            print(f"后台上传 {basename} 时出错 (RequestException): {e_req}")
        except Exception as e_api:
            # 处理其他所有错误
            print(f"后台上传 {basename} 时发生未知错误: {e_api}")
            traceback.print_exc()
    # --- [结束新增] ---


    def _scan_and_process(self):
        """扫描监控目录，处理新文件，返回本次成功处理的文件数"""
        processed_count_in_scan = 0 # 初始化计数器
        try:
            if not self.detector.folder or not os.path.exists(self.detector.folder):
                self.progress_signal.emit(0, "错误: 监控文件夹不存在")
                time.sleep(self.monitor_interval)
                return processed_count_in_scan # 返回0

            current_image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                current_image_paths.extend(glob(os.path.join(self.detector.folder, ext)))

            current_files_set = set(current_image_paths)
            new_files = sorted(list(current_files_set - self.processed_files))

            if new_files:
                self.progress_signal.emit(0, f"检测到 {len(new_files)} 个新文件，正在处理...")
                for file_path in new_files:
                    if not self._is_running: return processed_count_in_scan # 中断时返回当前计数
                    self.progress_signal.emit(0, f"处理新文件: {os.path.basename(file_path)}")
                    
                    # --- 改进的文件锁定检查 ---
                    if not self.is_file_ready(file_path):
                        self.progress_signal.emit(0, f"文件 {os.path.basename(file_path)} 尚在写入中，跳过此次")
                        continue # 跳过此文件，下次扫描再试
                    # --- 结束 ---

                    success = self._process_single_image(file_path)
                    if success:
                        # self.processed_files.add(file_path) # 移到 _process_single_image 内部
                        processed_count_in_scan += 1
                        if self.detector.cb_auto_delete.isChecked():
                            try:
                                os.remove(file_path)
                                self.progress_signal.emit(0, f"已删除原图: {os.path.basename(file_path)}")
                            except Exception as e:
                                self.progress_signal.emit(0, f"删除失败: {e}")
                    else:
                         print(f"处理文件失败或跳过: {os.path.basename(file_path)}")
                if processed_count_in_scan > 0: print(f"本次扫描处理了 {processed_count_in_scan} 个新文件。")

        except Exception as e:
            self.progress_signal.emit(0, f"监控扫描出错: {str(e)}")
            traceback.print_exc()
            time.sleep(self.monitor_interval)
        return processed_count_in_scan # 返回本次处理的数量

    def is_file_ready(self, file_path):
        """检查文件是否已写入完毕 (通过检查文件大小是否变化)"""
        try:
            if not os.path.exists(file_path):
                return False
                
            size_now = os.path.getsize(file_path)
            time.sleep(0.1) # 等待 100ms
            size_later = os.path.getsize(file_path)
            
            if size_now == size_later:
                # 尝试以写入模式打开，检查是否被锁定
                try:
                    with open(file_path, 'a'):
                        pass
                except IOError:
                    return False # 文件被锁定
                return True # 大小未变且未锁定
            else:
                return False # 文件大小仍在变化
        except Exception:
            return False # 访问文件时出错

    def stop(self):
        self._is_running = False

    def preprocess_templates(self):
        """预处理模板图"""
        temp_dir = os.path.join(self.output_dir, "模板中间结果")
        manifest_path = os.path.join(temp_dir, "cache_manifest.json")

        def normalize_data_structure(data):
            if isinstance(data, (list, tuple)): return [normalize_data_structure(item) for item in data]
            elif isinstance(data, dict): return {k: normalize_data_structure(v) for k, v in data.items()}
            else: return data

        current_state = { "template_paths": sorted(self.detector.template_paths), "threshold_conditions": self.threshold_conditions }

        cache_valid = False
        try:
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f: saved_state = json.load(f)
                if normalize_data_structure(saved_state) == normalize_data_structure(current_state):
                    required_files = []
                    if self.threshold_conditions.get("二值化阈值"): required_files.append("所有模板_二值化_最终合并结果.png")
                    if self.threshold_conditions.get("RGB二值化"): required_files.append("所有模板_RGB二值化_最终合并结果.png")
                    if self.threshold_conditions.get("灰度差阈值"): required_files.extend(["所有模板_最小灰度值.png", "所有模板_最大灰度值.png"])
                    if all(os.path.exists(os.path.join(temp_dir, f)) for f in required_files): cache_valid = True
        except Exception as e: print(f"读取缓存清单失败: {e}")

        if cache_valid:
            self.progress_signal.emit(0, "缓存命中，正在加载预处理结果...")
            if self.threshold_conditions.get("灰度差阈值"):
                min_p, max_p = os.path.join(temp_dir, "所有模板_最小灰度值.png"), os.path.join(temp_dir, "所有模板_最大灰度值.png")
                if os.path.exists(min_p): self.template_min = cv2.imread(min_p, cv2.IMREAD_GRAYSCALE)
                if os.path.exists(max_p): self.template_max = cv2.imread(max_p, cv2.IMREAD_GRAYSCALE)
            if self.threshold_conditions.get("二值化阈值"):
                bin_p = os.path.join(temp_dir, "所有模板_二值化_最终合并结果.png")
                if os.path.exists(bin_p): self.combined_binary_template = cv2.imread(bin_p, cv2.IMREAD_GRAYSCALE)
            if self.threshold_conditions.get("RGB二值化"):
                rgb_p = os.path.join(temp_dir, "所有模板_RGB二值化_最终合并结果.png")
                if os.path.exists(rgb_p): self.combined_rgb_binary_template = cv2.imread(rgb_p, cv2.IMREAD_GRAYSCALE)
            return

        self.progress_signal.emit(0, "缓存无效，正在重新处理模板...")
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        self.combined_binary_template, self.combined_rgb_binary_template = None, None
        self.template_min, self.template_max = None, None
        
        # --- 确保模板图至少有一个 ---
        if not self.detector.template_paths:
            self.progress_signal.emit(0, "错误: 模板路径列表为空，无法预处理。")
            return
            
        first_template_shape = None

        for i, template_path in enumerate(self.detector.template_paths):
            template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_gray is None: 
                self.progress_signal.emit(0, f"警告: 无法加载模板 {template_path}")
                continue
            template_rgb = cv2.imread(template_path)
            if template_rgb is None: 
                self.progress_signal.emit(0, f"警告: 无法加载模板 {template_path}")
                continue
                
            # --- 检查所有模板尺寸是否一致 ---
            if first_template_shape is None:
                first_template_shape = template_gray.shape
            elif template_gray.shape != first_template_shape:
                self.progress_signal.emit(0, f"错误: 模板 {template_path} 尺寸 {template_gray.shape} 与第一张模板 {first_template_shape} 不一致。")
                self.progress_signal.emit(0, "所有模板必须尺寸相同。")
                # 清理已处理的变量，防止使用不完整的数据
                self.template_min, self.template_max = None, None
                self.combined_binary_template, self.combined_rgb_binary_template = None, None
                return
            # --- 结束检查 ---

            if self.threshold_conditions.get("灰度差阈值"):
                if self.template_min is None: self.template_min, self.template_max = template_gray.copy(), template_gray.copy()
                else: self.template_min, self.template_max = np.minimum(self.template_min, template_gray), np.maximum(self.template_max, template_gray)
            if self.threshold_conditions.get("二值化阈值"):
                mask = np.zeros_like(template_gray)
                for low, high in self.threshold_conditions["二值化阈值"]: mask = cv2.bitwise_or(mask, cv2.inRange(template_gray, low, high))
                if self.combined_binary_template is None: self.combined_binary_template = mask
                else: self.combined_binary_template = cv2.bitwise_or(self.combined_binary_template, mask)
            if self.threshold_conditions.get("RGB二值化"):
                mask = np.zeros(template_gray.shape[:2], dtype=np.uint8)
                for r0, r1, g0, g1, b0, b1 in self.threshold_conditions["RGB二值化"]:
                    low, high = np.array([b0, g0, r0]), np.array([b1, g1, r1])
                    mask = cv2.bitwise_or(mask, cv2.inRange(template_rgb, low, high))
                if self.combined_rgb_binary_template is None: self.combined_rgb_binary_template = mask
                else: self.combined_rgb_binary_template = cv2.bitwise_or(self.combined_rgb_binary_template, mask)

        if self.threshold_conditions.get("灰度差阈值") and self.template_min is not None: cv2.imwrite(os.path.join(temp_dir, "所有模板_最小灰度值.png"), self.template_min)
        if self.threshold_conditions.get("灰度差阈值") and self.template_max is not None: cv2.imwrite(os.path.join(temp_dir, "所有模板_最大灰度值.png"), self.template_max)
        if self.threshold_conditions.get("二值化阈值") and self.combined_binary_template is not None: cv2.imwrite(os.path.join(temp_dir, "所有模板_二值化_最终合并结果.png"), self.combined_binary_template)
        if self.threshold_conditions.get("RGB二值化") and self.combined_rgb_binary_template is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.combined_rgb_binary_template = cv2.morphologyEx(self.combined_rgb_binary_template, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(os.path.join(temp_dir, "所有模板_RGB二值化_最终合并结果.png"), self.combined_rgb_binary_template)

        with open(manifest_path, 'w', encoding='utf-8') as f: json.dump(current_state, f, ensure_ascii=False, indent=4)
        self.progress_signal.emit(0, "模板处理完成，缓存已更新。")


    def _process_single_image(self, image_path):
        """检测单张图片，返回 True/False"""
        basename = os.path.splitext(os.path.basename(image_path))[0]
        # 使用 re.sub 清理文件名中的非法字符
        safe_basename = re.sub(r'[\\/*?:"<>|]', '_', basename)
        # --- 修改: 确保 output_dir 基于 self.output_dir ---
        current_image_output_dir = os.path.join(self.output_dir, f"结果_{safe_basename}")

        # 使用 try...finally 确保清理
        image, image_gray, combined_mask, result_img, filtered_diff_img, original_image_for_cropping = [None] * 6
        try:
            # --- 修改: 使用 current_image_output_dir ---
            if os.path.exists(current_image_output_dir): shutil.rmtree(current_image_output_dir)
            os.makedirs(current_image_output_dir, exist_ok=True)

            try:
                # 复制原图
                shutil.copy2(image_path, os.path.join(current_image_output_dir, "tested_image.jpg"))
            except IOError as e:
                self.progress_signal.emit(0, f"无法复制文件 (IOError): {e}")
                time.sleep(0.5)
                try: shutil.copy2(image_path, os.path.join(current_image_output_dir, "tested_image.jpg"))
                except Exception as e2: self.progress_signal.emit(0, f"复制文件最终失败: {e2}"); return False
            except Exception as e:
                 self.progress_signal.emit(0, f"复制文件时发生未知错误: {e}"); return False

            image = cv2.imread(image_path)
            if image is None: self.progress_signal.emit(0, f"错误: 无法加载检测图像 {image_path}"); return False

            # --- 图像校正 ---
            if self.use_alignment:
                self.progress_signal.emit(0, f"正在校正: {basename}")
                aligned_img = self.align_image(image)
                if aligned_img is None:
                    self.progress_signal.emit(0, f"校正失败，跳过: {basename}")
                    return False # 校正失败算处理失败
                image = aligned_img # 使用校正后的图像
                if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "校正后的图像.png"), image)
                
                # --- 新增: 覆盖 tested_image.jpg 为校正后的图像 ---
                # 确保缺陷查看器加载的是校正后的图像
                try:
                    cv2.imwrite(os.path.join(current_image_output_dir, "tested_image.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                except Exception as e_save:
                    self.progress_signal.emit(0, f"警告: 覆盖 tested_image.jpg 失败: {e_save}")
                # --- 结束新增 ---
                
            # --- 结束校正 ---

            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "灰度图.png"), image_gray)

            binary_masks = []
            t_start = time.time()

            # --- 阈值处理 (添加了尺寸检查) ---
            # 二值化
            if self.threshold_conditions.get("二值化阈值") and hasattr(self, 'combined_binary_template') and self.combined_binary_template is not None:
                if image_gray.shape == self.combined_binary_template.shape:
                    mask = np.zeros_like(image_gray)
                    for low, high in self.threshold_conditions["二值化阈值"]: mask = cv2.bitwise_or(mask, cv2.inRange(image_gray, low, high))
                    if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "二值化_合并结果.png"), mask)
                    diff = cv2.absdiff(mask, self.combined_binary_template)
                    diff_bin = cv2.bitwise_and(diff, mask)
                    if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "二值化差异图.png"), diff_bin)
                    binary_masks.append(diff_bin)
                else: print(f"警告: 二值化 - 尺寸不匹配 ({image_gray.shape} vs {self.combined_binary_template.shape}), 跳过。")
            # RGB二值化
            if self.threshold_conditions.get("RGB二值化") and hasattr(self, 'combined_rgb_binary_template') and self.combined_rgb_binary_template is not None:
                if image.shape[:2] == self.combined_rgb_binary_template.shape: # 比较 H, W
                    mask = np.zeros_like(image_gray)
                    for r0, r1, g0, g1, b0, b1 in self.threshold_conditions["RGB二值化"]:
                        low, high = np.array([b0, g0, r0]), np.array([b1, g1, r1])
                        mask = cv2.bitwise_or(mask, cv2.inRange(image, low, high))
                    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
                    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
                    if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "RGB二值化_合并结果.png"), mask)
                    diff = cv2.absdiff(mask, self.combined_rgb_binary_template)
                    diff_rgb = cv2.bitwise_and(diff, mask)
                    if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "RGB差异图.png"), diff_rgb)
                    binary_masks.append(diff_rgb)
                else: print(f"警告: RGB二值化 - 尺寸不匹配 ({image.shape[:2]} vs {self.combined_rgb_binary_template.shape}), 跳过。")
            # 灰度差
            if self.threshold_conditions.get("灰度差阈值") and hasattr(self, 'template_min') and hasattr(self, 'template_max'):
                if image_gray.shape == self.template_min.shape:
                    thresh = min(self.threshold_conditions["灰度差阈值"])
                    if len(self.detector.template_paths) == 1:
                        diff = cv2.absdiff(image_gray, self.template_min)
                        _, diff_gray = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
                    else:
                        low = np.clip(self.template_min.astype(np.int16) - thresh, 0, 255).astype(np.uint8)
                        high = np.clip(self.template_max.astype(np.int16) + thresh, 0, 255).astype(np.uint8)
                        if self.debug:
                            cv2.imwrite(os.path.join(current_image_output_dir, "灰度差_下限图.png"), low)
                            cv2.imwrite(os.path.join(current_image_output_dir, "灰度差_上限图.png"), high)
                        diff_gray = cv2.bitwise_or(cv2.compare(image_gray, low, cv2.CMP_LT), cv2.compare(image_gray, high, cv2.CMP_GT))
                    if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "灰度差_差异图.png"), diff_gray)
                    binary_masks.append(diff_gray)
                else: print(f"警告: 灰度差 - 尺寸不匹配 ({image_gray.shape} vs {self.template_min.shape}), 跳过。")

            # --- 组合掩码 ---
            if not binary_masks:
                self.progress_signal.emit(0, f"警告: {basename} 未生成有效掩码。")
                combined_mask = np.zeros_like(image_gray) # 全黑掩码
            else:
                combined_mask = binary_masks[0]
                op = cv2.bitwise_and if self.combo_method == "and" else cv2.bitwise_or
                for i in range(1, len(binary_masks)): combined_mask = op(combined_mask, binary_masks[i])
            
            # --- 新增: 应用Mark点最大外接矩形 ---
            if self.use_alignment and self.align_template_rects:
                try:
                    all_rects = np.array(self.align_template_rects)
                    # (x1, y1, x2, y2)
                    min_x = np.min(all_rects[:, 0])
                    min_y = np.min(all_rects[:, 1])
                    max_x = np.max(all_rects[:, 2])
                    max_y = np.max(all_rects[:, 3])
                    
                    # 增加一点 padding
                    pad = 5 
                    h, w = combined_mask.shape[:2]
                    x1 = max(0, int(min_x - pad))
                    y1 = max(0, int(min_y - pad))
                    x2 = min(w, int(max_x + pad))
                    y2 = min(h, int(max_y + pad))
                    
                    mark_bbox_mask = np.zeros_like(combined_mask)
                    cv2.rectangle(mark_bbox_mask, (x1, y1), (x2, y2), 255, -1) # 255 = 白色, -1 = 填充
                    
                    combined_mask = cv2.bitwise_and(combined_mask, mark_bbox_mask)
                    if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "差异图(Mark点BBox).png"), combined_mask)
                    
                except Exception as e_bbox:
                    print(f"应用Mark BBox失败: {e_bbox}")
            # --- 结束新增 ---

            # 总是保存差异总图 (Debug模式)
            if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "差异总图.png"), combined_mask)

            # --- 应用屏蔽罩 ---
            mask_path = os.path.join("config", "mask.png")
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_img is not None:
                    if mask_img.shape != combined_mask.shape:
                        print(f"警告: 屏蔽罩尺寸({mask_img.shape})与图像({combined_mask.shape})不匹配,将缩放.")
                        mask_img = cv2.resize(mask_img, (combined_mask.shape[1], combined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                    combined_mask = cv2.bitwise_and(combined_mask, mask_img)
                    if self.debug: cv2.imwrite(os.path.join(current_image_output_dir, "差异总图(带屏蔽).png"), combined_mask)

            t1_preprocessing_done = time.time()

            # --- 连通域分析及过滤 ---
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
            result_img = image.copy() # 用于绘制最终结果
            filtered_diff_img = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR) # 用于绘制过滤后的差异
            defect_count = 0
            defect_regions = [] # (x, y, w, h, area, ratio, cx, cy)

            if num_labels > 1:
                # ... (过滤逻辑基本不变，但增加了空检查) ...
                areas = stats[1:, cv2.CC_STAT_AREA]
                widths = stats[1:, cv2.CC_STAT_WIDTH]
                heights = stats[1:, cv2.CC_STAT_HEIGHT]

                area_cond = self.threshold_conditions.get("面积阈值", [])
                area_mask = np.ones_like(areas, dtype=bool) if not area_cond else np.zeros_like(areas, dtype=bool)
                for min_a, max_a in area_cond: area_mask |= (areas >= min_a) & (areas <= max_a)

                ratio_cond = self.threshold_conditions.get("比例阈值", [])
                ratio_mask = np.ones_like(areas, dtype=bool) if not ratio_cond else np.zeros_like(areas, dtype=bool)
                if ratio_cond:
                    min_wh = np.minimum(widths, heights) + 1e-6
                    ratios = np.maximum(widths, heights) / min_wh
                    for min_r, max_r in ratio_cond: ratio_mask |= (ratios >= min_r) & (ratios <= max_r)

                valid_indices = np.where(area_mask & ratio_mask)[0] + 1

                for i in valid_indices:
                    if not self._is_running: return False
                    x, y, w, h = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT+1]

                    # 亮度过滤
                    bright_cond = self.threshold_conditions.get("区域平均亮度阈值", [])
                    brightness_valid = True
                    if bright_cond:
                        region_mask = (labels == i)
                        region_pixels = image_gray[region_mask]
                        avg_bright = np.mean(region_pixels) if len(region_pixels) > 0 else 0
                        brightness_valid = any(min_b <= avg_bright <= max_b for min_b, max_b in bright_cond)
                    if not brightness_valid: continue

                    # RGB过滤
                    rgb_cond = self.threshold_conditions.get("RGB值阈值", [])
                    rgb_valid = True
                    if rgb_cond:
                        mask_rgb = (labels == i)
                        y_idx, x_idx = np.where(mask_rgb)
                        if len(y_idx) == 0: rgb_valid = False
                        else:
                            samp_ratio, max_samp = 0.1, 100
                            samp_count = max(1, min(max_samp, int(len(y_idx) * samp_ratio)))
                            rng = np.random.default_rng(42)
                            samp_indices = rng.choice(len(y_idx), samp_count, replace=False)
                            samp_pixels = image[y_idx[samp_indices], x_idx[samp_indices]]
                            b_min_s, b_max_s = np.min(samp_pixels[:, 0]), np.max(samp_pixels[:, 0])
                            g_min_s, g_max_s = np.min(samp_pixels[:, 1]), np.max(samp_pixels[:, 1])
                            r_min_s, r_max_s = np.min(samp_pixels[:, 2]), np.max(samp_pixels[:, 2])
                            rgb_valid = False
                            for r0, r1, g0, g1, b0, b1, ratio_t in rgb_cond:
                                overlap = (r_max_s>=r0 and r_min_s<=r1 and g_max_s>=g0 and g_min_s<=g1 and b_max_s>=b0 and b_min_s<=b1)
                                if overlap:
                                    in_rng = ((samp_pixels[:,2]>=r0)&(samp_pixels[:,2]<=r1)&
                                              (samp_pixels[:,1]>=g0)&(samp_pixels[:,1]<=g1)&
                                              (samp_pixels[:,0]>=b0)&(samp_pixels[:,0]<=b1))
                                    if np.mean(in_rng) >= ratio_t: rgb_valid = True; break
                    if not rgb_valid: continue

                    # AI过滤
                    ai_defect = True
                    if self.use_ai and self.detector.yolo_model:
                        pad = 10
                        roi = image[max(0, y-pad):min(image.shape[0], y+h+pad), max(0, x-pad):min(image.shape[1], x+w+pad)]
                        if roi.size > 0: ai_defect = len(self.detector.yolo_model(roi, verbose=False)[0].boxes) > 0
                    if not ai_defect: continue

                    # 通过所有检查
                    defect_count += 1
                    cx, cy = int(centroids[i][0]), int(centroids[i][1])
                    defect_regions.append((x, y, w, h, stats[i, cv2.CC_STAT_AREA], max(w, h) / (min(w, h) + 1e-6), cx, cy))

                    if self.debug:
                        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        cv2.putText(result_img, str(defect_count), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                        cv2.rectangle(filtered_diff_img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                        cv2.putText(filtered_diff_img, str(defect_count), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # 保存 Debug 图片
            if self.debug:
                cv2.imwrite(os.path.join(current_image_output_dir, "过滤后的差异图.png"), filtered_diff_img)
                cv2.imwrite(os.path.join(current_image_output_dir, "final_result.png"), result_img)

            t2_filtering_done = time.time()

            # --- 缺陷裁剪与保存 ---
            if defect_regions:
                # --- 修正: 缺陷裁剪应该使用校正后的图像 (image) ---
                # 因为 (x,y,w,h,cx,cy) 都是在校正后的图像上计算的
                original_image_for_cropping = image
                # --- 结束 ---

                t3_reload_done = time.time()
                defects_dir = os.path.join(current_image_output_dir, "defects")
                os.makedirs(defects_dir, exist_ok=True)

                tasks = []
                for i, (x, y, w, h, area, ratio, cx, cy) in enumerate(defect_regions):
                    crop_size = max(120, w, h)
                    x1 = max(0, int(cx - crop_size // 2))
                    y1 = max(0, int(cy - crop_size // 2))
                    x2 = min(original_image_for_cropping.shape[1], x1 + crop_size)
                    y2 = min(original_image_for_cropping.shape[0], y1 + crop_size)
                    x1 = max(0, x2 - crop_size)
                    y1 = max(0, y2 - crop_size)
                    tasks.append((x1, y1, x2, y2, i + 1, x, y, w, h, cx, cy))

                def process_defect(task, img, save_dir):
                    x1, y1, x2, y2, defect_id, x_orig, y_orig, w_orig, h_orig, cx, cy = task
                    roi = img[y1:y2, x1:x2].copy()
                    rel_x, rel_y = max(0, x_orig - x1), max(0, y_orig - y1)
                    rel_x2, rel_y2 = min(roi.shape[1], rel_x + w_orig), min(roi.shape[0], rel_y + h_orig)
                    if rel_x2 > rel_x and rel_y2 > rel_y:
                        cv2.rectangle(roi, (rel_x, rel_y), (rel_x2, rel_y2), (0, 0, 255), 1)
                    filename = f"{defect_id}_center({cx},{cy}).png"
                    # --- 添加 .png 后缀检查 ---
                    if not filename.lower().endswith(".png"): filename += ".png"
                    cv2.imwrite(os.path.join(save_dir, filename), roi)
                    return defect_id

                num_threads = min(8, max(2, len(tasks) // 10 + 1))
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    futures = [executor.submit(process_defect, task, original_image_for_cropping, defects_dir) for task in tasks]
                    # --- 修改: 简化错误处理 ---
                    concurrent.futures.wait(futures) # 等待所有任务完成
                    # 可以在这里检查 future.exception() 如果需要更详细的错误

                t4_crop_save_done = time.time()
                print(f"\n--- 性能分析 ({basename}) ---")
                print(f"预处理与掩码: {t1_preprocessing_done - t_start:.4f} s")
                print(f"连通域与过滤: {t2_filtering_done - t1_preprocessing_done:.4f} s")
                print(f"缺陷裁剪准备: {t3_reload_done - t2_filtering_done:.4f} s")
                print(f"缺陷裁剪保存: {t4_crop_save_done - t3_reload_done:.4f} s")
                print(f"--- 总计: {t4_crop_save_done - t_start:.4f} s ---")

            else: # 无缺陷
                 t4_crop_save_done = t2_filtering_done
                 print(f"\n--- 性能分析 ({basename}) ---")
                 print(f"预处理与掩码: {t1_preprocessing_done - t_start:.4f} s")
                 print(f"连通域与过滤: {t2_filtering_done - t1_preprocessing_done:.4f} s")
                 print(f"(无缺陷)")
                 print(f"--- 总计: {t4_crop_save_done - t_start:.4f} s ---")


            # --- [修改] 发送图像到外部API (在后台线程中) ---
            # 'image' 变量保存着已加载并（可能）校正后的图像
            
            # 传入 image.copy() 以避免主线程垃圾回收导致的问题
            upload_thread = threading.Thread(
                target=self._upload_image_in_background,
                args=(image.copy(), basename) # 必须使用 .copy()
            )
            upload_thread.daemon = True # 设为守护线程，主程序退出时它也会退出
            upload_thread.start()
            # --- [结束修改] ---
            

            # --- 成功处理后标记 ---
            self.processed_files.add(image_path)
            return True # 处理成功

        except Exception as process_error:
            self.progress_signal.emit(0, f"处理 {basename} 时出错: {process_error}")
            traceback.print_exc()
            # 尝试记录错误日志
            try:
                if 'current_image_output_dir' in locals() and os.path.exists(current_image_output_dir):
                    error_file = os.path.join(current_image_output_dir, "error.log")
                    with open(error_file, 'w', encoding='utf-8') as f: # 指定编码
                        f.write(f"处理文件 {image_path} 时发生错误:\n")
                        f.write(str(process_error) + "\n\n")
                        f.write(traceback.format_exc())
            except Exception as log_e: print(f"写入错误日志失败: {log_e}")
            return False # 处理失败
        finally:
             # --- 清理内存 ---
             image = None; image_gray = None; combined_mask = None; result_img = None
             filtered_diff_img = None; original_image_for_cropping = None
             gc.collect()