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
from glob import glob # 导入glob

class DetectionThread(QThread):
    progress_signal = pyqtSignal(int, str)  # 进度值, 消息
    finished_signal = pyqtSignal(bool, str)  # 是否成功, 消息
    intermediate_result_signal = pyqtSignal(str, np.ndarray)  # 中间结果名称, 图像数据
    
    def __init__(self, detector, image_paths, output_dir, threshold_conditions, 
                 use_ai, combo_method, debug, monitoring_mode=False, parent=None):
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
        # --- 结束新增 ---
    
    def run(self):
        try:
            # 1. 预处理模板图（所有模式都需要）
            self.preprocess_templates()
            if not self._is_running: return

            if self.monitoring_mode:
                # --- 监控模式 ---
                self.progress_signal.emit(0, "监控模式启动... 正在处理初始文件...")
                
                # 立即扫描并处理一次现有文件
                self._scan_and_process()

                while self._is_running:
                    self.progress_signal.emit(0, f"正在监控... {len(self.processed_files)}个文件已处理")
                    
                    # 使用循环休眠，以便快速响应停止信号
                    for _ in range(self.monitor_interval * 2):
                        if not self._is_running:
                            break
                        time.sleep(0.5)
                    
                    if not self._is_running:
                        break
                        
                    self._scan_and_process()
                
                self.finished_signal.emit(True, "监控已停止")

            else:
                # --- 单次运行模式 ---
                total = len(self.image_paths)
                for idx, image_path in enumerate(self.image_paths):
                    if not self._is_running:
                        self.progress_signal.emit(idx, "检测已终止")
                        return
                        
                    self.progress_signal.emit(idx + 1, f"处理中: {idx+1}/{total} - {os.path.basename(image_path)}")
                    
                    # 调用处理逻辑
                    self._process_single_image(image_path)
                    self.processed_files.add(image_path) # 标记为已处理
                
                self.finished_signal.emit(True, f"完成! 已处理{total}张图片")

        except Exception as e:
            traceback.print_exc()
            self.finished_signal.emit(False, f"检测失败: {str(e)}")

    def _scan_and_process(self):
        """扫描监控目录，处理新文件"""
        try:
            if not self.detector.folder or not os.path.exists(self.detector.folder):
                self.progress_signal.emit(0, "错误: 监控文件夹不存在")
                time.sleep(self.monitor_interval) # 等待文件夹出现
                return

            # 查找所有图片
            current_image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                current_image_paths.extend(glob(os.path.join(self.detector.folder, ext)))
            
            current_files_set = set(current_image_paths)
            new_files = sorted(list(current_files_set - self.processed_files)) # 找出新文件并排序

            if new_files:
                self.progress_signal.emit(0, f"检测到 {len(new_files)} 个新文件，正在处理...")
                for file_path in new_files:
                    if not self._is_running:
                        return # 响应停止信号
                    
                    self.progress_signal.emit(0, f"处理新文件: {os.path.basename(file_path)}")
                    self._process_single_image(file_path)
                    self.processed_files.add(file_path) # 处理完后加入集合
                    
                    # 如果勾选了自动删除，则处理完后删除
                    if self.detector.cb_auto_delete.isChecked():
                        try:
                            os.remove(file_path)
                            self.progress_signal.emit(0, f"已删除原图: {os.path.basename(file_path)}")
                        except Exception as e:
                            self.progress_signal.emit(0, f"删除失败: {e}")

        except Exception as e:
            self.progress_signal.emit(0, f"监控扫描出错: {str(e)}")
            traceback.print_exc()
            time.sleep(self.monitor_interval) # 出错时等待
        
    def stop(self):
        self._is_running = False
    
    def preprocess_templates(self):
        """
        预处理所有模板图，并增加了缓存机制。
        (此函数内容保持不变)
        """
        temp_dir = os.path.join(self.output_dir, "模板中间结果")
        manifest_path = os.path.join(temp_dir, "cache_manifest.json")

        def normalize_data_structure(data):
            if isinstance(data, (list, tuple)):
                return [normalize_data_structure(item) for item in data]
            elif isinstance(data, dict):
                return {k: normalize_data_structure(v) for k, v in data.items()}
            else:
                return data
        
        # 1. 定义当前处理任务的状态指纹
        current_state = {
            "template_paths": sorted(self.detector.template_paths),
            "threshold_conditions": self.threshold_conditions
        }
        
        # 2. 检查缓存是否有效
        cache_valid = False
        try:
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    saved_state = json.load(f)
                normalized_saved = normalize_data_structure(saved_state)
                normalized_current = normalize_data_structure(current_state)
                if normalized_saved == normalized_current:
                    required_files = []
                    if self.threshold_conditions.get("二值化阈值"):
                        required_files.append("所有模板_二值化_最终合并结果.png")
                    if self.threshold_conditions.get("RGB二值化"):
                        required_files.append("所有模板_RGB二值化_最终合并结果.png")
                    if self.threshold_conditions.get("灰度差阈值"):
                        required_files.extend(["所有模板_最小灰度值.png", "所有模板_最大灰度值.png"])
                    
                    all_files_exist = True
                    for filename in required_files:
                        file_path = os.path.join(temp_dir, filename)
                        if not os.path.exists(file_path):
                            all_files_exist = False
                            break
                    
                    if all_files_exist:
                        cache_valid = True
        
        except (json.JSONDecodeError, IOError, TypeError) as e:
            print(f"读取缓存清单失败: {e}")
            cache_valid = False

        # 3. 如果缓存有效则加载
        if cache_valid:
            self.progress_signal.emit(0, "缓存命中，正在加载预处理结果...")
            
            if self.threshold_conditions.get("灰度差阈值"):
                min_path = os.path.join(temp_dir, "所有模板_最小灰度值.png")
                max_path = os.path.join(temp_dir, "所有模板_最大灰度值.png")
                if os.path.exists(min_path):
                    self.template_min = cv2.imread(min_path, cv2.IMREAD_GRAYSCALE)
                if os.path.exists(max_path):
                    self.template_max = cv2.imread(max_path, cv2.IMREAD_GRAYSCALE)
            
            if self.threshold_conditions.get("二值化阈值"):
                bin_path = os.path.join(temp_dir, "所有模板_二值化_最终合并结果.png")
                if os.path.exists(bin_path):
                    self.combined_binary_template = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)
            
            if self.threshold_conditions.get("RGB二值化"):
                rgb_path = os.path.join(temp_dir, "所有模板_RGB二值化_最终合并结果.png")
                if os.path.exists(rgb_path):
                    self.combined_rgb_binary_template = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
            
            return
        
        # 4. 缓存无效或不存在，删除整个缓存目录并重新生成
        self.progress_signal.emit(0, "缓存无效，正在重新处理模板...")
        
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        self.combined_binary_template = None
        self.combined_rgb_binary_template = None
        self.template_min = None
        self.template_max = None
        
        for i, template_path in enumerate(self.detector.template_paths):
            template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_gray is None: continue
            template_rgb = cv2.imread(template_path)
            if template_rgb is None: continue
            
            if self.threshold_conditions.get("灰度差阈值"):
                if self.template_min is None:
                    self.template_min = template_gray.copy()
                    self.template_max = template_gray.copy()
                else:
                    self.template_min = np.minimum(self.template_min, template_gray)
                    self.template_max = np.maximum(self.template_max, template_gray)
            
            if self.threshold_conditions.get("二值化阈值"):
                binary_mask = np.zeros_like(template_gray)
                for low, high in self.threshold_conditions["二值化阈值"]:
                    template_mask = cv2.inRange(template_gray, low, high)
                    binary_mask = cv2.bitwise_or(binary_mask, template_mask)
                if self.combined_binary_template is None:
                    self.combined_binary_template = binary_mask
                else:
                    self.combined_binary_template = cv2.bitwise_or(self.combined_binary_template, binary_mask)
            
            if self.threshold_conditions.get("RGB二值化"):
                rgb_binary_mask = np.zeros(template_gray.shape[:2], dtype=np.uint8)
                for r_min, r_max, g_min, g_max, b_min, b_max in self.threshold_conditions["RGB二值化"]:
                    lower_bound_rgb = np.array([b_min, g_min, r_min])
                    upper_bound_rgb = np.array([b_max, g_max, r_max])
                    template_rgb_mask = cv2.inRange(template_rgb, lower_bound_rgb, upper_bound_rgb)
                    rgb_binary_mask = cv2.bitwise_or(rgb_binary_mask, template_rgb_mask)
                if self.combined_rgb_binary_template is None:
                    self.combined_rgb_binary_template = rgb_binary_mask
                else:
                    self.combined_rgb_binary_template = cv2.bitwise_or(self.combined_rgb_binary_template, rgb_binary_mask)
        
        # 5. 保存最终的聚合结果
        if self.threshold_conditions.get("灰度差阈值") and self.template_min is not None:
            cv2.imwrite(os.path.join(temp_dir, "所有模板_最小灰度值.png"), self.template_min)
        if self.threshold_conditions.get("灰度差阈值") and self.template_max is not None:
            cv2.imwrite(os.path.join(temp_dir, "所有模板_最大灰度值.png"), self.template_max)
        if self.threshold_conditions.get("二值化阈值") and self.combined_binary_template is not None:
            cv2.imwrite(os.path.join(temp_dir, "所有模板_二值化_最终合并结果.png"), self.combined_binary_template)
        
        if self.threshold_conditions.get("RGB二值化") and self.combined_rgb_binary_template is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.combined_rgb_binary_template = cv2.morphologyEx(
                self.combined_rgb_binary_template, cv2.MORPH_CLOSE, kernel)
            cv2.imwrite(os.path.join(temp_dir, "所有模板_RGB二值化_最终合并结果.png"), self.combined_rgb_binary_template)

        # 6. 更新缓存清单
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(current_state, f, ensure_ascii=False, indent=4)
        
        self.progress_signal.emit(0, "模板处理完成，缓存已更新。")


    def _process_single_image(self, image_path):
        """
        检测单张图片。
        (原 detect_single_image 方法)
        """
        
        # --- 新增: 创建独立的输出目录 ---
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join(self.output_dir, f"结果_{basename}")
        
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 复制原始图像
        try:
            shutil.copy2(image_path, os.path.join(output_dir, "tested_image.jpg"))
        except IOError as e:
            self.progress_signal.emit(0, f"无法复制文件: {e}")
            # 文件可能正在被写入，稍后重试
            time.sleep(0.5) 
            try:
                 shutil.copy2(image_path, os.path.join(output_dir, "tested_image.jpg"))
            except Exception as e2:
                 self.progress_signal.emit(0, f"复制文件最终失败: {e2}")
                 return False # 放弃处理此文件
        # --- 结束新增 ---

        image = cv2.imread(image_path)
        if image is None:
            self.progress_signal.emit(0, f"错误: 无法加载检测图像 {image_path}")
            return False
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.debug:
            cv2.imwrite(os.path.join(output_dir, "灰度图.png"), image_gray)

        intermediate_results = []
        binary_masks = []
        t_start = time.time()
        
        # 1. 处理二值化阈值条件
        if self.threshold_conditions.get("二值化阈值") and self.combined_binary_template is not None:
            image_binary_mask = np.zeros_like(image_gray)
            for low, high in self.threshold_conditions["二值化阈值"]:
                image_mask = cv2.inRange(image_gray, low, high)
                image_binary_mask = cv2.bitwise_or(image_binary_mask, image_mask)
                if self.debug:
                    cv2.imwrite(os.path.join(output_dir, f"二值化_阈值{low}-{high}.png"), image_mask)
            if self.debug:
                cv2.imwrite(os.path.join(output_dir, "二值化_合并结果.png"), image_binary_mask)
            diff = cv2.absdiff(image_binary_mask, self.combined_binary_template)
            diff_bin = cv2.bitwise_and(diff, image_binary_mask)
            if self.debug:
                cv2.imwrite(os.path.join(output_dir, "二值化差异图.png"), diff_bin)
            intermediate_results.append(("二值化差分", diff_bin.copy()))
            binary_masks.append(diff_bin)

        # 2. 处理RGB二值化条件
        if self.threshold_conditions.get("RGB二值化") and self.combined_rgb_binary_template is not None:
            rgb_mask = np.zeros_like(image_gray)
            for r_min, r_max, g_min, g_max, b_min, b_max in self.threshold_conditions["RGB二值化"]:
                lower_bound_rgb = np.array([b_min, g_min, r_min])
                upper_bound_rgb = np.array([b_max, g_max, r_max])
                image_rgb_mask = cv2.inRange(image, lower_bound_rgb, upper_bound_rgb)
                if self.debug:
                    cv2.imwrite(os.path.join(output_dir, f"RGB二值化_R{r_min}-{r_max}_G{g_min}-{g_max}_B{b_min}-{b_max}.png"), image_rgb_mask)
                rgb_mask = cv2.bitwise_or(rgb_mask, image_rgb_mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_CLOSE, kernel)
            if self.debug:
                cv2.imwrite(os.path.join(output_dir, "RGB二值化_合并结果.png"), rgb_mask)
            diff_rgb = cv2.absdiff(rgb_mask, self.combined_rgb_binary_template)
            detection_only_diff = cv2.bitwise_and(diff_rgb, rgb_mask)
            if self.debug:
                cv2.imwrite(os.path.join(output_dir, "RGB差异图.png"), detection_only_diff)
            intermediate_results.append(("RGB二值化差分", detection_only_diff.copy()))
            binary_masks.append(detection_only_diff)

        # 3. 处理灰度差阈值条件
        if self.threshold_conditions.get("灰度差阈值") and hasattr(self, 'template_min') and hasattr(self, 'template_max'):
            min_gray_threshold = min(self.threshold_conditions["灰度差阈值"])
            if len(self.detector.template_paths) == 1:
                diff = cv2.absdiff(image_gray, self.template_min)
                _, diff_gray = cv2.threshold(diff, min_gray_threshold, 255, cv2.THRESH_BINARY)
                if self.debug:
                    cv2.imwrite(os.path.join(output_dir, "灰度差_差异图.png"), diff_gray)
                intermediate_results.append(("灰度差阈值", diff_gray.copy()))
                binary_masks.append(diff_gray)
            else:
                lower_bound = np.clip(self.template_min.astype(np.int16) - min_gray_threshold, 0, 255).astype(np.uint8)
                upper_bound = np.clip(self.template_max.astype(np.int16) + min_gray_threshold, 0, 255).astype(np.uint8)
                if self.debug:
                    cv2.imwrite(os.path.join(output_dir, "灰度差_下限图.png"), lower_bound)
                    cv2.imwrite(os.path.join(output_dir, "灰度差_上限图.png"), upper_bound)
                out_of_range_lower = cv2.compare(image_gray, lower_bound, cv2.CMP_LT)
                out_of_range_upper = cv2.compare(image_gray, upper_bound, cv2.CMP_GT)
                diff_gray = cv2.bitwise_or(out_of_range_lower, out_of_range_upper)
                if self.debug:
                    cv2.imwrite(os.path.join(output_dir, "灰度差_差异图.png"), diff_gray)
                intermediate_results.append(("灰度差阈值", diff_gray.copy()))
                binary_masks.append(diff_gray)

        # 4. 组合中间结果
        if not binary_masks:
            self.progress_signal.emit(0, f"错误: {os.path.basename(image_path)} 未生成任何二值化掩码")
            return False

        combined_mask = binary_masks[0]
        for i in range(1, len(binary_masks)):
            if self.combo_method == "and":
                combined_mask = cv2.bitwise_and(combined_mask, binary_masks[i])
            else:  # "or"
                combined_mask = cv2.bitwise_or(combined_mask, binary_masks[i])
        cv2.imwrite(os.path.join(output_dir, "差异总图.png"), combined_mask)

        # 5. 应用屏蔽罩
        mask_path = os.path.join("config", "mask.png")
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                # 确保mask和图像尺寸一致
                if mask_img.shape != combined_mask.shape:
                    mask_img = cv2.resize(mask_img, (combined_mask.shape[1], combined_mask.shape[0]))
                combined_mask = cv2.bitwise_and(combined_mask, mask_img)
                cv2.imwrite(os.path.join(output_dir, "差异总图(带屏蔽).png"), combined_mask)
        
        t1_preprocessing_done = time.time()
        # 连通区域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)
        result_img = image.view()
        filtered_diff_img = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        defect_count = 0
        defect_regions = []

        # 向量化面积与比例判断
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            widths = stats[1:, cv2.CC_STAT_WIDTH]
            heights = stats[1:, cv2.CC_STAT_HEIGHT]

            area_mask = np.zeros_like(areas, dtype=bool)
            if not self.threshold_conditions["面积阈值"]:
                area_mask[:] = True
            else:
                for min_area, max_area in self.threshold_conditions["面积阈值"]:
                    area_mask |= (areas >= min_area) & (areas <= max_area)

            ratio_mask = np.zeros_like(areas, dtype=bool)
            if not self.threshold_conditions["比例阈值"]:
                ratio_mask[:] = True
            else:
                ratios = np.maximum(widths, heights) / (np.minimum(widths, heights) + 1e-5)
                for min_ratio, max_ratio in self.threshold_conditions["比例阈值"]:
                    ratio_mask |= (ratios >= min_ratio) & (ratios <= max_ratio)

            valid_indices = np.where(area_mask & ratio_mask)[0] + 1

            for i in valid_indices:
                if not self._is_running: return False

                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                # 5. 应用区域平均亮度阈值
                if self.threshold_conditions["区域平均亮度阈值"]:
                    region_mask = (labels == i)
                    region_pixels = image_gray[region_mask]
                    avg_brightness = np.mean(region_pixels) if len(region_pixels) > 0 else 0
                    brightness_valid = False
                    for min_bright, max_bright in self.threshold_conditions["区域平均亮度阈值"]:
                        if min_bright <= avg_brightness <= max_bright:
                            brightness_valid = True
                            break
                    if not brightness_valid:
                        continue

                # 3. 应用RGB阈值条件
                rgb_conditions = self.threshold_conditions["RGB值阈值"]
                rgb_valid = True
                if rgb_conditions:
                    mask = (labels == i)
                    y_indices, x_indices = np.where(mask)
                    if len(y_indices) == 0:
                        rgb_valid = False
                    else:
                        sample_ratio = 0.1
                        sample_count = max(1, min(100, int(len(y_indices) * sample_ratio)))
                        rng = np.random.default_rng(42)
                        indices = rng.choice(len(y_indices), sample_count, replace=False)
                        sampled_pixels = image[y_indices[indices], x_indices[indices]]
                        b_min, b_max = np.min(sampled_pixels[:, 0]), np.max(sampled_pixels[:, 0])
                        g_min, g_max = np.min(sampled_pixels[:, 1]), np.max(sampled_pixels[:, 1])
                        r_min, r_max = np.min(sampled_pixels[:, 2]), np.max(sampled_pixels[:, 2])
                        rgb_valid = False
                        for condition in rgb_conditions:
                            cond_rmin, cond_rmax, cond_gmin, cond_gmax, cond_bmin, cond_bmax, ratio_threshold = condition
                            range_overlap = (
                                (r_max >= cond_rmin) and (r_min <= cond_rmax) and
                                (g_max >= cond_gmin) and (g_min <= cond_gmax) and
                                (b_max >= cond_bmin) and (b_min <= cond_bmax)
                            )
                            if range_overlap:
                                in_range = (
                                    (sampled_pixels[:, 2] >= cond_rmin) &
                                    (sampled_pixels[:, 2] <= cond_rmax) &
                                    (sampled_pixels[:, 1] >= cond_gmin) &
                                    (sampled_pixels[:, 1] <= cond_gmax) &
                                    (sampled_pixels[:, 0] >= cond_bmin) &
                                    (sampled_pixels[:, 0] <= cond_bmax)
                                )
                                valid_ratio = np.mean(in_range)
                                if valid_ratio >= ratio_threshold:
                                    rgb_valid = True
                                    break

                # 4. AI过滤 (如果启用)
                is_defect = True
                if self.use_ai and self.detector.yolo_model and rgb_valid:
                    pad = 10
                    roi = image[max(0, y - pad):min(image.shape[0], y + h + pad),
                        max(0, x - pad):min(image.shape[1], x + w + pad)]
                    if roi.size > 0:
                        results = self.detector.yolo_model(roi, verbose=False)
                        if len(results[0].boxes) == 0:
                            is_defect = False
                
                if is_defect and rgb_valid:
                    defect_count += 1
                    center_x = x + w // 2
                    center_y = y + h // 2
                    new_w = int(w * 1.3)
                    new_h = int(h * 1.3)
                    new_x = max(0, int(x - (new_w - w) / 2))
                    new_y = max(0, int(y - (new_h - h) / 2))
                    new_w = min(new_w, image.shape[1] - new_x)
                    new_h = min(new_h, image.shape[0] - new_y)

                    if self.debug:   
                        cv2.rectangle(result_img, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 1)
                        cv2.putText(result_img, str(defect_count), (new_x, new_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        info_y = new_y + new_h + 20
                        area_text = f"area: {stats[i, cv2.CC_STAT_AREA]} px"
                        cv2.putText(result_img, area_text, (new_x, info_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                        ratio_text = f"L/W:{max(w, h) / (min(w, h) + 1e-5):.2f}"
                        cv2.putText(result_img, ratio_text, (new_x, info_y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                        cv2.rectangle(filtered_diff_img, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 1)
                        cv2.putText(filtered_diff_img, str(defect_count), (new_x, new_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        info_y = new_y + new_h + 20
                        cv2.putText(filtered_diff_img, area_text, (new_x, info_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                        cv2.putText(filtered_diff_img, ratio_text, (new_x, info_y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

                    defect_regions.append((new_x, new_y, new_w, new_h, stats[i, cv2.CC_STAT_AREA], max(w, h) / (min(w, h) + 1e-5), center_x, center_y))

        if self.debug:
            cv2.imwrite(os.path.join(output_dir, "过滤后的差异图.png"), filtered_diff_img)
            cv2.imwrite(os.path.join(output_dir, "final_result.png"), result_img)

        image = None
        gc.collect()
        t2_connected_components_done = time.time()
        
        reloaded_image = cv2.imread(image_path)
        if reloaded_image is None:
            self.progress_signal.emit(0, f"错误: 重新加载检测图像失败 {image_path}")
            return False
        
        t3_filtering_loop_done = time.time()
        defects_dir = os.path.join(output_dir, "defects")
        os.makedirs(defects_dir, exist_ok=True)

        tasks = []
        for i, (x, y, w, h, area, ratio, center_x, center_y) in enumerate(defect_regions):
            min_size = max(120, ((max(w, h) + 119) // 120) * 120)
            x1 = max(0, center_x - min_size // 2)
            y1 = max(0, center_y - min_size // 2)
            x2 = min(reloaded_image.shape[1], x1 + min_size)
            y2 = min(reloaded_image.shape[0], y1 + min_size)
            
            tasks.append((
                x1, y1, x2, y2, i + 1, x, y, w, h, center_x, center_y
            ))

        def process_defect(task, img, save_dir):
            x1, y1, x2, y2, defect_id, x, y, w, h, center_x, center_y = task
            roi = img[y1:y2, x1:x2].copy()
            
            defect_x1 = max(0, x - x1)
            defect_y1 = max(0, y - y1)
            defect_x2 = defect_x1 + w
            defect_y2 = defect_y1 + h
            
            if 0 <= defect_x1 < roi.shape[1] and 0 <= defect_y1 < roi.shape[0]:
                if defect_x2 > roi.shape[1]: defect_x2 = roi.shape[1]
                if defect_y2 > roi.shape[0]: defect_y2 = roi.shape[0]
                if defect_x2 > defect_x1 and defect_y2 > defect_y1:
                    cv2.rectangle(roi, (defect_x1, defect_y1), (defect_x2, defect_y2), (0, 0, 255), 1)
            
            filename = f"{defect_id}_center({center_x},{center_y}).png"
            cv2.imwrite(os.path.join(save_dir, filename), roi)
            return defect_id

        num_defects = len(tasks)
        num_threads = min(16, max(4, num_defects // 20))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_defect, task, reloaded_image, defects_dir) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    defect_id = future.result()
                except Exception as e:
                    print(f"处理缺陷时出错: {str(e)}")

        reloaded_image = None
        gc.collect()
        t4_filtering_loop_done = time.time()
        
        print(f"\n--- 性能分析 ({os.path.basename(image_path)}) ---")
        print(f"T1 - 预处理与掩码: {t1_preprocessing_done - t_start:.4f} s")
        print(f"T2 - 连通域: {t2_connected_components_done - t1_preprocessing_done:.4f} s")
        print(f"T3 - 重载: {t3_filtering_loop_done - t2_connected_components_done:.4f} s")
        print(f"T4 - 缺陷裁剪保存: {t4_filtering_loop_done - t3_filtering_loop_done:.4f} s")
        print(f"--- 总计: {t4_filtering_loop_done - t_start:.4f} s ---")
        return True