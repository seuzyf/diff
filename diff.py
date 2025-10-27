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
                            QProgressBar, QComboBox, QSlider, QRadioButton, QButtonGroup)
from PyQt5.QtGui import (QImage, QPixmap, QIntValidator, QDoubleValidator, QColor, QBrush,
                         QWheelEvent, QPainter, QPen, QFont, QMouseEvent, QTransform)
from PyQt5.QtCore import Qt, pyqtSignal, QPoint, QRect, QSize, QThread
from ultralytics import YOLO
import traceback
from glob import glob
import json
import time
import shutil 
import concurrent.futures
import gc

class DetectionThread(QThread):
    progress_signal = pyqtSignal(int, str)  # 进度值, 消息
    finished_signal = pyqtSignal(bool, str)  # 是否成功, 消息
    intermediate_result_signal = pyqtSignal(str, np.ndarray)  # 中间结果名称, 图像数据
    
    def __init__(self, detector, image_paths, output_dir, threshold_conditions, use_ai, combo_method, debug, parent=None):
        super().__init__(parent)
        self.detector = detector
        self.image_paths = image_paths
        self.output_dir = output_dir
        self.threshold_conditions = threshold_conditions
        self.use_ai = use_ai
        self.combo_method = combo_method
        self.debug = debug
        self._is_running = True
    
    def run(self):
        try:
            total = len(self.image_paths)
            
            # 预处理模板图（仅在检测开始时执行一次）
            self.preprocess_templates()
            
            for idx, image_path in enumerate(self.image_paths):
                if not self._is_running:
                    self.progress_signal.emit(0, "检测已终止")
                    return
                    
                self.progress_signal.emit(idx + 1, f"处理中: {idx+1}/{total} - {os.path.basename(image_path)}")
                
                basename = os.path.splitext(os.path.basename(image_path))[0]
                board_output_dir = os.path.join(self.output_dir, f"结果_{basename}")

                # 清空操作：如果目录已存在则删除整个目录
                if os.path.exists(board_output_dir):
                    shutil.rmtree(board_output_dir)  # 删除目录及其所有内容

                os.makedirs(board_output_dir, exist_ok=True)  # 重新创建目录
                
                # 复制原始图像
                shutil.copy2(image_path, os.path.join(board_output_dir, "tested_image.jpg"))
                
                # 执行检测
                result = self.detect_single_image(image_path, board_output_dir)
                
                if not result:
                    self.progress_signal.emit(idx + 1, f"处理失败: {os.path.basename(image_path)}")
            
            self.finished_signal.emit(True, f"完成! 已处理{total}张图片")
        except Exception as e:
            traceback.print_exc()
            self.finished_signal.emit(False, f"检测失败: {str(e)}")

        
    def stop(self):
        self._is_running = False
    
    def preprocess_templates(self):
        """
        预处理所有模板图，并增加了缓存机制。
        如果模板文件和处理参数未改变，则从缓存加载，避免重复计算。
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
        # 确保所有值都是可JSON序列化的
        current_state = {
            "template_paths": sorted(self.detector.template_paths),
            "threshold_conditions": self.threshold_conditions
        }
        print(current_state)
        
        # 2. 检查缓存是否有效
        cache_valid = False
        try:
            if os.path.exists(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    saved_state = json.load(f)
                    print(saved_state)
                # 深度比较状态是否一致
                # 统一数据结构后再比较
                normalized_saved = normalize_data_structure(saved_state)
                normalized_current = normalize_data_structure(current_state)
                if normalized_saved == normalized_current:
                    print("状态一致，检查缓存文件...")
                    # 状态一致，检查必要的缓存文件是否存在
                    required_files = []
                    
                    # 根据当前启用的条件添加所需文件
                    if self.threshold_conditions.get("二值化阈值"):
                        required_files.append("所有模板_二值化_最终合并结果.png")
                    if self.threshold_conditions.get("RGB二值化"):
                        required_files.append("所有模板_RGB二值化_最终合并结果.png")
                    if self.threshold_conditions.get("灰度差阈值"):
                        required_files.extend(["所有模板_最小灰度值.png", "所有模板_最大灰度值.png"])
                    
                    # 调试信息：打印需要检查的文件
                    print(f"需要检查的缓存文件: {required_files}")
                    
                    # 检查所有文件是否存在
                    all_files_exist = True
                    for filename in required_files:
                        file_path = os.path.join(temp_dir, filename)
                        if not os.path.exists(file_path):
                            print(f"缓存文件缺失: {file_path}")
                            all_files_exist = False
                            break
                    
                    if all_files_exist:
                        cache_valid = True
                        print("所有缓存文件存在，缓存有效")
                    else:
                        print("部分缓存文件缺失")
        
        except (json.JSONDecodeError, IOError, TypeError) as e:
            print(f"读取缓存清单失败: {e}")
            cache_valid = False

        # 3. 如果缓存有效则加载
        if cache_valid:
            print("缓存命中，从文件加载预处理结果...")
            self.progress_signal.emit(0, "缓存命中，正在加载预处理结果...")
            
            # 加载灰度上下限图（如果使用灰度差阈值）
            if self.threshold_conditions.get("灰度差阈值"):
                min_path = os.path.join(temp_dir, "所有模板_最小灰度值.png")
                max_path = os.path.join(temp_dir, "所有模板_最大灰度值.png")
                if os.path.exists(min_path):
                    self.template_min = cv2.imread(min_path, cv2.IMREAD_GRAYSCALE)
                    print(f"已加载模板最小灰度图: {min_path}")
                if os.path.exists(max_path):
                    self.template_max = cv2.imread(max_path, cv2.IMREAD_GRAYSCALE)
                    print(f"已加载模板最大灰度图: {max_path}")
            
            # 加载灰度二值化结果
            if self.threshold_conditions.get("二值化阈值"):
                bin_path = os.path.join(temp_dir, "所有模板_二值化_最终合并结果.png")
                if os.path.exists(bin_path):
                    self.combined_binary_template = cv2.imread(bin_path, cv2.IMREAD_GRAYSCALE)
                    print(f"已加载灰度二值化模板: {bin_path}")
            
            # 加载RGB二值化结果
            if self.threshold_conditions.get("RGB二值化"):
                rgb_path = os.path.join(temp_dir, "所有模板_RGB二值化_最终合并结果.png")
                if os.path.exists(rgb_path):
                    self.combined_rgb_binary_template = cv2.imread(rgb_path, cv2.IMREAD_GRAYSCALE)
                    print(f"已加载RGB二值化模板: {rgb_path}")
            
            return  # 加载成功后直接返回
        
        # 4. 缓存无效或不存在，删除整个缓存目录并重新生成
        print("缓存无效或不存在，正在重新处理模板...")
        self.progress_signal.emit(0, "缓存无效，正在重新处理模板...")
        
        # 删除整个缓存目录（如果存在）
        if os.path.exists(temp_dir):
            print(f"删除旧缓存目录: {temp_dir}")
            shutil.rmtree(temp_dir)
        
        # 重新创建缓存目录
        os.makedirs(temp_dir, exist_ok=True)
        print(f"创建新缓存目录: {temp_dir}")
        
        # 初始化聚合结果
        self.combined_binary_template = None
        self.combined_rgb_binary_template = None
        self.template_min = None
        self.template_max = None
        
        # 遍历所有模板图路径，一次只加载一张图
        for i, template_path in enumerate(self.detector.template_paths):
            template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if template_gray is None:
                print(f"无法读取灰度模板: {template_path}")
                continue
            
            template_rgb = cv2.imread(template_path)
            if template_rgb is None:
                print(f"无法读取RGB模板: {template_path}")
                continue
            
            # 只有在使用灰度差阈值时才计算灰度上下限
            if self.threshold_conditions.get("灰度差阈值"):
                if self.template_min is None:
                    self.template_min = template_gray.copy()
                    self.template_max = template_gray.copy()
                else:
                    self.template_min = np.minimum(self.template_min, template_gray)
                    self.template_max = np.maximum(self.template_max, template_gray)
            
            # 处理灰度二值化条件
            if self.threshold_conditions.get("二值化阈值"):
                binary_mask = np.zeros_like(template_gray)
                for low, high in self.threshold_conditions["二值化阈值"]:
                    template_mask = cv2.inRange(template_gray, low, high)
                    binary_mask = cv2.bitwise_or(binary_mask, template_mask)
                
                if self.combined_binary_template is None:
                    self.combined_binary_template = binary_mask
                else:
                    self.combined_binary_template = cv2.bitwise_or(self.combined_binary_template, binary_mask)
            
            # 处理RGB二值化条件
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
        # 只有在使用灰度差阈值时才保存灰度上下限图
        if self.threshold_conditions.get("灰度差阈值") and self.template_min is not None:
            min_path = os.path.join(temp_dir, "所有模板_最小灰度值.png")
            cv2.imwrite(min_path, self.template_min)
            print(f"已保存模板最小灰度图: {min_path}")
        
        if self.threshold_conditions.get("灰度差阈值") and self.template_max is not None:
            max_path = os.path.join(temp_dir, "所有模板_最大灰度值.png")
            cv2.imwrite(max_path, self.template_max)
            print(f"已保存模板最大灰度图: {max_path}")
        
        if self.threshold_conditions.get("二值化阈值") and self.combined_binary_template is not None:
            bin_path = os.path.join(temp_dir, "所有模板_二值化_最终合并结果.png")
            cv2.imwrite(bin_path, self.combined_binary_template)
            print(f"已保存灰度二值化模板: {bin_path}")
        
        if self.threshold_conditions.get("RGB二值化") and self.combined_rgb_binary_template is not None:
            # 对RGB二值化结果进行形态学闭合操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            self.combined_rgb_binary_template = cv2.morphologyEx(
                self.combined_rgb_binary_template, cv2.MORPH_CLOSE, kernel)
            
            rgb_path = os.path.join(temp_dir, "所有模板_RGB二值化_最终合并结果.png")
            cv2.imwrite(rgb_path, self.combined_rgb_binary_template)
            print(f"已保存RGB二值化模板: {rgb_path}")

        # 6. 更新缓存清单
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(current_state, f, ensure_ascii=False, indent=4)
        
        print(f"已更新缓存清单: {manifest_path}")
        print("模板处理完成，缓存已更新。")
        self.progress_signal.emit(0, "模板处理完成，缓存已更新。")


    def detect_single_image(self, image_path, output_dir):
        """
        检测单张图片。
        """
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
            self.progress_signal.emit(0, "错误: 未生成任何二值化掩码")
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
            # 提取所有区域的面积、宽、高
            areas = stats[1:, cv2.CC_STAT_AREA]
            widths = stats[1:, cv2.CC_STAT_WIDTH]
            heights = stats[1:, cv2.CC_STAT_HEIGHT]

            # 构建面积掩码
            area_mask = np.zeros_like(areas, dtype=bool)
            if not self.threshold_conditions["面积阈值"]:
                area_mask[:] = True
            else:
                for min_area, max_area in self.threshold_conditions["面积阈值"]:
                    area_mask |= (areas >= min_area) & (areas <= max_area)

            # 构建比例掩码
            ratio_mask = np.zeros_like(areas, dtype=bool)
            if not self.threshold_conditions["比例阈值"]:
                ratio_mask[:] = True
            else:
                ratios = np.maximum(widths, heights) / (np.minimum(widths, heights) + 1e-5)
                for min_ratio, max_ratio in self.threshold_conditions["比例阈值"]:
                    ratio_mask |= (ratios >= min_ratio) & (ratios <= max_ratio)

            # 合并有效区域索引
            valid_indices = np.where(area_mask & ratio_mask)[0] + 1

            # 遍历有效区域
            for i in valid_indices:
                if not self._is_running:
                    return False

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
                        # 绘图操作
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

                        # 过滤后的差异图绘图
                        cv2.rectangle(filtered_diff_img, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 0, 255), 1)
                        cv2.putText(filtered_diff_img, str(defect_count), (new_x, new_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        info_y = new_y + new_h + 20
                        cv2.putText(filtered_diff_img, area_text, (new_x, info_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
                        cv2.putText(filtered_diff_img, ratio_text, (new_x, info_y + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

                    # 保存缺陷区域信息
                    defect_regions.append((new_x, new_y, new_w, new_h, stats[i, cv2.CC_STAT_AREA], max(w, h) / (min(w, h) + 1e-5), center_x, center_y))

        # 保存过滤后的差异图
        if self.debug:
            cv2.imwrite(os.path.join(output_dir, "过滤后的差异图.png"), filtered_diff_img)
        
            # 保存最终结果
            cv2.imwrite(os.path.join(output_dir, "final_result.png"), result_img)

        image = None
        gc.collect()
        t2_connected_components_done = time.time()
        # 重新加载原图用于缺陷区域裁剪
        reloaded_image = cv2.imread(image_path)
        if reloaded_image is None:
            self.progress_signal.emit(0, f"错误: 重新加载检测图像失败 {image_path}")
            return False
        t3_filtering_loop_done = time.time()
        # 保存缺陷区域图像
        defects_dir = os.path.join(output_dir, "defects")
        os.makedirs(defects_dir, exist_ok=True)

        # 准备多线程处理的数据 - 只存储坐标信息
        tasks = []
        for i, (x, y, w, h, area, ratio, center_x, center_y) in enumerate(defect_regions):
            min_size = max(120, ((max(w, h) + 119) // 120) * 120)
            x1 = max(0, center_x - min_size // 2)
            y1 = max(0, center_y - min_size // 2)
            x2 = min(reloaded_image.shape[1], x1 + min_size)
            y2 = min(reloaded_image.shape[0], y1 + min_size)
            
            tasks.append((
                x1, y1, x2, y2,
                i + 1,  # 缺陷序号
                x, y, w, h,
                center_x, center_y
            ))

        # 使用线程池并行处理
        def process_defect(task, img, save_dir):
            x1, y1, x2, y2, defect_id, x, y, w, h, center_x, center_y = task
            roi = img[y1:y2, x1:x2].copy()
            
            # 在裁剪图上绘制缺陷矩形
            defect_x1 = max(0, x - x1)
            defect_y1 = max(0, y - y1)
            defect_x2 = defect_x1 + w
            defect_y2 = defect_y1 + h
            
            # 确保矩形在ROI范围内
            if 0 <= defect_x1 < roi.shape[1] and 0 <= defect_y1 < roi.shape[0]:
                if defect_x2 > roi.shape[1]:
                    defect_x2 = roi.shape[1]
                if defect_y2 > roi.shape[0]:
                    defect_y2 = roi.shape[0]
                
                if defect_x2 > defect_x1 and defect_y2 > defect_y1:
                    cv2.rectangle(roi, (defect_x1, defect_y1), (defect_x2, defect_y2), (0, 0, 255), 1)
            
            # 保存文件
            filename = f"{defect_id}_center({center_x},{center_y}).png"
            cv2.imwrite(os.path.join(save_dir, filename), roi)
            return defect_id

        # 动态调整线程数
        num_defects = len(tasks)
        num_threads = min(16, max(4, num_defects // 20))

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for task in tasks:
                futures.append(executor.submit(process_defect, task, reloaded_image, defects_dir))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    defect_id = future.result()
                except Exception as e:
                    print(f"处理缺陷时出错: {str(e)}")

        # 释放重新加载的图像
        reloaded_image = None
        gc.collect()
        t4_filtering_loop_done = time.time()
        
        print("\n--- 性能分析 ---")
        print(f"T1 - 图像预处理与掩码生成: {t1_preprocessing_done - t_start:.4f} 秒")
        print(f"T2 - 连通域分析: {t2_connected_components_done - t1_preprocessing_done:.4f} 秒")
        print(f"T3 - 重新加载: {t3_filtering_loop_done - t2_connected_components_done:.4f} 秒")
        print(f"T3 - 结果绘制与保存: {t4_filtering_loop_done - t3_filtering_loop_done:.4f} 秒")
        print(f"--- 总计耗时: {t3_filtering_loop_done - t_start:.4f} 秒 ---")
        return True


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
            painter.drawLine(final_x - 10, final_y, final_x + 10, final_y)
            painter.drawLine(final_x, final_y - 10, final_x, final_y + 10)
        
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


class DefectViewer(QDialog):
    def __init__(self, parent=None, output_dir=None):
        super().__init__(parent)
        self.setWindowTitle("缺陷查看器 (快捷键: ↑/↓切换, →标记误报)")
        self.resize(1600, 700)
        self.output_dir = output_dir
        self.defect_files = []
        self.defect_positions = []  # 存储缺陷位置 (x, y, w, h)
        self.false_positives = []   # 追踪误报状态 (True/False)
        self.current_index = 0
        self.result_dirs = []
        self.current_result_index = -1
        self.original_img = None
        self.original_img_rgb = None
        self.initUI()
    
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
        self.board_label.setFixedWidth(360)
        self.board_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        right_layout.addWidget(self.board_label)
        
        self.defect_label = QLabel("缺陷区域")
        self.defect_label.setAlignment(Qt.AlignCenter)
        self.defect_label.setFixedSize(360, 360)
        self.defect_label.setStyleSheet("border: 1px solid gray;")
        right_layout.addWidget(self.defect_label)
        
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
        
        if self.output_dir:
            self.load_result_dirs(self.output_dir)

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
            # 对于其他按键，调用父类的默认行为
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
        self.result_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
        
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
        
        image_path = self.find_image_in_directory(output_dir)
        if not image_path:
            QMessageBox.warning(self, "警告", f"目录中未找到检测图像: {output_dir}")
            return
        
        self.original_img = cv2.imread(image_path)
        if self.original_img is None:
            QMessageBox.critical(self, "错误", f"无法加载图像: {image_path}")
            return
        
        self.original_img_rgb = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2RGB)
        self.display_original_image()
        
        defects_dir = os.path.join(output_dir, "defects")
        if not os.path.exists(defects_dir):
            self.defect_list.clear() # 清空旧列表
            QMessageBox.warning(self, "警告", f"未找到缺陷目录: {defects_dir}")
            return
        
        # **重置所有列表**
        self.defect_files = []
        self.defect_positions = []
        self.false_positives = []
        self.defect_list.clear()
        
        defect_info = []
        for filename in os.listdir(defects_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                match = re.search(r'(\d+)_center\((\d+),(\d+)\)', filename)
                if match:
                    index = int(match.group(1))
                    center_x = int(match.group(2))
                    center_y = int(match.group(3))
                    
                    rect_match = re.search(r'rect\((\d+),(\d+),(\d+),(\d+)\)', filename)
                    if rect_match:
                        x, y, w, h = map(int, rect_match.groups())
                    else: # 默认值
                        x, y, w, h = center_x - 5, center_y - 5, 10, 10
                    
                    defect_info.append({
                        'index': index, 'path': os.path.join(defects_dir, filename),
                        'rect': (x, y, w, h)
                    })
        
        defect_info.sort(key=lambda x: x['index'])
        
        for info in defect_info:
            self.defect_files.append(info['path'])
            self.defect_positions.append(info['rect'])
            self.false_positives.append(False)  
            self.defect_list.addItem(f"缺陷 {info['index']}")
        
        if self.defect_files:
            self.current_index = 0
            self.defect_list.setCurrentRow(0)

        else:
            self.defect_label.clear()
            self.defect_label.setText("无缺陷")
            self.zoom_label.clear_defect()
            QMessageBox.information(self, "提示", "目录中未找到已保存的缺陷图片")

    def mark_false_positive(self):
        """标记/取消标记当前缺陷为误报"""
        if not self.defect_files or self.current_index >= len(self.defect_files):
            return
        
        # 切换误报状态
        current_status = self.false_positives[self.current_index]
        self.false_positives[self.current_index] = not current_status
        
        # 更新列表项的背景颜色
        item = self.defect_list.item(self.current_index)
        if self.false_positives[self.current_index]:
            # 标记为误报，设置绿色背景
            item.setBackground(QColor('lightgreen'))
        else:
            # 取消标记，恢复默认背景
            item.setBackground(QBrush())

    def generate_result_image(self):
        """根据当前未标记为误报的缺陷列表，重新生成结果图片"""
        if not self.output_dir or self.original_img is None:
            QMessageBox.warning(self, "警告", "请先加载包含原始图像的结果目录")
            return
        
        # 创建带标注的结果图像的副本
        result_img = self.original_img.view()
        
        # 仅为未标记为误报的缺陷添加标注
        valid_defect_counter = 1 # 用于为有效缺陷进行编号
        for i, pos in enumerate(self.defect_positions):
            # 检查该缺陷是否被标记为误报
            if not self.false_positives[i]:
                x, y, w, h = pos
                center_x = x + w // 2
                center_y = y + h // 2
                
                # 绘制十字标记
                cv2.drawMarker(result_img, (center_x, center_y), (0, 0, 255), 
                               cv2.MARKER_CROSS, 15, 1)
                
                # 绘制新的连续序号
                cv2.putText(result_img, str(valid_defect_counter), 
                            (center_x + 15, center_y - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                            (0, 0, 255), 2)
                
                valid_defect_counter += 1 # 序号递增
        
        # 保存结果
        result_path = os.path.join(self.output_dir, "updated_result.png")
        try:
            cv2.imwrite(result_path, result_img)
            
            # 在界面上显示新生成的结果
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_img_rgb.shape
            bytes_per_line = ch * w
            qimg = QImage(result_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.zoom_label.setPixmap(QPixmap.fromImage(qimg))
            
            QMessageBox.information(self, "成功", f"已生成更新后的结果图片:\n{result_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存图片失败: {str(e)}")
    
    def find_image_in_directory(self, directory):
        # 优先使用tested_image.jpg
        tested_image = os.path.join(directory, "tested_image.jpg")
        if os.path.exists(tested_image):
            return tested_image
        
        # 其次使用final_result.jpg
        final_result = os.path.join(directory, "final_result.jpg")
        if os.path.exists(final_result):
            return final_result
        
        # 最后查找其他图片
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(directory, filename)
        return None
    
    def display_original_image(self):
        h, w, ch = self.original_img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(self.original_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.zoom_label.setPixmap(QPixmap.fromImage(qimg))
    
    def update_defect_display(self, index):
        if not self.defect_files or index >= len(self.defect_files):
            return
        
        # 获取缺陷位置并设置到缩放标签
        x, y, w, h = self.defect_positions[index]
        self.zoom_label.set_current_defect(x, y, w, h)
        
        # 显示缺陷区域图片
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
    
    def on_list_click(self, row):
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
    


class PcbDefectDetector(QWidget):
    def __init__(self):
        super().__init__()
        self.template_paths = []
        self.image_path = None
        self.folder = None
        self.image_paths = []
        self.output_image = None
        self.threshold_ranges = []
        self.current_output_dir = ""
        self.detection_thread = None  # 初始化 detection_thread
        
        # 处理 YOLO 模型加载
        try:
            # 确保 Ultralytics 设置目录存在
            os.makedirs(os.path.join(os.getenv('APPDATA'), 'Ultralytics', exist_ok=True))
            
            # 尝试加载 YOLO 模型
            self.yolo_model = YOLO('best.pt') if os.path.exists('best.pt') else None
            
            # 如果模型加载失败，设置默认设置
            if self.yolo_model is None:
                self.create_default_ultralytics_settings()
        except Exception as e:
            print(f"YOLO 模型加载错误: {str(e)}")
            self.yolo_model = None
            self.create_default_ultralytics_settings()
            
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
        self.btn_clear_template = QPushButton('清除模板图')  # 新增清除按钮
        self.btn_load_template_folder = QPushButton('加载模板文件夹')
        self.btn_load_image = QPushButton('加载检测图')
        self.btn_clear_image = QPushButton('清除检测图')      # 新增清除按钮
        self.btn_load_folder = QPushButton('加载检测文件夹')
            
        self.btn_load_template.clicked.connect(self.load_template)
        self.btn_load_template_folder.clicked.connect(self.load_template_folder)  # 连接新按钮
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_load_folder.clicked.connect(self.load_image_folder)
        # 连接清除按钮的点击事件
        self.btn_clear_template.clicked.connect(self.clear_template)
        self.btn_clear_image.clicked.connect(self.clear_image)
        
        template_layout = QVBoxLayout(template_group)
        result_layout = QVBoxLayout(result_group)
        
        template_layout.addWidget(self.template_label, stretch=4)
        
        template_btn_layout = QHBoxLayout()

        # 移除了Mark点相关按钮
        
        template_btn_layout.addWidget(self.btn_load_template)
        template_btn_layout.addWidget(self.btn_load_template_folder)  # 添加新按钮
        template_btn_layout.addWidget(self.btn_clear_template) 
        template_layout.addLayout(template_btn_layout, stretch=1)
        
        result_layout.addWidget(self.result_label, stretch=4)
        
        result_btn_layout = QHBoxLayout()
        result_btn_layout.addWidget(self.btn_load_image)
        result_btn_layout.addWidget(self.btn_load_folder)
        result_btn_layout.addWidget(self.btn_clear_image)
        result_layout.addLayout(result_btn_layout, stretch=1)
        
        image_layout.addWidget(template_group, stretch=1)
        image_layout.addWidget(result_group, stretch=1)
        main_layout.addLayout(image_layout, stretch=3)
        
        param_group = QGroupBox("检测参数")
        param_layout = QGridLayout(param_group)
        
        # 组合方式设置
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
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("就绪")
        param_layout.addWidget(QLabel("进度:"), 0, 2)
        param_layout.addWidget(self.progress_bar, 0, 3)
        param_layout.addWidget(self.progress_label, 0, 4)
        
        # 阈值条件面板
        threshold_group = QGroupBox("阈值条件设置")
        threshold_layout = QVBoxLayout(threshold_group)
        
        # 添加阈值条件的按钮
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
        
        # 阈值条件容器
        self.threshold_scroll = QScrollArea()
        self.threshold_scroll.setWidgetResizable(True)
        self.threshold_container = QWidget()
        self.threshold_container_layout = QVBoxLayout(self.threshold_container)
        self.threshold_container_layout.setSpacing(5)
        self.threshold_scroll.setWidget(self.threshold_container)
        
        threshold_layout.addWidget(self.threshold_scroll)
        
        param_layout.addWidget(threshold_group, 1, 0, 1, 5)
        main_layout.addWidget(param_group, stretch=1)
        
        # 底部按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        self.cb_use_ai = QCheckBox('启用AI过滤')
        self.cb_debug = QCheckBox('保存中间结果图')
        self.cb_auto_delete = QCheckBox('自动删除原图')  # 新增勾选框
        btn_layout.addWidget(self.cb_auto_delete)       # 添加到布局中
        
        if not self.yolo_model:
            self.cb_use_ai.setEnabled(False)
            self.cb_use_ai.setText("启用AI (模型未加载)")
        
        btn_layout.addWidget(self.cb_use_ai)
        btn_layout.addWidget(self.cb_debug)
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
        
        # 添加默认阈值
        self.add_threshold_range("二值化阈值")
        self.add_threshold_range("面积阈值")
        self.add_threshold_range("比例阈值")
    

    def clear_template(self):
        """清除模板图"""
        self.template_paths = []

        self.btn_load_template.setText('加载单模板图')
        self.btn_load_template_folder.setText('加载模板文件夹')

    def clear_image(self):
        """清除检测图"""
        self.image_path = None
        self.image_paths = []
        self.folder = None
        self.btn_load_image.setText('加载检测图')
        self.btn_load_folder.setText('加载检测文件夹')
        self.progress_label.setText("就绪")

    def create_default_ultralytics_settings(self):
        """创建默认的 Ultralytics 设置文件"""
        settings_path = os.path.join(os.getenv('APPDATA'), 'Ultralytics', 'settings.json')
        default_settings = {
            "settings_version": "0.0.6",
            "openvino_msg": True,
            "runs_dir": os.path.join(os.getcwd(), "runs"),
            "weights_dir": os.path.join(os.getcwd(), "weights"),
            "datasets_dir": os.path.join(os.getcwd(), "datasets")
        }
        
        try:
            with open(settings_path, 'w') as f:
                json.dump(default_settings, f, indent=2)
        except Exception as e:
            print(f"无法创建 Ultralytics 设置文件: {str(e)}")

    def save_settings(self):
        settings = {
            'threshold_conditions': [],
            'use_ai': self.cb_use_ai.isChecked(),
            'combo_method': 'and' if self.radio_and.isChecked() else 'or',
            'debug': self.cb_debug.isChecked(),
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
            settings['threshold_conditions'].append(condition)
        
        os.makedirs('config', exist_ok=True)
        with open('config/settings.json', 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=2)

    def load_settings(self):
        try:
            if not os.path.exists('config/settings.json'):
                return False
                
            with open('config/settings.json', 'r', encoding='utf-8') as f:
                settings = json.load(f)
            
            # 清除现有阈值条件
            for widget in self.threshold_ranges[:]:
                self.remove_threshold_range(widget)
            
            # 先添加所有需要的阈值条件
            for condition in settings['threshold_conditions']:
                # 先添加控件但不设置值
                self.add_threshold_range(condition['type'])
            
            # 现在所有控件都已创建，可以安全地设置值
            for i, condition in enumerate(settings['threshold_conditions']):
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
            
            # 加载其他设置
            self.cb_use_ai.setChecked(settings.get('use_ai', False))
            if settings.get('combo_method', 'and') == 'and':
                self.radio_and.setChecked(True)
            else:
                self.radio_or.setChecked(True)
            self.cb_debug.setChecked(settings.get('debug', False))
            
            return True
        except Exception as e:
            print(f"加载设置失败: {str(e)}")
            traceback.print_exc()
            return False
    
    def closeEvent(self, event):
        self.stop_detection()
        self.save_settings()
        event.accept()

    def add_threshold_range(self, threshold_type):
        range_widget = ThresholdRangeWidget(threshold_type)
        range_widget.remove_btn.clicked.connect(lambda: self.remove_threshold_range(range_widget))
        self.threshold_container_layout.addWidget(range_widget)
        self.threshold_ranges.append(range_widget)
        return range_widget  # 返回创建的控件以便后续操作
    
    def remove_threshold_range(self, widget):
        self.threshold_ranges.remove(widget)
        widget.deleteLater()
    
    def load_template(self):
        """加载单个模板图"""
        path, _ = QFileDialog.getOpenFileName(self, "选择模板图", "", "图片文件 (*.png *.jpg *.jpeg)")
        if path:
            self.template_paths = [path]  # 单个模板图也存储在列表中
            self.load_and_display_image(path, self.template_label)
            self.btn_load_template.setText(f"已加载: {os.path.basename(path)}")
    
    def load_template_folder(self):
        """加载模板图文件夹"""
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
    
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择检测图", "", "图片文件 (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            self.image_paths = [path]
            self.load_and_display_image(path, self.result_label)
            self.btn_load_image.setText(f"已加载: {os.path.basename(path)}")
            self.btn_load_folder.setText("加载检测文件夹")
            self.progress_label.setText(f"已选择1张图片")
    
    def load_image_folder(self):
        self.folder = QFileDialog.getExistingDirectory(self, "选择检测图片文件夹", "")
        if self.folder:
            self.image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                self.image_paths.extend(glob(os.path.join(self.folder, ext)))
            if not self.image_paths:
                QMessageBox.warning(self, "警告", "文件夹中没有找到图片文件！")
                return
            self.image_path = self.image_paths[0]
            self.load_and_display_image(self.image_path, self.result_label)
            self.btn_load_folder.setText(f"已加载: {len(self.image_paths)}张图片")
            self.progress_label.setText(f"已选择{len(self.image_paths)}张图片")

    def reflash_image_folder(self):
        if self.folder:
            self.image_paths = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                self.image_paths.extend(glob(os.path.join(self.folder, ext)))
            if not self.image_paths:
                QMessageBox.warning(self, "警告", "文件夹中没有找到图片文件！")
                return
            self.image_path = self.image_paths[0]
            self.load_and_display_image(self.image_path, self.result_label)
            self.btn_load_folder.setText(f"已加载: {len(self.image_paths)}张图片")
            self.progress_label.setText(f"已选择{len(self.image_paths)}张图片")
    
    
    def load_and_display_image(self, img_path, label):
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("无法加载图像文件")
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(qimg))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败:\n{str(e)}")
    
    def start_detection(self):
        # 确保 detection_thread 已初始化
        if hasattr(self, 'detection_thread') and self.detection_thread and self.detection_thread.isRunning():
            QMessageBox.warning(self, "警告", "检测正在进行中，请等待完成或终止！")
            return
            
        if not self.template_paths or not self.image_paths:
            QMessageBox.warning(self, "警告", "请先加载模板图和检测图片！")
            return
            
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
            self.reflash_image_folder()
            # 确保创建新的 detection_thread
            self.detection_thread = DetectionThread(
                self,
                self.image_paths,
                self.current_output_dir,
                threshold_conditions,
                use_ai,
                combo_method,
                debug
            )
            
            self.detection_thread.progress_signal.connect(self.update_progress)
            self.detection_thread.finished_signal.connect(self.detection_finished)
            self.detection_thread.intermediate_result_signal.connect(self.handle_intermediate_result)
            
            self.btn_detect.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.btn_view_defects.setEnabled(False)
            self.progress_bar.setMaximum(len(self.image_paths))
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
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)
    
    def detection_finished(self, success, message):
        self.progress_label.setText(message)
        self.btn_detect.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_view_defects.setEnabled(True)

        # 自动删除原图逻辑
        if success and self.cb_auto_delete.isChecked():
            try:
                for img_path in self.image_paths:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                print("✅ 原图已自动删除。")
            except Exception as e:
                print(f"❌ 删除原图失败: {e}")

        if success:
            QMessageBox.information(self, "完成", "检测完成！")
        else:
            QMessageBox.warning(self, "警告", "检测未完成或失败！")
    
    def handle_intermediate_result(self, name, image):
        pass
    
    def get_threshold_conditions(self):
        conditions = {
            "二值化阈值": [],
            "RGB二值化": [],
            "面积阈值": [],
            "比例阈值": [],
            "灰度差阈值": [],
            "RGB值阈值": [],
            "区域平均亮度阈值": []
        }
        
        for widget in self.threshold_ranges:
            try:
                if widget.threshold_type == "二值化阈值":
                    low = int(widget.low_input.text())
                    high = int(widget.high_input.text())
                    if low > high:
                        low, high = high, low
                    conditions["二值化阈值"].append((low, high))
                
                elif widget.threshold_type == "RGB二值化":
                    r_min = int(widget.r_min.text())
                    r_max = int(widget.r_max.text())
                    g_min = int(widget.g_min.text())
                    g_max = int(widget.g_max.text())
                    b_min = int(widget.b_min.text())
                    b_max = int(widget.b_max.text())
                    
                    if r_min > r_max:
                        r_min, r_max = r_max, r_min
                    if g_min > g_max:
                        g_min, g_max = g_max, g_min
                    if b_min > b_max:
                        b_min, b_max = b_max, b_min
                    
                    conditions["RGB二值化"].append((r_min, r_max, g_min, g_max, b_min, b_max))
                
                elif widget.threshold_type == "面积阈值":
                    min_val = int(widget.min_input.text())
                    max_val = int(widget.max_input.text())
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                    conditions["面积阈值"].append((min_val, max_val))
                
                elif widget.threshold_type == "比例阈值":
                    min_ratio = float(widget.min_ratio_input.text())
                    max_ratio = float(widget.max_ratio_input.text())
                    if min_ratio > max_ratio:
                        min_ratio, max_ratio = max_ratio, min_ratio
                    conditions["比例阈值"].append((min_ratio, max_ratio))
                
                elif widget.threshold_type == "灰度差阈值":
                    threshold = int(widget.threshold_input.text())
                    conditions["灰度差阈值"].append(threshold)
                
                elif widget.threshold_type == "RGB值阈值":
                    r_min = int(widget.r_min.text())
                    r_max = int(widget.r_max.text())
                    g_min = int(widget.g_min.text())
                    g_max = int(widget.g_max.text())
                    b_min = int(widget.b_min.text())
                    b_max = int(widget.b_max.text())
                    ratio = float(widget.ratio_input.text())
                    
                    if r_min > r_max:
                        r_min, r_max = r_max, r_min
                    if g_min > g_max:
                        g_min, g_max = g_max, g_min
                    if b_min > b_max:
                        b_min, b_max = b_max, b_min
                    
                    conditions["RGB值阈值"].append((r_min, r_max, g_min, g_max, b_min, b_max, ratio))

                elif widget.threshold_type == "区域平均亮度阈值":
                    min_bright = int(widget.min_bright_input.text())
                    max_bright = int(widget.max_bright_input.text())
                    if min_bright > max_bright:
                        min_bright, max_bright = max_bright, min_bright
                    conditions["区域平均亮度阈值"].append((min_bright, max_bright))

            except ValueError:
                continue
        
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