# -- coding: utf-8 --
import sys
import os
import threading
import time
import numpy as np
import cv2
import traceback
from ctypes import *

# *** 修复2: 增强的SDK导入逻辑 ***
try:
    sdk_path = None
    if os.name == 'nt': # 仅限 Windows
        sdk_path_env = os.getenv('MVCAM_COMMON_RUNENV')
        
        if sdk_path_env and os.path.exists(os.path.join(sdk_path_env, "Samples/Python/MvImport")):
            sdk_path = os.path.join(sdk_path_env, "Samples/Python/MvImport")
            print(f"HIK SDK 路径 (来自环境变量): {sdk_path}")
        else:
            # 环境变量未设置或无效，尝试默认路径
            print("警告: 环境变量 'MVCAM_COMMON_RUNENV' 未设置或无效。")
            default_path = "C:/Program Files (x86)/MVS/Development/Samples/Python/MvImport"
            if not os.path.exists(default_path):
                 # 尝试 MVS_ROOT (MVS 4.x+)
                 mvs_root = os.getenv('MVS_ROOT', "C:/Program Files (x86)/MVS")
                 default_path = os.path.join(mvs_root, "Development/Samples/Python/MvImport")

            if os.path.exists(default_path):
                sdk_path = default_path
                print(f"HIK SDK 路径 (来自默认值): {sdk_path}")
            else:
                print(f"错误: HIK SDK MvImport 路径未找到于: {default_path}")
                print("请安装MVS并设置 MVCAM_COMMON_RUNENV 环境变量。")
    
    if sdk_path and sdk_path not in sys.path:
        sys.path.append(sdk_path)
        
    from MvCameraControl_class import *
    print(f"HIK SDK MvImport 加载成功")
    HIK_IMPORT_SUCCESS = True

except ImportError as e:
    print(f"错误: 无法导入 MvCameraControl_class。 {e}")
    HIK_IMPORT_SUCCESS = False
except Exception as e:
    print(f"加载 HIK SDK 时发生未知错误: {e}")
    HIK_IMPORT_SUCCESS = False

# 如果导入失败，定义虚假的占位符类，以防止主程序崩溃
if not HIK_IMPORT_SUCCESS:
    class MvCamera:
        def __init__(self): pass
        @staticmethod
        def MV_CC_Initialize(): return -1
        @staticmethod
        def MV_CC_Finalize(): pass
        @staticmethod
        def MV_CC_EnumDevices(t, l): return -1
        def MV_CC_CreateHandle(self, d): return -1
        def MV_CC_OpenDevice(self, a, b): return -1
        def MV_CC_GetOptimalPacketSize(self): return -1
        def MV_CC_SetIntValue(self, n, v): return -1
        def MV_CC_SetEnumValue(self, n, v): return -1
        def MV_CC_GetIntValue(self, n, v): return -1
        def MV_CC_StopGrabbing(self): return -1
        def MV_CC_CloseDevice(self): return -1
        def MV_CC_DestroyHandle(self): pass
        def MV_CC_StartGrabbing(self): return -1
        def MV_CC_GetImageBuffer(self, f, t): return -1
        def MV_CC_ConvertPixelType(self, p): return -1
        def MV_CC_FreeImageBuffer(self, f): return -1
        def MV_CC_SaveImageToFileEx(self, p): return -1

    MV_GIGE_DEVICE = 1
    MV_USB_DEVICE = 2
    PixelType_Gvsp_RGB8_Packed = 0x02080016
    PixelType_Gvsp_Mono8 = 0x01080001
    MV_TRIGGER_MODE_OFF = 0
    MV_ACCESS_Exclusive = 1
    MV_Image_Png = 4
    
    class MV_FRAME_OUT(Structure): pass
    class MV_CC_DEVICE_INFO_LIST(Structure): pass
    class MVCC_INTVALUE(Structure): pass
    class MV_SAVE_IMAGE_TO_FILE_PARAM_EX(Structure): pass
    class MV_CC_DEVICE_INFO(Structure): pass
    class POINTER(object): pass
    def cast(a,b): return type('dummy', (object,), {'contents': None})()
    def byref(a): pass
    def memset(a,b,c): pass
    def sizeof(a): return 0
    def create_string_buffer(a,b=None): return None
    class c_ubyte(object): pass

# --- 以上是导入和占位符 ---

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

class CameraManager:
    """封装SDK初始化和设备枚举"""
    def __init__(self):
        self.sdk_initialized = False
        if not HIK_IMPORT_SUCCESS:
            print("SDK未加载，CameraManager无法初始化。")
            return
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            print(f"SDK 初始化失败! ret[0x{ret:x}]")
        else:
            self.sdk_initialized = True
            print("SDK 初始化成功")

    def list_devices(self):
        """枚举所有可用的GIGE和USB设备"""
        if not self.sdk_initialized:
            return [], None

        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print(f"枚举设备失败! ret[0x{ret:x}]")
            return [], None

        if deviceList.nDeviceNum == 0:
            print("未找到设备")
            return [], None

        print(f"找到 {deviceList.nDeviceNum} 个设备:")
        devices = []
        for i in range(deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            display_name = ""
            
            # 必须检查nTLayerType
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                # 联合体 (union) 的访问方式
                strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName if c != 0])
                display_name = f"[GIGE] {strModeName}"
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                strModeName = ''.join([chr(c) for c in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if c != 0])
                display_name = f"[USB] {strModeName}"
            
            if display_name:
                print(f"  {i}: {display_name}")
                devices.append(display_name)
        
        return devices, deviceList

    def connect(self, deviceList, index):
        """连接到指定索引的设备"""
        if not self.sdk_initialized or not deviceList or index >= deviceList.nDeviceNum:
            return None

        cam = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print(f"创建句柄失败! ret[0x{ret:x}]")
            return None

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print(f"打开设备失败! ret[0x{ret:x}]")
            cam.MV_CC_DestroyHandle()
            return None

        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print(f"设置包大小失败! ret[0x{ret:x}]")
            else:
                print(f"获取包大小失败! ret[0x{nPacketSize:x}]")

        ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print(f"设置触发模式失败! ret[0x{ret:x}]")
            
        print("相机连接成功")
        return cam

    def disconnect(self, cam):
        """断开相机连接并销毁句柄"""
        if cam and HIK_IMPORT_SUCCESS:
            cam.MV_CC_StopGrabbing()
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            print("相机已断开")

    def __del__(self):
        """反初始化SDK"""
        if self.sdk_initialized:
            MvCamera.MV_CC_Finalize()
            print("SDK 已反初始化")

class CameraThread(QThread):
    """在独立线程中抓取相机图像"""
    new_frame_signal = pyqtSignal(QImage)
    frame_captured_signal = pyqtSignal(str, int, int)
    error_signal = pyqtSignal(str)

    def __init__(self, cam, parent=None):
        super().__init__(parent)
        self.cam = cam
        self._is_running = True
        self._capture_flag = False
        
        self.temp_dir = os.path.join("output", "temp_captures")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.p_rgb_buffer = None
        
        try:
            stWidth = MVCC_INTVALUE()
            stHeight = MVCC_INTVALUE()
            ret = self.cam.MV_CC_GetIntValue("Width", stWidth)
            if ret != 0: raise Exception(f"获取宽度失败! ret[0x{ret:x}]")
            ret = self.cam.MV_CC_GetIntValue("Height", stHeight)
            if ret != 0: raise Exception(f"获取高度失败! ret[0x{ret:x}]")
            
            self.width = stWidth.nCurValue
            self.height = stHeight.nCurValue
            
            if self.width <= 0 or self.height <= 0:
                raise Exception(f"无效的图像尺寸: {self.width}x{self.height}")

            self.n_rgb_buffer_size = self.width * self.height * 3
            self.p_rgb_buffer = (c_ubyte * self.n_rgb_buffer_size)()
            print(f"为预览分配RGB缓冲区: {self.width}x{self.height}")
            
            self.stConvertParam = MV_SAVE_IMAGE_PARAM_EX()
            self.stConvertParam.enPixelType = PixelType_Gvsp_RGB8_Packed
            self.stConvertParam.nWidth = self.width
            self.stConvertParam.nHeight = self.height
            self.stConvertParam.pData = self.p_rgb_buffer
            self.stConvertParam.nDataLen = self.n_rgb_buffer_size

        except Exception as e:
            self.error_signal.emit(f"相机线程初始化失败: {e}")
            self._is_running = False

    def _save_frame(self, stOutFrame, target_filename):
        """
        使用 MV_CC_SaveImageToFileEx 保存原始帧
        """
        try:
            save_path = os.path.join(self.temp_dir, target_filename)
            c_file_path = save_path.encode('ascii') 
            
            stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
            stSaveParam.enPixelType = stOutFrame.stFrameInfo.enPixelType
            stSaveParam.nWidth = stOutFrame.stFrameInfo.nWidth
            stSaveParam.nHeight = stOutFrame.stFrameInfo.nHeight
            stSaveParam.nDataLen = stOutFrame.stFrameInfo.nFrameLen
            stSaveParam.pData = stOutFrame.pBufAddr
            stSaveParam.enImageType = MV_Image_Png
            stSaveParam.pcImagePath = create_string_buffer(c_file_path, 256) # 分配足够空间
            stSaveParam.iMethodValue = 1
            stSaveParam.nQuality = 90
            
            print(f"正在保存图像至: {save_path}")
            mv_ret = self.cam.MV_CC_SaveImageToFileEx(stSaveParam)
            
            if mv_ret != 0:
                self.error_signal.emit(f"保存图像失败! MV_CC_SaveImageToFileEx ret[0x{mv_ret:x}]")
                return None
            else:
                print("保存图像成功!")
                return save_path
        
        except Exception as e:
            self.error_signal.emit(f"保存图像时发生Python错误: {e}")
            traceback.print_exc()
            return None

    def run(self):
        if not self._is_running:
            return

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.error_signal.emit(f"开始取流失败! ret[0x{ret:x}]")
            return

        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        while self._is_running:
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret == 0:
                info = stOutFrame.stFrameInfo
                
                # --- 预览转换 ---
                if self.p_rgb_buffer: # 仅当RGB缓冲区分配成功时
                    try:
                        self.stConvertParam.pBufAddr = stOutFrame.pBufAddr
                        self.stConvertParam.nBufLen = info.nFrameLen
                        self.stConvertParam.stFrameInfo = info

                        ret_convert = self.cam.MV_CC_ConvertPixelType(self.stConvertParam)
                        
                        if ret_convert == 0:
                            img_buf = (c_ubyte * self.stConvertParam.nDataLen).from_address(addressof(self.p_rgb_buffer))
                            q_img = QImage(img_buf, info.nWidth, info.nHeight, info.nWidth * 3, QImage.Format_RGB888)
                            self.new_frame_signal.emit(q_img.copy())
                        elif info.enPixelType == PixelType_Gvsp_Mono8:
                             q_img = QImage(stOutFrame.pBufAddr, info.nWidth, info.nHeight, info.nWidth, QImage.Format_Grayscale8)
                             self.new_frame_signal.emit(q_img.copy())
                        
                    except Exception as e:
                        print(f"预览转换时出错: {e}")

                # --- 捕获 ---
                if self._capture_flag:
                    self._is_running = False
                    self._capture_flag = False
                    
                    filename = f"capture_{int(time.time())}.png"
                    saved_path = self._save_frame(stOutFrame, filename)
                    
                    if saved_path:
                        self.frame_captured_signal.emit(saved_path, info.nWidth, info.nHeight)

                self.cam.MV_CC_FreeImageBuffer(stOutFrame)

            else:
                time.sleep(0.01)
        
        self.cam.MV_CC_StopGrabbing()
        print("取流线程已停止")

    def stop(self):
        self._is_running = False

    def capture_and_stop(self):
        self._capture_flag = True