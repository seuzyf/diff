# -- coding: utf-8 --
import sys
import os
import threading
import time
import numpy as np
import cv2
import traceback
from ctypes import *

# *** SDK导入逻辑 (保持不变) ***
try:
    sdk_path = None
    if os.name == 'nt': # 仅限 Windows
        sdk_path_env = os.getenv('MVCAM_COMMON_RUNENV')

        if sdk_path_env and os.path.exists(os.path.join(sdk_path_env, "Samples/Python/MvImport")):
            sdk_path = os.path.join(sdk_path_env, "Samples/Python/MvImport")
            print(f"HIK SDK 路径 (来自环境变量): {sdk_path}")
        else:
            print("警告: 环境变量 'MVCAM_COMMON_RUNENV' 未设置或无效。")
            default_path = "C:/Program Files (x86)/MVS/Development/Samples/Python/MvImport"
            if not os.path.exists(default_path):
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

# 如果导入失败，定义虚假的占位符类
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
    PixelType_Gvsp_BGR8_Packed = 0x02180015 # --- 新增: BGR格式 ---
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

from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker # --- 新增 QMutex ---
# from PyQt5.QtGui import QImage # 不再需要 QImage

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

            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
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
    # --- 修改: 发射 numpy 数组 (BGR格式) ---
    new_frame_signal = pyqtSignal(np.ndarray)
    frame_captured_signal = pyqtSignal(str, int, int) # 保存路径, 宽, 高
    error_signal = pyqtSignal(str)

    def __init__(self, cam, parent=None):
        super().__init__(parent)
        self.cam = cam
        self._is_running = True
        self._capture_flag = False
        self._mutex = QMutex() # --- 新增: 互斥锁保护标志位 ---

        self.temp_dir = os.path.join("output", "temp_captures")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.pDataForSaveImage = None # 存储用于保存的原始数据
        self.stFrameInfoForSave = None # 存储用于保存的帧信息

        # --- 修改: 缓冲区和转换参数 ---
        self.p_convert_buf = None # 用于存储转换后的BGR数据
        self.n_convert_buf_size = 0
        self.stConvertParam = MV_CC_PIXEL_CONVERT_PARAM() # 使用正确的转换参数结构体

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

            # 为 BGR 格式分配缓冲区
            self.n_convert_buf_size = self.width * self.height * 3
            self.p_convert_buf = (c_ubyte * self.n_convert_buf_size)()
            print(f"为预览分配BGR缓冲区: {self.width}x{self.height}")

        except Exception as e:
            self.error_signal.emit(f"相机线程初始化失败: {e}")
            self._is_running = False

    def _save_frame(self, target_filename):
        """
        使用 MV_CC_SaveImageToFileEx 保存捕获的帧
        """
        # --- 修改: 使用 self.pDataForSaveImage 和 self.stFrameInfoForSave ---
        if self.pDataForSaveImage is None or self.stFrameInfoForSave is None:
             self.error_signal.emit("保存失败: 捕获的数据无效")
             return None

        try:
            save_path = os.path.join(self.temp_dir, target_filename)
            c_file_path = save_path.encode('gbk') # 使用 gbk 编码支持中文路径

            stSaveParam = MV_SAVE_IMAGE_TO_FILE_PARAM_EX()
            stSaveParam.enPixelType = self.stFrameInfoForSave.enPixelType # 使用捕获时的像素格式
            stSaveParam.nWidth = self.stFrameInfoForSave.nWidth
            stSaveParam.nHeight = self.stFrameInfoForSave.nHeight
            stSaveParam.nDataLen = self.stFrameInfoForSave.nFrameLen
            stSaveParam.pData = self.pDataForSaveImage # 使用捕获时的原始数据指针
            stSaveParam.enImageType = MV_Image_Png # 保存为 PNG
            stSaveParam.pcImagePath = create_string_buffer(c_file_path, 256)
            stSaveParam.iMethodValue = 1
            stSaveParam.nQuality = 90 # PNG质量设置通常被忽略，但保留

            print(f"正在保存图像至: {save_path}")
            mv_ret = self.cam.MV_CC_SaveImageToFileEx(stSaveParam)

            if mv_ret != 0:
                self.error_signal.emit(f"保存图像失败! MV_CC_SaveImageToFileEx ret[0x{mv_ret:x}]")
                return None
            else:
                print("保存图像成功!")
                # 返回路径、宽度、高度
                return save_path, self.stFrameInfoForSave.nWidth, self.stFrameInfoForSave.nHeight

        except Exception as e:
            self.error_signal.emit(f"保存图像时发生Python错误: {e}")
            traceback.print_exc()
            return None
        finally:
            # 清理保存用的数据
            self.pDataForSaveImage = None
            self.stFrameInfoForSave = None

    def run(self):
        if not self._is_running:
            return

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            self.error_signal.emit(f"开始取流失败! ret[0x{ret:x}]")
            return

        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        while True:
            # --- 使用互斥锁保护 self._is_running 的读取 ---
            with QMutexLocker(self._mutex):
                if not self._is_running:
                    break
            # --- 结束修改 ---

            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret == 0:
                info = stOutFrame.stFrameInfo
                pData = stOutFrame.pBufAddr

                # --- 预览转换 ---
                if self.p_convert_buf: # 仅当BGR缓冲区分配成功时
                    try:
                        self.stConvertParam.nWidth = info.nWidth
                        self.stConvertParam.nHeight = info.nHeight
                        self.stConvertParam.pSrcData = pData
                        self.stConvertParam.nSrcDataLen = info.nFrameLen
                        self.stConvertParam.enSrcPixelType = info.enPixelType
                        # --- 目标格式改为 BGR ---
                        self.stConvertParam.enDstPixelType = PixelType_Gvsp_BGR8_Packed
                        self.stConvertParam.pDstBuffer = self.p_convert_buf
                        self.stConvertParam.nDstBufferSize = self.n_convert_buf_size

                        ret_convert = self.cam.MV_CC_ConvertPixelType(self.stConvertParam)

                        if ret_convert == 0:
                            # --- 转换为 Numpy 数组 ---
                            # 注意 ctypes 指针到 numpy 的转换
                            img_buff = (c_ubyte * self.n_convert_buf_size).from_address(addressof(self.p_convert_buf))
                            # 创建一个 numpy 数组视图，无需复制内存
                            img_bgr = np.frombuffer(img_buff, dtype=np.uint8).reshape(info.nHeight, info.nWidth, 3)
                            # --- 发射 numpy 数组 ---
                            self.new_frame_signal.emit(img_bgr)
                        else:
                             # 如果转换失败，尝试直接处理 Mono8
                             if info.enPixelType == PixelType_Gvsp_Mono8:
                                 img_buff_mono = (c_ubyte * info.nFrameLen).from_address(addressof(pData))
                                 img_mono = np.frombuffer(img_buff_mono, dtype=np.uint8).reshape(info.nHeight, info.nWidth)
                                 # 转换为 BGR
                                 img_bgr = cv2.cvtColor(img_mono, cv2.COLOR_GRAY2BGR)
                                 self.new_frame_signal.emit(img_bgr)
                             # else: # 其他不支持的格式暂时不处理
                             #    print(f"不支持的像素格式用于直接预览: {info.enPixelType}")


                    except Exception as e:
                        print(f"预览转换时出错: {e}")
                        traceback.print_exc() # 打印详细错误

                # --- 捕获 ---
                with QMutexLocker(self._mutex):
                    capture_now = self._capture_flag
                    if capture_now:
                        self._capture_flag = False # 重置标志位
                        # --- 存储原始数据和信息用于保存 ---
                        # 需要复制数据，因为缓冲区会被相机覆盖
                        # 注意：这里假设 nFrameLen 不会超过一个合理的大小
                        try:
                           # 确保pData有效且长度合理
                           if pData is not None and info.nFrameLen > 0 and info.nFrameLen < (100 * 1024 * 1024): # 限制最大100MB
                                self.pDataForSaveImage = (c_ubyte * info.nFrameLen)()
                                memmove(self.pDataForSaveImage, pData, info.nFrameLen) # 复制内存
                                self.stFrameInfoForSave = info # 复制结构体
                           else:
                               self.pDataForSaveImage = None
                               self.stFrameInfoForSave = None
                               print(f"警告: 无效的帧数据用于保存。pData: {pData}, FrameLen: {info.nFrameLen}")
                        except Exception as copy_e:
                            self.pDataForSaveImage = None
                            self.stFrameInfoForSave = None
                            print(f"复制帧数据时出错: {copy_e}")
                        # --- 结束存储 ---
                        self._is_running = False # 准备退出循环

                # --- 如果捕获了，则尝试保存并退出 ---
                if capture_now and self.pDataForSaveImage:
                    filename = f"capture_{int(time.time())}.png"
                    save_result = self._save_frame(filename)
                    if save_result:
                        save_path, width, height = save_result
                        self.frame_captured_signal.emit(save_path, width, height)
                    # 不论保存成功与否，都要释放缓冲区并退出
                    self.cam.MV_CC_FreeImageBuffer(stOutFrame)
                    break # 跳出 while 循环

                # 正常释放缓冲区
                ret_free = self.cam.MV_CC_FreeImageBuffer(stOutFrame)
                if ret_free != 0:
                    # 如果释放失败，可能需要停止并重启流
                    print(f"释放缓冲区失败! ret[0x{ret_free:x}]")
                    # break # 考虑是否需要退出

            else:
                 # 获取图像失败，短暂等待
                 # print(f"获取图像失败 ret[0x{ret:x}]") # 减少打印
                 time.sleep(0.01)

        # 循环结束后停止取流
        self.cam.MV_CC_StopGrabbing()
        print("相机取流线程已停止")

    def stop(self):
        # --- 使用互斥锁保护 self._is_running 的写入 ---
        with QMutexLocker(self._mutex):
            self._is_running = False

    def capture_and_stop(self):
        # --- 使用互斥锁保护 self._capture_flag 的写入 ---
        with QMutexLocker(self._mutex):
            self._capture_flag = True
            # 不需要在这里设置 is_running = False，run循环会处理