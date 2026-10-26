#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""循迹 + 交通标志识别融合主程序（智能小车综合赛道）。

单摄像头 640x480：
  - 主线程 ~20Hz：LineDetector 逐帧循迹检测（CPU 开销小）；
  - 后台线程：YOLOX 11 类标志检测（416² 推理 50~300ms，不阻塞控制环）；
  - Arbiter 仲裁：循迹 PD / 标志任务记忆→执行区触发→动作执行 / 红灯抢占；
  - 底盘串口统一下发 (x, y, z)。

方向约定：x>0 前进；y>0 左移；逻辑 z>0=左转。--z-flip 为物理方向兜底。

用法:
    python run.py                          # 车上正常运行
    python run.py --debug                  # 显示融合调试窗口
    python run.py --no-drive --debug       # 无车台架验证（只打印指令）
    python run.py --port COM3 --camera 1   # 指定串口/摄像头
    python self_check.py                   # 无车离线断言
"""
import argparse
import logging
import os
import sys
import threading
import time

import cv2

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
if CASE_DIR not in sys.path:
    sys.path.insert(0, CASE_DIR)

from comm.chassis import ChassisController
from comm.usb_camera import DEFAULT_SCAN_DEVICES, USBCamera
from core.arbiter import Arbiter
from core.line_follower import FollowerController, LineDetector
from detection.traffic_sign_config import COCO_SIGN_MAP, DIS_THR, MODEL_PATH
from detection.traffic_sign_detector import TrafficSignDetector
from detection.yolox_utils import putText as zh_putText

logger = logging.getLogger('fused')


# =====================================================================
# 无车底盘（安全验证）
# =====================================================================
class NoDriveChassis:
    """只打印控制量，不访问串口，供安全验证识别与决策链路。"""

    def __init__(self):
        self.port = 'NO-DRIVE'
        self._last_command = None

    def send_speed(self, x_mms, y_mms, z_dps):
        command = (int(x_mms), int(y_mms), int(z_dps))
        if command != self._last_command:
            print('[NO-DRIVE] X={} Y={} Z={}'.format(*command))
            self._last_command = command
        return True

    def stop(self):
        return self.send_speed(0, 0, 0)

    def read_status(self):
        return None

    def close(self):
        pass


# =====================================================================
# 标志检测后台线程：latest-frame-in / latest-result-out
# =====================================================================
class SignDetectionWorker:
    """在后台线程中跑 YOLOX 推理，主控制环非阻塞读写。"""

    def __init__(self, detector):
        self.detector = detector
        self._lock = threading.Lock()
        self._frame = None
        self._frame_seq = 0
        self._result = None          # (frame_seq, sign, conf, dets)
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stopped = True

    def submit(self, frame):
        """提交最新帧（覆盖式，主线程每帧调用）。"""
        with self._lock:
            self._frame = frame
            self._frame_seq += 1

    def latest_result(self):
        with self._lock:
            return self._result

    def _loop(self):
        while not self._stopped:
            with self._lock:
                frame = self._frame
                seq = self._frame_seq
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                sign, conf, dets = self.detector.detect_signs(frame)
            except Exception:
                logger.exception('标志检测线程异常，本帧跳过')
                time.sleep(0.1)
                sign, conf, dets = None, 0.0, None
            with self._lock:
                self._result = (seq, sign, conf, dets)


# =====================================================================
# 工具函数
# =====================================================================
def get_detection_context(dets, frame_shape, confirmed_sign):
    """提取已确认标志在画面中的位置，供执行区触发与绕锥方向判断。"""
    if dets is None or frame_shape is None or not confirmed_sign:
        return None
    candidates = [det for det in dets
                  if COCO_SIGN_MAP.get(int(det[5])) == confirmed_sign]
    if not candidates:
        return None
    best = max(candidates, key=lambda item: float(item[4]))
    center_x = (float(best[0]) + float(best[2])) / 2.0
    bbox_h = float(best[3]) - float(best[1])
    return {
        'center_x_norm': center_x / max(float(frame_shape[1]), 1.0),
        'bbox_h_ratio': bbox_h / max(float(frame_shape[0]), 1.0),
        'bbox': tuple(float(value) for value in best[:4]),
    }


def to_physical(x, y, z_logic, z_flip):
    """逻辑速度 → 底盘物理速度的唯一边界。z_flip 时转向整体取反。"""
    z = -z_logic if z_flip else z_logic
    return int(x), int(y), int(z)


def reconnect_camera(camera, preferred_device):
    """摄像头掉线后重新扫描打开，兼容重插后 /dev/video 编号变化。"""
    camera.release()
    candidates = []
    for device in (preferred_device, camera.device, *DEFAULT_SCAN_DEVICES):
        if device is not None and device not in candidates:
            candidates.append(device)
    for device in candidates:
        if os.name != 'nt' and isinstance(device, int):
            if not os.path.exists('/dev/video%d' % device):
                continue
        if camera.open(device):
            return True
    return False


# =====================================================================
# 命令行参数
# =====================================================================
def parse_args():
    p = argparse.ArgumentParser(description='循迹 + 交通标志识别融合（智能小车综合赛道）')
    # 硬件
    p.add_argument('--camera', type=int, default=None, help='USB 摄像头编号（默认自动扫描 0~5）')
    p.add_argument('--port', type=str, default=None, help='底盘串口，如 COM3 / /dev/ttyACM0')
    p.add_argument('--baud', type=int, default=115200, help='串口波特率')
    p.add_argument('--no-drive', action='store_true', help='只打印指令，不连接底盘')
    p.add_argument('--debug', action='store_true', help='显示融合调试窗口')
    p.add_argument('--max-frames', type=int, default=None, help='最大运行帧数（调试用）')
    p.add_argument('--target-fps', type=int, default=20, help='控制环目标频率（默认 20Hz）')
    # 标志检测
    p.add_argument('--model', default=MODEL_PATH, help='11 类 ONNX 模型路径')
    p.add_argument('--confidence', type=float, default=DIS_THR, help='识别置信度阈值')
    p.add_argument('--confirm-frames', type=int, default=2, help='同类标志连续确认次数')
    p.add_argument('--roundabout-confidence', type=float, default=0.80,
                   help='大环岛标志独立置信度阈值；低于此值完全忽略（默认0.80）')
    p.add_argument('--uturn-confidence', type=float, default=0.75,
                   help='掉头标志独立置信度阈值；低于此值完全忽略（默认0.75）')
    p.add_argument('--turn-confidence', type=float, default=0.75,
                   help='左转/右转标志独立置信度阈值；低于此值完全忽略（默认0.75）')
    # 循迹视觉
    p.add_argument('--black', action='store_true', help='黑线白底（默认）')
    p.add_argument('--white', action='store_true', help='白线黑底')
    p.add_argument('--binary-mode', choices=('fixed', 'otsu', 'adaptive'), default='otsu')
    p.add_argument('--threshold', type=int, default=100, help='fixed 模式灰度阈值')
    p.add_argument('--adaptive-block', type=int, default=31)
    p.add_argument('--adaptive-c', type=float, default=8.0)
    p.add_argument('--roi-top', type=float, default=0.45)
    p.add_argument('--n-scan-rows', type=int, default=12)
    p.add_argument('--scan-start', type=float, default=0.25)
    p.add_argument('--crop-bottom', type=float, default=0.50)
    p.add_argument('--crop-top', type=float, default=0.75)
    p.add_argument('--track-half', type=float, default=70.0)
    p.add_argument('--work-width', type=int, default=320)
    p.add_argument('--corner-width', type=float, default=0.55, help='横条最小宽度占窗比例')
    p.add_argument('--cross-sym', type=float, default=0.35, help='十字路口对称判据')
    p.add_argument('--cross-min-ext', type=float, default=0.15,
                   help='十字判据短边伸出最小占图宽比例(防直角弯肘部误判 cross)')
    # 循迹控制
    p.add_argument('--speed', type=int, default=250, help='巡航速度 mm/s（协议限幅 ±300）')
    p.add_argument('--max-z', type=int, default=800)
    p.add_argument('--kp', type=float, default=10.0)
    p.add_argument('--kd', type=float, default=1.2)
    p.add_argument('--ka', type=float, default=3.5)
    p.add_argument('--err-alpha', type=float, default=0.6)
    p.add_argument('--z-rate', type=float, default=120.0)
    p.add_argument('--lost-hold', type=int, default=10)
    p.add_argument('--search-frames', type=int, default=15)
    p.add_argument('--startup-frames', type=int, default=5)
    p.add_argument('--ramp-frames', type=int, default=20)
    p.add_argument('--min-move-speed', type=int, default=0)
    # 转向执行（左右转复用直角弯三段式）
    p.add_argument('--corner-speed', type=int, default=40)
    p.add_argument('--corner-z', type=int, default=400)
    p.add_argument('--corner-min-frames', type=int, default=6)
    p.add_argument('--corner-max-frames', type=int, default=150)
    p.add_argument('--corner-exit-err', type=float, default=35.0)
    p.add_argument('--corner-exit-angle', type=float, default=30.0)
    p.add_argument('--corner-turn-sec', type=float, default=0.0,
                   help='转弯固定旋转段时长秒（0=自动按 900/corner-z 换算≈90°，与掉头标定同源；转多了调小，转少了调大）')
    # 融合行为
    p.add_argument('--slow-speed', type=int, default=140, help='限速段速度 mm/s')
    p.add_argument('--crosswalk-speed', type=int, default=100, help='人行横道低速通过速度 mm/s')
    p.add_argument('--crosswalk-sec', type=float, default=3.0, help='人行横道低速持续秒数')
    p.add_argument('--memory-hold', type=float, default=4.0, help='任务记忆保持秒数（超时丢弃）')
    p.add_argument('--cooldown', type=float, default=4.0, help='动作完成后同标志冷却秒数')
    p.add_argument('--approach-scale', type=float, default=0.5, help='记忆任务后接近执行区的减速比')
    p.add_argument('--exec-trigger', type=float, default=0.60,
                   help='横条 nearness 达到该值视为到达执行区（0~1）')
    p.add_argument('--no-match-corner-side', action='store_true',
                   help='转向触发不校验横条偏向（默认校验：左转要求横条偏左）')
    p.add_argument('--lead-sec', type=float, default=0.3, help='环岛的记忆延时触发秒数')
    p.add_argument('--turn-hold', type=float, default=20.0,
                   help='转向任务最长记忆保持秒数：只在路口(横条/岔口线索)触发转向，一直等不到路口才丢弃')
    p.add_argument('--lane-change-side', choices=('left', 'right'), default='left',
                   help='变道方向（按本轮公布规则选择）')
    p.add_argument('--lc-delay', type=float, default=3.0,
                   help='见过掉头标志后，确认十字路口再循迹多少秒开始平移')
    p.add_argument('--lc-max-sec', type=float, default=8.0,
                   help='平移最长秒数（一直找不到新线时强制回循迹的安全兜底）')
    # 雪糕筒封岔（专用状态机：确认雪糕筒即进入；锥在哪家岔，就走另一家岔）
    p.add_argument('--cone-trigger-h', type=float, default=0.25,
                   help='雪糕筒 bbox 高占画面比例达到该值视为已到岔口')
    p.add_argument('--cone-lead-sec', type=float, default=2.5,
                   help='看到雪糕筒后未等到岔口横条的兜底入岔秒数')
    p.add_argument('--cone-hold-sec', type=float, default=2.0,
                   help='接近期间雪糕筒消失超过该秒数按误检放弃')
    p.add_argument('--cone-branch-sec', type=float, default=5.0,
                   help='雪糕筒入岔循线总时长秒（偏长防早触发漏岔，过岔后仅低速循线）')
    p.add_argument('--cone-decide-sec', type=float, default=0.3,
                   help='确认雪糕筒后触发入岔前的判定稳定秒数（防变道刚结束姿态未稳时误判封堵侧）')
    p.add_argument('--cone-total', type=int, default=2,
                   help='全程雪糕筒数量，处理满后永久忽略路锥检测（0=关闭该功能）')
    p.add_argument('--cone-missing-timeout-sec', type=float, default=20.0,
                   help='锥桶进度停在1/2多久后兜底视为2/2（默认20s）')
    p.add_argument('--uturn-sec', type=float, default=7.2,
                   help='原地掉头时长秒（确认标志后 --uturn-delay 触发，总转角≈360°，实车标定值）')
    p.add_argument('--uturn-delay', type=float, default=2.0,
                   help='确认掉头标志后多少秒开始原地掉头（写死时间触发，不等横条）')
    p.add_argument('--uturn-follow-sec', type=float, default=4.5,
                   help='原地掉头结束后循迹多少秒执行右转（0=不右转，转完直接回循迹）')
    p.add_argument('--uturn-right', action='store_true', default=True, help='向右掉头（默认向右）')
    # 岔路（有转向标志时入岔；无标志直行通过岔口）
    p.add_argument('--turn-style', choices=('auto', 'branch', 'corner'), default='auto',
                   help='标志转向方式：auto=按触发线索自动选(默认，岔口→循线入岔，横条/兜底→三段式直角弯) branch=闭环循线入岔 corner=三段式直角弯')
    p.add_argument('--no-first-turn-right', dest='first_turn_right',
                   action='store_false', default=True,
                   help='关闭写死规则"启动后第一个转向标志(左/右)统一按右转处理"')
    p.add_argument('--branch-speed', type=int, default=120, help='入岔循线速度 mm/s')
    p.add_argument('--branch-sec', type=float, default=2.5, help='入岔动作总时长秒')
    p.add_argument('--branch-lost-z', type=int, default=150, help='入岔失线缓转角速度 度/s')
    p.add_argument('--branch-lost-sec', type=float, default=1.0, help='入岔失线超时秒')
    # Y 形岔口选路（60~85° 分叉；无标志时无条件走固定一侧）
    p.add_argument('--fork-side', choices=('left', 'right', 'none'), default='left',
                   help='Y 形岔口无条件选路：left=左岔(沿原方向,默认) right=右岔(新路线) none=关闭')
    p.add_argument('--fork-min-ratio', type=float, default=0.5,
                   help='岔口分支段宽至少占本行最宽段该比例才视为真分支(防噪点误选)')
    p.add_argument('--fork-hold-frames', type=int, default=15,
                   help='检测到岔口后保持选边并屏蔽横条误判的帧数(20Hz 下 15≈0.75s)')
    # 小环岛入口一次性右入（完成全部锥桶且静默后，反Y→正Y才允许触发）
    p.add_argument('--no-fork-right-once', action='store_true',
                   help='关闭环岛入口"反Y→正Y 一次性右入"(默认开启，全程仅触发一次)')
    p.add_argument('--fork-right-window', type=float, default=5.0,
                   help='反Y确认后见正Y的最大有效间隔（默认5.0s，无最小间隔）')
    p.add_argument('--fork-right-confirm-frames', type=int, default=3,
                   help='反Y和正Y各自需要连续确认的帧数（默认3帧）')
    p.add_argument('--fork-right-sec', type=float, default=3.0,
                   help='一次性右入的右侧选线/小幅右转保持秒数（默认3s）')
    p.add_argument('--fork-right-z', type=int, default=80,
                   help='小环岛右入保持期的最小右转角速度绝对值（实际输出z<=-80）')
    p.add_argument('--fork-right-cone-clear-sec', type=float, default=20.0,
                   help='处理完/兜底补全锥桶后连续未见锥桶多久才启用反Y→正Y右入（默认20s）')
    # 无标志直角弯自动转弯（只认单侧横条，岔口对称横条不触发）
    p.add_argument('--no-auto-corner', action='store_true',
                   help='关闭无标志直角弯自动转弯（岔口被误判单侧横条时用）')
    p.add_argument('--auto-corner-trigger', type=float, default=0.75,
                   help='自动直角弯横条 nearness 阈值(0~1，比标志转向更靠近才转)')
    p.add_argument('--auto-corner-frames', type=int, default=3,
                   help='自动直角弯连续确认帧数（防误触发）')
    # 环岛（入口左转 → 绕环，第1路口右转找线，第2路口左转出环）
    p.add_argument('--rb-loop-speed', type=int, default=120, help='绕环循迹速度 mm/s')
    p.add_argument('--rb-junc-trigger', type=float, default=0.5,
                   help='环上路口横条 nearness 计数门槛（第1/第2路口识别）')
    p.add_argument('--rb-memory-hold', type=float, default=40.0,
                   help='环岛标志任务等待入口横条的最长秒数')
    p.add_argument('--rb-first-straight-sec', type=float, default=2.0,
                   help='旧版兼容参数（第1路口现由右转找线参数接管）')
    p.add_argument('--rb-first-right-speed', type=int, default=100,
                   help='大环岛第1路口右转找线时的前进速度 mm/s')
    p.add_argument('--rb-first-right-z', type=int, default=160,
                   help='大环岛第1路口右转角速度绝对值（程序自动输出负值）')
    p.add_argument('--rb-first-right-min-sec', type=float, default=0.8,
                   help='大环岛第1路口固定右转的最短秒数')
    p.add_argument('--rb-first-right-max-sec', type=float, default=4.0,
                   help='大环岛第1路口连续右转最长秒数（超时停车等线，不恢复错误循迹）')
    p.add_argument('--rb-first-line-frames', type=int, default=4,
                   help='大环岛第1路口找到主线后连续稳定多少帧才恢复循迹')
    p.add_argument('--rb-first-line-error', type=float, default=45.0,
                   help='大环岛第1路口重捕获主线的最大横向误差 px')
    p.add_argument('--rb-first-line-angle', type=float, default=35.0,
                   help='大环岛第1路口重捕获主线的最大角度误差 deg')
    p.add_argument('--rb-junc-confirm-frames', type=int, default=3,
                   help='大环岛路口横条连续确认帧数（防单帧误计数）')
    p.add_argument('--rb-junc-clear-frames', type=int, default=10,
                   help='大环岛计数后连续无横条多少帧才允许重新计数')
    p.add_argument('--rb-junc-min-gap-sec', type=float, default=3.0,
                   help='大环岛相邻两次路口计数的最小间隔秒数（防同路口重复）')
    p.add_argument('--rb-max-sec', type=float, default=60.0,
                   help='大环岛动作最长秒数（安全兜底，默认60s）')
    p.add_argument('--z-flip', action='store_true',
                   help='物理转向与约定相反时整体取反（预期不需要，架空验证后再用）')
    return p.parse_args()


# =====================================================================
# 调试画面
# =====================================================================
def draw_debug(frame, det, dets, status, cmd, fps, arbiter, work_width=320):
    """合成调试窗口：相机帧 + 循迹 overlay + 检测框 + 二值图 + 状态条。"""
    disp_w = 480
    h, w = frame.shape[:2]
    scale = disp_w / w
    disp = cv2.resize(frame, (disp_w, int(h * scale)))

    # 标志检测框（最新一帧推理结果）
    if dets is not None and len(dets) > 0:
        for d in dets:
            if float(d[4]) < DIS_THR:
                continue
            x0, y0, x1, y1 = [int(v * scale) for v in d[:4]]
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
            name = COCO_SIGN_MAP.get(int(d[5]), '?')
            disp = zh_putText(disp, '%s %.0f%%' % (name, float(d[4]) * 100),
                              (x0, max(y0 - 20, 14)), color=(0, 255, 0), font_size=16)

    # 循迹 overlay（坐标从工作图换算到显示图）
    s = disp.shape[1] / float(work_width)
    cx0 = disp.shape[1] // 2
    cv2.line(disp, (cx0, 0), (cx0, disp.shape[0]), (255, 0, 0), 1)
    if det['is_valid']:
        for (x, y, bw) in det['points']:
            cv2.circle(disp, (int(x * s), int(y * s)), 3, (0, 255, 0), -1)
        if det['points']:
            y_top = min(p_[1] for p_ in det['points'])
            y_bot = max(p_[1] for p_ in det['points'])
            x_top = int((det['a'] * y_top + det['b']) * s)
            x_bot = int((det['a'] * y_bot + det['b']) * s)
            cv2.line(disp, (x_top, int(y_top * s)), (x_bot, int(y_bot * s)),
                     (0, 255, 0), 2)
        cx, cy = det['centroid']
        cv2.circle(disp, (int(cx * s), int(cy * s)), 6, (0, 255, 255), -1)
    corner = det.get('corner')
    if corner is not None:
        y = int(corner['row'] * s)
        half = int(corner['width'] * s / 2)
        cx_c = int(corner['cx'] * s)
        is_cross = corner['kind'] == 'cross'
        color = (255, 0, 255) if is_cross else (0, 0, 255)
        label = 'CROSS' if is_cross else ('CORNER-R' if corner['side'] > 0 else 'CORNER-L')
        cv2.line(disp, (cx_c - half, y), (cx_c + half, y), color, 3)
        cv2.putText(disp, '%s %.2f' % (label, corner['nearness']),
                    (max(cx_c - 40, 4), max(y - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if det.get('fork'):
        disp = zh_putText(disp, '岔口选边生效中', (6, 70),
                          color=(0, 200, 255), font_size=15)
    elif det.get('junction') == 'merge':
        disp = zh_putText(disp, '汇合口:守住当前线', (6, 70),
                          color=(255, 200, 0), font_size=15)

    # 二值图（翻转显示：黑线为黑）
    binary = det.get('binary')
    if binary is not None:
        import numpy as np
        bh, bw = binary.shape[:2]
        bdisp = cv2.bitwise_not(binary)
        bdisp = cv2.resize(bdisp, (disp_w, int(disp_w * bh / bw)))
        bdisp = cv2.cvtColor(bdisp, cv2.COLOR_GRAY2BGR)
        disp = np.vstack([disp, bdisp])

    # 状态条
    bar = disp.copy()
    cv2.rectangle(bar, (0, 0), (disp.shape[1], 52), (20, 24, 40), -1)
    disp = cv2.addWeighted(bar, 0.7, disp, 0.3, 0)
    task = status.get('pending_task')
    line1 = 'STATE=%s follower=%s cmd=(%d,%d,%d) fps=%.0f' % (
        status['state'], status.get('follower_state'), cmd[0], cmd[1], cmd[2], fps)
    cv2.putText(disp, line1, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 255), 1)
    if status.get('fork_right_used'):
        fork_gate = '已用'
    elif status.get('fork_right_armed'):
        fork_gate = '就绪'
    elif status.get('cone_handled', 0) < status.get('cone_total', 0):
        wait_sec = status.get('cone_completion_wait_remaining', 0.0)
        fork_gate = '等锥%.1fs' % wait_sec if wait_sec > 0 else '等锥'
    else:
        fork_gate = '静默%.1fs' % status.get('fork_right_quiet_remaining', 0.0)
    cone_progress = '%d/%d%s' % (
        status.get('cone_handled', 0), status.get('cone_total', 0),
        '(补)' if status.get('cone_completion_assumed') else '')
    lane_gate = '开' if status.get('uturn_sign_seen') else '锁'
    line2 = 'cruise=%d limited=%s 锥=%s 右入=%s 平移=%s' % (
        status['cruise_speed'], status['speed_limited'], cone_progress,
        fork_gate, lane_gate)
    if task:
        line2 += ' 待执行:%s' % task['sign']
    if status.get('current_action'):
        line2 += ' 正在:%s' % status['current_action']
    disp = zh_putText(disp, line2, (6, 42), color=(255, 191, 0), font_size=15)
    for index, log_line in enumerate(arbiter.recent_logs(2)):
        disp = zh_putText(disp, log_line[-40:], (6, disp.shape[0] - 26 + index * 16),
                          color=(148, 163, 184), font_size=13)
    return disp


# =====================================================================
# 主流程
# =====================================================================
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s %(name)s] %(message)s')

    polarity = 'white' if args.white else 'black'
    start_time = time.time()

    # 1. 摄像头
    camera = USBCamera(device=args.camera, width=640, height=480, fps=30)
    if not camera.open():
        logger.error('无法打开 USB 摄像头（已尝试 0~5），请检查连接')
        return 1
    logger.info('摄像头已打开: device=%s size=%s', camera.device, camera.actual_size)

    # 2. 底盘
    chassis = None
    worker = None
    try:
        if args.no_drive:
            chassis = NoDriveChassis()
            logger.warning('NO-DRIVE 安全模式：不会连接底盘或驱动车轮')
        else:
            logger.info('检测到串口: %s', ChassisController.list_ports())
            chassis = ChassisController(port=args.port, baudrate=args.baud)
            chassis.connect()
            logger.info('底盘已连接: %s @ %d', chassis.port, args.baud)

        # 3. 循迹检测器 + 控制器
        detector_line = LineDetector(
            work_width=args.work_width, roi_top_ratio=args.roi_top,
            n_scan_rows=args.n_scan_rows, scan_start_ratio=args.scan_start,
            crop_bottom_frac=args.crop_bottom, crop_top_frac=args.crop_top,
            track_half=args.track_half, polarity=polarity,
            binary_mode=args.binary_mode, fixed_threshold=args.threshold,
            adaptive_block=args.adaptive_block, adaptive_c=args.adaptive_c,
            corner_width_ratio=args.corner_width,
            fork_side={'left': -1, 'right': 1, 'none': 0}[args.fork_side],
            fork_min_ratio=args.fork_min_ratio,
            fork_hold_frames=args.fork_hold_frames)
        detector_line.cross_sym_ratio = float(args.cross_sym)
        detector_line.cross_min_ext_ratio = float(args.cross_min_ext)

        follower = FollowerController(
            max_z=args.max_z, kp=args.kp, kd=args.kd, ka=args.ka,
            err_alpha=args.err_alpha, z_rate_limit=args.z_rate,
            lost_hold=args.lost_hold, search_frames=args.search_frames,
            startup_frames=args.startup_frames, ramp_frames=args.ramp_frames,
            min_move_speed=args.min_move_speed,
            corner_speed=args.corner_speed, corner_z=args.corner_z,
            corner_min_frames=args.corner_min_frames,
            corner_max_frames=args.corner_max_frames,
            corner_exit_err=args.corner_exit_err,
            corner_exit_angle=args.corner_exit_angle,
            corner_turn_sec=args.corner_turn_sec)

        # 4. 标志检测器（后台线程）
        sign_detector = TrafficSignDetector(
            model_path=args.model, conf_thr=args.confidence,
            confirm_frames=max(1, args.confirm_frames),
            class_conf_thresholds={
                '环岛': args.roundabout_confidence,
                '掉头': args.uturn_confidence,
                '左转': args.turn_confidence,
                '右转': args.turn_confidence,
            })
        worker = SignDetectionWorker(sign_detector)
        worker.start()

        # 5. 仲裁器（line_detector 传入供岔路选线）
        arbiter = Arbiter(
            follower, line_detector=detector_line,
            cruise_speed=args.speed, slow_speed=args.slow_speed,
            crosswalk_speed=args.crosswalk_speed, crosswalk_sec=args.crosswalk_sec,
            memory_hold_sec=args.memory_hold, cooldown_sec=args.cooldown,
            approach_scale=args.approach_scale, exec_trigger=args.exec_trigger,
            match_corner_side=not args.no_match_corner_side,
            lead_sec=args.lead_sec, lane_side=args.lane_change_side,
            lc_delay_sec=args.lc_delay, lc_max_sec=args.lc_max_sec,
            cone_trigger_h=args.cone_trigger_h,
            cone_lead_sec=args.cone_lead_sec, cone_hold_sec=args.cone_hold_sec,
            cone_branch_sec=args.cone_branch_sec, cone_total=args.cone_total,
            cone_decide_sec=args.cone_decide_sec,
            uturn_sec=args.uturn_sec, uturn_right=args.uturn_right,
            uturn_delay_sec=args.uturn_delay,
            uturn_follow_sec=args.uturn_follow_sec,
            turn_hold_sec=args.turn_hold,
            turn_style=args.turn_style,
            first_turn_right=args.first_turn_right,
            branch_speed=args.branch_speed, branch_sec=args.branch_sec,
            branch_lost_z=args.branch_lost_z, branch_lost_sec=args.branch_lost_sec,
            auto_corner=not args.no_auto_corner,
            auto_corner_trigger=args.auto_corner_trigger,
            auto_corner_frames=args.auto_corner_frames,
            rb_loop_speed=args.rb_loop_speed,
            rb_junc_trigger=args.rb_junc_trigger, rb_max_sec=args.rb_max_sec,
            fork_right_once=not args.no_fork_right_once,
            fork_right_window_sec=args.fork_right_window,
            fork_right_confirm_frames=args.fork_right_confirm_frames,
            fork_right_sec=args.fork_right_sec,
            fork_right_z=args.fork_right_z,
            fork_right_cone_clear_sec=args.fork_right_cone_clear_sec,
            cone_missing_timeout_sec=args.cone_missing_timeout_sec,
            rb_memory_hold_sec=args.rb_memory_hold,
            rb_first_straight_sec=args.rb_first_straight_sec,
            rb_junc_confirm_frames=args.rb_junc_confirm_frames,
            rb_junc_clear_frames=args.rb_junc_clear_frames,
            rb_junc_min_gap_sec=args.rb_junc_min_gap_sec,
            rb_first_right_speed=args.rb_first_right_speed,
            rb_first_right_z=args.rb_first_right_z,
            rb_first_right_min_sec=args.rb_first_right_min_sec,
            rb_first_right_max_sec=args.rb_first_right_max_sec,
            rb_first_line_frames=args.rb_first_line_frames,
            rb_first_line_error=args.rb_first_line_error,
            rb_first_line_angle=args.rb_first_line_angle,
            roundabout_confidence=args.roundabout_confidence,
            uturn_confidence=args.uturn_confidence,
            turn_confidence=args.turn_confidence)
        logger.info('初始化完成: 极性=%s 巡航=%d 限速=%d z_flip=%s 岔口选路=%s',
                    polarity, args.speed, args.slow_speed, args.z_flip,
                    args.fork_side)

        frame_interval = 1.0 / max(1, args.target_fps)
        last_t = time.time()
        last_det_seq = 0
        latest_dets = None
        frame_count = 0
        no_frame_count = 0
        reconnect_attempts = 0
        last_reconnect_try = 0.0
        send_fail = 0
        fps = 0.0
        fps_n = 0
        fps_t = time.time()
        status_t = 0.0
        cmd = (0, 0, 0)
        status = arbiter.status()

        while True:
            loop_start = time.time()
            now = loop_start
            dt = max(min(now - last_t, 0.5), 1e-3)
            last_t = now

            frame = camera.read()
            if frame is None:
                no_frame_count += 1
                # 断流约 1 秒后开始自动重连（车辆已由"禁止盲走"逻辑保持停车），
                # 每秒试一次，连续 15 次失败才放弃退出。
                if (no_frame_count >= 20
                        and now - last_reconnect_try >= 1.0):
                    last_reconnect_try = now
                    reconnect_attempts += 1
                    if reconnect_attempts > 15:
                        logger.error('摄像头重连 %d 次均失败，停车退出',
                                     reconnect_attempts - 1)
                        chassis.stop()
                        break
                    logger.warning('相机断流，第 %d 次重连（车辆保持停车）',
                                   reconnect_attempts)
                    if reconnect_camera(camera, args.camera):
                        logger.info('摄像头重连成功: device=%s size=%s',
                                    camera.device, camera.actual_size)
                        no_frame_count = 0
                        reconnect_attempts = 0
                        # 重新识别后才允许执行标志动作；清循迹预测窗强制全窗重捕获
                        sign_detector.reset()
                        detector_line._prev_cx = None
            else:
                no_frame_count = 0
                reconnect_attempts = 0
                worker.submit(frame)

            det = detector_line.process(frame) if frame is not None else \
                detector_line._empty_result()

            # 后台检测结果：只对新序号产生一次 sign_event（去重）
            sign_event = None
            result = worker.latest_result()
            if result is not None and result[0] != last_det_seq:
                last_det_seq = result[0]
                _, sign, conf, dets = result
                latest_dets = dets
                if sign is not None and frame is not None:
                    context = get_detection_context(dets, frame.shape, sign)
                    sign_event = {'sign': sign, 'conf': conf, 'context': context}
                    logger.info('确认标志: %s (%.1f%%)', sign, conf * 100.0)

            x, y, z_logic, status = arbiter.update(dt, det, sign_event, now)
            cmd = to_physical(x, y, z_logic, args.z_flip)
            if frame is None and status['state'] != 'executing':
                # 相机失效禁止盲走：开环执行器有界可继续，其余状态立即停车
                cmd = (0, 0, 0)

            if chassis.send_speed(*cmd):
                send_fail = 0
            else:
                send_fail += 1
                if send_fail >= 10:
                    logger.error('串口发送持续失败(%d次)，停车退出', send_fail)
                    chassis.stop()
                    break

            if args.debug and frame is not None:
                disp = draw_debug(frame, det, latest_dets, status, cmd, fps,
                                  arbiter, work_width=args.work_width)
                cv2.imshow('fused (line + sign)', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info('调试窗口按 q 退出')
                    break

            # 帧率控制与统计
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            fps_n += 1
            if time.time() - fps_t >= 1.0:
                fps = fps_n / (time.time() - fps_t)
                fps_n = 0
                fps_t = time.time()

            if time.time() - status_t >= 1.0:
                status_t = time.time()
                real = ''
                st = chassis.read_status()
                if st:
                    real = ' 实速x/y=%+d/%+d 实转z=%+.0f°/s 电压=%.1fV' % (
                        st['real_x'], st['real_y'],
                        st['real_z'] * 57.3, st['battery_voltage'])
                logger.info('状态=%s/%s err=%.1fpx 角=%.1f° cmd=(%d,%d,%d) '
                            'cruise=%d FPS=%.0f 失帧=%d%s',
                            status['state'], status.get('follower_state'),
                            det['error_px'], det['angle_deg'],
                            cmd[0], cmd[1], cmd[2], status['cruise_speed'],
                            fps, no_frame_count, real)

            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                logger.info('达到最大帧数 %d，退出', args.max_frames)
                break

    except KeyboardInterrupt:
        logger.info('用户中断')
    except Exception:
        logger.exception('运行异常')
        return 1
    finally:
        cv2.destroyAllWindows()
        if worker is not None:
            worker.stop()
        if chassis is not None:
            chassis.stop()
            chassis.close()
        camera.release()
        logger.info('程序已安全退出，运行 %.1fs', time.time() - start_time)
    return 0


if __name__ == '__main__':
    sys.exit(main())
