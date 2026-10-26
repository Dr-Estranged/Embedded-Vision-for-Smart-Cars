#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""底盘串口驱动（本地组件，可独立运行）。

协议与 qt_car_monitor.py（实验1.3）完全一致：
  - 指令帧(11 字节): 0x7B 0x00 0x00 [x_hi x_lo] [y_hi y_lo] [z_hi z_lo] BCC 0x7D
      BCC = 第0~8字节按位异或；
      x/y: 前进/横移  单位 mm/s，限幅 ±300；
      z  : 转向      单位 度/s，限幅 ±1500。
  - 方向约定: x>0 前进 / x<0 后退；y>0 左移 / y<0 右移；z>0 左转 / z<0 右转。
  - 状态帧(24 字节): 0x7B ... [BCC] 0x7D，BCC = 第0~21字节异或，位于第22字节。

用法示例：
    chassis = ChassisController()
    chassis.connect()                    # 自动识别串口
    chassis.send_speed(150, 0, 300)      # 前进150mm/s 并左转300度/s (z>0=左转)
    chassis.send_speed(150, 0, -300)     # 前进150mm/s 并右转300度/s (z<0=右转)
    chassis.send_ros_vel(0.15, -0.5)     # 或使用 ROS 风格速度(线性m/s, 角速度rad/s)
    chassis.stop()
    chassis.close()
"""
import math
import time

import serial
import serial.tools.list_ports

CMD_HEAD = 0x7B
CMD_TAIL = 0x7D


class ChassisController:
    """通过标准串口控制履带/轮式底盘。"""

    def __init__(self, port=None, baudrate=115200, timeout=0.1,
                 x_limit=300, y_limit=300, z_limit=1500):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.limits = {'x': x_limit, 'y': y_limit, 'z': z_limit}
        self.ser = None
        self._rx = bytearray()

    # ------------------------------------------------------------------
    # 串口发现
    # ------------------------------------------------------------------
    @staticmethod
    def list_ports():
        """返回 (设备名, 描述) 列表。"""
        return [(p.device, p.description)
                for p in serial.tools.list_ports.comports()]

    @staticmethod
    def auto_select_port():
        """优先返回常见底盘串口(/dev/ttyACM0~10 或 /dev/ttyUSB*)，否则返回第一个可用串口。"""
        ports = serial.tools.list_ports.comports()
        if not ports:
            return None
        # 优先匹配常见底盘设备名（Linux 板卡典型环境，ACM0~ACM10 逐个尝试）
        preferred = ['/dev/ttyACM%d' % i for i in range(11)]
        preferred += ['/dev/ttyUSB0', '/dev/ttyUSB1']
        for name in preferred:
            for p in ports:
                if p.device == name:
                    return p.device
        for p in ports:
            if p.vid is not None:
                return p.device
        return ports[0].device

    # ------------------------------------------------------------------
    # 连接
    # ------------------------------------------------------------------
    @property
    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    def connect(self, port=None):
        """连接底盘串口，port 为空时自动识别。"""
        port = port or self.port or self.auto_select_port()
        if port is None:
            raise ConnectionError('未发现可用串口，请检查 USB 转串口连接')
        self.ser = serial.Serial(port, self.baudrate, timeout=self.timeout)
        self.port = port
        return self.is_connected

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 发送指令
    # ------------------------------------------------------------------
    def _clamp(self, value, axis):
        limit = self.limits[axis]
        return max(-limit, min(limit, value))

    def send_speed(self, x=0, y=0, z=0):
        """按底盘原始约定发送速度：x/y 单位 mm/s，z 单位 度/s。"""
        if not self.is_connected:
            return False
        x = int(round(self._clamp(x, 'x')))
        y = int(round(self._clamp(y, 'y')))
        z = int(round(self._clamp(z, 'z')))

        cmd = bytearray(11)
        cmd[0] = CMD_HEAD
        cmd[1] = 0x00
        cmd[2] = 0x00
        cmd[3] = (x >> 8) & 0xFF
        cmd[4] = x & 0xFF
        cmd[5] = (y >> 8) & 0xFF
        cmd[6] = y & 0xFF
        cmd[7] = (z >> 8) & 0xFF
        cmd[8] = z & 0xFF
        bcc = 0
        for i in range(9):
            bcc ^= cmd[i]
        cmd[9] = bcc
        cmd[10] = CMD_TAIL
        try:
            self.ser.write(cmd)
            self.ser.flush()
            return True
        except OSError:
            return False

    def send_ros_vel(self, linear_x=0.0, angular_z=0.0):
        """ROS 风格速度转换为底盘指令再下发。

        linear_x: 前进速度 m/s（正=前进）；
        angular_z: 转向角速度 rad/s（正=逆时针=左转，与 ROS 一致）。
        内部转换为底盘约定：x 前正，z 左正。
        """
        x_mms = linear_x * 1000.0
        z_dps = -math.degrees(angular_z)   # ROS 正=左转 -> 底盘 z 负=左转
        return self.send_speed(x=round(x_mms), y=0, z=round(z_dps))

    def stop(self, repeat=3, interval=0.05):
        """连续发送几次零速指令，确保底盘停下。"""
        for _ in range(repeat):
            self.send_speed(0, 0, 0)
            time.sleep(interval)

    # ------------------------------------------------------------------
    # 状态回读（可选，用于显示电池电压/实速等）
    # ------------------------------------------------------------------
    @staticmethod
    def _u16(a, b):
        return (a << 8) | b

    @staticmethod
    def _s16(a, b):
        v = (a << 8) | b
        return v - 65536 if v > 32767 else v

    def read_status(self):
        """从串口缓存中解析一帧 24 字节状态，无完整有效帧返回 None。

        返回字段：flag_stop, real_x/y/z(mm/s 或 度/s 折算), acc_x/y/z,
                  ang_vel_x/y/z, battery_voltage(V)。
        """
        if not self.is_connected:
            return None
        try:
            if self.ser.in_waiting:
                self._rx.extend(self.ser.read(self.ser.in_waiting))
        except OSError:
            return None

        s16 = self._s16
        u16 = self._u16
        while len(self._rx) >= 24:
            start = self._rx.find(bytes([CMD_HEAD]))
            if start == -1:
                self._rx.clear()
                return None
            if start > 0:
                del self._rx[:start]
            if len(self._rx) < 24:
                return None
            frame = bytes(self._rx[:24])
            del self._rx[:24]
            if frame[0] != CMD_HEAD or frame[23] != CMD_TAIL:
                continue
            bcc = 0
            for i in range(22):
                bcc ^= frame[i]
            if bcc != frame[22]:
                continue
            return {
                'flag_stop': frame[1],
                'real_x': s16(frame[2], frame[3]),
                'real_y': s16(frame[4], frame[5]),
                'real_z': s16(frame[6], frame[7]) / 1000.0,
                'acc_x': s16(frame[8], frame[9]),
                'acc_y': s16(frame[10], frame[11]),
                'acc_z': s16(frame[12], frame[13]),
                'ang_vel_x': s16(frame[14], frame[15]),
                'ang_vel_y': s16(frame[16], frame[17]),
                'ang_vel_z': s16(frame[18], frame[19]),
                'battery_voltage': u16(frame[20], frame[21]) / 1000.0,
            }
        return None