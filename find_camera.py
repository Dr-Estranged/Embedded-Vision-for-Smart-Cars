#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摄像头枚举工具：找出本机可用的摄像头编号与后端。

用法：python find_camera.py
输出每个 编号 x 后端 的组合是否能打开并读到帧。
然后在 run.py 中用 --camera <编号> 指定能用的那个。
"""
import cv2

BACKENDS = [(cv2.CAP_ANY, 'ANY默认')]
if hasattr(cv2, 'CAP_DSHOW'):
    BACKENDS.append((cv2.CAP_DSHOW, 'DSHOW'))
if hasattr(cv2, 'CAP_MSMF'):
    BACKENDS.append((cv2.CAP_MSMF, 'MSMF'))


def main():
    found = []
    for idx in range(8):
        for backend, name in BACKENDS:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ok, frame = False, None
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    break
            if ok and frame is not None:
                h, w = frame.shape[:2]
                print('可用: --camera %d  后端=%s  分辨率=%dx%d' % (idx, name, w, h))
                found.append(idx)
            else:
                print('编号 %d 后端=%s 能打开但读不到帧' % (idx, name))
            cap.release()
    if not found:
        print('没有找到可用摄像头，请检查：')
        print('  1. USB 摄像头是否插好（重新插拔/换 USB 口）')
        print('  2. 是否被其他程序占用（浏览器/微信/相机应用/Teams/OBS）')
        print('  3. Windows 设置 -> 隐私 -> 摄像头 -> 允许桌面应用访问')
        print('  4. 设备管理器中摄像头设备是否正常（驱动）')
    else:
        print('\n建议命令: python run.py --camera %d --no-drive --debug' % found[0])


if __name__ == '__main__':
    main()
