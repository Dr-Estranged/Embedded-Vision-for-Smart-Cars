#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOX 后处理工具（参考 01-通用物体检测/components/utils.py）。

包含：中文字体绘制 putText、图像预处理 preproc、NMS、demoPostprocess。

包路径：detection.yolox_utils
"""
import os
import sys

import cv2
import numpy as np

# 允许本模块被单独执行（python detection/yolox_utils.py）时仍能正确导入兄弟模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from detection.traffic_sign_config import FONT_PATH, FONT_SIZE


def putText(img, text, org=(0, 0), font_path=FONT_PATH,
            color=(0, 0, 255), font_size=FONT_SIZE):
    """
    在图片上显示文字（优先支持中文，PIL 缺失时回退 cv2 原生绘制）。
    :param img: 输入的 img, 通过 cv2 读取（BGR）
    :param text: 要显示的文字
    :param org: 文字左上角坐标
    :param font_path: 字体路径
    :param color: 字体颜色, (B,G,R)
    :param font_size: 字号
    """
    try:
        from PIL import ImageFont, ImageDraw, Image
    except ImportError:
        # PIL 不可用：回退到 cv2 绘制（不支持中文，仅能显示 ASCII）
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    b, g, r = color
    a = 0
    draw.text(org, text, font=ImageFont.truetype(font_path, font_size), fill=(b, g, r, a))
    img = np.array(img_pil)
    return img


def preproc(img, input_size, swap=(2, 0, 1)):
    """YOLOX 标准预处理：等比缩放 + 114 灰边填充 + CHW 转换。"""
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def nms(boxes, scores, nms_thr):
    """单类别 NMS（Numpy 实现）。"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclassNms(boxes, scores, nms_thr, score_thr):
    """多类别 NMS（Numpy 实现）。"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]

        keep = nms(valid_boxes, valid_scores, nms_thr)
        if len(keep) > 0:
            cls_inds = np.ones((len(keep), 1)) * cls_ind
            dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
            final_dets.append(dets)

    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def demoPostprocess(outputs, img_size, p6=False):
    """YOLOX 后处理：解码网格偏移与尺度，得到原图坐标。"""
    grids = []
    expanded_strides = []

    strides = [8, 16, 32, 64] if p6 else [8, 16, 32]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs