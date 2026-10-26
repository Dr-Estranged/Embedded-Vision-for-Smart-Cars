#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交通标志检测器（YOLOX，实验1.5）。

基于 01-通用物体检测 的 YOLOX ONNX 推理流程：
    preproc → onnx推理 → demoPostprocess → multiclassNms → 可视化

并在此之上实现"多帧确认"机制：
    同一交通标志连续多帧命中且类别一致才输出为已确认标志，
    用于抑制偶发误识别与单帧抖动。

包路径：detection.traffic_sign_detector
"""
import time

import numpy as np

from detection.onnx_infer import OnnxRun
from detection import yolox_utils as yu
from detection.visualize import vis
from detection.traffic_sign_config import (
    MODEL_PATH, INPUT_SIZE, NMS_THR, SCORE_THR, DIS_THR,
    COCO_CLASSES, COCO_SIGN_MAP,
)


class TrafficSignDetector:
    """YOLOX 交通标志检测器。"""

    def __init__(self, model_path=MODEL_PATH,
                 conf_thr=DIS_THR,
                 confirm_frames=3,
                 forget_sec=1.5,
                 class_conf_thresholds=None):
        """
        :param model_path: ONNX 模型路径
        :param conf_thr: 参与决策/显示的置信度阈值
        :param confirm_frames: 连续确认帧数（多帧确认，抗误检）
        :param forget_sec: 无结果超过该时间则重置确认计数（秒）
        :param class_conf_thresholds: 单独类别置信度阈值，如 {'环岛': 0.75}
        """
        self.onnx_run = OnnxRun(model_name='YOLOX', model_path=model_path)
        self.conf_thr = conf_thr
        self.confirm_frames = max(1, int(confirm_frames))
        self.forget_sec = forget_sec
        self.class_conf_thresholds = dict(class_conf_thresholds or {})

        # 多帧确认状态
        self._last_sign = None
        self._last_time = 0.0
        self._confirm_count = 0

    # ============== 对外接口 ==============

    def detect(self, img: np.ndarray):
        """
        检测一帧中的交通标志。

        返回:
            ann_img:  标注后的图像（含检测框与中文标签）
            sign:     已确认的交通标志名，None 表示未确认
            conf:     已确认标志的置信度
            dets:     原始检测结果 (N x 6) 或 None
        """
        input_data, ratio = self._preprocess(img)
        net_outs = self.onnx_run.inference(input_data)
        dets = self._postprocess(net_outs[0], ratio)

        ann_img = self._annotate(img, dets)
        sign, conf = self._confirm(dets)
        return ann_img, sign, conf, dets

    def detect_signs(self, img: np.ndarray):
        """轻量检测：跳过 _annotate 绘图，供后台检测线程使用（省 CPU）。

        返回:
            sign: 已确认的交通标志名，None 表示未确认
            conf: 已确认标志的置信度
            dets: 原始检测结果 (N x 6) 或 None
        """
        input_data, ratio = self._preprocess(img)
        net_outs = self.onnx_run.inference(input_data)
        dets = self._postprocess(net_outs[0], ratio)
        sign, conf = self._confirm(dets)
        return sign, conf, dets

    def reset(self):
        """重置多帧确认状态（如切换场景/重新开始）。"""
        self._last_sign = None
        self._last_time = 0.0
        self._confirm_count = 0

    # ============== 内部实现 ==============

    def _preprocess(self, img: np.ndarray):
        if img is None or img.size == 0:
            raise ValueError('输入图像为空')
        img_src, ratio = yu.preproc(img, INPUT_SIZE)
        return img_src[None, :, :, :], ratio

    def _postprocess(self, net_out: np.ndarray, ratio: float):
        """
        将网络输出解码为原图坐标系下的检测框。
        返回 dets: (N x 6) [x0, y0, x1, y1, score, cls_id] 或 None
        """
        predictions = yu.demoPostprocess(net_out, INPUT_SIZE, p6=False)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio

        return yu.multiclassNms(boxes_xyxy, scores, nms_thr=NMS_THR, score_thr=SCORE_THR)

    def _annotate(self, img: np.ndarray, dets):
        """在图像上绘制检测框与标签。"""
        ann_img = img.copy()
        if dets is None or len(dets) == 0:
            return ann_img
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        ann_img = vis(ann_img, final_boxes, final_scores, final_cls_inds,
                      conf=self.conf_thr, class_names=COCO_CLASSES)
        return ann_img

    def _confirm(self, dets):
        """
        多帧确认：在 dets 中查找置信度达标且属于交通标志的类别。
        连续 confirm_frames 帧命中同一标志 → 返回 (标志名, 置信度)。
        """
        now = time.time()
        best_sign = None
        best_score = 0.0

        if dets is not None and len(dets) > 0:
            for det in dets:
                score = float(det[4])
                cls_id = int(det[5])
                sign_name = COCO_SIGN_MAP.get(cls_id)
                if sign_name is None:
                    continue
                # 全局阈值仍适用于全部类别；某些易误检类别可设置更高的
                # 独立阈值。环岛默认由主程序设置为 0.75。
                min_score = max(float(self.conf_thr),
                                float(self.class_conf_thresholds.get(
                                    sign_name, self.conf_thr)))
                if score < min_score or score <= best_score:
                    continue
                best_score = score
                best_sign = sign_name

        if best_sign is not None:
            if best_sign == self._last_sign and (now - self._last_time) < self.forget_sec:
                self._confirm_count += 1
            else:
                # 标志变化或超时未识别 → 重新计数
                self._confirm_count = 1
            self._last_sign = best_sign
            self._last_time = now

            if self._confirm_count >= self.confirm_frames:
                return best_sign, best_score
            return None, best_score

        # 当前帧未检测到标志：超时后清空计数
        if (now - self._last_time) > self.forget_sec:
            self._confirm_count = 0
            self._last_sign = None
        return None, 0.0


# 示例使用方式（离线单张图片测试）
if __name__ == '__main__':
    import cv2
    import sys
    detector = TrafficSignDetector()
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        image = cv2.imread(img_path)
        if image is None:
            print(f'无法读取图片: {img_path}')
            sys.exit(1)
        ann, sign, conf, _ = detector.detect(image)
        print(f'确认标志: {sign} 置信度: {conf:.2f}')
        cv2.imshow('detect', ann)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('用法: python detection/traffic_sign_detector.py <图片路径>')
