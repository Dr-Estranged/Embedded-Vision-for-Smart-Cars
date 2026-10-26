# -*- coding: utf-8 -*-
"""
交通标志识别配置（实验1.5）。

说明：
- 默认使用 yolox_nano.onnx（COCO 80 类通用检测模型，参考 01-通用物体检测），
  其中与交通相关的类别：
      class 9  = 交通灯
      class 11 = 停车标志
- 若需识别 左转/右转/直行/限速 等更多交通标志，可训练/加载专用交通标志模型，
  并把 TRAFFIC_SIGN_CLASSES 配置为对应模型的类别顺序，
  COCO_SIGN_MAP 与 SIGN_BEHAVIORS 决定了"检测类别 → 交通行为"的映射。

包路径：detection.traffic_sign_config
"""
import os

# ============== 资源路径（相对本文件所在目录） ==============
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, '..', 'resources', 'model_zoo', 'sign11_yolox_nano.onnx')
FONT_PATH = os.path.join(_BASE_DIR, '..', 'resources', 'font', 'simsun.ttc')

# ============== 检测参数 ==============
INPUT_SIZE = (416, 416)      # YOLOX 输入尺寸 (h, w)
NMS_THR = 0.45               # NMS 阈值
SCORE_THR = 0.1              # 后处理置信度下限（保留候选）
DIS_THR = 0.30               # 可视化/参与决策的置信度阈值



# ============== 11 类模型类别顺序（必须与训练时完全一致） ==============
TRAFFIC_SIGN_CLASSES = (
    "cone",
    "crossroads",
    "end_speed_limit_40",
    "green_pedestrian",
    "pedestrian_crossing",
    "red_pedestrian",
    "round_about",
    "speed_limit40",
    "turn_left",
    "turn_right",
    "U_turn",
)

# 模型内部英文标签 → 决策层中文标签。
DISPLAY_NAMES = (
    "路锥", "十字路口", "解除40限速", "行人绿灯", "人行横道",
    "行人红灯", "环岛", "限速40", "左转", "右转", "掉头",
)
COCO_SIGN_MAP = {i: name for i, name in enumerate(DISPLAY_NAMES)}

# 兼容旧命名：detector 可视化仍引用 COCO_CLASSES，指向同一类别元组
COCO_CLASSES = DISPLAY_NAMES

# ============== 交通标志 → 车辆行为（与 core.behavior_decision 中一致） ==============
SIGN_BEHAVIORS = {
    '路锥': 'cone_fork',
    '十字路口': 'lane_change',
    '解除40限速': 'end_speed_limit',
    '行人绿灯': 'green_light',
    '人行横道': 'pedestrian_crossing',
    '行人红灯': 'red_light',
    '环岛': 'roundabout',
    '限速40': 'speed_limit',
    '左转': 'left_turn',
    '右转': 'right_turn',
    '掉头': 'u_turn',
}

# 可视化字体参数
FONT_SIZE = 20
