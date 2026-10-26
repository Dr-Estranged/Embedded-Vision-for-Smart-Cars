#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视觉循迹检测与控制器（融合版，改写自 04-基于视觉的道路循迹-学生版）。

本文件包含两部分：

  1) LineDetector：线条检测器，与循迹实验版逐字一致（梯形 ROI 掩膜、
     fixed/手写Otsu 二值化、形态学、连通域过滤、扫描行最宽连续段、
     直线拟合、直角弯/十字路口横条识别）。process() 返回 dict，其中
     corner 字段（None 或 {'kind','side','row','width','cx','nearness'}）
     在融合版中兼任"标志转向动作到达执行区"的触发信号。

  2) FollowerController：无相机/串口依赖的循迹控制器（由原 LineFollower
     重构而来）。每帧调用 step(det, dt, base_speed) 返回速度命令而不是
     直接发串口；巡航速度由上层仲裁器按限速/人行横道等约束动态传入。
     begin_turn()/turn_step() 提供直角转弯三段执行（固定角速度 →
     PD 辅助回正 → 超时停车），供仲裁器在标志转向任务中复用。

方向约定（与 04/05 实验可执行代码一致，注释中有过时说法以此为准）：
  x>0 前进；y>0 左移；逻辑 z>0 = 左转、z<0 = 右转。
  本类输出的 z 已经是"逻辑 z"（内部沿用 z_invert=True 的默认换算）。
"""
import logging
import math

try:
    import cv2
    import numpy as np
except ImportError:  # 允许在无 OpenCV 的机器上跑 FollowerController 的离线自检
    cv2 = None
    np = None

logger = logging.getLogger(__name__)


# =====================================================================
# 1. 线条检测器（追踪式）——与循迹实验版逐字一致，勿动
# =====================================================================
class LineDetector:
    """从画面中提取巡线路径中心与方向（处理在降采样小图上进行）。"""

    def __init__(self, work_width=320, roi_top_ratio=0.45,
                 n_scan_rows=12, min_seg_width=2,
                 polarity='black',
                 crop_bottom_frac=0.25, crop_top_frac=0.60,
                 track_half=50.0, scan_start_ratio=0.25,
                 binary_mode='otsu', fixed_threshold=100,
                 adaptive_block=31, adaptive_c=8.0,
                 corner_width_ratio=0.55, corner_rows=16,
                 fork_side=0, fork_min_ratio=0.5, fork_hold_frames=15):
        self.work_width = work_width
        self.roi_top_ratio = roi_top_ratio          # 垂直方向：只处理底部这段(车前方地面)
        self.n_scan_rows = n_scan_rows              # 扫描行数
        self.min_seg_width = min_seg_width          # 过滤过窄噪点段
        self.polarity = polarity                    # 'black'=黑线白底 / 'white'=白线黑底
        # 梯形裁切：底部保留比例 / 顶部保留比例（相对整幅宽）
        self.crop_bottom_frac = crop_bottom_frac    # 底部(近车头)窗口窄
        self.crop_top_frac = crop_top_frac          # 顶部(远处)窗口宽，留转弯余量
        self.track_half = track_half                # 滑动搜索窗半宽(px, 工作图坐标)
        self.scan_start_ratio = scan_start_ratio    # 扫描起点在 ROI 内的比例(0~1)
        self.binary_mode = binary_mode
        self.fixed_threshold = int(np.clip(fixed_threshold, 0, 255))
        self.adaptive_block = max(3, int(adaptive_block) | 1)
        self.adaptive_c = float(adaptive_c)
        # 直角弯识别：某一扫描行的最宽线段超过窗宽这个比例，即认为是横向拐角条
        self.corner_width_ratio = float(corner_width_ratio)
        self.corner_rows = int(corner_rows)
        # 十字路口判据：横条两侧伸出长度的比值超过它即视为左右对称
        self.cross_sym_ratio = 0.35
        # 十字判据的绝对长度门槛：短边伸出必须 ≥ 图宽×该比例才算真十字。
        # 直角弯横条在竖线另一侧有一小段"肘部"伸出，远处时长边也不长，
        # 没有绝对门槛会被对称判据误判成 cross（远近标签跳变）。
        self.cross_min_ext_ratio = 0.15
        # 十字路口判据：横条上方中央这个半宽范围内若仍有线，说明正前方路是通的
        self.cross_ahead_half = 25
        # 横条行拟合点剔除：横向岔条(直角分叉/十字口)与线连通成超宽段时，
        # 段中心偏向横条一侧，参与拟合会把线拽向岔路。行宽超过
        # max(中位宽×bar_drop_mult, 中位宽+bar_drop_margin) 的点不参与拟合。
        self.bar_drop_mult = 2.5
        self.bar_drop_margin = 12

        # 上一帧车头参考行处的线中心(工作图 x)，兼作本帧搜索窗中心
        self._prev_cx = None

        # 岔路选线（由仲裁器的岔路执行器动态设置，平时必须为 0）：
        #   0 = 每行取最宽段(默认循迹)
        #  +1 = 每行取最右侧有效段(锁定右边岔路)
        #  -1 = 每行取最左侧有效段(锁定左边岔路)
        self.branch_side = 0

        # Y 形岔口选路（无标志时无条件走固定一侧，branch_side=0 时才生效）：
        #  某扫描行出现 ≥2 个"足够宽"的段(宽度 ≥ 本行最宽段×fork_min_ratio，
        #  细噪点被排除)即判定为岔口行，取 fork_side 侧那条作为该行的线。
        #  -1 = 锁定左岔(沿原方向) / +1 = 锁定右岔(新岔路) / 0 = 关闭。
        #  检测到岔口后 fork_hold_frames 内持续选边并屏蔽 corner 横条结果，
        #  防止分叉汇合区被误判为单侧直角弯而触发自动转弯。
        # 反Y(两线汇合成一条)与正Y 一样会出现双段行，靠"段间距趋势"区分：
        #  远处间距大=分岔(锁边)，远处间距小=汇合(守当前线，不锁边)。
        self.fork_side = 1 if fork_side > 0 else (-1 if fork_side < 0 else 0)
        self.fork_min_ratio = float(fork_min_ratio)
        self.fork_hold_frames = max(0, int(fork_hold_frames))
        self._fork_seen = False       # 本帧检测到分岔(正Y)
        self._merge_seen = False      # 本帧检测到汇合(反Y)
        self._fork_hold = 0           # 岔口保持计数(帧)
        self._junction = None         # 本帧判别结果 None/'fork'/'merge'(调试用)

    # ------------------------------------------------------------------
    def process(self, frame):
        """处理一帧，返回检测结果 dict。"""
        empty = self._empty_result()
        if frame is None or frame.size == 0:
            return empty

        h, w = frame.shape[:2]
        scale = self.work_width / w
        work = cv2.resize(frame, (self.work_width, int(h * scale)))
        wh, ww = work.shape[:2]

        roi_top = int(wh * self.roi_top_ratio)
        roi = work[roi_top:, :]
        roi_h, roi_w = roi.shape[:2]
        if roi_h < 10:
            return empty

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # 梯形 ROI 掩膜：每一行只保留画面中央的一段；ROI 顶部(远处)宽度
        # 比例为 crop_top_frac，底部(近处)为 crop_bottom_frac，中间按行线性变化。
        center = ww / 2.0
        frac = np.linspace(self.crop_top_frac, self.crop_bottom_frac, roi_h)
        half = frac * ww * 0.5
        left_bound = center - half
        right_bound = center + half
        cols = np.arange(ww)
        inside = (cols[None, :] >= left_bound[:, None]) & (cols[None, :] <= right_bound[:, None])

        # 二值化：fixed 固定阈值 / otsu 手写 Otsu / adaptive 局部自适应（可选拓展）。
        # 无论哪种模式，都只能在 inside 内产生 255，窗外必须保持 0。
        in_vals = blur[inside]
        if in_vals.size < 50:
            return self._empty_result(binary=np.zeros_like(blur), roi_top=roi_top)

        if self.binary_mode == 'adaptive':
            binary = self._adaptive_binary(blur, inside)
        else:
            if self.binary_mode == 'fixed':
                threshold = self.fixed_threshold
            else:
                threshold = self._otsu(in_vals)
            # 退化保护必须保留：近乎纯黑/纯白画面不得误判为线路。
            if threshold <= 5 or threshold >= 250:
                self._prev_cx = None
                return self._empty_result(binary=None, roi_top=roi_top)
            binary = self._apply_global_threshold(blur, inside, threshold)

        # 前景占比校验：一条线只应占窗内很小比例，占满整窗说明是假检(纯色/大面积暗区)
        fg_ratio = binary[inside].mean() / 255.0
        if fg_ratio < 0.005 or fg_ratio > 0.35:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # --- 3. 连通域几何过滤：保留"条带状"线分量（对曲线弯更宽容） ---
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        line_mask = np.zeros_like(binary)
        min_h = roi_h * 0.25          # 线至少要覆盖 ROI 25% 的高度(曲线弯放宽)
        for i in range(1, labels.max() + 1):
            x_, y_, bw_, bh_, area = stats[i]
            if area < 25:
                continue
            # 剔除又矮又窄的碎块；保留横向长条(直角弯拐角)与纵向线。
            if bh_ < min_h and bw_ < min_h:
                continue
            # 面积极度低→细带；面积高但相对包围盒仍细长(弯段绕行)→也接受
            fill = area / float(bw_ * bh_)          # 0~1，实心色块接近1
            if bw_ > 8 and bh_ > 8 and fill > 0.85:
                continue                            # 实心大色块→剔除(防假检)
            line_mask[labels == i] = 255
        binary = line_mask
        if int(binary.max()) == 0:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # --- 4. 扫描线 + 滑动搜索窗 ---
        # 预测位置作为搜索窗中心；丢线后重捕获时清空预测窗
        self._fork_seen = False
        self._merge_seen = False
        pred = self._prev_cx if self._prev_cx is not None else center
        points = self._scan_lines(binary, roi_top, ww, inside, pred)
        if len(points) < 3:
            # 重捕获：不受预测限制，直接在全裁切范围内找
            points = self._scan_lines(binary, roi_top, ww, inside, None)
        if len(points) < 3:
            self._prev_cx = None
            return self._empty_result(binary=binary, roi_top=roi_top)

        # 拟合、横向误差与方向角：x = a*y + b；error_px 线在右为正；
        # 方向角 atan(-a)；EMA 更新 _prev_cx 供下一帧滑窗预测。
        ys = np.asarray([p[1] for p in points], dtype=np.float64)
        xs = np.asarray([p[0] for p in points], dtype=np.float64)
        a, b = np.polyfit(ys, xs, 1)
        ref_y = roi_top + roi_h - 1
        cx_fit = a * ref_y + b
        error_px = cx_fit - ww / 2.0
        angle_deg = math.degrees(math.atan(-a))

        # 拐角方向(side)判定的历史参考线位：取本帧 EMA 更新前的 _prev_cx
        # (前期干净循迹累积的历史值)。横条逼近时本帧拟合易被横条行带偏、
        # 底部外推放大误差，直接用本帧 cx_fit 判 side 会把左右判反
        cx_hist = self._prev_cx
        if self._prev_cx is None:
            self._prev_cx = cx_fit
        else:
            self._prev_cx = 0.6 * cx_fit + 0.4 * self._prev_cx

        # Y 岔口：本帧见到分岔行则刷新保持计数；见到汇合(反Y)则立即清掉
        # 保持计数——汇合口不锁边、不屏蔽横条，防止旁路并入时把车带偏。
        # hold 期间屏蔽 corner 横条，防止分叉区被误判为单侧直角弯触发自动转弯。
        if self._fork_seen:
            self._fork_hold = self.fork_hold_frames
        elif self._merge_seen:
            self._fork_hold = 0
        elif self._fork_hold > 0:
            self._fork_hold -= 1
        fork_active = self.fork_side != 0 and self._fork_hold > 0

        corner = None if fork_active else \
            self._detect_corner(binary, inside, roi_top, cx_fit, cx_hist)

        return {
            'is_valid': True,
            'centroid': (cx_fit, float(ref_y)),
            'error_px': float(error_px),          # 线在右 → 正 → 右转
            'angle_deg': float(angle_deg),
            'a': a, 'b': b,                     # 主直线参数 x=a*y+b（调试画线用）
            'points': points,                   # 参与拟合的点
            'binary': binary,
            'roi_top': roi_top,
            'line_type': self.polarity,
            'corner': corner,
            'fork': bool(fork_active),          # 岔口选边生效中（调试显示用）
            'junction': self._junction,         # None/'fork'/'merge'（调试显示用）
        }

    # ------------------------------------------------------------------
    def _detect_corner(self, binary, inside, roi_top, cx_ref, cx_hist=None):
        """识别直角拐角：找横跨检测窗大半宽度的横向线段。

        正常巡线时每行只有一小段线；直角弯处会出现一条很宽的横条。
        取最靠近车头的那一条作为拐角，用它相对车头线中心的偏向判断转向方向。
        side 以 cx_hist(历史线位，前期干净循迹的 EMA)为基准——横条逼近时
        本帧拟合(cx_ref)易被横条行带偏，会把转向方向判反；无历史才退回
        cx_ref。corner/cross 分类仍用 cx_ref(判据基准须与本帧拟合一致)。

        返回 None 或 {'side': +1右/-1左, 'row', 'width', 'cx', 'nearness'}。
        nearness 为 0~1，越接近 1 表示拐角越靠近车头。
        """
        roi_h = binary.shape[0]
        rows = np.unique(np.linspace(0, roi_h - 1, self.corner_rows).astype(int))
        found = None
        for rel_y in rows:
            cols = np.nonzero(inside[rel_y])[0]
            if cols.size == 0:
                continue
            l0, r0 = int(cols[0]), int(cols[-1]) + 1
            win_w = r0 - l0
            if win_w < 10:
                continue

            nz = np.flatnonzero(binary[rel_y, l0:r0])
            if nz.size == 0:
                continue
            breaks = np.where(np.diff(nz) > 1)[0] + 1
            widest = max(np.split(nz, breaks), key=len)
            seg_w = int(widest[-1] - widest[0] + 1)
            if seg_w < self.corner_width_ratio * win_w:
                continue

            # 保留最靠近车头(rel_y 最大)的横条
            if found is None or rel_y > found['rel_y']:
                found = {
                    'rel_y': int(rel_y),
                    'width': seg_w,
                    'left': l0 + int(widest[0]),
                    'right': l0 + int(widest[-1]),
                    'cx': l0 + (int(widest[0]) + int(widest[-1])) / 2.0,
                }

        if found is None:
            return None

        kind, sym = self._classify_corner(binary, found, cx_ref)
        side_ref = cx_hist if cx_hist is not None else cx_ref
        return {
            'kind': kind,                      # 'corner' 直角弯 / 'cross' 十字路口
            'sym': sym,                        # cross 是否由对称判据认定(真十字)
            'side': 1 if found['cx'] >= side_ref else -1,
            'row': found['rel_y'] + roi_top,
            'width': found['width'],
            'cx': found['cx'],
            'nearness': found['rel_y'] / max(1, roi_h - 1),
        }

    def _classify_corner(self, binary, bar, cx_ref):
        """区分直角弯和十字路口，返回 (kind, symmetric)。

        直角弯的横条只朝一侧伸出，且横条上方没有线；
        十字路口的横条左右基本对称，且正前方仍有线继续向远处延伸。
        多个判据满足任意一个就按十字路口处理，宁可直行也不误转。

        symmetric 标记 cross 是否由"左右对称/超宽双侧"判据认定(真十字)；
        仅靠"上方贯通"判定的 cross 为 False——单侧横条+贯通线是侧向开口
        的 Y/T 岔口(如本赛道右开口 Y：左岔长线贯通+右侧直角横条)，不是
        真十字路口，变道等动作不得把它当执行区。

        对称判据带绝对长度门槛（短边伸出 ≥ 图宽×cross_min_ext_ratio）：
        直角弯肘部那小段伸出在远处会让 short/long 比值超过 cross_sym_ratio，
        没有绝对门槛会把直角弯误判成 cross（远近标签跳变）。
        """
        img_width = binary.shape[1]
        min_ext = img_width * self.cross_min_ext_ratio
        left_ext = cx_ref - bar['left']
        right_ext = bar['right'] - cx_ref
        long_ext = max(left_ext, right_ext)
        short_ext = min(left_ext, right_ext)
        if (long_ext > 0 and short_ext >= min_ext
                and short_ext / long_ext >= self.cross_sym_ratio):
            return 'cross', True

        # 兜底：横条本身很宽且两侧都有明显伸出，也按十字路口直行
        bar_width = bar['right'] - bar['left']
        if (bar_width > img_width * 0.65
                and left_ext > min_ext
                and right_ext > min_ext):
            return 'cross', True

        # 横条上方中央若仍有前景，说明直行方向是通的(但非对称，不是真十字)
        ahead = binary[:max(0, bar['rel_y'] - 2), :]
        if ahead.shape[0] >= 3:
            lo = max(0, int(cx_ref) - self.cross_ahead_half)
            hi = min(ahead.shape[1], int(cx_ref) + self.cross_ahead_half + 1)
            band = ahead[:, lo:hi]
            # 上方大部分行有线即视为贯通，不要求全部
            if band.size and (band.max(axis=1) > 0).mean() >= 0.4:
                return 'cross', False

        return 'corner', False

    # ------------------------------------------------------------------
    def _scan_lines(self, binary, roi_top, ww, inside, pred):
        """对每一扫描行，在"裁切窗 ∩ 预测窗"内找线，返回 [(x, y, w), ...]。

        y 为整图坐标；pred 为 None 时(重捕获)只用裁切窗。
        两趟式：先逐行收集候选段，再整帧判别 分岔(正Y)/汇合(反Y)，最后按
        判别结果选段——分岔锁 fork_side 侧分支，汇合守离预测最近的段，
        普通路面取最宽段(原行为)。
        """
        roi_h = binary.shape[0]
        rows = np.linspace(int(roi_h * self.scan_start_ratio),
                           roi_h - 1, self.n_scan_rows).astype(int)
        row_data = []                               # (rel_y, l0, segments)
        for rel_y in rows:
            mask_row = np.nonzero(inside[rel_y])[0]
            if mask_row.size == 0:
                continue
            l0, r0 = int(mask_row[0]), int(mask_row[-1]) + 1
            # 岔路选线模式(branch_side!=0)必须在整行梯形窗内找段：
            # 岔路分支很快岔出主线 ±track_half 的预测窗，若仍按预测窗裁切，
            # "最靠岔路侧的段"永远只能是窗内的主线，车会循主线冲进被封堵的岔路。
            if pred is not None and self.branch_side == 0:
                l0 = max(l0, int(round(pred - self.track_half)))
                r0 = min(r0, int(round(pred + self.track_half)) + 1)
            if r0 - l0 < 1:
                continue

            nz = np.flatnonzero(binary[rel_y, l0:r0])
            if nz.size == 0:
                continue
            breaks = np.where(np.diff(nz) > 1)[0] + 1
            # 段按 x 从小到大(从左到右)排列
            row_data.append((rel_y, l0, np.split(nz, breaks)))

        self._junction = None
        if self.branch_side == 0 and self.fork_side != 0:
            self._junction = self._classify_junction(row_data)
            if self._junction == 'fork':
                self._fork_seen = True
            elif self._junction == 'merge':
                self._merge_seen = True

        points = []
        for rel_y, l0, segments in row_data:
            best_segment = self._select_segment(segments, self._junction, pred, l0)
            if best_segment is None:
                continue
            bw = int(best_segment[-1]) - int(best_segment[0]) + 1
            if bw < self.min_seg_width:
                continue
            cx = l0 + (int(best_segment[0]) + int(best_segment[-1]) + 1) // 2
            points.append((cx, rel_y + roi_top, bw))
        return self._drop_bar_points(points)

    def _drop_bar_points(self, points):
        """从拟合点中剔除"横条行"。

        直角分叉/十字口的横向条与线连通成超宽段，段中心偏向横条一侧，
        参与拟合会把线拽向岔路——右开口 Y(左岔长线贯通+右侧直角横条)
        会被拉成直行甚至偏右，无法进入左岔。剔除超宽行后由横条上、
        下方的正常行拟合，车自然沿贯通线(左岔)平滑通过。
        横条本身由 _detect_corner 在二值图上独立检测，不受影响；
        正常行不足 3 个时放弃剔除（贴着大路口时保护拟合点数）。
        """
        if len(points) < 4:
            return points
        widths = sorted(p[2] for p in points)
        med = widths[len(widths) // 2]
        limit = max(med * self.bar_drop_mult, med + self.bar_drop_margin)
        kept = [p for p in points if p[2] < limit]
        return kept if len(kept) >= 3 else points

    def _select_segment(self, segments, junction, pred, l0):
        """按当前模式从一行的候选段(按 x 从左到右)中选出要循的那一条。"""
        if self.branch_side > 0:
            # 右岔模式：取最右侧有效段，把右岔分支当作"那条线"
            for cand in reversed(segments):
                if cand.size >= self.min_seg_width:
                    return cand
            return segments[-1]
        if self.branch_side < 0:
            # 左岔模式：取最左侧有效段
            for cand in segments:
                if cand.size >= self.min_seg_width:
                    return cand
            return segments[0]
        if junction == 'fork':
            # 分岔(正Y)：锁定 fork_side 侧的真分支(细噪点已被比例门槛排除)
            valid = [s for s in segments if s.size >= self.min_seg_width]
            if len(valid) >= 2:
                widest = max(s.size for s in valid)
                substantial = [s for s in valid
                               if s.size >= widest * self.fork_min_ratio]
                if len(substantial) >= 2:
                    return substantial[-1] if self.fork_side > 0 else substantial[0]
            return max(segments, key=len)
        if junction == 'merge' and pred is not None:
            # 汇合(反Y)：旁路并入，锁定离预测位置最近的段，守住当前这条线
            valid = [s for s in segments if s.size >= self.min_seg_width]
            if valid:
                target = pred - l0
                return min(valid,
                           key=lambda s: abs((int(s[0]) + int(s[-1])) / 2.0 - target))
        return max(segments, key=len)

    def _classify_junction(self, row_data):
        """整帧判别 Y 岔口(分岔) / 反Y(汇合)，返回 'fork' / 'merge' / None。

        判据是双段行的"段间距随距离的变化趋势"：
          分岔(正Y)：岔点在近处，两分支向远处张开 → 远处行间距 > 近处行间距；
          汇合(反Y)：并线点在远处，两线向远处收拢 → 远处行间距 < 近处行间距。
        至少 2 个双段行才能判趋势；单行双段多半是噪点，不判(保持普通循迹)，
        也避免汇合口偶发双段被误判成岔口而触发 fork_hold 锁边。
        远近间距基本持平也不判：那是横向岔条(直角分叉)造成的偶发双段，
        横向岔条由 _drop_bar_points + 横条检测处理，不走岔口锁边逻辑。
        "足够宽" = 宽度 ≥ 本行最宽段的 fork_min_ratio：同一行两条岔线深度
        相同、成像线宽相近，而断线碎块/噪点远细于真线，天然被比例门槛排除。
        """
        pairs = []                                  # (rel_y, 两外侧段间距)
        for rel_y, _l0, segments in row_data:
            valid = [s for s in segments if s.size >= self.min_seg_width]
            if len(valid) < 2:
                continue
            widest = max(s.size for s in valid)
            substantial = [s for s in valid
                           if s.size >= widest * self.fork_min_ratio]
            if len(substantial) < 2:
                continue
            gap = int(substantial[-1][0]) - int(substantial[0][-1])
            pairs.append((rel_y, gap))
        if len(pairs) < 2:
            return None
        pairs.sort()
        gap_far = pairs[0][1]                       # 最远双段行(y 最小)
        gap_near = pairs[-1][1]                     # 最近双段行(y 最大)
        if gap_far > gap_near + 2:
            return 'fork'
        if gap_near > gap_far + 2:
            return 'merge'
        # 间距趋势平坦：多是横向岔条(直角分叉)造成的偶发双段，并非真正的
        # 双线并行，不判岔口——既不锁边也不按汇合处理(避免误记录"见过反Y")。
        return None

    def _apply_global_threshold(self, blur, inside, threshold):
        """根据极性应用一个全局阈值，并保证窗外为 0。"""
        binary = np.zeros_like(blur)
        # 黑线白底：inside 且灰度小于 threshold 的像素设为 255。
        # 白线黑底：inside 且灰度大于 threshold 的像素设为 255。
        if self.polarity == 'black':
            binary[inside & (blur < threshold)] = 255
        else:
            binary[inside & (blur > threshold)] = 255
        return binary

    @staticmethod
    def _otsu(arr):
        """手写 Otsu，返回使类间方差最大的灰度阈值。"""
        hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        total = hist.sum()
        if total == 0:
            return 0
        sum_all = float(np.dot(np.arange(256), hist))
        sum_b, w_b = 0.0, 0.0
        best_t, max_var = 0, -1.0

        for t in range(256):
            w_b += hist[t]
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break

            sum_b += t * hist[t]

            m_b = sum_b / w_b
            m_f = (sum_all - sum_b) / w_f

            var = w_b * w_f * (m_b - m_f) ** 2
            if var > max_var:
                max_var = var
                best_t = t

        return best_t

    def _adaptive_binary(self, blur, inside):
        """（基础拓展，可选）：局部自适应二值化。

        黑线：blur < local_mean - adaptive_c；白线：blur > local_mean + adaptive_c。
        结果只能在 inside 内为 255。禁止直接调用 cv2.adaptiveThreshold。
        """
        local_mean = cv2.boxFilter(blur, ddepth=-1,
                                   ksize=(self.adaptive_block, self.adaptive_block),
                                   normalize=True, borderType=cv2.BORDER_REPLICATE)
        binary = np.zeros_like(blur)
        if self.polarity == 'black':
            binary[inside & (blur < local_mean - self.adaptive_c)] = 255
        else:
            binary[inside & (blur > local_mean + self.adaptive_c)] = 255
        return binary

    def _empty_result(self, binary=None, roi_top=0):
        return {
            'is_valid': False,
            'centroid': None,
            'error_px': 0.0,
            'angle_deg': 0.0,
            'a': 0.0, 'b': 0.0,
            'points': [],
            'binary': binary,
            'roi_top': roi_top,
            'line_type': self.polarity,
            'corner': None,
            'fork': False,
            'junction': None,
        }


# =====================================================================
# 2. 循迹控制器（无相机/串口依赖，供仲裁器逐帧调用）
# =====================================================================
class FollowerController:
    """循迹控制器：step() 输入检测结果，输出速度命令（不直接发串口）。

    参数（均为底盘串口约定单位）：
      max_z       最大转向速度 度/s
      kp / kd / ka  横向误差比例/微分/方向角前馈增益
      err_alpha   误差/角度低通滤波系数(0~1)
      lost_hold   失线低速直行的帧数上限
      search_frames 失线旋转搜索的帧数上限
      startup_frames 起步确认帧数：连续检测到线这么多帧后才前进(默认5)
      ramp_frames    起步后速度从0平滑加速到目标的帧数(默认20，约1秒)
      min_move_speed 克服电机死区的最小前进速度(0=不限制)
      corner_*     直角转弯三段执行参数（供 begin_turn/turn_step 使用）
    """

    def __init__(self, max_z=800,
                 kp=10.0, kd=1.2, ka=3.5,
                 err_alpha=0.6, z_rate_limit=120.0,
                 lost_hold=10, search_frames=15,
                 startup_frames=5, ramp_frames=20,
                 min_move_speed=0,
                 corner_speed=40, corner_z=400,
                 corner_min_frames=6, corner_max_frames=150,
                 corner_exit_err=35.0, corner_exit_angle=30.0,
                 corner_turn_sec=0.0,
                 z_invert=True):
        self.max_z = max_z
        self.kp = kp
        self.kd = kd
        self.ka = ka
        self.err_alpha = err_alpha
        self.z_rate_limit = z_rate_limit    # 每帧最大转向增量(°/s)，防猛甩
        self.lost_hold = lost_hold
        self.search_frames = search_frames
        self.startup_frames = startup_frames
        self.ramp_frames = ramp_frames
        self.min_move_speed = max(0, min_move_speed)
        # z_invert 沿用循迹实验默认 True：PD 输出经取反后得到逻辑 z(z>0=左转)
        self.z_invert = z_invert

        # 直角转弯（被仲裁器转向任务复用）
        self.corner_speed = corner_speed            # 转弯时的前进速度 mm/s
        self.corner_z = corner_z                    # 转弯时的转向角速度 度/s
        self.corner_min_frames = corner_min_frames  # 至少转这么多帧再判断是否转完
        self.corner_max_frames = corner_max_frames  # 超时上限，到点必须停车
        self.corner_exit_err = corner_exit_err      # 退出转弯所需的横向误差阈值
        self.corner_exit_angle = corner_exit_angle  # 退出转弯所需的方向角阈值
        # 固定旋转段标定时长(秒)：0=自动按 900/corner_z 换算——实车标定
        # z=500 转 360° 需 7.2s(掉头标定)，即 90°≈900(z·s)，与 corner_z 联动
        self.corner_turn_sec = float(corner_turn_sec)
        self._corner_turning = False
        self._corner_side = 0
        self._corner_frames = 0
        self._corner_elapsed = 0.0       # 转弯固定段累计秒数(标定转角用)

        self._prev_err = 0.0
        self._filtered_err = 0.0
        self._filtered_angle = 0.0
        self._has_prev = False      # 是否已有上一帧有效误差(首帧不微分)
        self._last_z = 0.0
        self._last_sign = 1         # 最后一次有效误差方向(失线搜索用)
        self._lost_count = 0
        self._start_seen = 0        # 起步期连续有效帧计数
        self._started = False       # 起步确认是否完成(完成后才前进)
        self._run_frames = 0        # 起步后已运行帧数(速度斜坡用)

    # ------------------------------------------------------------------
    def step(self, det, dt, base_speed):
        """处理一帧循迹，返回 (state, x_mm/s, y_mm/s, z_逻辑°/s)。

        base_speed 由上层（仲裁器）按限速/横道/接近减速等约束动态给定。
        det 为 LineDetector.process() 的返回 dict（无效帧 is_valid=False）。
        """
        dt = max(dt, 1e-3)
        err = det['error_px']
        angle = det['angle_deg']

        if det['is_valid']:
            state = 'run'
            self._lost_count = 0
            self._last_sign = 1 if err >= 0 else -1

            if not self._has_prev:
                # 首帧：用真实误差初始化滤波器，避免 derr 尖峰把转向打满
                self._filtered_err = err
                self._filtered_angle = angle
                self._prev_err = err
                self._has_prev = True

            # 低通滤波（对误差与角度统一滤波）
            self._filtered_err = (self.err_alpha * err +
                                  (1 - self.err_alpha) * self._filtered_err)
            self._filtered_angle = (self.err_alpha * angle +
                                    (1 - self.err_alpha) * self._filtered_angle)
            err = self._filtered_err
            angle = self._filtered_angle

            # PD + 方向角前馈
            derr = (err - self._prev_err) / dt
            z_raw = self.kp * err + self.kd * derr + self.ka * angle
            z = max(-self.max_z, min(self.max_z, z_raw))

            # 转向速率限制：单帧最多变化 z_rate_limit °/s，防车身猛甩
            dz = z - self._last_z
            if abs(dz) > self.z_rate_limit:
                z = self._last_z + self.z_rate_limit * (1 if dz > 0 else -1)
            self._last_z = z

            if self.z_invert:          # 输出逻辑 z（z>0 左转 / z<0 右转）
                z = -z

            # 起步确认：连续 startup_frames 帧检测有效才允许前进
            if not self._started:
                self._start_seen += 1
                self._prev_err = err
                if self._start_seen >= self.startup_frames:
                    self._started = True      # 本帧起开始前进
                    self._run_frames = 0
                    return 'start-ok', 0, 0, int(z)
                return 'start', 0, 0, 0       # 静止确认：等线稳定

            # 起步斜坡 + 弯道降速
            self._run_frames += 1
            ramp = min(1.0, self._run_frames / max(1, self.ramp_frames))

            z_ratio = abs(z) / max(1, self.max_z)
            speed_limit = float(base_speed)
            if abs(err) > 40:
                speed_limit = min(speed_limit, base_speed * 0.3)

            # 转向越强速度越低
            speed = speed_limit * ramp * (1.0 - 0.5 * z_ratio)
            speed = max(0.0, min(float(base_speed), speed))

            # 电机存在启动死区，低于该值只会堵转发热而不转动
            if 0 < speed < self.min_move_speed:
                speed = min(float(self.min_move_speed), float(base_speed))

            self._prev_err = err
            return state, int(speed), 0, int(z)

        # ---- 无效帧 ----
        self._has_prev = False
        self._filtered_err = 0.0
        self._filtered_angle = 0.0
        self._last_z = 0.0
        if not self._started:
            # 起步期间失线：不前进也不旋转搜索，原地停车等线出现
            self._start_seen = 0
            return 'start-lost', 0, 0, 0
        state, z, speed = self._handle_lost(base_speed)
        return state, int(speed), 0, int(z)

    # ------------------------------------------------------------------
    def begin_turn(self, side):
        """进入直角转弯执行状态。side=+1 右转 / -1 左转。"""
        self._corner_turning = True
        self._corner_side = 1 if side > 0 else -1
        self._corner_frames = 0
        self._corner_elapsed = 0.0
        self._lost_count = 0
        logger.info('开始转向执行：方向=%s', '右' if self._corner_side > 0 else '左')

    def turn_step(self, det, dt):
        """直角转弯逐帧动作，返回 (state, speed, z_逻辑, done)。

        两段式：固定角速度旋转段(严格转满标定转角) → PD 捕获回正段；
        超过 corner_max_frames 一律停车(done=True 且 speed=0)，防止原地空转。

        固定段严格转满标定时长 corner_turn_sec 秒（0=自动 900/corner_z，
        与掉头标定 7.2s·z500=360° 同源，即 90°≈900 z·s），期间不做任何
        PD/退出判断——两条前车之鉴：
          1) 十字路口横条在 0~90° 旋转全程都是"有效线"，靠"丢线"判断
             转离原线永远等不到，车冲过 90° 转到 100°+ 把横条甩到身后
             再也收不回（过冲）；
          2) 直角弯丢线后，斜着的横条只转了十几度就会以"干净线"形态
             短暂出现，提前 PD/退出会只转 ~15° 就回正开走、随即丢线。
        标定时长到点后：线回正(err/angle 达标且不是单侧横条)即退出；
        线可见未回正用 PD 辅助拉正——欠转继续拉、过转运回，双向自校正；
        线不可见则保持固定角速度继续转直到捕获(补足欠转)。
        """
        dt = max(dt, 1e-3)
        self._corner_frames += 1
        self._corner_elapsed += dt

        if self._corner_frames > self.corner_max_frames:
            logger.warning('转向超过 %d 帧仍未重新捕获线路，停车',
                           self.corner_max_frames)
            self._corner_turning = False
            self._last_z = 0.0
            return 'turn-timeout', 0, 0, True

        err = det['error_px']
        angle = det['angle_deg']
        corner = det.get('corner')
        line_ok = (det['is_valid']
                   and (corner is None or corner['kind'] == 'cross'))

        # 固定旋转段严格按标定时长转满 90°，到点才允许 PD/退出
        turn_sec = (self.corner_turn_sec if self.corner_turn_sec > 0
                    else 900.0 / max(self.corner_z, 1))
        fixed_done = self._corner_elapsed >= turn_sec

        # 回正退出：固定段结束 + 线重新回正(误差小、角度正、不是单侧横条)
        if (self._corner_frames >= self.corner_min_frames
                and fixed_done
                and line_ok
                and abs(err) <= self.corner_exit_err
                and abs(angle) <= self.corner_exit_angle):
            logger.info('转向完成，用时 %d 帧', self._corner_frames)
            self._corner_turning = False
            self._last_z = 0.0
            self._has_prev = False      # 重新初始化滤波，避免出弯瞬间微分冲击
            return 'turn-exit', self.corner_speed, 0, True

        # 固定段结束且线重新可见：PD 辅助跟踪新线回正（欠转拉正/过转运回）
        if (self._corner_frames >= self.corner_min_frames
                and fixed_done
                and line_ok):
            derr = (err - self._prev_err) / dt
            z_raw = self.kp * err + self.kd * derr + self.ka * angle
            z = max(-self.max_z, min(self.max_z, z_raw))
            dz = z - self._last_z
            if abs(dz) > self.z_rate_limit:
                z = self._last_z + self.z_rate_limit * (1 if dz > 0 else -1)
            self._last_z = z
            if self.z_invert:
                z = -z
            self._prev_err = err
            return 'turn-track', self.corner_speed, int(z), False

        # 固定角速度旋转：side=+1(右转) → 逻辑 z 取负
        z = -self._corner_side * self.corner_z
        state = 'turn-R' if self._corner_side > 0 else 'turn-L'
        return state, self.corner_speed, int(z), False

    def force_started(self):
        """动作执行器闭环接管时调用：跳过起步静止确认，直接带斜坡起步。

        岔路/环岛执行器是"行进中接管"，若走正常起步确认会原地停 5 帧，
        在岔口/环入口停顿既顿挫又容易错过入弯时机。
        """
        self._started = True
        self._start_seen = self.startup_frames
        self._run_frames = 0

    @property
    def turning(self):
        return self._corner_turning

    # ------------------------------------------------------------------
    def reset_after_action(self):
        """标志动作（转向/掉头/变道/绕锥/环岛/红灯停车）结束后调用：
        清空全部控制状态，重新起步确认 + 速度斜坡，避免微分冲击与猛冲。"""
        self._corner_turning = False
        self._corner_frames = 0
        self._corner_side = 0
        self._corner_elapsed = 0.0
        self._prev_err = 0.0
        self._filtered_err = 0.0
        self._filtered_angle = 0.0
        self._has_prev = False
        self._last_z = 0.0
        self._lost_count = 0
        self._start_seen = 0
        self._started = False
        self._run_frames = 0

    # ------------------------------------------------------------------
    def _handle_lost(self, base_speed):
        """失线处理(分三阶段，全部低速安全执行)，返回 (state, z, speed)：
        1) lost-hold   短暂低速直行，等线重新回进视野；
        2) lost-search 朝最后一次见线的方向旋转搜索，超过半程后自动换向；
        3) lost-stop   仍未找回则停车等待。
        所有非零速度都有明确帧数上限；不允许无限前进或无限旋转。
        """
        self._lost_count += 1

        if self._lost_count <= self.lost_hold:
            # 阶段 1: 短暂低速直行
            speed = base_speed * 0.3
            return 'lost-hold', 0, speed

        if self._lost_count <= self.lost_hold + self.search_frames:
            # 阶段 2: 旋转搜索，超过半程后换向
            search_progress = self._lost_count - self.lost_hold
            if search_progress > self.search_frames / 2:
                sign = -self._last_sign
            else:
                sign = self._last_sign
            z = sign * self.max_z * 0.5
            return 'lost-search', z, 0

        # 阶段 3: 停车等待
        return 'lost-stop', 0, 0
