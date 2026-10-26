#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""融合仲裁器：循迹 + 交通标志行为的统一决策。

优先级（每 tick 自上而下）：
  1) 行人红灯：任意时刻确认即抢占一切（含动作执行中），立即停车等待绿灯；
     红/绿灯永远不受冷却屏蔽（安全）。
  2) RED_WAIT：停车直到确认行人绿灯。
  3) EXECUTING：推进转向/岔路/环岛/时序执行器，只响应红灯；屏蔽横条事件与失线保护。
  4) MEMORIZED：已记忆任务、未到执行区——继续循迹但按比例减速；
     转向任务两线索触发：横条到达执行区（corner.nearness）/ 正Y岔口
     连续确认——只在路口才转，不设时间兜底盲转；任务最长保持
     turn_hold_sec 秒。其余任务超过 memory_hold_sec 未触发则丢弃。
  5) FOLLOW：默认循迹；限速/解除限速/人行横道等立即生效类直接调制巡航速度；
     无标志时遇单侧直角弯横条（kind='corner'）连续确认后自动直角转弯；
     岔路/十字路口的对称横条（kind='cross'）不自动转，需转向标志引导入岔。

岔路口与环岛（本赛道两个关键场景）：
  岔口几何：直行道贯通 + 右侧岔路（非 Y 形）。
  * 有右转标志 → 走右边岔路：记忆任务 → 岔口横条触发（岔口直行道贯通，
    横条被判 kind='cross'、side 不可靠，此时信任标志方向）→ BranchExecutor
    让检测器每行取最右侧线段锁定岔线，PD 闭环循线驶入，不开环猛转
    （平滑岔路用直角弯的固定角速度必然转过/不够）。
  * 无右转标志 → 直行通过岔口（'cross' 横条不触发任何转向，循迹直行），
    之后遇单侧直角弯横条（kind='corner'）自动触发 TurnExecutor 三段式直角转弯。
  * 环岛标志 → 入口T口横条触发 → RoundaboutExecutor（写死动作链）：
    入口T口三段式左转 → 闭环循迹绕环（岔口选边关闭，直行过第 1 个
    路口）→ 第 2 个路口三段式左转出环 → 回正常循迹。环上路口按
    横条 nearness 计数（离开路口重新武装，一个路口只计一次）。
  * 无标志相切环岛（几何序列触发）：正Y 默认走左(原方向)；处理完全部
    锥桶且连续 fork_right_cone_clear_sec 秒未再检测到锥桶后，才启用入口
    识别。随后反Y/正Y都需连续多帧确认，正Y必须在反Y确认后的
    fork_right_window_sec 秒内出现——不再设置最小时间间隔——
    判定为环岛入口，唯一一次切右入环，fork_right_sec 后恢复左侧并永久锁存；
    若锥桶进度停在 1/2 达 cone_missing_timeout_sec，则自动视为全部通过，
    再从该时刻开始计算上述连续无锥桶时间；
    出环岛时再遇 反Y→正Y 序列因锁存不再触发，仍走左(原方向)。

方向约定：x>0 前进；y>0 左移；逻辑 z>0 = 左转、z<0 = 右转
（与 04/05 实验可执行代码一致；run.py 出口处有 --z-flip 物理兜底）。
"""
import logging

from detection.traffic_sign_config import SIGN_BEHAVIORS

logger = logging.getLogger(__name__)

# 立即生效、不进记忆/执行流程的行为
# （人行横道按本轮规则为"确认后低速通过 N 秒"的短时速度调制，也属立即生效）
IMMEDIATE_BEHAVIORS = {'speed_limit', 'end_speed_limit', 'green_light', 'red_light',
                       'pedestrian_crossing'}

# 需要"横条到达执行区"触发的行为（环岛同转向：只在路口横条触发）
# 掉头/变道不在此列：本赛道掉头点与十字路口处无可依赖横条，确认标志后
# 均按时间触发（写死）
CORNER_TRIGGER_BEHAVIORS = {'left_turn', 'right_turn', 'roundabout'}


class TurnExecutor:
    """转向执行器（左/右转）：包装 FollowerController 的三段式直角转弯。

    固定角速度转弯 → 线重新可见后 PD 辅助回正 → 回正退出 / 超时停车。
    比纯计时开环可靠：转完能真正重新捕获线路。
    """

    def __init__(self, follower, side):
        self.follower = follower
        self.side = 1 if side > 0 else -1

    def start(self):
        self.follower.begin_turn(self.side)

    def step(self, det, dt):
        state, speed, z, done = self.follower.turn_step(det, dt)
        return (int(speed), 0, int(z)), state, done


class SequenceExecutor:
    """时序执行器（掉头/环岛/变道/绕锥）：按 ((秒, x, y, z), ...) 逐段推进。

    用 dt 累计计时而非墙钟，方便离线断言测试。
    """

    def __init__(self, sequence):
        self.sequence = tuple(sequence)
        self.elapsed = 0.0

    def start(self):
        self.elapsed = 0.0

    def step(self, det, dt):
        self.elapsed += max(dt, 0.0)
        acc = 0.0
        for index, (sec, x, y, z) in enumerate(self.sequence):
            acc += sec
            if self.elapsed < acc:
                return (int(x), int(y), int(z)), 'seq-%d' % index, False
        return (0, 0, 0), 'seq-done', True


class BranchExecutor:
    """岔路执行器（有转向标志时走岔路）：闭环循线驶入岔路，不开环猛转。

    平滑岔路不是 90° 弯，固定角速度开环转弯必然转过/转不够；正确做法是
    "选线"：把 LineDetector.branch_side 设为岔路侧，检测器每行改取最靠
    岔路侧的有效线段，岔路分支即成为被循的那条"线"，PD 自然驶入。

    安全边界：总时长 duration_sec 到点结束；失线超过 lost_sec 也结束。
    结束/被取消时都会把 branch_side 还原为 0（正常最宽段循迹）。
    """

    def __init__(self, follower, detector, side,
                 speed=120, duration_sec=2.5, lost_z=150, lost_sec=1.0):
        self.follower = follower
        self.detector = detector
        # side=0：不改检测器选线，正常循最宽段（雪糕筒过岔用——锥挡住错岔，
        # 唯一连通线即正确路线，无需反向选线）
        self.side = 0 if side == 0 else (1 if side > 0 else -1)
        self.speed = int(speed)
        self.duration_sec = float(duration_sec)
        self.lost_z = int(lost_z)
        self.lost_sec = float(lost_sec)

    def start(self):
        self.elapsed = 0.0
        self.lost = 0.0
        if self.detector is not None and self.side != 0:
            self.detector.branch_side = self.side
        self.follower.force_started()      # 行进中接管，不停车重新确认

    def step(self, det, dt):
        dt = max(dt, 0.0)
        self.elapsed += dt
        valid = det is not None and det.get('is_valid')
        self.lost = 0.0 if valid else self.lost + dt
        if self.elapsed >= self.duration_sec or self.lost >= self.lost_sec:
            self._release()
            return (0, 0, 0), 'branch-done', True
        if valid:
            _, x, y, z = self.follower.step(det, dt, self.speed)
            return (x, y, z), 'branch-follow', False
        # 岔线短暂丢失：向岔路侧缓转等待重捕获（不进入旋转搜索）
        return (int(self.speed * 0.3), 0, int(-self.side * self.lost_z)), \
            'branch-lost', False

    def cancel(self):
        """被红灯等抢占时调用：务必还原检测器选线模式。"""
        self._release()

    def _release(self):
        if self.detector is not None:
            self.detector.branch_side = 0


class RoundaboutExecutor:
    """环岛执行器（本赛道写死动作链）：
    入口T口三段式左转 → 闭环循迹绕环 → 第 1 个路口主动右转找线 →
    第 2 个路口三段式左转出环 → 完成回正常循迹。

    环上路口用横条检测计数（corner/cross 且 nearness>=junc_trigger）。
    横条必须连续 confirm_frames 帧才确认；计数后进入"未武装"状态，既要
    连续 miss_frames 帧无合格横条，又要距离上次计数至少 min_gap_sec 秒，
    才能重新武装，防止同一路口闪断后被重复计成第二个路口。入环
    左转刚结束时入口横条可能仍在视野，同样以未武装状态进入绕环段，
    防止把入口横条误计为第 1 个路口。
    绕环段把检测器岔口选边关闭(fork_side=0)。第 1 个路口一旦进入
    连续确认阶段便输出固定右转角速度；确认后至少右转 first_right_min_sec，
    等横条消失且主线连续 first_line_frames 帧稳定后才恢复闭环循迹。
    first_right_max_sec 是右转找线的安全上限。结束/被取消时恢复选边原值。
    """

    def __init__(self, follower, detector=None, follow_speed=120,
                 junc_trigger=0.5, miss_frames=10, max_sec=60.0,
                 first_straight_sec=2.0, confirm_frames=3,
                 min_gap_sec=3.0, first_right_speed=100,
                 first_right_z=160, first_right_min_sec=0.8,
                 first_right_max_sec=4.0, first_line_frames=4,
                 first_line_error=45.0, first_line_angle=35.0):
        self.follower = follower
        self.detector = detector
        self.follow_speed = int(follow_speed)
        self.junc_trigger = float(junc_trigger)
        self.miss_frames = max(1, int(miss_frames))
        self.confirm_frames = max(1, int(confirm_frames))
        self.min_gap_sec = max(0.0, float(min_gap_sec))
        self.max_sec = float(max_sec)
        # 保留旧属性/参数，避免已有启动脚本因 --rb-first-straight-sec 失效；
        # 第1路口现已由“固定右转 + 稳定找线”接管，不再使用直行保护计时。
        self.first_straight_sec = max(0.0, float(first_straight_sec))
        self.first_right_speed = max(0, int(first_right_speed))
        self.first_right_z = abs(int(first_right_z))
        self.first_right_min_sec = max(0.0, float(first_right_min_sec))
        self.first_right_max_sec = max(self.first_right_min_sec,
                                       float(first_right_max_sec))
        self.first_line_frames = max(1, int(first_line_frames))
        self.first_line_error = max(0.0, float(first_line_error))
        self.first_line_angle = max(0.0, float(first_line_angle))
        self._fork_side_save = None

    def start(self):
        self.phase = 'enter'              # enter → ring → exit
        self.elapsed = 0.0
        self.junctions = 0
        self._armed = False               # 入口横条仍在视野，先不武装
        self._miss = 0
        self._near_count = 0              # 横条连续确认帧数
        self._last_junction_elapsed = None
        self._first_right_active = False  # 第1路口固定右转找线阶段
        self._first_right_elapsed = 0.0
        self._first_line_count = 0
        self._first_right_timeout_logged = False
        self.follower.force_started()
        self.follower.begin_turn(-1)      # 入口T口：左转
        if self.detector is not None:
            self._fork_side_save = self.detector.fork_side
            self.detector.fork_side = 0
        logger.info('环岛：入口T口左转；之后绕环，第1路口右转找线、第2路口左转出环')

    def step(self, det, dt):
        dt = max(dt, 0.0)
        self.elapsed += dt
        if self.elapsed >= self.max_sec:
            logger.warning('环岛：超时 %.0fs 强制结束', self.max_sec)
            self._release()
            return (0, 0, 0), 'rb-max', True
        if self.phase in ('enter', 'exit'):
            _, speed, z, done = self.follower.turn_step(det, dt)
            state = 'rb-enter' if self.phase == 'enter' else 'rb-exit'
            if not done:
                return (int(speed), 0, int(z)), state, False
            if self.phase == 'enter':
                self.phase = 'ring'
                self._armed = False       # 入口横条还在视野，离开后才武装
                self._miss = 0
                self._near_count = 0
                self.follower.reset_after_action()
                self.follower.force_started()
                logger.info('环岛：入环左转完成，开始绕环')
                return (int(speed), 0, 0), 'rb-ring', False
            logger.info('环岛：出环左转完成')
            self._release()
            return (0, 0, 0), 'rb-done', True
        # ring：闭环循迹绕环 + 横条路口计数
        corner = det.get('corner') if det and det.get('is_valid') else None
        near_bar = (corner is not None
                    and corner['nearness'] >= self.junc_trigger)
        # 只有计数器已经离开入口横条并重新武装，才把横条当作第1路口候选；
        # 否则入环口残留横条会让车辆刚入环就误向右打。
        first_junction_candidate = (self.junctions == 0 and near_bar
                                    and self._armed)
        if near_bar:
            self._miss = 0
            self._near_count += 1
            if self._armed and self._near_count >= self.confirm_frames:
                self._armed = False
                self._near_count = 0
                self.junctions += 1
                self._last_junction_elapsed = self.elapsed
                if self.junctions >= 2:
                    self.phase = 'exit'
                    self.follower.begin_turn(-1)   # 第 2 路口：左转出环
                    logger.info('环岛：第 2 个路口，左转出环')
                    return (0, 0, 0), 'rb-exit', False
                self._first_right_active = True
                self._first_right_elapsed = 0.0
                self._first_line_count = 0
                self._first_right_timeout_logged = False
                logger.info('环岛：第 1 个路口，固定右转找线 '
                            '(x=%d z=-%d，稳定%d帧后恢复循迹)',
                            self.first_right_speed, self.first_right_z,
                            self.first_line_frames)
        else:
            self._near_count = 0
            self._miss += 1
            gap_ok = (self._last_junction_elapsed is None
                      or self.elapsed - self._last_junction_elapsed >= self.min_gap_sec)
            if (not self._first_right_active
                    and self._miss >= self.miss_frames and gap_ok):
                self._armed = True

        # 第1路口：从候选确认帧起就明确给右转速度，绝不让受左支路影响的
        # 循迹拟合输出左转。确认后必须先达到最短右转时间，再以“横条消失、
        # 误差和角度均正常”的连续帧作为抓到正确主线的依据。
        if first_junction_candidate and not self._first_right_active:
            return (self.first_right_speed, 0, -self.first_right_z), \
                'rb-first-right', False
        if self._first_right_active:
            self._first_right_elapsed += dt
            line_ok = (det is not None and det.get('is_valid')
                       and not near_bar
                       and abs(float(det.get('error_px', 0.0))) <= self.first_line_error
                       and abs(float(det.get('angle_deg', 0.0))) <= self.first_line_angle)
            if self._first_right_elapsed >= self.first_right_min_sec and line_ok:
                self._first_line_count += 1
            else:
                self._first_line_count = 0
            found_line = self._first_line_count >= self.first_line_frames
            timed_out = self._first_right_elapsed >= self.first_right_max_sec
            if found_line:
                self._first_right_active = False
                self._first_line_count = 0
                # 找线帧不同时作为第二路口的“清空帧”；恢复循迹后重新累计，
                # 进一步隔离第1路口与第2路口。
                self._miss = 0
                self._near_count = 0
                self._armed = False
                self.follower.reset_after_action()
                self.follower.force_started()
                logger.info('环岛：第 1 个路口已找到稳定主线，恢复绕环循迹')
                if det is not None and det.get('is_valid'):
                    _, x, y, z = self.follower.step(det, dt, self.follow_speed)
                    return (x, y, z), 'rb-ring', False
                return (int(self.follow_speed * 0.3), 0, 0), 'rb-rlost', False
            if timed_out:
                # 不把控制权交还给可能仍被左支路吸偏的循迹拟合。停止等待，
                # 摄像头仍继续检测；一旦稳定主线累计够帧，下一帧正常恢复。
                if not self._first_right_timeout_logged:
                    logger.warning('环岛：第 1 个路口右转找线 %.1fs 超时，'
                                   '停车等待稳定主线', self.first_right_max_sec)
                    self._first_right_timeout_logged = True
                return (0, 0, 0), 'rb-first-wait-line', False
            return (self.first_right_speed, 0, -self.first_right_z), \
                'rb-first-right', False

        if det is not None and det.get('is_valid'):
            _, x, y, z = self.follower.step(det, dt, self.follow_speed)
            return (x, y, z), 'rb-ring', False
        # 环线短暂丢失：低速直行等待重捕获（不旋转搜索）
        return (int(self.follow_speed * 0.3), 0, 0), 'rb-rlost', False

    def cancel(self):
        """被红灯等抢占时调用：务必还原检测器岔口选边。"""
        self._release()

    def _release(self):
        if self.detector is not None and self._fork_side_save is not None:
            self.detector.fork_side = self._fork_side_save
            self._fork_side_save = None


class UturnExecutor:
    """掉头执行器（本赛道写死动作链）：
    原地旋转(spin_sec, z=spin_z) → 闭环循迹 follow_sec → 三段式直角转弯
    (turn_side) → 完成回正常循迹。

    原地 360° 自转后车头朝向不变，循迹段用 PD 闭环沿线行进（不开环计时
    走位，保证 1s 后车确实在线上），转弯段复用直角转弯的固定角速度
    +PD 辅助回正，转完真正重新捕获线路。
    follow_sec<=0 或 turn_side==0 时退化为纯原地掉头（转完即完成）。
    """

    def __init__(self, follower, spin_sec, spin_z, follow_speed,
                 follow_sec=0.0, turn_side=0):
        self.follower = follower
        self.spin_sec = float(spin_sec)
        self.spin_z = int(spin_z)
        self.follow_speed = int(follow_speed)
        self.follow_sec = float(follow_sec)
        self.turn_side = 1 if turn_side > 0 else (-1 if turn_side < 0 else 0)

    def start(self):
        self.phase = 'spin'
        self.elapsed = 0.0
        self.follower.force_started()      # 后续循迹段行进中接管，不停车确认
        logger.info('掉头：开始原地旋转 %.1fs（之后循迹 %.1fs %s）',
                    self.spin_sec, self.follow_sec,
                    '再右转' if self.turn_side > 0 else
                    ('再左转' if self.turn_side < 0 else '后回循迹'))

    def step(self, det, dt):
        dt = max(dt, 0.0)
        if self.phase == 'spin':
            self.elapsed += dt
            if self.elapsed < self.spin_sec:
                # 原地掉头：x/y 恒 0，z 恒定旋转（顺时针=逻辑 z<0）
                return (0, 0, self.spin_z), 'uturn-spin', False
            if self.follow_sec <= 0 or self.turn_side == 0:
                return (0, 0, 0), 'uturn-done', True
            self.phase, self.elapsed = 'follow', 0.0
            # 清掉自转前的滤波/微分状态，循迹段从干净状态带斜坡起步
            self.follower.reset_after_action()
            self.follower.force_started()
        if self.phase == 'follow':
            self.elapsed += dt
            if self.elapsed >= self.follow_sec:
                self.phase = 'turn'
                self.follower.begin_turn(self.turn_side)
                logger.info('掉头：循迹 %.1fs 结束，开始直角转弯', self.follow_sec)
            elif det is not None and det.get('is_valid'):
                _, x, y, z = self.follower.step(det, dt, self.follow_speed)
                return (x, y, z), 'uturn-follow', False
            else:
                # 线短暂丢失：低速直行等待重捕获（不进入旋转搜索）
                return (int(self.follow_speed * 0.3), 0, 0), 'uturn-flost', False
        if self.phase == 'turn':
            _, speed, z, done = self.follower.turn_step(det, dt)
            return (int(speed), 0, int(z)), 'uturn-turn', done


class LaneShiftExecutor:
    """变道执行器（本赛道写死动作）：x=0 纯横向平移(lateral>0 向左)，
    直到新线的黑线进入视野且基本居中 → 完成回循迹。

    两阶段（起步时旧线仍在视野内，不能直接拿"看见线"当完成条件）：
      leave: 平移脱离旧线——旧线连续 lost_frames 帧丢失，或偏差
             side*err >= abandon_err（旧线已滑到画面边缘）→ 进入 seek；
      seek : 继续平移，新线有效且 |err|<=capture_err 连续 capture_frames
             帧 → 完成；
    安全兜底：max_sec 到点强制完成（交还仲裁回循迹）。
    """

    def __init__(self, follower, lateral, max_sec=8.0, abandon_err=70,
                 capture_err=40, lost_frames=3, capture_frames=2):
        self.follower = follower
        self.lateral = int(lateral)
        self.side = 1 if lateral > 0 else -1
        self.max_sec = float(max_sec)
        self.abandon_err = float(abandon_err)
        self.capture_err = float(capture_err)
        self.lost_frames = max(1, int(lost_frames))
        self.capture_frames = max(1, int(capture_frames))

    def start(self):
        self.phase = 'leave'
        self.elapsed = 0.0
        self._lost = 0
        self._cap = 0
        self.follower.force_started()      # 行进中接管，不停车重新确认
        logger.info('变道：开始纯横向平移（%s），直到前方见线',
                    '左' if self.side > 0 else '右')

    def step(self, det, dt):
        dt = max(dt, 0.0)
        self.elapsed += dt
        if det is not None and det.get('is_valid'):
            valid, err = True, float(det.get('error_px', 0.0))
        else:
            valid, err = False, 0.0
        if self.elapsed >= self.max_sec:
            logger.info('变道：平移超时 %.1fs，强制回循迹', self.max_sec)
            return (0, 0, 0), 'shift-max', True
        if self.phase == 'leave':
            if valid:
                self._lost = 0
                if self.side * err >= self.abandon_err:
                    self.phase = 'seek'     # 旧线已甩到画面边缘，视为脱离
            else:
                self._lost += 1
                if self._lost >= self.lost_frames:
                    self.phase = 'seek'     # 旧线真正丢失
        else:  # seek
            if valid and abs(err) <= self.capture_err:
                self._cap += 1
                if self._cap >= self.capture_frames:
                    logger.info('变道：新线已入视野居中，平移结束')
                    return (0, 0, 0), 'shift-done', True
            else:
                self._cap = 0
        return (0, self.lateral, 0), 'shift-' + self.phase, False


class Arbiter:
    """循迹与标志行为的仲裁状态机。

    update(dt, det, sign_event, now) -> (x, y, z逻辑, status)
      det:        LineDetector.process() 的返回 dict（无效帧 is_valid=False）
      sign_event: None 或 {'sign': '左转', 'conf': 0.9,
                           'context': {'center_x_norm': .., 'bbox_h_ratio': ..}}
      now:        墙钟秒（由调用方注入，便于测试）
    """

    def __init__(self, follower, line_detector=None,
                 cruise_speed=250, slow_speed=140,
                 crosswalk_speed=100, crosswalk_sec=3.0,
                 memory_hold_sec=4.0, cooldown_sec=4.0,
                 approach_scale=0.5, exec_trigger=0.60,
                 match_corner_side=True, lead_sec=0.3,
                 lane_side='left', cone_trigger_h=0.25,
                 cone_lead_sec=2.5, cone_hold_sec=2.0,
                 cone_branch_sec=5.0, cone_total=2,
                 cone_decide_sec=0.3,
                 uturn_sec=7.2, uturn_z=500, uturn_right=False,
                 uturn_delay_sec=2.0,
                 uturn_follow_sec=1.0, uturn_turn_side=1,
                 turn_hold_sec=20.0,
                 turn_style='auto', first_turn_right=True,
                 branch_speed=120, branch_sec=2.5,
                 branch_lost_z=150, branch_lost_sec=1.0,
                 auto_corner=True, auto_corner_trigger=0.75,
                 auto_corner_frames=3,
                 rb_loop_speed=120, rb_junc_trigger=0.5, rb_max_sec=60.0,
                 fork_right_once=True, fork_right_window_sec=5.0,
                 fork_right_confirm_frames=3,
                 fork_right_sec=3.0, fork_right_z=80,
                 lane_x=100, lane_y=160, lane_straight_x=180,
                 lc_delay_sec=3.0, lc_max_sec=8.0,
                 fork_right_cone_clear_sec=20.0,
                 cone_missing_timeout_sec=20.0,
                 rb_memory_hold_sec=40.0, rb_first_straight_sec=2.0,
                 rb_junc_confirm_frames=3, rb_junc_clear_frames=10,
                 rb_junc_min_gap_sec=3.0, rb_first_right_speed=100,
                 rb_first_right_z=160, rb_first_right_min_sec=0.8,
                 rb_first_right_max_sec=4.0, rb_first_line_frames=4,
                 rb_first_line_error=45.0, rb_first_line_angle=35.0,
                 roundabout_confidence=0.80, uturn_confidence=0.75,
                 turn_confidence=0.75):
        self.follower = follower
        self.line_detector = line_detector      # 岔路选线用（可为 None）
        self.normal_speed = int(cruise_speed)
        self.slow_speed = int(slow_speed)
        self.crosswalk_speed = int(crosswalk_speed)
        self.crosswalk_sec = float(crosswalk_sec)
        self.memory_hold_sec = float(memory_hold_sec)
        self.cooldown_sec = float(cooldown_sec)
        self.approach_scale = float(approach_scale)
        self.exec_trigger = float(exec_trigger)
        self.match_corner_side = bool(match_corner_side)
        self.lead_sec = float(lead_sec)
        self.cone_trigger_h = float(cone_trigger_h)
        # 雪糕筒封岔专用状态机参数（与通用标志任务参数互不共用）
        self.cone_lead_sec = float(cone_lead_sec)
        self.cone_hold_sec = float(cone_hold_sec)
        self.cone_branch_sec = float(cone_branch_sec)
        self.cone_total = max(0, int(cone_total))
        # 触发前的判定稳定时间：防止刚结束变道等动作、姿态未稳时同拍误触发
        self.cone_decide_sec = float(cone_decide_sec)
        # 动作参数（掉头写死动作链：确认标志 uturn_delay_sec 后原地旋转
        # 一周(spin)=x/y 恒 0、z 恒定，时长 uturn_sec 按实车标定总转角≈360°；
        # 随后闭环循迹 uturn_follow_sec 秒，再执行一次三段式直角转弯
        # (uturn_turn_side, 0=不转直接回循迹)；uturn_right=True 顺时针(z<0)）
        self.uturn_sec = float(uturn_sec)
        self.uturn_z = int(uturn_z)
        self.uturn_right = bool(uturn_right)
        self.uturn_delay_sec = float(uturn_delay_sec)
        self.uturn_follow_sec = float(uturn_follow_sec)
        self.uturn_turn_side = 1 if uturn_turn_side > 0 else \
            (-1 if uturn_turn_side < 0 else 0)
        # 转向任务的记忆保持：转向只在路口触发（横条/岔口线索），不设时间
        # 兜底盲转——路中间原地盲转比漏转糟得多。任务最长保持 turn_hold_sec
        # 秒，一直等不到路口线索才丢弃（防误检标志长期占用 memorized）
        self.turn_hold_sec = float(turn_hold_sec)
        # 岔路（有转向标志时入岔；turn_style='auto' 按触发线索自动选择：
        # 岔口线索→闭环循线入岔，横条/时间兜底→三段式直角转弯）
        self.turn_style = str(turn_style)
        # 写死规则：启动后第一个转向标志(左/右)统一按右转处理——本轮 YOLO
        # 对左/右转标志区分不准，第一站固定右转；之后按识别类别
        self.first_turn_right = bool(first_turn_right)
        self.branch_speed = int(branch_speed)
        self.branch_sec = float(branch_sec)
        self.branch_lost_z = int(branch_lost_z)
        self.branch_lost_sec = float(branch_lost_sec)
        # 无标志直角弯自动转弯（只认单侧横条 kind='corner'，岔口 cross 不触发）
        self.auto_corner = bool(auto_corner)
        self.auto_corner_trigger = float(auto_corner_trigger)
        self.auto_corner_frames = max(1, int(auto_corner_frames))
        # 环岛（入口T口左转 → 闭环绕环，第1路口固定右转找到稳定主线，
        # 第2路口左转出环；环上路口按横条 nearness 计数并做重复抑制）
        self.rb_loop_speed = int(rb_loop_speed)
        self.rb_junc_trigger = float(rb_junc_trigger)
        self.rb_max_sec = float(rb_max_sec)
        self.rb_memory_hold_sec = max(0.0, float(rb_memory_hold_sec))
        self.rb_first_straight_sec = max(0.0, float(rb_first_straight_sec))
        self.rb_junc_confirm_frames = max(1, int(rb_junc_confirm_frames))
        self.rb_junc_clear_frames = max(1, int(rb_junc_clear_frames))
        self.rb_junc_min_gap_sec = max(0.0, float(rb_junc_min_gap_sec))
        self.rb_first_right_speed = max(0, int(rb_first_right_speed))
        self.rb_first_right_z = abs(int(rb_first_right_z))
        self.rb_first_right_min_sec = max(0.0, float(rb_first_right_min_sec))
        self.rb_first_right_max_sec = max(self.rb_first_right_min_sec,
                                          float(rb_first_right_max_sec))
        self.rb_first_line_frames = max(1, int(rb_first_line_frames))
        self.rb_first_line_error = max(0.0, float(rb_first_line_error))
        self.rb_first_line_angle = max(0.0, float(rb_first_line_angle))
        self.roundabout_confidence = max(0.0, min(1.0,
                                                  float(roundabout_confidence)))
        self.uturn_confidence = max(0.0, min(1.0, float(uturn_confidence)))
        self.turn_confidence = max(0.0, min(1.0, float(turn_confidence)))
        # 环岛入口一次性右入：反Y/正Y分别连续确认若干帧，且正Y必须在
        # window_sec 内出现，才唯一一次切到右侧入环；之后恢复默认选边并
        # 永久锁存。无最小间隔，多帧确认仍用于压制几何分类的单帧抖动。
        self.fork_right_once = bool(fork_right_once)
        self.fork_right_window_sec = max(0.0, float(fork_right_window_sec))
        self.fork_right_confirm_frames = max(1, int(fork_right_confirm_frames))
        self.fork_right_sec = float(fork_right_sec)
        self.fork_right_z = abs(int(fork_right_z))
        # 小环岛入口识别门槛：全部锥桶处理完后，还要连续一段时间没有再次
        # 检测到锥桶，才允许反Y→正Y消耗唯一一次右入机会。
        self.fork_right_cone_clear_sec = max(0.0, float(fork_right_cone_clear_sec))
        # 只差最后一个锥桶时的兜底：等待超时仍未完成，就按全部通过处理。
        self.cone_missing_timeout_sec = max(0.0, float(cone_missing_timeout_sec))
        self._fork_side_default = (getattr(line_detector, 'fork_side', 0)
                                   if line_detector is not None else 0)
        self.lane_x = int(lane_x)
        self.lane_y = int(lane_y) if lane_side == 'left' else -int(lane_y)
        self.lane_straight_x = int(lane_straight_x)
        # 变道（十字路口，写死动作：确认后循迹 lc_delay_sec 秒 → x=0 纯横向
        # 平移，直到新线入视野居中；lc_max_sec 安全兜底）
        self.lc_delay_sec = float(lc_delay_sec)
        self.lc_max_sec = float(lc_max_sec)

        self.log = []
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self.state = 'follow'
        self.cruise_speed = self.normal_speed
        self.pending_task = None
        self.current_behavior = None
        self._cancel_executor()
        self._memorized_at = 0.0
        self._cooldown_until = {}
        self._crosswalk_until = 0.0
        self._auto_cooldown_until = 0.0
        self._auto_corner_count = 0
        self._turn_fork_frames = 0        # 转向任务：正Y岔口连续确认帧数
        self._first_turn_done = False     # 首个转向标志写死右转是否已用掉
        self._uturn_sign_seen = False     # 十字路口平移只有见过掉头标志后才解锁
        self._merge_seen_at = None        # 最近一次见到反Y(汇合)的时刻
        self._fork_merge_frames = 0       # 反Y连续确认帧数
        self._fork_fork_frames = 0        # 正Y连续确认帧数
        self._fork_right_used = False     # 一次性右入是否已用掉(锁存)
        self._fork_right_until = 0.0      # 右入选边恢复的截止时刻(0=未激活)
        self._fork_right_armed = False    # 锥桶进度+静默时间满足后才置 True
        self._fork_last_cone_seen_at = None  # 任意状态下最近一次确认锥桶的时刻
        self._all_cones_done_at = None    # 最后一个锥桶动作完成/超时补全的时刻
        self._cone_partial_since = None   # 已通过部分锥桶、等待最后一个的起始时刻
        self._cone_completion_assumed = False  # 是否由超时兜底补成全部通过
        self._last_update_at = 0.0        # 仅用于状态显示
        self._last_follower_state = 'start'
        # 雪糕筒封岔专用状态机运行时状态
        self.cone_handled = 0           # 已处理的雪糕筒数
        self._cone_blocked_side = 0     # 被封堵岔路侧：-1 左 / +1 右 / 0 未知
        self._cone_first_seen = 0.0
        self._cone_last_seen = 0.0
        self._cone_last_h = 0.0
        self._cone_debug = (0.5, 0.5)   # 最近一次封堵侧判定的 (锥x, 基准)
        self._cone_saved_fork_side = None   # cone_approach 期间暂存的岔口选边
        if self.line_detector is not None:
            self.line_detector.branch_side = 0
            self.line_detector.fork_side = self._fork_side_default
        self.follower.reset_after_action()

    def _cancel_executor(self):
        """中止当前执行器（红灯抢占/复位）：有清理动作的执行器必须清理。"""
        executor = getattr(self, 'executor', None)   # __init__ 首次 reset 时尚无此属性
        if executor is not None and hasattr(executor, 'cancel'):
            executor.cancel()
        self.executor = None

    # ------------------------------------------------------------------
    def update(self, dt, det, sign_event, now):
        self._last_update_at = float(now)
        # ---- 0) 一次性右入环岛的选边限时恢复（任何状态下都按时恢复默认侧） ----
        if self._fork_right_until and now >= self._fork_right_until:
            self._restore_fork_side()

        sign = None
        conf = 0.0
        context = None
        if sign_event:
            sign = sign_event.get('sign')
            conf = float(sign_event.get('conf', 0.0))
            context = sign_event.get('context')

        # 环岛类别容易在无标志场景误检：低于独立阈值的结果完全忽略，
        # 不进入任务记忆，更不会等待横条或执行入环动作。检测器已有同样
        # 的前置过滤，此处保留第二层防护，避免其他事件来源绕过检测器。
        if sign == '环岛' and conf < self.roundabout_confidence:
            self._log('忽略低置信度环岛: %.1f%% < %.1f%%'
                      % (conf * 100.0, self.roundabout_confidence * 100.0))
            sign = None
            context = None
        if sign == '掉头' and conf < self.uturn_confidence:
            self._log('忽略低置信度掉头: %.1f%% < %.1f%%'
                      % (conf * 100.0, self.uturn_confidence * 100.0))
            sign = None
            context = None
        if sign in ('左转', '右转') and conf < self.turn_confidence:
            self._log('忽略低置信度%s: %.1f%% < %.1f%%'
                      % (sign, conf * 100.0, self.turn_confidence * 100.0))
            sign = None
            context = None

        # 必须在红灯/执行中等提前 return 之前记录：只要识别线程确认了锥桶，
        # 小环岛入口的“连续未见锥桶”计时就应立即清零。
        if sign == '路锥':
            self._note_cone_seen(now)
        if sign == '掉头' and not self._uturn_sign_seen:
            self._uturn_sign_seen = True
            # 解锁时清掉此前误识别十字路口留下的冷却，避免真正的后续
            # 十字路口恰好落在冷却窗口内而被继续屏蔽。
            self._cooldown_until.pop('十字路口', None)
            self._log('已确认掉头标志：从现在起允许十字路口平移动作')

        # 第一个锥桶完成后长期没有完成第二个时，按赛道时序兜底补成 2/2。
        # 放在所有提前 return 之前，使红灯/其他动作期间计时也不会停住；但若
        # 当前确实正在接近或通过第二个锥桶，则等待该状态机自行给出结果。
        self._check_cone_completion_timeout(now)

        # ---- 1) 红灯抢占一切 ----
        if sign == '行人红灯' and self.state != 'red_wait':
            self._enter_red_wait(now, conf)

        # ---- 2) 红灯等待 ----
        if self.state == 'red_wait':
            if sign == '行人绿灯':
                self._release_red_wait(now, conf)
                # 放行后落入正常循迹（follower 已复位，重新起步确认）
            else:
                return 0, 0, 0, self.status()

        # ---- 3) 动作执行中：只响应红灯，其余标志留待完成后再次确认 ----
        if self.state == 'executing':
            cmd, phase_state, done = self.executor.step(det, dt)
            self._last_follower_state = phase_state
            if not done:
                return cmd[0], cmd[1], cmd[2], self.status()
            self._finish_execute(now)
            # 完成当拍落入循迹（follower 已复位，起步确认期间输出 0）

        # ---- 冷却过滤（红/绿灯不屏蔽） ----
        if sign is not None and sign not in ('行人红灯', '行人绿灯'):
            if now < self._cooldown_until.get(sign, 0.0):
                sign = None

        # 十字路口模型容易被其他标志误触发：在掉头标志真正出现前，彻底
        # 屏蔽 lane_change 任务，不进入记忆、更不会在3秒后执行横向平移。
        if sign == '十字路口' and not self._uturn_sign_seen:
            self._log('忽略十字路口：尚未确认掉头标志，平移动作未解锁')
            self._cooldown_until['十字路口'] = now + self.cooldown_sec
            sign = None

        # ---- 3.5) 雪糕筒封岔专用状态机（与环岛/转向/岔路等判断完全隔离） ----
        if sign == '路锥' or self.state in ('cone_approach', 'cone_branch'):
            cone_cmd = self._cone_step(dt, det, sign, context, now)
            if sign == '路锥':
                sign = None      # 路锥事件只进专用状态机，不进通用记忆/立即生效流程
            if cone_cmd is not None:
                return cone_cmd[0], cone_cmd[1], cone_cmd[2], self.status()
            # 返回 None：本拍已回落 follow（完成/误检丢弃/被忽略），继续后续正常流程

        # ---- 立即生效类 ----
        if sign is not None:
            behavior = SIGN_BEHAVIORS.get(sign)
            if behavior in IMMEDIATE_BEHAVIORS:
                self._apply_immediate(now, sign, behavior, conf)
                sign = None

        # ---- 4) 已记忆任务：更新上下文 / 检查执行区触发 / 超时丢弃 ----
        if self.state == 'memorized':
            self._refresh_pending_context(sign, context)
            self._check_memory(now, det)
            if self.state == 'executing':
                cmd, phase_state, done = self.executor.step(det, dt)
                self._last_follower_state = phase_state
                if not done:
                    return cmd[0], cmd[1], cmd[2], self.status()
                self._finish_execute(now)

        # ---- 5) 循迹（FOLLOW 或 MEMORIZED 接近段） ----
        if self.state == 'follow' and sign is not None:
            self._memorize(now, sign, conf, context)

        # ---- 5.5) 无标志直角弯自动转弯（标志任务优先：本拍已记忆则不自动转） ----
        if self.state == 'follow' and self.auto_corner:
            self._check_auto_corner(now, det)
            if self.state == 'executing':
                cmd, phase_state, done = self.executor.step(det, dt)
                self._last_follower_state = phase_state
                if not done:
                    return cmd[0], cmd[1], cmd[2], self.status()
                self._finish_execute(now)

        # ---- 5.6) 小环岛一次性右入（锥桶门槛后，反Y→正Y才允许触发） ----
        if self.state == 'follow':
            self._check_fork_right_once(now, det)

        cruise = self._effective_cruise(now)
        if self.state == 'memorized':
            cruise = cruise * self.approach_scale
        follower_state, x, y, z = self.follower.step(det, dt, cruise)
        if self._fork_right_until:
            # 小环岛右入不能只依赖视觉选线：实车上普通PD在第一拍之后会
            # 很快把右转指令衰减到接近0。保持期内强制一个小幅右转下限；
            # 若PD已经给出更强的右转则保留，不削弱它。
            z = min(int(z), -self.fork_right_z)
            follower_state = 'fork-right'
        self._last_follower_state = follower_state
        return x, y, z, self.status()

    # ------------------------------------------------------------------
    def _effective_cruise(self, now):
        cruise = self.cruise_speed
        if now < self._crosswalk_until:
            cruise = min(cruise, self.crosswalk_speed)
        return cruise

    def _apply_immediate(self, now, sign, behavior, conf):
        if behavior == 'speed_limit':
            if self.cruise_speed != self.slow_speed:
                self._log('%s: 限速 %d mm/s (%.2f)' % (sign, self.slow_speed, conf))
            self.cruise_speed = self.slow_speed
        elif behavior == 'end_speed_limit':
            if self.cruise_speed != self.normal_speed:
                self._log('%s: 恢复 %d mm/s (%.2f)' % (sign, self.normal_speed, conf))
            self.cruise_speed = self.normal_speed
        elif behavior == 'green_light':
            self._log('%s: 保持通行 (%.2f)' % (sign, conf))
        elif behavior == 'pedestrian_crossing':
            self._crosswalk_until = now + self.crosswalk_sec
            self._log('%s: 低速通过 %.1fs (%.2f)' % (sign, self.crosswalk_sec, conf))
        if behavior != 'red_light':
            self._cooldown_until[sign] = now + self.cooldown_sec

    def _memorize(self, now, sign, conf, context):
        behavior = SIGN_BEHAVIORS.get(sign)
        if behavior is None or behavior in IMMEDIATE_BEHAVIORS:
            return
        # 写死规则：启动后第一个转向标志(左/右)统一按右转处理——本轮 YOLO
        # 对左/右转标志区分不准，第一站固定右转；之后的转向标志按识别类别。
        if (self.first_turn_right and not self._first_turn_done
                and behavior in ('left_turn', 'right_turn')):
            self._first_turn_done = True
            if behavior != 'right_turn':
                self._log('首个转向标志写死改写: %s -> 右转 (%.2f)'
                          % (sign, conf))
                behavior = 'right_turn'
        self.pending_task = {
            'behavior': behavior,
            'sign': sign,
            'conf': float(conf),
            'memorized_at': now,
            'context': dict(context or {}),
        }
        self._memorized_at = now
        self._turn_fork_frames = 0
        self.state = 'memorized'
        self._log('记忆任务: %s -> %s (%.2f)' % (sign, behavior, conf))

    def _refresh_pending_context(self, sign, context):
        """记忆期间同标志再次确认时刷新上下文（如路锥距离逐渐变近）。"""
        if (sign is None or context is None or not self.pending_task
                or sign != self.pending_task['sign']):
            return
        self.pending_task['context'].update(context)

    def _check_memory(self, now, det):
        if not self.pending_task:
            self.state = 'follow'
            return
        behavior = self.pending_task['behavior']
        corner = det.get('corner') if det and det.get('is_valid') else None
        triggered = False

        if behavior in ('left_turn', 'right_turn'):
            want = -1 if behavior == 'left_turn' else 1
            cue = None
            if corner is not None and corner['nearness'] >= self.exec_trigger:
                if not self.match_corner_side:
                    cue = 'bar'
                else:
                    # 岔口直行道贯通时横条对称(kind='cross')、side 不可靠：
                    # 已确认转向标志即信任标志方向，不再要求横条偏向匹配
                    if corner['side'] == want or corner['kind'] == 'cross':
                        cue = 'bar'
            # 岔口线索：正Y分岔(两分支向远处张开)连续 2 帧 → 触发入岔。
            # 覆盖"前向Y岔口没有横条"的场景——那种岔口横条检测永远为
            # None，纯等横条必然 4s 超时丢弃、直行冲过岔口。
            if det is not None and det.get('is_valid') \
                    and det.get('junction') == 'fork':
                self._turn_fork_frames += 1
            else:
                self._turn_fork_frames = 0
            if cue is None and self._turn_fork_frames >= 2:
                cue = 'fork'
            # 转向不用时间兜底盲转：看不见路口线索就绝不转——盲转(路中间
            # 原地旋转)比漏转糟得多。任务保持到 turn_hold_sec 才丢弃(下方)。
            if cue is not None:
                self.pending_task['cue'] = cue   # 供 _start_execute 选执行器
                triggered = True
            # 横条已非常近且明确指向反方向(单侧 corner 且 side 相反)：
            # 说明该标志与此弯无关(误检/为下一站而设)。立即丢弃任务回 follow，
            # 否则 memorized 状态会屏蔽自动直角弯，等 4s 超时车已冲过弯。
            if (not triggered and self.match_corner_side
                    and corner is not None and corner['kind'] == 'corner'
                    and corner['side'] != want
                    and corner['nearness'] >= self.auto_corner_trigger):
                self._log('横条明确指向%s，与%s矛盾，丢弃任务交给自动直角弯'
                          % ('右' if corner['side'] > 0 else '左',
                             self.pending_task['sign']))
                # 丢弃后冷却：标志仍持续确认时立即重记忆会让 memorized 状态
                # 长期占用，挡住自动直角弯与后续标志（如右Y 前的误检十字
                # 挡住掉头标志）
                self._cooldown_until[self.pending_task['sign']] = \
                    now + self.cooldown_sec
                self.pending_task = None
                self.state = 'follow'
                return
        elif behavior == 'u_turn':
            # 本赛道掉头点无横条：确认标志后写死按时间触发（uturn_delay_sec，
            # 满足"确认后 1s 内"），不等任何横条/距离条件
            triggered = (now - self._memorized_at) >= self.uturn_delay_sec
        elif behavior == 'lane_change':
            # 写死时间触发（同掉头）：确认十字路口后继续循迹 lc_delay_sec
            # 秒即开始平移，不等横条（本赛道十字路口处无可依赖的横条特征）
            triggered = (now - self._memorized_at) >= self.lc_delay_sec
        elif behavior == 'roundabout':
            # 入口T口横条触发（同转向：只在路口触发，无时间兜底盲转；
            # 任务按 turn_hold_sec 长保持，等不到横条才丢弃）
            triggered = (corner is not None
                         and corner['nearness'] >= self.exec_trigger)

        if triggered:
            self._start_execute(now)
        else:
            # 转向/环岛任务保持更久(turn_hold_sec)：标志常提前几秒出现，
            # 必须等到路口线索(横条/岔口)才执行；其余任务按 memory_hold_sec。
            if behavior == 'roundabout':
                hold = self.rb_memory_hold_sec
            elif behavior in ('left_turn', 'right_turn'):
                hold = self.turn_hold_sec
            else:
                hold = self.memory_hold_sec
            if now - self._memorized_at > hold:
                self._log('任务超时丢弃: %s' % self.pending_task['sign'])
                # 丢弃后冷却：标志仍持续确认时不立刻重记忆，否则 memorized
                # 状态反复占用会挡住后续标志的记忆（如右Y 前的误检十字挡住掉头）
                if self.pending_task['sign']:
                    self._cooldown_until[self.pending_task['sign']] = \
                        now + self.cooldown_sec
                self.pending_task = None
                self.state = 'follow'

    def _start_execute(self, now):
        if not self.pending_task:
            self.state = 'follow'
            return
        behavior = self.pending_task['behavior']
        context = self.pending_task.get('context') or {}

        if behavior == 'left_turn':
            self.executor = self._make_turn_executor(
                -1, cue=self.pending_task.get('cue'))
        elif behavior == 'right_turn':
            self.executor = self._make_turn_executor(
                +1, cue=self.pending_task.get('cue'))
        elif behavior == 'u_turn':
            # 写死动作链：原地旋转一周 → 循迹 uturn_follow_sec → 直角转弯
            # (uturn_turn_side)。顺时针(右)=逻辑 z<0；循迹段 PD 闭环保证
            # 1s 后车仍在线上，转弯段转完真正重新捕获线路后回正常循迹。
            z = -self.uturn_z if self.uturn_right else self.uturn_z
            self.executor = UturnExecutor(
                self.follower, spin_sec=self.uturn_sec, spin_z=z,
                follow_speed=int(self.cruise_speed),
                follow_sec=self.uturn_follow_sec,
                turn_side=self.uturn_turn_side)
        elif behavior == 'roundabout':
            self.executor = RoundaboutExecutor(
                self.follower, detector=self.line_detector,
                follow_speed=self.rb_loop_speed,
                junc_trigger=self.rb_junc_trigger,
                max_sec=self.rb_max_sec,
                first_straight_sec=self.rb_first_straight_sec,
                confirm_frames=self.rb_junc_confirm_frames,
                miss_frames=self.rb_junc_clear_frames,
                min_gap_sec=self.rb_junc_min_gap_sec,
                first_right_speed=self.rb_first_right_speed,
                first_right_z=self.rb_first_right_z,
                first_right_min_sec=self.rb_first_right_min_sec,
                first_right_max_sec=self.rb_first_right_max_sec,
                first_line_frames=self.rb_first_line_frames,
                first_line_error=self.rb_first_line_error,
                first_line_angle=self.rb_first_line_angle)
            # 标志引导的环岛已接管全部路口选择，"反Y→正Y 一次性右入"的
            # 几何逻辑(为无标志相切环岛准备)永久锁存，防止出环后误触发
            self._fork_right_used = True
        elif behavior == 'lane_change':
            # 写死动作：x=0 纯横向平移(lane_y 已含方向，默认向左)，
            # 先脱离旧线再平移到新线入视野居中 → 完成回循迹
            self.executor = LaneShiftExecutor(
                self.follower, lateral=self.lane_y, max_sec=self.lc_max_sec)
        else:
            self._log('未知动作，安全停车: %s' % behavior)
            self.executor = SequenceExecutor(((0.20, 0, 0, 0),))

        self.current_behavior = behavior
        self.state = 'executing'
        self.executor.start()
        self._log('到达执行区，开始执行: %s' % self.pending_task['sign'])

    def _make_turn_executor(self, side, cue=None):
        """标志转向的执行器。turn_style='auto'(默认)按触发线索选择：
        岔口(正Y)线索 → 岔路模式(闭环循线入岔，平滑岔路不猛转)；
        横条线索(带横条的直角路口) → 三段式直角转弯——横条路口用岔路
        模式会被 _drop_bar_points 剔除的横条行干扰、锁住贯通直线径直
        冲过路口。--turn-style branch/corner 可强制固定一种。"""
        style = self.turn_style
        if style == 'auto':
            style = 'branch' if cue == 'fork' else 'corner'
        if style == 'branch':
            return BranchExecutor(self.follower, self.line_detector, side,
                                  speed=self.branch_speed,
                                  duration_sec=self.branch_sec,
                                  lost_z=self.branch_lost_z,
                                  lost_sec=self.branch_lost_sec)
        return TurnExecutor(self.follower, side=side)

    def _check_auto_corner(self, now, det):
        """无标志直角弯：单侧横条(kind='corner')连续确认且足够近 → 自动直角转弯。

        只认单侧横条；岔口/十字路口的对称横条(kind='cross')不触发——
        保证"无标志直行过岔口、有标志才入岔"，与标志引导互不干扰。
        """
        if now < self._auto_cooldown_until:
            self._auto_corner_count = 0
            return
        corner = det.get('corner') if det and det.get('is_valid') else None
        if (corner is not None and corner['kind'] == 'corner'
                and corner['nearness'] >= self.auto_corner_trigger):
            self._auto_corner_count += 1
        else:
            self._auto_corner_count = 0
        if self._auto_corner_count < self.auto_corner_frames:
            return
        self._auto_corner_count = 0
        side = corner['side'] if corner['side'] in (-1, 1) else 1
        self.pending_task = {'behavior': 'auto_corner', 'sign': None,
                             'conf': 0.0, 'memorized_at': now, 'context': {}}
        self.current_behavior = 'auto_corner'
        self.executor = TurnExecutor(self.follower, side=side)
        self.state = 'executing'
        self.executor.start()
        self._log('自动直角转弯: %s转 (nearness=%.2f)'
                  % ('右' if side > 0 else '左', corner['nearness']))

    def _check_fork_right_once(self, now, det):
        """无标志小环岛入口的一次性右入。

        先处理满 cone_total 个锥桶，再连续 fork_right_cone_clear_sec 秒没有
        检测到锥桶，才开始接收几何序列。启用前的反Y不会被记忆。启用后
        反Y、正Y必须分别连续 confirm_frames 帧成立，而且正Y确认时刻必须
        在反Y确认后的 window_sec 秒内，才切到右侧闭环入环；无最小间隔。

        出环岛时同样会先见反Y 再见正Y，但 _fork_right_used 已锁存，
        不再触发——岔口选边保持默认侧(左)，沿原方向出环。
        """
        if not self.fork_right_once or self._fork_right_used:
            return
        if not self._fork_right_armed:
            if not self._fork_right_gate_ready(now):
                # 门槛开启前的误判反Y绝不能带进真正入口的时间窗口。
                self._clear_fork_right_sequence()
                return
            self._fork_right_armed = True
            self._clear_fork_right_sequence()
            self._log('小环岛右入已启用：%d 个锥桶处理完成，连续 %.1fs 未见锥桶'
                      % (self.cone_handled, self.fork_right_cone_clear_sec))
        if self.line_detector is None or not det or not det.get('is_valid'):
            # 丢帧不能参与“连续多帧确认”，但已确认的反Y仍可在设定窗口
            # 内等待正Y，适应两种几何之间短暂只有单线的正常过渡。
            self._fork_merge_frames = 0
            self._fork_fork_frames = 0
            return

        # 已确认反Y超过最大窗口：整个序列作废。当前帧可以作为下一次
        # 反Y候选的第1帧，但旧序列绝不能与后续正Y拼接。
        if (self._merge_seen_at is not None
                and now - self._merge_seen_at > self.fork_right_window_sec):
            self._clear_fork_right_sequence()

        junction = det.get('junction')
        if junction == 'merge':
            self._fork_fork_frames = 0
            if self._merge_seen_at is None:
                self._fork_merge_frames += 1
                if self._fork_merge_frames >= self.fork_right_confirm_frames:
                    self._merge_seen_at = now
                    self._fork_merge_frames = 0
                    self._log('小环岛入口候选: 反Y连续%d帧确认，等待%.1fs内正Y'
                              % (self.fork_right_confirm_frames,
                                 self.fork_right_window_sec))
        elif junction == 'fork':
            self._fork_merge_frames = 0
            if self._merge_seen_at is None:
                self._fork_fork_frames = 0
                return
            self._fork_fork_frames += 1
            interval = now - self._merge_seen_at
            if (self._fork_fork_frames >= self.fork_right_confirm_frames
                    and interval <= self.fork_right_window_sec):
                self._fork_right_used = True
                self._fork_right_until = now + self.fork_right_sec
                self.line_detector.fork_side = 1
                self._log('环岛入口: 反Y/正Y各连续%d帧，间隔%.2fs，'
                          '唯一一次右入(保持%.1fs)'
                          % (self.fork_right_confirm_frames, interval,
                             self.fork_right_sec))
                self._clear_fork_right_sequence()
        else:
            # None/其他几何会打断“连续帧”，不能用零散误判拼出确认结果。
            self._fork_merge_frames = 0
            self._fork_fork_frames = 0

    def _clear_fork_right_sequence(self):
        """清除反Y→正Y候选序列及两侧连续帧计数。"""
        self._merge_seen_at = None
        self._fork_merge_frames = 0
        self._fork_fork_frames = 0

    def _note_cone_seen(self, now):
        """记录任意一次确认锥桶，并重新开始小环岛入口的静默计时。"""
        self._fork_last_cone_seen_at = float(now)
        if not self._fork_right_used:
            if self._fork_right_armed:
                self._log('再次检测到锥桶：小环岛右入暂停，重新静默计时')
            self._fork_right_armed = False
            self._clear_fork_right_sequence()

    def _fork_right_gate_ready(self, now):
        """全部锥桶已完成且此后连续未见锥桶达到设定时间。"""
        if (self.cone_total <= 0 or self.cone_handled < self.cone_total
                or self._all_cones_done_at is None):
            return False
        quiet_since = self._all_cones_done_at
        if self._fork_last_cone_seen_at is not None:
            quiet_since = max(quiet_since, self._fork_last_cone_seen_at)
        return now - quiet_since >= self.fork_right_cone_clear_sec

    def _check_cone_completion_timeout(self, now):
        """只差最后一个锥桶且等待超时时，兜底视为全部锥桶已经通过。"""
        if (self.cone_total <= 1
                or self.cone_handled != self.cone_total - 1
                or self._cone_partial_since is None
                or self.state in ('cone_approach', 'cone_branch')
                or now - self._cone_partial_since < self.cone_missing_timeout_sec):
            return
        old_count = self.cone_handled
        self.cone_handled = self.cone_total
        self._all_cones_done_at = float(now)
        self._cone_partial_since = None
        self._cone_completion_assumed = True
        self._fork_right_armed = False
        self._clear_fork_right_sequence()
        self._log('锥桶进度停在 %d/%d 超过 %.1fs，兜底视为 %d/%d；开始无锥桶静默计时'
                  % (old_count, self.cone_total, self.cone_missing_timeout_sec,
                     self.cone_total, self.cone_total))

    def _cone_completion_wait_remaining(self, now):
        """返回等待最后一个锥桶的兜底剩余秒数，供调试显示。"""
        if (self.cone_total <= 1
                or self.cone_handled != self.cone_total - 1
                or self._cone_partial_since is None):
            return 0.0
        return max(0.0, self.cone_missing_timeout_sec
                   - (now - self._cone_partial_since))

    def _fork_right_quiet_remaining(self, now):
        """返回小环岛入口门槛尚需的无锥桶秒数，供调试显示。"""
        if (self.cone_total <= 0 or self.cone_handled < self.cone_total
                or self._all_cones_done_at is None):
            return self.fork_right_cone_clear_sec
        quiet_since = self._all_cones_done_at
        if self._fork_last_cone_seen_at is not None:
            quiet_since = max(quiet_since, self._fork_last_cone_seen_at)
        return max(0.0, self.fork_right_cone_clear_sec - (now - quiet_since))

    def _restore_fork_side(self):
        """右入保持时间到：恢复默认岔口选边(左)。锁存已置，不会再右入。"""
        self._fork_right_until = 0.0
        if self.line_detector is not None:
            self.line_detector.fork_side = self._fork_side_default
        self._log('右入辅助结束: 取消小幅右转，岔口选边恢复默认，'
                  '此后正Y只走默认侧')

    def _cone_blocked_side_of(self, context, det):
        """按雪糕筒相对"路中心"的左右判定封堵侧：-1 左 / +1 右 / 0 无位置信息。

        基准优先级：岔口横条中心 > 画面中心。
        不能用循迹线中心(centroid)：本车 fork_side 默认锁一侧岔线，接近
        岔口时循迹目标已偏向某侧岔路，centroid 随之偏到该岔，位于岔口的
        路锥相对它会被判反（实测：锥在左岔被误判为封右岔）。
        也不能只用画面中心：车身偏斜接近岔口时整个岔口偏在画面一侧，
        挡左岔的锥也会落在画面中线右侧，导致封堵侧判反。
        """
        if not context:
            return 0
        cone_x = float(context.get('center_x_norm', 0.5))
        ref = 0.5
        ww = getattr(self.line_detector, 'work_width', 0) \
            if self.line_detector is not None else 0
        if det is not None and det.get('is_valid') and ww:
            corner = det.get('corner')
            if corner is not None:
                ref = float(corner['cx']) / ww
        self._cone_debug = (cone_x, ref)
        return -1 if cone_x < ref else 1

    # ------------------------------------------------------------------
    # 雪糕筒封岔专用状态机（本赛道仅 cone_total 个雪糕筒，全部在岔路口）
    # ------------------------------------------------------------------
    def _cone_step(self, dt, det, sign, context, now):
        """推进雪糕筒封岔状态机，返回 (x, y, z)；None 表示本拍落入正常流程。

        cone_approach：确认雪糕筒即进入——按雪糕筒相对路中心的左/右记录
          封堵侧（锥在左 = 左岔禁行），减速循迹接近岔口；期间持续用最新
          检测刷新封堵侧。满足任一触发（岔口横条到达 / 锥足够近 / lead
          兜底）即入岔；雪糕筒消失超过 cone_hold_sec 按误检丢弃（不动作、
          不计数）。
        cone_branch：BranchExecutor 闭环循线驶入未封堵岔路（不开环猛转），
          到点/失线结束 → 计数 + 冷却 → 回 follow。
        仅在 follow 状态下才会被雪糕筒触发；执行/记忆等其他状态下路锥
        事件直接忽略，不影响环岛/转向/变道等既有判断。
        """
        # follow 下新确认的雪糕筒：立即进入专用状态机（不占通用任务记忆）
        if (sign == '路锥' and self.state == 'follow'
                and self.cone_handled < self.cone_total):
            if not context:
                return None              # 无位置信息（罕见）：等下一次确认
            self._cone_blocked_side = self._cone_blocked_side_of(context, det)
            self._cone_first_seen = now
            self._cone_last_seen = now
            self._cone_last_h = float(context.get('bbox_h_ratio', 0.0))
            self.state = 'cone_approach'
            self.current_behavior = 'cone_fork'
            # 接近期间暂停岔口选边锁定：否则 fork_side 默认侧会在入岔决策前
            # 就把车拉进某条岔路（实测：左锁使车提前驶入被锥封堵的左岔）
            if self.line_detector is not None:
                self._cone_saved_fork_side = self.line_detector.fork_side
                self.line_detector.fork_side = 0
            self._log('确认雪糕筒(%d/%d)：封堵%s岔，减速接近岔口 (锥x=%.2f 基准=%.2f)' % (
                self.cone_handled + 1, self.cone_total,
                '左' if self._cone_blocked_side < 0 else '右',
                self._cone_debug[0], self._cone_debug[1]))

        if self.state == 'cone_approach':
            # 最新一次目击刷新封堵侧与距离（越靠近岔口估计越准）
            if sign == '路锥' and context:
                self._cone_blocked_side = self._cone_blocked_side_of(context, det)
                self._cone_last_seen = now
                self._cone_last_h = float(context.get(
                    'bbox_h_ratio', self._cone_last_h))
            # 雪糕筒消失超时：按误检丢弃，不动作、不计数，直接回循迹
            if now - self._cone_last_seen > self.cone_hold_sec:
                self._log('雪糕筒 %.1fs 未再出现，按误检丢弃' % self.cone_hold_sec)
                self.state = 'follow'
                self.current_behavior = None
                self._cone_restore_fork_side()
                return None
            corner = det.get('corner') if det and det.get('is_valid') else None
            # 触发前先给 cone_decide_sec 让封堵侧判定用最新检测稳定下来：
            # 刚进状态机那一拍可能刚结束变道等动作，姿态/基准都不可靠
            seen_for = now - self._cone_first_seen
            triggered = seen_for >= self.cone_decide_sec and (
                (corner is not None and corner['nearness'] >= self.exec_trigger)
                or self._cone_last_h >= self.cone_trigger_h
                or seen_for >= self.cone_lead_sec)
            if triggered:
                # 不反向选线：锥本身挡在错岔上，画面中唯一连通的线就是要走的
                # 路，正常循最宽段即可（实测反向选线会选中对侧噪段/旧线，把车
                # 拽向错误方向；首个岔口真线在锥侧，反向选线必然选错）
                self.executor = BranchExecutor(
                    self.follower, self.line_detector, 0,
                    speed=self.branch_speed, duration_sec=self.cone_branch_sec,
                    lost_z=self.branch_lost_z, lost_sec=self.branch_lost_sec)
                self.state = 'cone_branch'
                self.executor.start()
                self._log('雪糕筒到岔口(锥在%s侧)，循线通过 (锥x=%.2f 基准=%.2f)' % (
                    '左' if self._cone_blocked_side < 0 else '右',
                    self._cone_debug[0], self._cone_debug[1]))
                cmd, phase_state, done = self.executor.step(det, dt)
                self._last_follower_state = phase_state
                if not done:
                    return cmd
                self._finish_cone(now)
                return None
            # 未到岔口：减速循迹接近（本拍由状态机接管，屏蔽其他标志）
            cruise = self._effective_cruise(now) * self.approach_scale
            follower_state, x, y, z = self.follower.step(det, dt, cruise)
            self._last_follower_state = follower_state
            return (x, y, z)

        if self.state == 'cone_branch':
            cmd, phase_state, done = self.executor.step(det, dt)
            self._last_follower_state = phase_state
            if not done:
                return cmd
            self._finish_cone(now)
            return None

        return None

    def _cone_restore_fork_side(self):
        """恢复进入 cone_approach 时暂存的岔口选边（幂等）。

        注意：暂存值可能已过期——cone_approach/cone_branch 期间环岛
        一次性右入的保持时间(_fork_right_until)可能到期，此时
        _restore_fork_side 已把 fork_side 恢复为默认侧，不能再恢复为
        暂存的 +1，否则会让车在后续 Y 岔口误走右侧（实测：第一锥后
        fork_side 错误恢复为 +1，导致第二岔口前的 Y 岔口车走右岔冲向锥）。
        """
        if self._cone_saved_fork_side is None or self.line_detector is None:
            self._cone_saved_fork_side = None
            return
        saved = self._cone_saved_fork_side
        self._cone_saved_fork_side = None
        # 暂存的 +1 来自环岛右入，但右入保持已过期(_fork_right_used 锁存、
        # _fork_right_until 已清零)→ 恢复为默认侧，不能恢复为 +1
        if saved > 0 and self._fork_right_used and not self._fork_right_until:
            self.line_detector.fork_side = self._fork_side_default
        else:
            self.line_detector.fork_side = saved

    def _finish_cone(self, now):
        """雪糕筒岔路通过：计数 + 同锥冷却 + 防自动直角弯余波，回落循迹。"""
        self.cone_handled += 1
        if (self.cone_total > 0 and self.cone_handled >= self.cone_total
                and self._all_cones_done_at is None):
            self._all_cones_done_at = float(now)
            self._cone_partial_since = None
            self._fork_right_armed = False
            self._clear_fork_right_sequence()
        elif 0 < self.cone_handled < self.cone_total:
            # 每完成一个但尚未全部完成的锥桶，都从该时刻重新等待最后一个。
            self._cone_partial_since = float(now)
        self._log('雪糕筒岔路通过 (%d/%d)%s' % (
            self.cone_handled, self.cone_total,
            '，本图雪糕筒已全部处理' if self.cone_handled >= self.cone_total else ''))
        self._cooldown_until['路锥'] = now + self.cooldown_sec
        self._auto_cooldown_until = now + self.cooldown_sec
        self.state = 'follow'
        self.current_behavior = None
        self.executor = None
        self._cone_restore_fork_side()
        self.follower.reset_after_action()

    def _finish_execute(self, now):
        sign = self.pending_task['sign'] if self.pending_task else None
        self._log('动作完成: %s' % (sign or self.current_behavior))
        if sign:
            self._cooldown_until[sign] = now + self.cooldown_sec
        # 任何动作完成后都设自动转弯冷却，防止同一横条/余波立刻重触发
        self._auto_cooldown_until = now + self.cooldown_sec
        self.state = 'follow'
        self.pending_task = None
        self.current_behavior = None
        self.executor = None
        self.follower.reset_after_action()

    def _enter_red_wait(self, now, conf):
        self.state = 'red_wait'
        self.pending_task = None
        self.current_behavior = 'red_light'
        self._cancel_executor()
        self._cone_restore_fork_side()
        self.follower.reset_after_action()
        self._log('行人红灯: 立即停车等待绿灯 (%.2f)' % conf)

    def _release_red_wait(self, now, conf):
        self.state = 'follow'
        self.current_behavior = None
        self.follower.reset_after_action()
        self._log('行人绿灯: 重新起步 (%.2f)' % conf)

    # ------------------------------------------------------------------
    def status(self):
        return {
            'state': self.state,
            'pending_task': self.pending_task,
            'current_action': self.current_behavior,
            'cruise_speed': self._effective_cruise_now(),
            'speed_limited': self.cruise_speed == self.slow_speed,
            'follower_state': self._last_follower_state,
            'uturn_sign_seen': self._uturn_sign_seen,
            'cone_handled': self.cone_handled,
            'cone_total': self.cone_total,
            'fork_right_armed': self._fork_right_armed,
            'fork_right_used': self._fork_right_used,
            'fork_right_quiet_remaining': self._fork_right_quiet_remaining(
                self._last_update_at),
            'cone_completion_assumed': self._cone_completion_assumed,
            'cone_completion_wait_remaining': self._cone_completion_wait_remaining(
                self._last_update_at),
        }

    def _effective_cruise_now(self):
        # 状态显示用；_crosswalk_until 的精确判定在 update 内以注入 now 完成，
        # 这里仅返回当前持续巡航速度（限速调制后）。
        return self.cruise_speed

    def _log(self, msg):
        logger.info('%s', msg)
        self.log.append(msg)
        if len(self.log) > 100:
            self.log.pop(0)

    def recent_logs(self, n=8):
        return self.log[-n:]
