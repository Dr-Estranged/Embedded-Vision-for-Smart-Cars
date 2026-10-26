#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""不连接摄像头和底盘，验证融合仲裁器的状态机与动作映射。

直接注入合成的循迹检测结果与时间戳，覆盖：
  - 11 类标志 → 行为映射完整性
  - 限速持续 / 解除恢复、人行横道短时低速
  - 转向任务：记忆 → 横条触发执行区 → 逻辑 z 方向 → 完成后冷却
  - 横条偏向不匹配不触发、记忆超时丢弃
  - 掉头 / 变道 / 环岛 / 绕锥的触发与动作方向
  - 红灯执行中抢占、绿灯放行、红灯无冷却
  - 动作结束后重新起步确认
  - 转向执行器超时停车

运行：python self_check.py
"""
import os
import sys

CASE_DIR = os.path.dirname(os.path.abspath(__file__))
if CASE_DIR not in sys.path:
    sys.path.insert(0, CASE_DIR)

from core.arbiter import Arbiter
from core.line_follower import FollowerController
from detection.traffic_sign_config import SIGN_BEHAVIORS, TRAFFIC_SIGN_CLASSES


class FakeDetector:
    """最小检测器替身：带 branch_side / fork_side 属性。

    无 numpy 的机器上无法实例化真 LineDetector；仲裁器只读写这两个属性。
    """

    def __init__(self):
        self.branch_side = 0
        self.fork_side = -1


# ------------------------------------------------------------------
def make_det(valid=True, err=0.0, angle=0.0, corner=None, junction=None):
    return {'is_valid': valid, 'centroid': (160.0, 100.0),
            'error_px': err, 'angle_deg': angle, 'a': 0.0, 'b': 160.0,
            'points': [(160, 90 + i * 5, 4) for i in range(5)],
            'binary': None, 'roi_top': 50, 'line_type': 'black',
            'corner': corner, 'junction': junction}


def make_corner(kind='corner', side=-1, nearness=0.7):
    return {'kind': kind, 'side': side, 'row': 200, 'width': 120,
            'cx': 160.0, 'nearness': nearness}


def new_arbiter(**kw):
    follower = FollowerController(startup_frames=5, ramp_frames=20)
    # 测试默认关闭"首个转向标志写死右转"（各用例按识别类别验证），
    # 该规则由专门用例显式开启验证
    kw.setdefault('first_turn_right', False)
    return Arbiter(follower, **kw)


def tick(arb, t, det=None, sign=None, conf=0.9, context=None, dt=0.05):
    event = {'sign': sign, 'conf': conf, 'context': context} if sign else None
    return arb.update(dt, det if det is not None else make_det(), event, t)


def start_follow(arb, t0=0.0, n=8):
    """让 follower 完成起步确认（5 帧）并进入 run。"""
    cmd = None
    for i in range(n):
        cmd = tick(arb, t0 + i * 0.05)
    return cmd


# ------------------------------------------------------------------
def main():
    # 0. 映射完整性
    assert len(TRAFFIC_SIGN_CLASSES) == 11
    assert len(SIGN_BEHAVIORS) == 11
    assert set(SIGN_BEHAVIORS.values()) == {
        'cone_fork', 'lane_change', 'end_speed_limit', 'green_light',
        'pedestrian_crossing', 'red_light', 'roundabout', 'speed_limit',
        'left_turn', 'right_turn', 'u_turn'}
    print('PASS: 11 类标志映射完整')

    # 1. 基础循迹：起步确认后前进
    arb = new_arbiter()
    x, y, z, st = start_follow(arb)
    assert st['state'] == 'follow' and x > 0, (x, st)
    print('PASS: 起步确认后循迹前进 x=%d' % x)

    # 2. 限速持续 / 解除恢复
    arb = new_arbiter()
    start_follow(arb)
    tick(arb, 10.0, sign='限速40')
    assert arb.cruise_speed == 140
    for i in range(1, 5):                      # 无新标志也持续限速
        tick(arb, 10.0 + i * 0.05)
        assert arb.cruise_speed == 140
    tick(arb, 11.0, sign='解除40限速')
    assert arb.cruise_speed == 250
    print('PASS: 限速 140 持续生效，解除后恢复 250')

    # 3. 人行横道：短时低速通过
    arb = new_arbiter()
    start_follow(arb)
    tick(arb, 20.0, sign='人行横道')
    assert arb._effective_cruise(20.1) == 100
    assert arb._effective_cruise(23.2) == 250  # 默认 3 秒后恢复
    print('PASS: 人行横道确认后低速 3 秒，之后恢复')

    # 4. 左转完整流程(岔路模式)：记忆→减速→横条触发→闭环循线(线在左 z>0)→完成→冷却
    arb = new_arbiter(branch_sec=0.3, turn_style='branch')
    start_follow(arb)
    _, _, _, st = tick(arb, 30.0, sign='左转')
    assert st['state'] == 'memorized'
    x, _, _, _ = tick(arb, 30.05)              # 接近段减速
    assert x <= 125, x
    det_left = make_det(corner=make_corner(kind='corner', side=-1, nearness=0.7))
    _, _, _, st = tick(arb, 30.10, det=det_left)
    assert st['state'] == 'executing', st
    _, _, z, _ = tick(arb, 30.15, det=make_det(err=-40.0))  # 岔线在左 → 逻辑左转 z>0
    assert z > 0, z
    finish_t = None
    for i in range(2, 30):                     # 到点 branch_sec 完成
        _, _, _, st = tick(arb, 30.10 + i * 0.05,
                           det=make_det(err=0.0, angle=0.0))
        if st['state'] != 'executing':
            finish_t = 30.10 + i * 0.05
            break
    assert finish_t is not None and st['state'] == 'follow'
    _, _, _, st = tick(arb, finish_t + 1.0, sign='左转')   # 冷却期内忽略
    assert st['state'] == 'follow' and st['pending_task'] is None
    print('PASS: 左转(岔路模式) 记忆→横条触发→闭环 z>0→完成→4s 冷却')

    # 5. 横条偏向不匹配不触发；转向无时间兜底盲转——等不到路口线索就
    #    一直保持，到 turn_hold_sec(20s) 才丢弃；关闭校验后横条即可触发
    arb = new_arbiter()
    start_follow(arb)
    tick(arb, 40.0, sign='左转')
    det_right = make_det(corner=make_corner(kind='corner', side=+1, nearness=0.7))
    _, _, _, st = tick(arb, 40.05, det=det_right)
    assert st['state'] == 'memorized'          # 右横条不触发左转
    _, _, _, st = tick(arb, 43.05)
    assert st['state'] == 'memorized'          # 无 3s 盲转兜底
    _, _, _, st = tick(arb, 59.9)
    assert st['state'] == 'memorized'          # 20s 保持未到
    _, _, _, st = tick(arb, 60.2)              # 超过 turn_hold 20s → 丢弃
    assert st['state'] == 'follow' and st['pending_task'] is None
    arb = new_arbiter(match_corner_side=False)
    start_follow(arb)
    tick(arb, 40.0, sign='左转')
    _, _, _, st = tick(arb, 40.05, det=det_right)
    assert st['state'] == 'executing'
    print('PASS: 横条偏向校验 + 转向无盲转兜底(20s 保持后丢弃)')

    # 5.9 前向Y岔口(无横条)：junction='fork' 连续 2 帧即触发入岔，不等兜底
    arb = new_arbiter(branch_sec=0.2)
    start_follow(arb)
    tick(arb, 49.0, sign='右转')
    _, _, _, st = tick(arb, 49.05, det=make_det(junction='fork'))
    assert st['state'] == 'memorized'          # 单帧岔口不触发
    _, _, _, st = tick(arb, 49.10, det=make_det(junction='fork'))
    assert st['state'] == 'executing', st      # 连续 2 帧岔口 → 触发入岔
    print('PASS: 前向Y岔口(无横条)junction=fork 连续2帧触发入岔')

    # 5.95 写死规则：启动后首个转向标志(左或右)统一按右转；其后按识别类别。
    #      auto 执行器选择：横条(cross)线索 → 三段式直角转弯(固定 z<0)
    arb = new_arbiter(first_turn_right=True)
    start_follow(arb)
    tick(arb, 90.0, sign='左转')               # 首个转向标志 → 写死改写为右转
    assert arb.pending_task['behavior'] == 'right_turn'
    det_cross = make_det(corner=make_corner(kind='cross', nearness=0.7))
    _, _, z, st = tick(arb, 90.05, det=det_cross)
    assert st['state'] == 'executing' and z < 0, (z, st)  # 三段式右转固定 z<0
    assert arb.follower._corner_side == 1      # TurnExecutor(直角弯)，非岔路选线
    t = 90.05
    while arb.state == 'executing' and t < 98.5:  # 线一直丢 → 超时出弯回循迹
        t += 0.05
        tick(arb, t, det=make_det(valid=False))
    assert arb.state == 'follow', t
    tick(arb, t + 5.0, sign='左转')            # 冷却已过；第二个左转不再改写
    assert arb.pending_task['behavior'] == 'left_turn'
    print('PASS: 首个转向标志写死右转 + 横条线索走三段式直角转弯')

    # 5.5 岔口对称横条(kind='cross')：side 不可靠，有右转标志即信任标志方向触发入岔
    arb = new_arbiter(branch_sec=0.2)
    start_follow(arb)
    tick(arb, 45.0, sign='右转')
    _, _, _, st = tick(arb, 45.05,
                       det=make_det(corner=make_corner(kind='cross', side=-1,
                                                       nearness=0.7)))
    assert st['state'] == 'executing', st
    # 无标志过岔口：cross 横条不触发任何转向，保持循迹直行
    arb = new_arbiter()
    start_follow(arb)
    for i in range(6):
        _, _, _, st = tick(arb, 46.0 + i * 0.05,
                           det=make_det(corner=make_corner(kind='cross', side=+1,
                                                           nearness=0.9)))
    assert st['state'] == 'follow' and arb.executor is None
    print('PASS: 有标志岔口凭标志方向入岔；无标志岔口直行通过')

    # 5.7 标志与横条矛盾：左转任务 + 明确右侧 corner(nearness≥0.75) → 立即丢弃
    #     任务，自动直角弯接管右转（不再等 4s 超时导致冲弯/反向转）
    arb = new_arbiter()
    start_follow(arb)
    tick(arb, 47.0, sign='左转')
    det_cr = make_det(corner=make_corner(kind='corner', side=+1, nearness=0.8))
    _, _, _, st = tick(arb, 47.05, det=det_cr)
    assert st['state'] == 'follow' and arb.pending_task is None   # 矛盾即丢弃
    _, _, _, st = tick(arb, 47.10, det=det_cr)                    # 自动弯确认中
    assert st['state'] == 'follow'
    _, _, z, st = tick(arb, 47.15, det=det_cr)                    # 第 3 帧触发
    assert st['state'] == 'executing' and z < 0, (z, st)          # 自动右转接管
    print('PASS: 标志与横条矛盾时立即丢弃任务，自动直角弯接管')

    # 5.8 十字路口平移必须先见过掉头标志；解锁后再按写死时间触发。
    gate_arb = new_arbiter()
    start_follow(gate_arb)
    _, _, _, st = tick(gate_arb, 47.0, sign='十字路口')
    assert st['state'] == 'follow' and st['pending_task'] is None
    assert not st['uturn_sign_seen']
    _, _, _, st = tick(gate_arb, 47.1, sign='掉头')
    assert st['uturn_sign_seen'] and st['pending_task']['behavior'] == 'u_turn'

    # 解锁后的变道：memorized 期间横条形状无关，到 lc_delay_sec(默认3s)
    # 才开始纯横向平移；脱离旧线→见新线居中→回循迹；完成后冷却。
    arb = new_arbiter()
    start_follow(arb)
    arb._uturn_sign_seen = True
    tick(arb, 48.0, sign='十字路口')
    det_asym = make_det(corner=dict(make_corner(kind='cross', side=+1,
                                                nearness=0.9), sym=False))
    _, _, _, st = tick(arb, 50.9, det=det_asym)      # 3s 未到，横条无关
    assert st['state'] == 'memorized'
    x, y, z, st = tick(arb, 51.05, det=det_asym)     # 到点纯时间触发
    assert st['state'] == 'executing' and (x, y, z) == (0, 160, 0), (x, y, z, st)
    assert st['follower_state'] == 'shift-leave'
    # 旧线仍有效且不偏 → 继续平移；旧线偏到画面边缘(err≥70) → 进入 seek
    _, _, _, st = tick(arb, 51.10)
    assert st['follower_state'] == 'shift-leave'
    _, _, _, st = tick(arb, 51.15, det=make_det(err=80.0))
    assert st['follower_state'] == 'shift-seek'
    # 新线入视野且居中连续 2 帧 → 完成回循迹
    tick(arb, 51.20, det=make_det(err=10.0))
    _, _, _, st = tick(arb, 51.25, det=make_det(err=10.0))
    assert st['state'] == 'follow', st
    _, _, _, st = tick(arb, 51.30, sign='十字路口')  # 冷却期内不重记忆
    assert st['state'] == 'follow' and arb.pending_task is None
    _, _, _, st = tick(arb, 51.35, sign='掉头')      # 掉头不受冷却影响
    assert st['state'] == 'memorized'
    assert arb.pending_task['behavior'] == 'u_turn'
    print('PASS: 十字路口在掉头标志前禁用；解锁后3s触发平移并正常完成')

    # 6. 掉头动作链(写死)：确认后 uturn_delay_sec(默认2.0s)触发 → 原地旋转
    #    → 闭环循迹 uturn_follow_sec → 三段式右转 → 回循迹
    arb = new_arbiter(uturn_sec=0.4, uturn_follow_sec=0.3)
    start_follow(arb)
    tick(arb, 50.0, sign='掉头')
    _, _, _, st = tick(arb, 51.5)
    assert st['state'] == 'memorized'                   # 延时 2.0s 未到不触发
    x, y, z, st = tick(arb, 52.05)                      # 到点纯时间触发(无横条)
    assert st['state'] == 'executing' and (x, y, z) == (0, 0, 500)  # 原地旋转(默认左)
    phases = []
    t = 52.05
    for i in range(20):                     # 1.0s：覆盖 spin 0.4 + follow 0.3
        t += 0.05
        _, _, _, st = tick(arb, t)
        phases.append(st['follower_state'])
    assert 'uturn-spin' in phases and 'uturn-follow' in phases, phases
    assert st['follower_state'] == 'uturn-turn'           # 已进入转弯段
    # 核心回归：标定转角未满时，原线仍有效且对齐也不得退出/PD 拉回，
    # 必须保持固定角速度右转(z<0)——否则只转了几度就被"循迹"扼杀
    _, _, z, st = tick(arb, t + 0.05)
    assert st['follower_state'] == 'uturn-turn' and z < 0, (z, st)
    # 固定段严格转满标定转角(≈2.25s@corner_z=400)：期间丢线也不提前
    # 退出；到点后线回正即出弯回循迹
    for i in range(3):
        tick(arb, t + 0.10 + i * 0.05, det=make_det(valid=False))
    t += 0.25
    for i in range(60):
        t += 0.05
        _, _, _, st = tick(arb, t, det=make_det(err=0.0, angle=0.0))
        if arb.state == 'follow':
            break
    assert arb.state == 'follow', st
    print('PASS: 掉头链 确认后2s触发→原地旋转→循迹→右转(转满90°才回正)→回循迹')

    # 6.5 顺时针(--uturn-right) + follow_sec=0 退化为纯原地掉头(转完即回循迹)
    arb = new_arbiter(uturn_right=True, uturn_sec=0.3, uturn_follow_sec=0.0)
    start_follow(arb)
    tick(arb, 55.0, sign='掉头')
    x, y, z, st = tick(arb, 57.05)
    assert st['state'] == 'executing' and (x, y, z) == (0, 0, -500)
    t = 57.05
    while arb.state == 'executing' and t < 60.0:
        t += 0.05
        _, _, _, st = tick(arb, t)
        assert st['follower_state'] != 'uturn-turn'     # follow_sec=0 无后续段
    assert arb.state == 'follow'
    print('PASS: 顺时针原地掉头 (0, 0, -500)，follow_sec=0 时转完直接回循迹')

    # 7. 红灯执行中抢占 → 绿灯放行 → 红灯无冷却
    _, _, _, st = tick(arb, 50.5, sign='行人红灯')
    assert st['state'] == 'red_wait'
    assert tick(arb, 52.0)[:3] == (0, 0, 0)
    _, _, _, st = tick(arb, 52.5, sign='行人绿灯')
    assert st['state'] == 'follow'
    _, _, _, st = tick(arb, 53.0, sign='行人红灯')   # 放行后短期内红灯仍立即生效
    assert st['state'] == 'red_wait' and arb.update(0.05, make_det(), None, 53.1)[:3] == (0, 0, 0)
    print('PASS: 红灯抢占执行动作、绿灯放行、红灯不受冷却屏蔽')

    # 8. 动作结束后重新起步确认（前 4 帧 x==0，第 6 帧 x>0）
    arb = new_arbiter(uturn_sec=0.3, uturn_follow_sec=0.0)
    start_follow(arb)
    tick(arb, 60.0, sign='掉头')
    _, _, _, st = tick(arb, 62.1)              # 延时 2.0s 到点触发
    assert st['state'] == 'executing'
    t = 62.1
    while arb.state == 'executing':            # 跑完原地旋转(follow_sec=0)
        t += 0.05
        tick(arb, t)
    speeds = []
    for i in range(1, 7):                      # 完成后重新起步确认
        x, _, _, _ = tick(arb, t + i * 0.05)
        speeds.append(x)
    assert speeds[:4] == [0, 0, 0, 0] and speeds[5] > 0, speeds
    print('PASS: 动作结束重新起步确认 %s' % speeds)

    # 9. 变道安全兜底：平移全程找不到新线 → lc_max_sec 到点强制完成回循迹
    arb = new_arbiter(lc_max_sec=0.5)
    start_follow(arb)
    arb._uturn_sign_seen = True
    tick(arb, 70.0, sign='十字路口')
    x, y, z, st = tick(arb, 73.05)               # 3s 延时到点触发
    assert st['state'] == 'executing' and (x, y, z) == (0, 160, 0), (x, y, z)
    t = 73.05
    while arb.state == 'executing' and t < 75.0:  # 全程无有效线 → 超时兜底
        t += 0.05
        _, _, _, st = tick(arb, t, det=make_det(valid=False))
    assert arb.state == 'follow', st
    print('PASS: 变道找不到新线时 lc_max_sec 超时强制回循迹')

    # 9.9 大环岛独立置信度门槛：低于80%完全忽略，恰好80%允许记忆。
    arb = new_arbiter(roundabout_confidence=0.80)
    start_follow(arb)
    _, _, _, st = tick(arb, 79.0, sign='环岛', conf=0.799)
    assert st['state'] == 'follow', st
    _, _, _, st = tick(arb, 79.1, sign='环岛', conf=0.80)
    assert st['state'] == 'memorized', st
    print('PASS: 大环岛独立置信度阈值：79.9%忽略，80.0%允许触发')

    # 9.95 掉头独立置信度门槛：低于75%不得执行，也不得解锁十字路口。
    arb = new_arbiter(uturn_confidence=0.75)
    start_follow(arb)
    _, _, _, st = tick(arb, 79.2, sign='掉头', conf=0.749)
    assert st['state'] == 'follow' and not arb._uturn_sign_seen, st
    _, _, _, st = tick(arb, 79.3, sign='掉头', conf=0.75)
    assert st['state'] == 'memorized' and arb._uturn_sign_seen, st
    print('PASS: 掉头独立置信度阈值：74.9%忽略，75.0%允许触发并解锁平移')

    # 9.97 左右转独立置信度门槛：低于75%不记忆，恰好75%允许记忆。
    for sign in ('左转', '右转'):
        arb = new_arbiter(turn_confidence=0.75)
        start_follow(arb)
        _, _, _, st = tick(arb, 79.4, sign=sign, conf=0.749)
        assert st['state'] == 'follow' and arb.pending_task is None, (sign, st)
        _, _, _, st = tick(arb, 79.5, sign=sign, conf=0.75)
        assert st['state'] == 'memorized', (sign, st)
    print('PASS: 左转/右转独立置信度阈值：74.9%忽略，75.0%允许触发')

    # 10. 环岛写死动作链：横条触发 → 入口T口左转(z>0，岔口选边关闭) →
    #     绕环循迹 → 第1路口固定右转找线 → 第2路口左转出环 → 完成
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, rb_memory_hold_sec=40.0,
                      rb_first_straight_sec=2.0, rb_max_sec=60.0,
                      rb_junc_confirm_frames=3, rb_junc_clear_frames=5,
                      rb_junc_min_gap_sec=0.5,
                      rb_first_right_speed=100, rb_first_right_z=160,
                      rb_first_right_min_sec=0.2,
                      rb_first_right_max_sec=0.6,
                      rb_first_line_frames=3)
    start_follow(arb)
    tick(arb, 80.0, sign='环岛')
    _, _, _, st = tick(arb, 115.0)                    # 35s 后仍保留环岛任务
    assert st['state'] == 'memorized'
    _, _, z, st = tick(arb, 115.05,
                       det=make_det(corner=make_corner(kind='cross',
                                                       nearness=0.7)))
    assert st['state'] == 'executing' and z > 0, (z, st)   # 入口T口左转
    assert arb.executor.max_sec == 60.0
    assert arb.executor.first_straight_sec == 2.0
    assert det_line.fork_side == 0                          # 绕环段选边关闭
    t = 115.05
    for i in range(3):                            # 入环转弯丢线(固定段照转不误)
        t += 0.05
        tick(arb, t, det=make_det(valid=False))
    for i in range(60):                           # 标定时长(≈2.25s)到点回正→入环完成
        t += 0.05
        _, _, _, st = tick(arb, t, det=make_det(err=0.0, angle=0.0))
        if st['follower_state'] == 'rb-ring':
            break
    assert st['follower_state'] == 'rb-ring', st
    for i in range(6):                            # 离开入口横条→计数器武装
        t += 0.05
        tick(arb, t)
    # 第1路口故意给出强烈左偏拟合；必须覆盖循迹结果，明确输出固定右转。
    det_j = make_det(err=-100.0,
                     corner=make_corner(kind='cross', nearness=0.7))
    # 连续3帧才确认第1路口；候选帧起就右转，第三帧进入右转找线阶段。
    for i in range(3):
        x, y, z, st = tick(arb, t + 0.05, det=det_j)
        t += 0.05
        assert st['state'] == 'executing'
        assert st['follower_state'] == 'rb-first-right', st
        assert (x, y, z) == (100, 0, -160), (x, y, z, st)
        assert arb.executor.junctions == (1 if i == 2 else 0)

    # 同一路口持续出现很多帧，绝不能被重复计成第2路口。
    for i in range(8):
        t += 0.05
        _, _, z, st = tick(arb, t, det=det_j)
        assert arb.executor.junctions == 1 and st['follower_state'] != 'rb-exit'
        assert st['follower_state'] == 'rb-first-right' and z == -160

    # 横条只闪断2帧（小于 clear_frames=5）后重新出现，仍是同一路口。
    for i in range(2):
        t += 0.05
        tick(arb, t)
    for i in range(3):
        t += 0.05
        x, y, z, st = tick(arb, t, det=det_j)
        assert arb.executor.junctions == 1 and st['follower_state'] != 'rb-exit'
    assert st['follower_state'] == 'rb-first-wait-line'
    assert (x, y, z) == (0, 0, 0)       # 找线超时只停车，不恢复可能左转的循迹

    # 真正离开第1路口：横条消失且主线连续稳定3帧后，才结束固定右转并
    # 恢复闭环循迹；随后重新累计5个无横条帧，才允许计数第2路口。
    for i in range(3):
        t += 0.05
        _, _, z, st = tick(arb, t, det=make_det(err=5.0, angle=3.0))
        if i < 2:
            assert st['follower_state'] == 'rb-first-wait-line' and z == 0
    assert st['follower_state'] == 'rb-ring', st
    assert not arb.executor._first_right_active
    for i in range(5):
        t += 0.05
        tick(arb, t)
    # 第2路口同样连续3帧确认，第三帧才允许左转出环。
    for i in range(3):
        t += 0.05
        _, _, _, st = tick(arb, t, det=det_j)
        if i < 2:
            assert st['follower_state'] != 'rb-exit'
    assert arb.executor.junctions == 2
    assert st['follower_state'] == 'rb-exit', st
    _, _, z, st = tick(arb, t + 0.05, det=det_j)  # 出环固定角速度左转
    t += 0.05
    assert z > 0, z
    for i in range(3):                            # 出环丢线(固定段照转不误)
        t += 0.05
        tick(arb, t, det=make_det(valid=False))
    for i in range(60):                           # 标定时长到点回正→出环完成
        t += 0.05
        tick(arb, t, det=make_det(err=0.0, angle=0.0))
        if arb.state == 'follow':
            break
    assert arb.state == 'follow'
    assert det_line.fork_side == -1                 # 岔口选边已还原
    print('PASS: 大环岛第1路口固定右转至稳定找线；路口去重后第2路口才左转')

    # 10.5 无标志直角弯：单侧横条连续 3 帧确认 → 自动右转(z<0)；完成后冷却不重触发
    arb = new_arbiter(auto_corner_frames=3)
    start_follow(arb)
    det_rc = make_det(corner=make_corner(kind='corner', side=+1, nearness=0.8))
    for i in range(2):
        _, _, _, st = tick(arb, 100.0 + i * 0.05, det=det_rc)
        assert st['state'] == 'follow'                   # 未满 3 帧不触发
    _, _, z, st = tick(arb, 100.10, det=det_rc)
    assert st['state'] == 'executing' and z < 0, (z, st)  # 自动右转：固定角速度 z<0
    t = 100.10
    for i in range(4):                           # 转弯中原线丢失(真正转离)
        t += 0.05
        tick(arb, t, det=make_det(valid=False))
    while arb.state == 'executing':              # 线重新捕获回正后完成
        t += 0.05
        tick(arb, t)
    for i in range(3):                                   # 冷却期内同一弯不重触发
        _, _, _, st = tick(arb, t + 0.1 + i * 0.05, det=det_rc)
    assert st['state'] == 'follow'
    print('PASS: 无标志直角弯自动右转 + 完成后冷却')

    # 10.55 十字路口横条全程可见的左转(回归)：固定段不等丢线——横条在
    #       0~90° 全程有效，等丢线会过冲 100°+ 收不回。标定时长
    #       (900/corner_z，corner_z=400 时≈2.25s)到点后 PD 接管回正退出
    arb = new_arbiter()
    start_follow(arb)
    tick(arb, 105.0, sign='左转')
    _, _, z, st = tick(arb, 105.05,
                       det=make_det(corner=make_corner(kind='cross',
                                                       nearness=0.7)))
    assert st['state'] == 'executing' and z > 0, (z, st)  # 三段式左转固定 z>0
    t = 105.05
    for i in range(90):                 # 线全程有效且对齐(横条→新线不过渡丢线)
        t += 0.05
        _, _, _, st = tick(arb, t, det=make_det(err=0.0, angle=0.0))
        if st['state'] != 'executing':
            break
    assert st['state'] == 'follow', (t, st)   # ~2.25s 标定时长到点回正退出
    assert t - 105.05 > 1.5                   # 不是过早秒退
    print('PASS: 横条全程可见时固定段按标定时长结束，不过冲 100°+')

    # 10.57 直角弯"假线"不得提前结束固定段(回归)：直角弯丢线后，斜着的
    #       横条只转十几度就会以干净线/误判 cross 形态短暂出现——若提前
    #       PD/退出，车只转 ~15° 就回正开走随即丢线(实车复现)。固定段
    #       必须严格转满标定转角(900/corner_z≈2.25s@400)，期间任何
    #       "有效线"都不得触发 PD/退出
    arb = new_arbiter(auto_corner_frames=3)
    start_follow(arb)
    det_rc = make_det(corner=make_corner(kind='corner', side=+1, nearness=0.8))
    for i in range(3):
        _, _, _, st = tick(arb, 120.0 + i * 0.05, det=det_rc)
    assert st['state'] == 'executing'                 # 自动右转已触发
    t = 120.10
    for i in range(4):                                # 原线丢失，固定段照转
        t += 0.05
        tick(arb, t, det=make_det(valid=False))
    det_fake = make_det(err=10.0, angle=5.0,          # 斜横条扮"回正干净线"
                        corner=make_corner(kind='cross', side=+1, nearness=0.8))
    _, _, z, st = tick(arb, t + 0.05, det=det_fake)   # 不得提前 PD/退出
    t += 0.05
    assert st['follower_state'] == 'turn-R' and z < 0, (z, st)
    _, _, z, st = tick(arb, t + 0.05,                 # 干净线形态同样不得提前退出
                       det=make_det(err=10.0, angle=5.0))
    t += 0.05
    assert st['follower_state'] == 'turn-R' and z < 0, (z, st)
    for i in range(60):                               # 标定时长到点才 PD 回正退出
        t += 0.05
        _, _, _, st = tick(arb, t, det=make_det(err=10.0, angle=5.0))
        if st['state'] == 'follow':
            break
    assert st['state'] == 'follow', st
    assert t - 120.10 > 2.0                           # 确实转满了 90°(≈2.25s)
    print('PASS: 固定段严格转满标定转角，假线不提前回正(15°早退不复现)')

    # 10.6 岔路执行器设置/释放检测器选线；红灯抢占也释放
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, branch_sec=0.2, turn_style='branch')
    start_follow(arb)
    tick(arb, 110.0, sign='右转')
    _, _, _, st = tick(arb, 110.05,
                       det=make_det(corner=make_corner(kind='cross', nearness=0.7)))
    assert st['state'] == 'executing' and det_line.branch_side == 1
    t = 110.05
    while arb.state == 'executing':
        t += 0.05
        tick(arb, t)
    assert det_line.branch_side == 0                     # 正常结束后还原
    det_line2 = FakeDetector()
    arb = new_arbiter(line_detector=det_line2, branch_sec=5.0, turn_style='branch')
    start_follow(arb)
    tick(arb, 120.0, sign='右转')
    tick(arb, 120.05, det=make_det(corner=make_corner(kind='cross', nearness=0.7)))
    assert det_line2.branch_side == 1
    _, _, _, st = tick(arb, 120.2, sign='行人红灯')
    assert st['state'] == 'red_wait' and det_line2.branch_side == 0   # 抢占也还原
    print('PASS: 岔路选线设置/释放（含红灯抢占）')

    # 10.7 无标志小环岛一次性右入：完成全部锥桶并连续6s未见锥桶后才启用；
    #      启用前的反Y必须丢弃，启用后的反Y→正Y才唯一一次切右。
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line,
                      fork_right_window_sec=5.0,
                      fork_right_confirm_frames=3, fork_right_sec=3.0,
                      fork_right_z=80,
                      fork_right_cone_clear_sec=6.0, cone_total=2)
    start_follow(arb)
    # 未完成两个锥桶：反Y→正Y完整序列也不能触发，且反Y不留存。
    tick(arb, 130.0, det=make_det(junction='merge'))
    tick(arb, 132.0, det=make_det(junction='fork'))
    assert det_line.fork_side == -1
    assert arb._merge_seen_at is None and not arb._fork_right_armed

    # 模拟第二个锥桶在 t=140 完成；静默不足6s时的反Y仍必须丢弃。
    arb.cone_handled = 2
    arb._all_cones_done_at = 140.0
    arb._fork_last_cone_seen_at = 140.0
    tick(arb, 145.9, det=make_det(junction='merge'))
    assert arb._merge_seen_at is None and not arb._fork_right_armed
    # 满6s后门槛开启，但之前的反Y已经清空，直接正Y仍不触发。
    tick(arb, 146.0)
    assert arb._fork_right_armed
    tick(arb, 146.1, det=make_det(junction='fork'))
    assert det_line.fork_side == -1

    # 两帧零散反Y会被空白帧打断，不能确认入口候选。
    tick(arb, 146.2, det=make_det(junction='merge'))
    tick(arb, 146.25, det=make_det(junction='merge'))
    tick(arb, 146.3)
    assert arb._merge_seen_at is None
    # 启用后，反Y/正Y各连续3帧，5s窗口内唯一一次切右入环。
    for t in (147.0, 147.05, 147.10):
        tick(arb, t, det=make_det(junction='merge'))
    assert abs(arb._merge_seen_at - 147.10) < 1e-6
    for t in (147.40, 147.45, 147.50):
        x, y, z, st = tick(arb, t, det=make_det(junction='fork'))
    assert det_line.fork_side == 1
    assert z <= -80 and st['follower_state'] == 'fork-right', (z, st)
    # 即使随后普通PD对居中直线只想输出z≈0，右入保持期仍给小幅右转。
    _, _, z, st = tick(arb, 147.60, det=make_det(err=0.0, angle=0.0))
    assert z == -80 and st['follower_state'] == 'fork-right', (z, st)
    # 保持期内维持右侧；3s 到点恢复默认左侧
    tick(arb, 150.0, det=make_det(junction='fork'))
    assert det_line.fork_side == 1
    tick(arb, 150.6)
    assert det_line.fork_side == -1
    # 锁存：出环岛再见 反Y→正Y 序列，不触发(走左)
    tick(arb, 160.0, det=make_det(junction='merge'))
    tick(arb, 161.0, det=make_det(junction='fork'))
    assert det_line.fork_side == -1
    print('PASS: 两锥+静默门槛后，反Y/正Y各连续3帧且5s内才右入')

    # 10.75 只保留5s最大时间窗：没有最小间隔；超过5s的旧反Y不能
    #       与之后的正Y拼接。
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_total=2,
                      fork_right_cone_clear_sec=0.0,
                      fork_right_window_sec=5.0,
                      fork_right_confirm_frames=3, fork_right_sec=1.0)
    start_follow(arb)
    arb.cone_handled = 2
    arb._all_cones_done_at = 100.0
    arb._fork_last_cone_seen_at = 100.0
    tick(arb, 100.0)                                  # 门槛就绪
    for t in (101.00, 101.05, 101.10):               # 反Y确认于101.10
        tick(arb, t, det=make_det(junction='merge'))
    for t in (101.15, 101.20, 101.25):               # 仅0.15s：无最小间隔，合法
        tick(arb, t, det=make_det(junction='fork'))
    assert det_line.fork_side == 1 and arb._fork_right_used

    # 另起状态机验证超过5s无效。
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_total=2,
                      fork_right_cone_clear_sec=0.0,
                      fork_right_window_sec=5.0,
                      fork_right_confirm_frames=3, fork_right_sec=1.0)
    start_follow(arb)
    arb.cone_handled = 2
    arb._all_cones_done_at = 100.0
    arb._fork_last_cone_seen_at = 100.0
    tick(arb, 100.0)
    for t in (101.00, 101.05, 101.10):
        tick(arb, t, det=make_det(junction='merge'))
    for t in (106.15, 106.20, 106.25):               # >5s：旧序列作废
        tick(arb, t, det=make_det(junction='fork'))
    assert det_line.fork_side == -1 and not arb._fork_right_used
    for t in (107.00, 107.05, 107.10):
        tick(arb, t, det=make_det(junction='merge'))
    for t in (111.40, 111.45, 111.50):               # 4.40s：有效
        tick(arb, t, det=make_det(junction='fork'))
    assert det_line.fork_side == 1 and arb._fork_right_used
    print('PASS: 小环岛序列无最小间隔，仅限制反Y后5.0s内确认正Y')

    # 10.8 第二锥漏过兜底：1/2 保持20s后自动视为2/2；从补全时刻再
    #      连续20s无锥桶才启用，补全前/静默期内的反Y都不能残留。
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_total=2,
                      cone_missing_timeout_sec=20.0,
                      fork_right_cone_clear_sec=20.0,
                      fork_right_window_sec=5.0,
                      fork_right_confirm_frames=3, fork_right_sec=3.0)
    start_follow(arb)
    arb._finish_cone(200.0)                          # 第一个锥桶完成：1/2
    assert arb.cone_handled == 1 and arb._cone_partial_since == 200.0
    tick(arb, 219.9, det=make_det(junction='merge')) # 尚未满20s，不补全
    assert arb.cone_handled == 1 and arb._merge_seen_at is None
    _, _, _, st = tick(arb, 220.0)                   # 满20s，兜底补成2/2
    assert arb.cone_handled == 2 and arb._cone_completion_assumed
    assert arb._all_cones_done_at == 220.0
    assert abs(st['fork_right_quiet_remaining'] - 20.0) < 1e-6
    tick(arb, 239.9, det=make_det(junction='merge')) # 静默不足20s，反Y丢弃
    assert not arb._fork_right_armed and arb._merge_seen_at is None
    tick(arb, 240.0)                                 # 静默满20s，仅启用门槛
    assert arb._fork_right_armed
    tick(arb, 240.1, det=make_det(junction='fork'))  # 之前反Y不残留，不能右入
    assert det_line.fork_side == -1
    for t in (240.20, 240.25, 240.30):
        tick(arb, t, det=make_det(junction='merge'))
    for t in (240.60, 240.65, 240.70):
        tick(arb, t, det=make_det(junction='fork'))
    assert det_line.fork_side == 1

    # 若第二个锥桶在超时前正常完成，则不走“假定完成”分支。
    arb2 = new_arbiter(cone_total=2, cone_missing_timeout_sec=20.0)
    start_follow(arb2)
    arb2._finish_cone(300.0)
    arb2._finish_cone(310.0)
    assert arb2.cone_handled == 2 and not arb2._cone_completion_assumed
    assert arb2._all_cones_done_at == 310.0 and arb2._cone_partial_since is None
    print('PASS: 锥桶1/2超时20s补成2/2，再静默20s才启用小环岛右入')

    # 11. 雪糕筒封岔（专用状态机）：入岔不反向选线，循最宽连通段过岔
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_branch_sec=0.3)
    start_follow(arb)
    _, _, _, st = tick(arb, 90.0, sign='路锥',
                       context={'center_x_norm': 0.25, 'bbox_h_ratio': 0.10})
    assert st['state'] == 'cone_approach' and det_line.branch_side == 0, st
    assert det_line.fork_side == 0                   # 接近期间暂停岔口选边锁定
    x, _, _, _ = tick(arb, 90.05)                    # 接近段减速循迹
    assert 0 < x <= 125, x
    _, _, _, st = tick(arb, 90.10,                   # 判定稳定期内横条到达不触发
                       det=make_det(corner=make_corner(kind='cross', nearness=0.7)))
    assert st['state'] == 'cone_approach' and det_line.branch_side == 0, st
    _, _, _, st = tick(arb, 90.35,                   # 过 0.3s 稳定期 → 触发入岔
                       det=make_det(corner=make_corner(kind='cross', nearness=0.7)))
    assert st['state'] == 'cone_branch' and det_line.branch_side == 0, st  # 不反向选线
    _, _, z, _ = tick(arb, 90.40, det=make_det(err=40.0))   # 线在右 → 逻辑右转 z<0
    assert z < 0, z
    t = 90.35
    while arb.state == 'cone_branch':                # 到点完成
        t += 0.05
        tick(arb, t)
    assert arb.state == 'follow' and det_line.branch_side == 0
    assert det_line.fork_side == -1                  # 完成后恢复默认选边
    assert arb.cone_handled == 1
    _, _, _, st = tick(arb, t + 0.5, sign='路锥',    # 冷却期内同一锥不重触发
                       context={'center_x_norm': 0.25, 'bbox_h_ratio': 0.40})
    assert st['state'] == 'follow'
    # 第二岔口：右锥 → 同样不反向选线，循线通过
    _, _, _, st = tick(arb, t + 6.0, sign='路锥',
                       context={'center_x_norm': 0.75, 'bbox_h_ratio': 0.30})
    assert st['state'] == 'cone_approach', st        # 稳定期内不触发
    _, _, _, st = tick(arb, t + 6.35, sign='路锥',
                       context={'center_x_norm': 0.75, 'bbox_h_ratio': 0.30})
    assert st['state'] == 'cone_branch' and det_line.branch_side == 0, st
    print('PASS: 雪糕筒封岔 入岔不反向选线(循最宽连通段)，完成计数+冷却')

    # 11b. 右锥 → 锥够近(bbox)+过稳定期 → 触发入岔（不选线）
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_branch_sec=0.3)
    start_follow(arb)
    tick(arb, 95.0, sign='路锥', context={'center_x_norm': 0.75, 'bbox_h_ratio': 0.10})
    _, _, _, st = tick(arb, 95.35, sign='路锥',
                       context={'center_x_norm': 0.75, 'bbox_h_ratio': 0.30})
    assert st['state'] == 'cone_branch' and det_line.branch_side == 0, st
    print('PASS: 雪糕筒封岔 右锥→锥近触发→循线入岔')

    # 11c. 既无横条锥也不变大 → cone_lead_sec 兜底触发入岔
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_branch_sec=0.2, cone_lead_sec=0.5)
    start_follow(arb)
    tick(arb, 100.0, sign='路锥', context={'center_x_norm': 0.25, 'bbox_h_ratio': 0.05})
    _, _, _, st = tick(arb, 100.30)
    assert st['state'] == 'cone_approach', st
    _, _, _, st = tick(arb, 100.60)
    assert st['state'] == 'cone_branch' and det_line.branch_side == 0, st
    print('PASS: 雪糕筒封岔 横条漏检时 lead 兜底入岔')

    # 11d. 接近期间雪糕筒消失超时 → 按误检丢弃回循迹（不动作、不计数）
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_hold_sec=0.4, cone_lead_sec=99.0)
    start_follow(arb)
    tick(arb, 105.0, sign='路锥', context={'center_x_norm': 0.25, 'bbox_h_ratio': 0.05})
    _, _, _, st = tick(arb, 105.20)
    assert st['state'] == 'cone_approach' and det_line.fork_side == 0, st
    _, _, _, st = tick(arb, 105.50)          # 距上次目击 0.5s > cone_hold_sec
    assert st['state'] == 'follow' and arb.cone_handled == 0, st
    assert det_line.fork_side == -1                  # 丢弃后恢复默认选边
    print('PASS: 雪糕筒误检丢弃（不影响其他逻辑）')

    # 11e. 处理满 cone_total 个后 → 路锥检测被彻底忽略
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_branch_sec=0.1, cone_total=1)
    start_follow(arb)
    tick(arb, 110.0, sign='路锥', context={'center_x_norm': 0.25, 'bbox_h_ratio': 0.30})
    assert arb.state == 'cone_approach'              # 稳定期内不同拍触发
    _, _, _, st = tick(arb, 110.35)                  # 过稳定期 → 触发入岔
    assert st['state'] == 'cone_branch' and det_line.branch_side == 0, st
    t = 110.35
    while arb.state == 'cone_branch':
        t += 0.05
        tick(arb, t)
    assert arb.cone_handled == 1
    assert arb._all_cones_done_at is not None
    _, _, _, st = tick(arb, t + 5.0, sign='路锥',   # 冷却已过，但数量已满
                       context={'center_x_norm': 0.75, 'bbox_h_ratio': 0.30})
    assert st['state'] == 'follow' and det_line.branch_side == 0, st
    print('PASS: 雪糕筒数量上限（处理满后忽略路锥）')

    # 11f2. 封堵侧基准：有岔口横条时用横条中心（锥在画面中线右、但在横条
    #       中心左 → 仍判封左岔）；无横条时才退回画面中心
    det_line = FakeDetector()
    det_line.work_width = 320                  # 启用横条中心判定基准
    arb = new_arbiter(line_detector=det_line, cone_branch_sec=0.2,
                      cone_lead_sec=0.5)
    start_follow(arb)
    det_bar = make_det(corner=make_corner(kind='cross', nearness=0.3))
    det_bar['corner']['cx'] = 192.0            # 横条中心偏右(0.60)
    tick(arb, 115.0, det=det_bar, sign='路锥',
         context={'center_x_norm': 0.55, 'bbox_h_ratio': 0.05})
    _, _, _, st = tick(arb, 115.60, det=det_bar)   # lead 兜底触发入岔
    # 封堵侧判左(不判反)，仅用于日志；入岔不再反向选线
    assert st['state'] == 'cone_branch' and det_line.branch_side == 0, st
    assert arb._cone_blocked_side == -1
    print('PASS: 雪糕筒封堵侧按岔口横条中心判定（锥偏画面右但仍封左岔）')

    # 11f3. 判定稳定期内用最新目击刷新封堵侧（首拍误判可在触发前纠正）
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_branch_sec=0.3,
                      cone_lead_sec=0.5)
    start_follow(arb)
    tick(arb, 117.0, sign='路锥', context={'center_x_norm': 0.75, 'bbox_h_ratio': 0.10})
    assert arb._cone_blocked_side == 1               # 首拍判右
    _, _, _, st = tick(arb, 117.20, sign='路锥',     # 触发前刷新为左
                       context={'center_x_norm': 0.25, 'bbox_h_ratio': 0.10})
    assert arb._cone_blocked_side == -1 and st['state'] == 'cone_approach', st
    _, _, _, st = tick(arb, 117.55)                  # lead 兜底触发入岔
    assert st['state'] == 'cone_branch' and det_line.branch_side == 0, st
    print('PASS: 雪糕筒封堵侧在判定稳定期内可被最新目击纠正')

    # 11f. 其他标志任务记忆期间，雪糕筒事件被忽略（不影响环岛/转向等判断）
    arb = new_arbiter()
    start_follow(arb)
    tick(arb, 120.0, sign='左转')
    _, _, _, st = tick(arb, 120.05, sign='路锥',
                       context={'center_x_norm': 0.25, 'bbox_h_ratio': 0.30})
    assert st['state'] == 'memorized' and st['pending_task']['sign'] == '左转', st
    print('PASS: 雪糕筒不干扰已记忆的其他标志任务')

    # 11g. 入岔期间红灯抢占 → 选线释放、停车等待
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_branch_sec=5.0)
    start_follow(arb)
    tick(arb, 125.0, sign='路锥', context={'center_x_norm': 0.25, 'bbox_h_ratio': 0.30})
    _, _, _, _ = tick(arb, 125.35)                   # 过稳定期触发入岔
    assert det_line.branch_side == 0                 # 入岔不反向选线
    _, _, _, st = tick(arb, 125.45, sign='行人红灯')
    assert st['state'] == 'red_wait' and det_line.branch_side == 0, st
    assert det_line.fork_side == -1                  # 红灯抢占也恢复默认选边
    print('PASS: 雪糕筒入岔期间红灯抢占并释放选线')

    # 11h. 门槛已经就绪后若再次看见锥桶：立即撤销就绪、清除已记录反Y，
    #      从这次锥桶重新连续计时；重新满时直接见正Y也不得误触发。
    det_line = FakeDetector()
    arb = new_arbiter(line_detector=det_line, cone_total=2,
                      fork_right_cone_clear_sec=1.0,
                      fork_right_window_sec=5.0,
                      fork_right_confirm_frames=3, fork_right_sec=1.0)
    start_follow(arb)
    arb.cone_handled = 2
    arb._all_cones_done_at = 130.0
    arb._fork_last_cone_seen_at = 130.0
    tick(arb, 131.0)                                  # 静默1s，门槛就绪
    assert arb._fork_right_armed
    for t in (131.10, 131.15, 131.20):
        tick(arb, t, det=make_det(junction='merge'))
    assert abs(arb._merge_seen_at - 131.20) < 1e-6
    tick(arb, 131.3, sign='路锥',                     # 数量已满仍要重置静默计时
         context={'center_x_norm': 0.25, 'bbox_h_ratio': 0.30})
    assert not arb._fork_right_armed and arb._merge_seen_at is None
    tick(arb, 132.2, det=make_det(junction='fork'))   # 尚不足1s
    assert det_line.fork_side == -1
    tick(arb, 132.3, det=make_det(junction='fork'))   # 满1s，仅重新启用；无反Y
    assert arb._fork_right_armed and det_line.fork_side == -1
    for t in (132.40, 132.45, 132.50):
        tick(arb, t, det=make_det(junction='merge'))
    for t in (132.80, 132.85, 132.90):
        tick(arb, t, det=make_det(junction='fork'))
    assert det_line.fork_side == 1
    print('PASS: 就绪后再见锥桶会清除反Y并重新静默计时，不会提前右入')


    # 12. 转向执行器超时停车
    follower = FollowerController(corner_max_frames=10, corner_min_frames=2)
    follower.begin_turn(side=1)
    det_inv = make_det(valid=False)
    done = False
    state = speed = None
    for _ in range(11):
        state, speed, z, done = follower.turn_step(det_inv, 0.05)
    assert done and speed == 0 and state == 'turn-timeout', (state, speed, done)
    print('PASS: 转向超时强制停车')

    # 13. Y 形岔口选线：合成 60° 分叉图，左锁直行 / 右锁入岔 / 屏蔽横条误判
    try:
        import cv2
        import numpy as np
        from core.line_follower import LineDetector
    except ImportError:
        print('SKIP: 无 numpy/cv2，跳过 Y 岔口选线测试')
    else:
        # 640x480 白底黑线：主干垂直向上，在画面中部 60° 分叉，
        # 左岔沿原方向直行，右岔向右上分出（工作图坐标 = 像素/2）。
        fork_frame = np.full((480, 640, 3), 255, np.uint8)
        cv2.line(fork_frame, (320, 479), (320, 340), (0, 0, 0), 8)   # 主干
        cv2.line(fork_frame, (320, 340), (320, 216), (0, 0, 0), 8)   # 左岔=原方向
        cv2.line(fork_frame, (320, 340), (460, 216), (0, 0, 0), 8)   # 右岔=新路线

        det_left = LineDetector(fork_side=-1).process(fork_frame)
        assert det_left['is_valid'] and det_left['junction'] == 'fork'
        assert det_left['fork']
        assert max(p[0] for p in det_left['points']) < 170, det_left['points']
        assert abs(det_left['error_px']) < 15 and abs(det_left['angle_deg']) < 8, \
            (det_left['error_px'], det_left['angle_deg'])
        assert det_left['corner'] is None          # 岔口期间屏蔽横条，防误触发自动转弯

        det_right = LineDetector(fork_side=+1).process(fork_frame)
        assert det_right['is_valid'] and det_right['junction'] == 'fork'
        assert det_right['fork']
        assert max(p[0] for p in det_right['points']) > 175, det_right['points']
        assert det_right['angle_deg'] > det_left['angle_deg'] + 3, \
            (det_right['angle_deg'], det_left['angle_deg'])

        det_off = LineDetector(fork_side=0).process(fork_frame)
        assert det_off['is_valid'] and not det_off['fork']
        print('PASS: Y 岔口左锁直行(err≈0)/右锁入岔/横条屏蔽/关闭时不受影响')

        # 13.5 反Y 汇合：左侧旁路并入主线，车沿主线(中央)行驶。
        # 汇合口不得锁左(否则被旁路带偏=平移)，须守住当前线(离预测最近的段)。
        merge_frame = np.full((480, 640, 3), 255, np.uint8)
        cv2.line(merge_frame, (320, 479), (320, 216), (0, 0, 0), 8)  # 主线=车在走
        cv2.line(merge_frame, (200, 479), (320, 340), (0, 0, 0), 8)  # 左旁路汇入

        det_merge = LineDetector(fork_side=-1).process(merge_frame)
        assert det_merge['is_valid'] and det_merge['junction'] == 'merge', \
            det_merge['junction']
        assert not det_merge['fork']               # 汇合不锁边、不屏蔽横条
        assert min(p[0] for p in det_merge['points']) > 140, det_merge['points']
        assert abs(det_merge['error_px']) < 15, det_merge['error_px']

        # 汇合后一帧(纯直线)：岔口保持计数未被污染，横条检测正常
        straight = np.full((480, 640, 3), 255, np.uint8)
        cv2.line(straight, (320, 479), (320, 216), (0, 0, 0), 8)
        det_straight = LineDetector(fork_side=-1).process(straight)
        assert det_straight['is_valid'] and det_straight['junction'] is None
        assert not det_straight['fork']
        print('PASS: 反Y 汇合判 merge、守当前线(err≈0)、不清线不锁边')

        # 13.6 横条分类距离稳定性 + sym 标记：直角弯肘部短伸出不得误判 cross
        ld = LineDetector()
        bin_empty = np.zeros((100, 320), np.uint8)      # 横条上方无线
        # 右直角弯远端：左伸 45px(肘部)、右伸 90px ——
        # 旧判据 45/90=0.5≥0.35 会误判 cross，新判据短边<15%图宽判 corner
        bar_elbow = {'left': 115, 'right': 250, 'rel_y': 50}
        kind, sym = ld._classify_corner(bin_empty, bar_elbow, 160)
        assert kind == 'corner' and not sym
        # 真十字路口：两侧伸出都长且对称 → 仍判 cross 且 sym=True(可变道)
        bar_cross = {'left': 60, 'right': 260, 'rel_y': 50}
        kind, sym = ld._classify_corner(bin_empty, bar_cross, 160)
        assert kind == 'cross' and sym
        # 侧向开口 Y/T：单侧横条 + 上方有竖线贯通 → cross 但 sym=False(非真十字)
        bin_ahead = np.zeros((100, 320), np.uint8)
        bin_ahead[:45, 155:165] = 255                   # 横条上方中央竖线
        bar_side = {'left': 150, 'right': 250, 'rel_y': 50}
        kind, sym = ld._classify_corner(bin_ahead, bar_side, 160)
        assert kind == 'cross' and not sym
        print('PASS: 直角弯不误判 cross；真十字 sym=True；侧向开口 sym=False')

        # 13.7 右开口 Y(左岔长线贯通 + 右侧直角横条)：剔除横条行拟合点，
        # 拟合沿左岔上左延伸(向左进入)，不被横向岔条拽成直行/偏右。
        # 几何与车上调试图一致：主干垂直，岔点在中部，左岔 60° 上左长线，
        # 右岔水平直角横条(与线连通成超宽段)。
        right_y = np.full((480, 640, 3), 255, np.uint8)
        cv2.line(right_y, (320, 479), (320, 340), (0, 0, 0), 8)   # 主干
        cv2.line(right_y, (320, 340), (200, 140), (0, 0, 0), 8)   # 左岔=长线
        cv2.line(right_y, (320, 340), (470, 340), (0, 0, 0), 8)   # 右岔=直角

        det_y = LineDetector(fork_side=-1).process(right_y)
        assert det_y['is_valid']
        assert max(p[2] for p in det_y['points']) < 25, det_y['points']  # 横条行已剔除
        assert det_y['angle_deg'] < -3, det_y['angle_deg']   # 拟合沿左岔上左(向左转)
        assert det_y['junction'] != 'merge'                  # 不得误记录"见过反Y"
        # 未剔除横条行时横条中心(约x=195)会把拟合拽成右倾(angle>0)→直行出岔
        assert max(p[0] for p in det_y['points']) < 170, det_y['points']
        print('PASS: 右开口Y 剔除横条行后拟合向左进入左岔(angle=%.1f°)'
              % det_y['angle_deg'])

        # 13.8 岔路选线模式(branch_side!=0)不被预测窗裁剪：预测窗偏左时
        # 右岔(x≈160~230)在 pred±track_half 窗外，若仍裁窗则取不到右岔段。
        ld_branch = LineDetector(fork_side=0)
        ld_branch.branch_side = 1                    # 右岔选线模式
        ld_branch._prev_cx = 100.0                   # 车身偏左：预测窗[50,150]
        det_b = ld_branch.process(fork_frame)
        assert det_b['is_valid']
        assert max(p[0] for p in det_b['points']) > 175, det_b['points']
        print('PASS: 岔路选线模式在整行梯形窗内找段（预测窗偏左仍捕获右岔）')

    print('\n全部断言通过。方向约定: x>0 前进, y>0 左移, 逻辑 z>0 左转。')


if __name__ == '__main__':
    main()
