"""
小区停车分配：场景建模、步行/行车几何、匈牙利精确解与 PSO 近似解。

坐标系：原点在地块左下角 (0, 0)，x、y 单位为米；地块范围由 lot.width、lot.height 限定。

注释约定：
    - 分段使用「# --- 标题 ---」与常量组说明；
    - 对外可调用的函数使用文档字符串说明入参/返回值；
    - 行内注释说明非显而易见的约束或与前端约定的字段（见 articles/parking-pso 中 app.js）。
"""
from __future__ import annotations

import copy
import heapq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

# --- PSO 默认超参数 ---
N_PARTICLES_DEFAULT = 40
N_ITER_DEFAULT = 600
W_DEFAULT = 0.7
C1_DEFAULT = 1.5
C2_DEFAULT = 1.5
V_MAX_DEFAULT = 0.25

# --- 车位与建筑几何（与前端 app.js 中 B、SNAP_* 保持一致）---
# 车位贴合内环：约 5 m 车长沿路、2.5 m 车宽（轴对齐绘制）
SLOT_SNAP_MARGIN = 0.45
# 东、西停车带：沿路内缩，与 SLOT_ROAD_INSET 一致
SLOT_ROAD_INSET = 2.55
# 南、北停车带：车宽方向半宽 + 边距；与东西带共用同一内缩时会产生视觉空隙，故单独参数
SLOT_BERTH_WIDTH = 2.5
SLOT_HALF_BERTH_W = SLOT_BERTH_WIDTH / 2.0

# 步行绕障：非目的楼矩形 footprint（对应前端 B.bw × B.bh）
BUILDING_FOOTPRINT_W = 11.0
BUILDING_FOOTPRINT_H = 7.0
# 车位在平面中的轴对齐尺寸（米），对应前端 B.sw × B.sh；绘图与引擎一致
SLOT_VIS_LENGTH_M = 5.0
SLOT_VIS_WIDTH_M = 2.5


def default_scenario() -> Dict[str, Any]:
    """返回约 100 m × 100 m 的演示场景（由 30 m 教学沙盘等比放大，拓扑不变）。"""
    lot_w, lot_h = 100.0, 100.0
    k = lot_w / 30.0
    entrance = [7.0 * k, 6.0 * k]
    inner = {
        "x_min": 7.0 * k,
        "x_max": 23.0 * k,
        "y_min": 6.0 * k,
        "y_max": 24.0 * k,
    }
    obstacle = {
        "x_min": 13.0 * k,
        "x_max": 17.0 * k,
        "y_min": 8.0 * k,
        "y_max": 22.0 * k,
    }
    buildings = [
        [8.0 * k, 27.0 * k],
        [15.0 * k, 27.0 * k],
        [22.0 * k, 27.0 * k],
        [15.0 * k, 3.0 * k],
        [22.0 * k, 3.0 * k],
        [3.0 * k, 8.0 * k],
        [3.0 * k, 15.0 * k],
        [3.0 * k, 22.0 * k],
        [27.0 * k, 8.0 * k],
        [27.0 * k, 15.0 * k],
        [27.0 * k, 22.0 * k],
    ]
    slot_ys = np.linspace(obstacle["y_min"] + 1.2, obstacle["y_max"] - 1.2, 6)
    slot_half_length = 2.55
    left_x = inner["x_min"] + slot_half_length
    right_x = inner["x_max"] - slot_half_length
    slots = []
    for y in slot_ys:
        slots.append([left_x, float(y)])
        slots.append([right_x, float(y)])
    n_b = len(buildings)
    n_veh = 12
    vehicle_destinations = [i % n_b for i in range(n_veh)] if n_b else []
    return {
        "lot": {"width": lot_w, "height": lot_h},
        "entrance": entrance,
        "inner": inner,
        "obstacle": obstacle,
        "buildings": buildings,
        "slots": slots,
        "n_veh": n_veh,
        "vehicle_destinations": vehicle_destinations,
        "pso": {
            "n_particles": N_PARTICLES_DEFAULT,
            "n_iter": N_ITER_DEFAULT,
            "w": W_DEFAULT,
            "c1": C1_DEFAULT,
            "c2": C2_DEFAULT,
            "v_max": V_MAX_DEFAULT,
        },
        "constraints": {
            "snap_slots_to_inner_road": True,
            "snap_entrance_to_inner": True,
        },
        "display": {
            "length_unit": "m",
            "time_unit": "s",
            "scale_bar_m": 20.0,
            "coord_note": "平面坐标 1 单位 = 1 m",
        },
    }


def _seg_seg_intersect_xy(
    ax: float,
    ay: float,
    bx: float,
    by: float,
    cx: float,
    cy: float,
    dx: float,
    dy: float,
) -> bool:
    """线段 AB 与 CD 相交（含端点落在另一线段上）。"""
    o1 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    o2 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
    o3 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    o4 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    tol = 1e-9
    if abs(o1) <= tol:
        if min(ax, bx) - tol <= cx <= max(ax, bx) + tol and min(ay, by) - tol <= cy <= max(ay, by) + tol:
            return True
    if abs(o2) <= tol:
        if min(ax, bx) - tol <= dx <= max(ax, bx) + tol and min(ay, by) - tol <= dy <= max(ay, by) + tol:
            return True
    if abs(o3) <= tol:
        if min(cx, dx) - tol <= ax <= max(cx, dx) + tol and min(cy, dy) - tol <= ay <= max(cy, dy) + tol:
            return True
    if abs(o4) <= tol:
        if min(cx, dx) - tol <= bx <= max(cx, dx) + tol and min(cy, dy) - tol <= by <= max(cy, dy) + tol:
            return True
    return False


def segment_intersects_axis_rect(
    p1: np.ndarray, p2: np.ndarray, rect: Dict[str, float]
) -> bool:
    """线段是否与轴对齐矩形相交（含穿过内部）。纯标量边相交检测，避免热路径上反复分配小数组。"""
    eps = 1e-3
    x_min_i = float(rect["x_min"]) + eps
    x_max_i = float(rect["x_max"]) - eps
    y_min_i = float(rect["y_min"]) + eps
    y_max_i = float(rect["y_max"]) - eps
    if x_min_i >= x_max_i or y_min_i >= y_max_i:
        return False
    p1x, p1y = float(p1[0]), float(p1[1])
    p2x, p2y = float(p2[0]), float(p2[1])
    edges = (
        (x_min_i, y_min_i, x_max_i, y_min_i),
        (x_max_i, y_min_i, x_max_i, y_max_i),
        (x_max_i, y_max_i, x_min_i, y_max_i),
        (x_min_i, y_max_i, x_min_i, y_min_i),
    )
    for x0, y0, x1, y1 in edges:
        if _seg_seg_intersect_xy(p1x, p1y, p2x, p2y, x0, y0, x1, y1):
            return True
    midx = 0.5 * (p1x + p2x)
    midy = 0.5 * (p1y + p2y)
    return (x_min_i < midx < x_max_i) and (y_min_i < midy < y_max_i)


def segment_intersects_obstacle(
    p1: np.ndarray, p2: np.ndarray, obs: Dict[str, float]
) -> bool:
    return segment_intersects_axis_rect(p1, p2, obs)


def _building_axis_box(cx: float, cy: float) -> Dict[str, float]:
    hw = BUILDING_FOOTPRINT_W / 2.0
    hh = BUILDING_FOOTPRINT_H / 2.0
    return {"x_min": cx - hw, "x_max": cx + hw, "y_min": cy - hh, "y_max": cy + hh}


def walk_blocking_boxes(
    obs: Dict[str, float], buildings_pos: np.ndarray, dest_bi: int
) -> List[Dict[str, float]]:
    """花坛 + 除目的楼外的所有居民楼矩形（步行不可穿行）。"""
    boxes: List[Dict[str, float]] = [obs]
    n = int(buildings_pos.shape[0])
    for i in range(n):
        if i == dest_bi:
            continue
        bx, by = float(buildings_pos[i, 0]), float(buildings_pos[i, 1])
        boxes.append(_building_axis_box(bx, by))
    return boxes


def segment_clear_boxes(
    p1: np.ndarray, p2: np.ndarray, boxes: List[Dict[str, float]]
) -> bool:
    return not any(segment_intersects_axis_rect(p1, p2, b) for b in boxes)


def _point_strictly_inside_rect(p: np.ndarray, r: Dict[str, float], eps: float = 1e-2) -> bool:
    return (
        float(r["x_min"]) + eps < p[0] < float(r["x_max"]) - eps
        and float(r["y_min"]) + eps < p[1] < float(r["y_max"]) - eps
    )


def corner_waypoint_valid(p: np.ndarray, boxes: List[Dict[str, float]]) -> bool:
    for b in boxes:
        if _point_strictly_inside_rect(p, b):
            return False
    return True


def _rect_corners(box: Dict[str, float]) -> List[np.ndarray]:
    return [
        np.array([box["x_min"], box["y_min"]], dtype=float),
        np.array([box["x_min"], box["y_max"]], dtype=float),
        np.array([box["x_max"], box["y_min"]], dtype=float),
        np.array([box["x_max"], box["y_max"]], dtype=float),
    ]


def _all_corner_points(boxes: List[Dict[str, float]]) -> List[np.ndarray]:
    pts: List[np.ndarray] = []
    for b in boxes:
        pts.extend(_rect_corners(b))
    return pts


def _polyline_length(pts: List[np.ndarray]) -> float:
    return sum(float(np.linalg.norm(pts[i] - pts[i - 1])) for i in range(1, len(pts)))


def _polyline_segments_clear(pts: List[np.ndarray], boxes: List[Dict[str, float]]) -> bool:
    for i in range(1, len(pts)):
        if not segment_clear_boxes(pts[i - 1], pts[i], boxes):
            return False
    return True


def _dedup_valid_corners(boxes: List[Dict[str, float]]) -> List[np.ndarray]:
    seen_keys = set()
    uniq: List[np.ndarray] = []
    for c in _all_corner_points(boxes):
        key = (round(float(c[0]), 3), round(float(c[1]), 3))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if not corner_waypoint_valid(c, boxes):
            continue
        uniq.append(c)
    return uniq


def _simplify_colinear_polyline(pts: List[np.ndarray]) -> List[np.ndarray]:
    """去掉共线中间点，便于绘制且长度不变。"""
    if len(pts) <= 2:
        return pts
    out: List[np.ndarray] = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = out[-1], pts[i], pts[i + 1]
        v1 = b - a
        v2 = c - b
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        if abs(cross) > 1e-5:
            out.append(b)
    out.append(pts[-1])
    return out


def _visibility_graph_shortest_path(
    s: np.ndarray,
    t: np.ndarray,
    boxes: List[Dict[str, float]],
) -> Optional[Tuple[float, List[np.ndarray]]]:
    """
    多边形障碍最短路的经典简化：顶点 = 起点、终点、各矩形障碍角点；
    边 = 两端点连线不穿任何障碍；边权 = 欧氏长度；Dijkstra 求最短路。
    （障碍为轴对齐矩形时，欧氏最短路可仅由这些顶点构成。）
    """
    corners = _dedup_valid_corners(boxes)
    nodes: List[np.ndarray] = [np.asarray(s, dtype=float).copy()]
    for c in corners:
        nodes.append(np.asarray(c, dtype=float).copy())
    nodes.append(np.asarray(t, dtype=float).copy())
    n = len(nodes)
    goal = n - 1

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        pi = nodes[i]
        for j in range(i + 1, n):
            pj = nodes[j]
            if segment_clear_boxes(pi, pj, boxes):
                wij = float(np.linalg.norm(pi - pj))
                adj[i].append((j, wij))
                adj[j].append((i, wij))

    dist_arr = np.full(n, np.inf, dtype=float)
    dist_arr[0] = 0.0
    parent = np.full(n, -1, dtype=np.int64)
    pq: List[Tuple[float, int]] = [(0.0, 0)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist_arr[u] + 1e-12:
            continue
        if u == goal:
            break
        for v, w in adj[u]:
            nd = d_u + w
            if nd < dist_arr[v]:
                dist_arr[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if not np.isfinite(dist_arr[goal]):
        return None

    seq: List[int] = []
    cur = int(goal)
    while cur >= 0:
        seq.append(cur)
        cur = int(parent[cur])
    seq.reverse()
    raw = [nodes[i] for i in seq]
    simplified = _simplify_colinear_polyline(raw)
    return float(_polyline_length(simplified)), simplified


def walking_plan(
    slot_xy: np.ndarray,
    building_xy: np.ndarray,
    obs: Dict[str, float],
    buildings_pos: np.ndarray,
    dest_bi: int,
    boxes: Optional[List[Dict[str, float]]] = None,
) -> Tuple[float, List[np.ndarray]]:
    """
    车位 → 目的楼：避开花坛与非目的楼矩形。
    主算法：可见性图 + Dijkstra（标准平面多边形障碍欧氏最短路顶点集）；
    若无解（数值/退化）则回退到绕花坛折线模板。

    若传入 ``boxes``（与 ``walk_blocking_boxes(obs, buildings_pos, dest_bi)`` 相同），
    可避免在批量预计算时重复构造障碍列表。
    """
    if boxes is None:
        boxes = walk_blocking_boxes(obs, buildings_pos, dest_bi)
    s = np.asarray(slot_xy, dtype=float)
    t = np.asarray(building_xy, dtype=float)

    if segment_clear_boxes(s, t, boxes):
        pts0 = [s, t]
        return float(_polyline_length(pts0)), pts0

    vg = _visibility_graph_shortest_path(s, t, boxes)
    if vg is not None:
        return vg[0], vg[1]

    margin = 0.5
    x_left = obs["x_min"] - margin
    x_right = obs["x_max"] + margin

    def side_polylines(x_side: float) -> List[List[np.ndarray]]:
        slot_y = float(s[1])
        b_y = float(t[1])
        out: List[List[np.ndarray]] = []
        if b_y <= obs["y_min"] - margin or b_y >= obs["y_max"] + margin:
            p1 = np.array([x_side, slot_y])
            p2 = np.array([x_side, b_y])
            pts = [s, p1, p2, t]
            if _polyline_segments_clear(pts, boxes):
                out.append(pts)
            return out
        y_up = obs["y_max"] + margin
        y_down = obs["y_min"] - margin
        p1u = np.array([x_side, slot_y])
        p2u = np.array([x_side, y_up])
        p3u = np.array([t[0], y_up])
        pts_u = [s, p1u, p2u, p3u, t]
        p1d = np.array([x_side, slot_y])
        p2d = np.array([x_side, y_down])
        p3d = np.array([t[0], y_down])
        pts_d = [s, p1d, p2d, p3d, t]
        for pts in (pts_u, pts_d):
            if _polyline_segments_clear(pts, boxes):
                out.append(pts)
        return out

    fb: List[Tuple[float, List[np.ndarray]]] = []
    for pts in side_polylines(x_left) + side_polylines(x_right):
        fb.append((_polyline_length(pts), pts))
    if fb:
        best_fb = min(fb, key=lambda x: x[0])
        return float(best_fb[0]), best_fb[1]

    pts_last = [s, t]
    return float(_polyline_length(pts_last)), pts_last


def build_road_segments(inner: Dict[str, float]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """仅内环矩形四边；入口为单独一点，不画竖向引道。"""
    ix0, ix1 = inner["x_min"], inner["x_max"]
    iy0, iy1 = inner["y_min"], inner["y_max"]
    return [
        ((ix0, iy0), (ix1, iy0)),
        ((ix1, iy0), (ix1, iy1)),
        ((ix1, iy1), (ix0, iy1)),
        ((ix0, iy1), (ix0, iy0)),
    ]


def driving_distance_from_entrance(
    slot_xy: np.ndarray, inner: Dict[str, float], entrance: np.ndarray
) -> float:
    """
    行车距离（示意模型）：沿内环行驶至车位一侧再水平驶入。

    入口 (ex, ey) 视为贴内环边（normalize 时已吸附到矩形四边）。
    车位中心 x 小于内环水平中点时走左侧接入点 (ix0, sy)，否则走 (ix1, sy)；
    sy 为车位纵坐标限制在 [iy0, iy1] 内。环上两点间取矩形边界较短弧长。
    """
    ix0, ix1 = inner["x_min"], inner["x_max"]
    iy0, iy1 = inner["y_min"], inner["y_max"]
    ex, ey = float(entrance[0]), float(entrance[1])
    sx, sy = float(slot_xy[0]), float(slot_xy[1])
    mid_x = (ix0 + ix1) / 2.0
    sy_c = min(max(sy, iy0), iy1)

    if sx < mid_x:
        d_ring = perimeter_distance_between(ex, ey, ix0, sy_c, inner)
        into_slot = abs(sx - ix0)
    else:
        d_ring = perimeter_distance_between(ex, ey, ix1, sy_c, inner)
        into_slot = abs(ix1 - sx)

    return d_ring + into_slot


def walking_distance(
    slot_xy: np.ndarray,
    building_xy: np.ndarray,
    obs: Dict[str, float],
    buildings_pos: np.ndarray,
    dest_bi: int,
    boxes: Optional[List[Dict[str, float]]] = None,
) -> float:
    d, _ = walking_plan(
        slot_xy, building_xy, obs, buildings_pos, dest_bi, boxes=boxes
    )
    return d


def walking_path_polyline(
    slot_xy: np.ndarray,
    building_xy: np.ndarray,
    obs: Dict[str, float],
    buildings_pos: np.ndarray,
    dest_bi: int,
    boxes: Optional[List[Dict[str, float]]] = None,
) -> List[List[float]]:
    """与 walking_distance 一致的折线（避开花坛与非目的楼），供 Web / matplotlib 绘制。"""
    _, pts = walking_plan(
        slot_xy, building_xy, obs, buildings_pos, dest_bi, boxes=boxes
    )
    return [[float(p[0]), float(p[1])] for p in pts]


def closest_point_on_segment(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float]:
    dx = x2 - x1
    dy = y2 - y1
    l2 = dx * dx + dy * dy
    if l2 < 1e-18:
        return x1, y1
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
    return x1 + t * dx, y1 + t * dy


def snap_point_to_inner_perimeter(
    x: float, y: float, inner: Dict[str, float], lot_w: float, lot_h: float
) -> Tuple[float, float]:
    """将点投影到内环矩形道路中心线（四边）上最近一点。"""
    ix0, ix1 = float(inner["x_min"]), float(inner["x_max"])
    iy0, iy1 = float(inner["y_min"]), float(inner["y_max"])
    strips = [
        (ix0, iy0, ix1, iy0),
        (ix1, iy0, ix1, iy1),
        (ix1, iy1, ix0, iy1),
        (ix0, iy1, ix0, iy0),
    ]
    best_x, best_y = x, y
    best_d = 1e30
    for x0, y0, x1, y1 in strips:
        qx, qy = closest_point_on_segment(x, y, x0, y0, x1, y1)
        d = (x - qx) ** 2 + (y - qy) ** 2
        if d < best_d:
            best_d = d
            best_x, best_y = qx, qy
    return max(0.0, min(lot_w, best_x)), max(0.0, min(lot_h, best_y))


def arc_length_from_bl_ccw(px: float, py: float, inner: Dict[str, float]) -> float:
    """内环上一点相对左下角 (ix0,iy0) 沿逆时针周界的弧长 ∈ [0, L)。"""
    ix0, ix1 = float(inner["x_min"]), float(inner["x_max"])
    iy0, iy1 = float(inner["y_min"]), float(inner["y_max"])
    w, h = ix1 - ix0, iy1 - iy0
    L = 2.0 * w + 2.0 * h
    if L < 1e-9:
        return 0.0
    tol = 0.04
    if abs(py - iy0) <= tol and ix0 - tol <= px <= ix1 + tol:
        return max(0.0, min(w, px - ix0))
    if abs(px - ix1) <= tol and iy0 - tol <= py <= iy1 + tol:
        return w + max(0.0, min(h, py - iy0))
    if abs(py - iy1) <= tol and ix0 - tol <= px <= ix1 + tol:
        return w + h + max(0.0, min(w, ix1 - px))
    if abs(px - ix0) <= tol and iy0 - tol <= py <= iy1 + tol:
        return w + h + w + max(0.0, min(h, iy1 - py))
    qx, qy = snap_point_to_inner_perimeter(px, py, inner, 1e9, 1e9)
    return arc_length_from_bl_ccw(qx, qy, inner)


def perimeter_distance_between(
    ax: float, ay: float, bx: float, by: float, inner: Dict[str, float]
) -> float:
    """沿内环矩形边界，从 (ax,ay) 到 (bx,by) 的最短路长（两点先吸附到边界，再取双向较短弧）。"""
    qax, qay = snap_point_to_inner_perimeter(ax, ay, inner, 1e9, 1e9)
    qbx, qby = snap_point_to_inner_perimeter(bx, by, inner, 1e9, 1e9)
    ix0, ix1 = float(inner["x_min"]), float(inner["x_max"])
    iy0, iy1 = float(inner["y_min"]), float(inner["y_max"])
    w, h = ix1 - ix0, iy1 - iy0
    L = 2.0 * w + 2.0 * h
    if L < 1e-9:
        return 0.0
    s1 = arc_length_from_bl_ccw(qax, qay, inner)
    s2 = arc_length_from_bl_ccw(qbx, qby, inner)
    d = abs(s1 - s2)
    return float(min(d, L - d))


def snap_slot_to_road(
    x: float,
    y: float,
    inner: Dict[str, float],
    lot_w: float,
    lot_h: float,
    margin: float = SLOT_SNAP_MARGIN,
) -> Tuple[float, float]:
    """
    将车位中心投影到内环内侧四条停车带线段上（与边平行）。
    东西带按车长方向内缩（SLOT_ROAD_INSET）；南北带按车宽方向内缩（半车宽+边距），
    避免轴对齐矩形在横带上「离停车线过远」。
    """
    xm, xM = float(inner["x_min"]), float(inner["x_max"])
    ym, yM = float(inner["y_min"]), float(inner["y_max"])
    iw = xM - xm
    ih = yM - ym
    if iw < 2 * margin + 0.2 or ih < 2 * margin + 0.2:
        return max(0.0, min(lot_w, x)), max(0.0, min(lot_h, y))
    inset_ew = max(0.4, min(SLOT_ROAD_INSET, iw / 2.0 - margin - 0.1))
    inset_ns = max(0.4, min(SLOT_HALF_BERTH_W + margin, ih / 2.0 - margin - 0.1))
    strips: List[Tuple[float, float, float, float]] = [
        (xm + margin, ym + inset_ns, xM - margin, ym + inset_ns),
        (xm + margin, yM - inset_ns, xM - margin, yM - inset_ns),
        (xm + inset_ew, ym + margin, xm + inset_ew, yM - margin),
        (xM - inset_ew, ym + margin, xM - inset_ew, yM - margin),
    ]
    best_x, best_y = x, y
    best_d = 1e30
    for x0, y0, x1, y1 in strips:
        qx, qy = closest_point_on_segment(x, y, x0, y0, x1, y1)
        d = (x - qx) ** 2 + (y - qy) ** 2
        if d < best_d:
            best_d = d
            best_x, best_y = qx, qy
    return max(0.0, min(lot_w, best_x)), max(0.0, min(lot_h, best_y))


def normalize_scenario(raw: Dict[str, Any]) -> Dict[str, Any]:
    """深拷贝并补齐场景字段，应用车位/入口吸附、n_veh 上界与 vehicle_destinations 规范化。"""
    s = copy.deepcopy(raw)
    lot = s.get("lot") or {}
    lw = float(lot.get("width", 100))
    lh = float(lot.get("height", 100))
    s["lot"] = {"width": lw, "height": lh}
    s.setdefault("entrance", [7.0, 6.0])
    s.setdefault("inner", default_scenario()["inner"])
    s.setdefault("obstacle", default_scenario()["obstacle"])
    s["buildings"] = [[float(p[0]), float(p[1])] for p in s.get("buildings") or []]
    s["slots"] = [[float(p[0]), float(p[1])] for p in s.get("slots") or []]
    # 车位吸附停车带、入口吸附内环：与前端一致，始终开启；请求中若为 false 也会被覆盖
    snap_road = True
    snap_ent = True
    s["constraints"] = {
        "snap_slots_to_inner_road": snap_road,
        "snap_entrance_to_inner": snap_ent,
    }
    inner = s["inner"]
    if snap_ent:
        ex, ey = float(s["entrance"][0]), float(s["entrance"][1])
        s["entrance"] = list(snap_point_to_inner_perimeter(ex, ey, inner, lw, lh))
    if snap_road and s["slots"]:
        s["slots"] = [
            list(snap_slot_to_road(float(p[0]), float(p[1]), inner, lw, lh))
            for p in s["slots"]
        ]
    n_veh = int(s.get("n_veh", 12))
    n_slot = len(s["slots"])
    if n_slot == 0:
        s["n_veh"] = 0
    else:
        s["n_veh"] = max(1, min(n_veh, n_slot))
    pso = s.get("pso") or {}
    s["pso"] = {
        "n_particles": int(pso.get("n_particles", N_PARTICLES_DEFAULT)),
        "n_iter": int(pso.get("n_iter", N_ITER_DEFAULT)),
        "w": float(pso.get("w", W_DEFAULT)),
        "c1": float(pso.get("c1", C1_DEFAULT)),
        "c2": float(pso.get("c2", C2_DEFAULT)),
        "v_max": float(pso.get("v_max", V_MAX_DEFAULT)),
    }
    disp_def = default_scenario().get("display") or {}
    disp_in = s.get("display") or {}
    s["display"] = {
        "length_unit": str(disp_in.get("length_unit", disp_def.get("length_unit", "m"))),
        "time_unit": str(disp_in.get("time_unit", disp_def.get("time_unit", "s"))),
        "scale_bar_m": float(disp_in.get("scale_bar_m", disp_def.get("scale_bar_m", 20.0))),
        "coord_note": str(
            disp_in.get("coord_note", disp_def.get("coord_note", "平面坐标 1 单位 = 1 m"))
        ),
    }
    _normalize_vehicle_destinations(s)
    return s


def _normalize_vehicle_destinations(s: Dict[str, Any]) -> None:
    """每辆车对应一栋目的楼（buildings 下标 0..n_b-1），长度与 n_veh 一致。"""
    n_b = len(s.get("buildings") or [])
    n_veh = int(s.get("n_veh", 0))
    if n_b <= 0 or n_veh <= 0:
        s["vehicle_destinations"] = []
        return
    raw = s.get("vehicle_destinations")
    out: List[int] = []
    if isinstance(raw, list):
        for i in range(n_veh):
            if i < len(raw):
                try:
                    bi = int(raw[i])
                except (TypeError, ValueError):
                    bi = i % n_b
            else:
                bi = i % n_b
            out.append(max(0, min(n_b - 1, bi)))
    else:
        out = [i % n_b for i in range(n_veh)]
    s["vehicle_destinations"] = out


def precompute_from_normalized(
    s: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, int, int, List[List[Dict[str, float]]]]:
    """
    由已 ``normalize_scenario`` 的场景计算行车向量与步行矩阵。
    避免在 ``run_pso`` 路径上重复深拷贝归一化。
    返回的 ``boxes_by_bi`` 供结果路径折线复用，避免再次调用 ``walk_blocking_boxes``。
    """
    inner = s["inner"]
    obs = s["obstacle"]
    slots_pos = np.array(s["slots"], dtype=float)
    buildings_pos = np.array(s["buildings"], dtype=float)
    n_slot = slots_pos.shape[0]
    n_b = buildings_pos.shape[0]
    entrance = np.array(s["entrance"], dtype=float)
    if n_slot == 0 or n_b == 0:
        return slots_pos, buildings_pos, n_slot, n_b, []
    drive_dist = np.array(
        [driving_distance_from_entrance(slot_xy, inner, entrance) for slot_xy in slots_pos],
        dtype=float,
    )
    # 同一目的楼对应同一组障碍矩形，按目的楼索引缓存，避免对每个车位重复构造
    boxes_by_bi = [
        walk_blocking_boxes(obs, buildings_pos, bi) for bi in range(n_b)
    ]
    walk_mat = np.zeros((n_slot, n_b), dtype=float)
    for si, slot_xy in enumerate(slots_pos):
        for bi, building_xy in enumerate(buildings_pos):
            walk_mat[si, bi] = walking_distance(
                slot_xy,
                building_xy,
                obs,
                buildings_pos,
                bi,
                boxes=boxes_by_bi[bi],
            )
    return drive_dist, walk_mat, n_slot, n_b, boxes_by_bi


def decode_particle(position: np.ndarray, n_veh: int, n_slot: int) -> np.ndarray:
    """按粒子各维排序，排名第 k 的车辆使用车位 slot_perm[k]（与原脚本一致）。"""
    order = np.argsort(position, kind="stable")
    assign = np.empty(n_veh, dtype=int)
    slot_perm = np.arange(n_slot, dtype=int)
    for k in range(n_veh):
        veh_idx = int(order[k])
        assign[veh_idx] = int(slot_perm[k])
    return assign


def _pack_optimize_result(
    s: Dict[str, Any],
    *,
    gbest_value: float,
    history_best: List[float],
    best_assign: np.ndarray,
    slots_pos: np.ndarray,
    buildings_pos: np.ndarray,
    obs: Dict[str, float],
    veh_targets: np.ndarray,
    boxes_by_bi: List[List[Dict[str, float]]],
    inner: Dict[str, float],
    optimizer: str,
) -> Dict[str, Any]:
    n_veh = int(best_assign.shape[0])
    paths: List[List[List[float]]] = []
    for i in range(n_veh):
        ti = int(veh_targets[i])
        slot_xy = slots_pos[int(best_assign[i])]
        bxy = buildings_pos[ti]
        paths.append(
            walking_path_polyline(
                slot_xy,
                bxy,
                obs,
                buildings_pos,
                ti,
                boxes=boxes_by_bi[ti],
            )
        )
    road_segments = build_road_segments(inner)
    return {
        "scenario": s,
        "gbest_value": float(gbest_value),
        "history_best": history_best,
        "assign": [int(x) for x in best_assign.tolist()],
        "veh_targets": [int(x) for x in veh_targets.tolist()],
        "paths": paths,
        "road_segments": [
            [[float(a[0]), float(a[1])], [float(b[0]), float(b[1])]]
            for a, b in road_segments
        ],
        "optimizer": optimizer,
    }


def run_optimize(
    scenario: Dict[str, Any],
    seed: Optional[int] = None,
    method: str = "exact",
) -> Dict[str, Any]:
    """
    停车分配优化。

    - ``method=\"exact\"``：最小化 Σ(行车时间+步行时间)，每车一车位的**线性分配**，用匈牙利算法求**全局最优**
      （与当前 ``drive_dist`` / ``walk_mat`` 可加模型一致）。
    - ``method=\"pso\"``：粒子群启发式，不保证全局最优，但 ``history_best`` 可画收敛曲线。

    ``seed`` 仅对 ``pso`` 有效：``None`` 时每次使用系统随机熵；传入整数可复现同一场景下的 PSO 轨迹。
    ``exact`` 结果与种子无关。
    """
    m = str(method or "exact").strip().lower()
    if m not in ("exact", "pso"):
        m = "exact"

    s = normalize_scenario(scenario)
    inner = s["inner"]
    obs = s["obstacle"]
    slots_pos = np.array(s["slots"], dtype=float)
    buildings_pos = np.array(s["buildings"], dtype=float)
    n_slot = int(slots_pos.shape[0])
    n_b = int(buildings_pos.shape[0])
    n_veh = int(s["n_veh"])
    err: Dict[str, Any] = {
        "error": "需要至少一个车位、一栋楼，且车辆数大于 0。",
        "scenario": s,
        "gbest_value": None,
        "history_best": [],
        "assign": [],
        "veh_targets": [],
        "paths": [],
        "road_segments": [],
        "optimizer": m,
    }
    if n_slot == 0 or n_b == 0 or n_veh == 0:
        return err

    n_veh = min(n_veh, n_slot)
    s["n_veh"] = n_veh
    _normalize_vehicle_destinations(s)
    veh_targets = np.array(s["vehicle_destinations"], dtype=int)

    drive_dist, walk_mat, _, _, boxes_by_bi = precompute_from_normalized(s)
    v_car, v_walk = 10.0, 1.5

    if m == "exact":
        # 代价矩阵 cost[i,j]：车 i 使用车位 j 的行车时间 + 步行时间；匈牙利算法求全局最小权匹配
        cost = drive_dist.reshape(1, -1) / v_car + walk_mat[:, veh_targets].T / v_walk
        row_ind, col_ind = linear_sum_assignment(cost)
        best_assign = np.empty(n_veh, dtype=int)
        best_assign[row_ind] = col_ind
        gbest_value = float(cost[row_ind, col_ind].sum())
        history_best = [gbest_value]
        return _pack_optimize_result(
            s,
            gbest_value=gbest_value,
            history_best=history_best,
            best_assign=best_assign,
            slots_pos=slots_pos,
            buildings_pos=buildings_pos,
            obs=obs,
            veh_targets=veh_targets,
            boxes_by_bi=boxes_by_bi,
            inner=inner,
            optimizer="exact",
        )

    rng = np.random.default_rng() if seed is None else np.random.default_rng(int(seed))
    pso = s["pso"]
    n_particles = pso["n_particles"]
    n_iter = pso["n_iter"]
    w, c1, c2, v_max = pso["w"], pso["c1"], pso["c2"], pso["v_max"]

    def objective(position: np.ndarray) -> float:
        assign = decode_particle(position, n_veh, n_slot)
        drive_total = drive_dist[assign].sum() / v_car
        walk_dists = walk_mat[assign, veh_targets]
        walk_total = walk_dists.sum() / v_walk
        return float(drive_total + walk_total)

    positions = rng.random((n_particles, n_veh))
    velocities = rng.standard_normal((n_particles, n_veh)) * 0.1
    pbest_positions = positions.copy()
    pbest_values = np.array([objective(pos) for pos in positions])
    gbest_idx = int(np.argmin(pbest_values))
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_value = float(pbest_values[gbest_idx])
    history_best = [gbest_value]

    for _ in range(n_iter):
        for i in range(n_particles):
            r1 = rng.random(n_veh)
            r2 = rng.random(n_veh)
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )
            velocities[i] = np.clip(velocities[i], -v_max, v_max)
            positions[i] = np.clip(positions[i] + velocities[i], 0.0, 1.0)
            val = objective(positions[i])
            if val < pbest_values[i]:
                pbest_values[i] = val
                pbest_positions[i] = positions[i].copy()
        gbest_idx = int(np.argmin(pbest_values))
        if pbest_values[gbest_idx] < gbest_value:
            gbest_value = float(pbest_values[gbest_idx])
            gbest_position = pbest_positions[gbest_idx].copy()
        history_best.append(gbest_value)

    best_assign = decode_particle(gbest_position, n_veh, n_slot)
    return _pack_optimize_result(
        s,
        gbest_value=gbest_value,
        history_best=history_best,
        best_assign=best_assign,
        slots_pos=slots_pos,
        buildings_pos=buildings_pos,
        obs=obs,
        veh_targets=veh_targets,
        boxes_by_bi=boxes_by_bi,
        inner=inner,
        optimizer="pso",
    )


def run_pso(
    scenario: Dict[str, Any],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    粒子群近似优化（便于观察 ``history_best`` 收敛曲线）。
    ``seed is None`` 时初值与随机项每次不同；需要复现时传入整数种子。
    需要**全局最优**时请用 ``run_optimize(..., method=\"exact\")``。
    """
    return run_optimize(scenario, seed=seed, method="pso")
