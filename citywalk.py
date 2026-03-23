# -*- coding: utf-8 -*-
"""
Citywalk 路线规划：调用高德 REST API，提供 /plan、/locate_city、/search_image。

停车分配接口由 parking_api 蓝图挂载（/api/default、/api/optimize），在下方 register_blueprint 注册。
"""
import json
import logging
import time
from typing import List, Dict, Tuple, Optional
from math import radians, cos, sin, asin, sqrt

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

from parking_api import parking_bp

# ==================== 基础配置 ====================
import os

app = Flask(__name__, static_folder='.', static_url_path='')

# 跨域：公开接口，允许任意 Origin；由 flask-cors 统一加响应头，避免与各路由重复设置
CORS(app, resources={
    r"/plan": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": False  # origins 为 * 时不可携带凭据
    },
    r"/locate_city": {
        "origins": "*",
        "methods": ["GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": False
    },
    r"/search_image": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": False
    },
    r"/api/.*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": False
    }
})

app.register_blueprint(parking_bp)

# 静态文件服务 - 支持前端直接访问
@app.route('/')
def index():
    """主页 - 返回 index.html"""
    return app.send_static_file('index.html')

# 注意：高德 Key 建议在控制台配置域名白名单；需开通地理编码、步行路径、周边搜索等权限
AMAP_KEY = "083ed6a4ffab4ab6aded4ecc383a30bb"

# 高德静态地图（卫星/路网底图）专用 Key
AMAP_STATIC_MAP_KEY = "a84ad5d57a96deaa8116818ef1a1ab67"

# ==================== 核心配置（含POI过滤规则） ====================
ROUTE_SAMPLE_INTERVAL = 500  # 每500米在最短路线上取1个采样点
MAX_SAMPLE_POINTS = 8  # 最多8个采样点
POI_SEARCH_RADIUS = 800  # 每个采样点搜索周边800米（适度范围，兼顾质量和覆盖）
POI_PER_SAMPLE = 3  # 每个采样点取3个POI
# 注意：选定POI后会重新规划路线经过这些POI，所以不需要严格的距离限制

# POI类型图标映射
POI_TYPE_ICONS = {
    "咖啡": "☕",
    "咖啡馆": "☕",
    "咖啡店": "☕",
    "咖啡屋": "☕",
    "甜品": "🍰",
    "甜品店": "🍰",
    "奶茶": "🧋",
    "奶茶店": "🧋",
    "饮品店": "🥤",
    "面包": "🥐",
    "面包店": "🥐",
    "烘焙": "🥐",
    "蛋糕": "🎂",
    "蛋糕店": "🎂",
    "花店": "💐",
    "花艺": "💐",
    "鲜花": "💐",
    "公园": "🌳",
    "景区": "🏞️",
    "绿地": "🌿",
    "博物馆": "🏛️",
    "纪念馆": "🏛️",
    "美术馆": "🎨",
    "艺术": "🎨",
    "展览": "🖼️",
    "文创": "✨",
    "创意": "✨",
    "商场": "🛍️",
    "购物": "🛍️",
    "书店": "📚",
    "图书": "📚",
    "餐厅": "🍽️",
    "餐饮": "🍴",
    "酒吧": "🍸",
    "历史": "🏯",
    "古迹": "🏯",
    "故居": "🏠",
}

# 核心：无效/低价值POI过滤规则（分维度）
# 1. 完全排除的POI类型（关键词匹配）
EXCLUDE_POI_TYPES = [
    # 住宅类
    "住宅", "小区", "公寓", "别墅", "商住楼", "保障房", "安置房",
    # 工业/仓储类
    "工厂", "仓库", "物流园", "产业园", "工业园", "加工", "制造",
    # 汽修/加油类
    "加油站", "汽修", "汽配", "洗车", "轮胎", "保养",
    # 医疗/康养类（非景点）
    "医院", "诊所", "药店", "养老院", "康复中心", "体检中心",
    # 生活服务（低价值）
    "家政", "保洁", "搬家", "快递", "干洗", "理发", "美容", "足疗", "按摩", "SPA", "便利店",
    # 金融/政务类
    "银行", "ATM", "营业厅", "邮局", "派出所", "政务中心", "税务局",
    # 其他低价值
    "彩票", "烟酒行", "充电站", "收费站", "停车场", "施工", "围挡","厕所","公共厕所"
]

# 2. 低匹配度POI关键词（即使命中目标类型，也排除）
LOW_VALUE_KEYWORDS = [
    "办公", "写字楼", "招商", "出租", "售房", "中介", "装修", "建材",
    "批发", "仓储", "配送", "后厨", "员工通道", "内部", "临时"
]

# 3. 有效POI类型权重（用于筛选高价值POI）
VALID_POI_WEIGHT = {
    "无偏好": {"咖啡馆": 5, "甜品店": 5, "花店": 5, "公园": 5, "商场": 5, "面包店": 5},
    "自然": {"公园": 10, "景区": 10, "湿地公园": 10, "森林公园": 10, "绿地": 8},
    "历史": {"纪念馆": 10, "博物馆": 10, "历史古迹": 10, "名人故居": 10, "文博馆": 8},
    "文创": {"美术馆": 10, "创意园区": 10, "艺术中心": 8, "文创空间": 8, "展览馆": 7},
    "花店": {"花店": 10, "花艺店": 9, "鲜花店": 9, "花艺馆": 8},
    "咖啡": {"咖啡馆": 10, "咖啡屋": 9, "咖啡店": 9, "咖啡体验馆": 8},
    "甜品": {"甜品店": 10, "奶茶店": 9, "糖水铺": 8, "饮品店": 7},
    "烘焙": {"面包店": 10, "烘焙店": 9, "蛋糕店": 9, "西点店": 8},
    "商场": {"商场": 10, "购物中心": 9, "购物广场": 8, "商业中心": 7}
}

# 日志配置（生产环境使用INFO级别）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ==================== 通用工具函数 ====================
def api_request_with_retry(url: str, params: dict, max_retries: int = 3, timeout: int = 10) -> Optional[dict]:
    """统一的API请求函数，带重试机制"""
    for retry in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "1":
                return data
            logging.warning(f"API返回错误码，重试{retry + 1}/{max_retries}：{data.get('info', '未知错误')}")
            time.sleep(0.3)
        except requests.exceptions.Timeout:
            logging.warning(f"API请求超时，重试{retry + 1}/{max_retries}")
            time.sleep(0.5)
        except requests.exceptions.RequestException as e:
            logging.warning(f"API请求异常，重试{retry + 1}/{max_retries}：{str(e)}")
            time.sleep(0.5)
        except Exception as e:
            logging.error(f"API请求未知错误：{str(e)}")
            break
    return None


# ==================== 地理工具函数 ====================
def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """计算两点间距离（米）"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000  # 地球半径（米）
    return c * r


def get_geo_code(address: str, city: str = None) -> Tuple[float, float]:
    """地址转经纬度（支持全国任意城市）"""
    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": AMAP_KEY,
        "address": address,
        "output": "json"
    }
    # 如果指定了城市，则添加城市参数
    if city:
        params["city"] = city

    for retry in range(3):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "1" and len(data.get("geocodes", [])) > 0:
                lng, lat = map(float, data["geocodes"][0]["location"].split(","))
                return lng, lat
            time.sleep(0.5)
        except Exception as e:
            logging.warning(f"地理编码重试{retry + 1}失败：{str(e)}")
            time.sleep(0.5)

    # 兜底：返回北京核心区坐标（避免解析失败导致400）
    logging.error(f"地址 {address} 解析失败，使用北京核心区兜底坐标")
    return 116.4074, 39.9042  # 北京市中心经纬度


def is_poi_in_target_city(poi: Dict, target_city: str = None) -> bool:
    """校验POI是否在目标城市（支持全国任意城市）"""
    if not target_city:
        return True  # 未指定城市时，接受所有POI
    city = poi.get("cityname", "") or poi.get("pname", "")
    # 支持多种匹配方式：完全匹配、包含匹配
    return target_city in city or city in target_city


def get_city_from_location(lng: float, lat: float) -> Optional[str]:
    """通过坐标逆地理编码获取城市名"""
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {
        "key": AMAP_KEY,
        "location": f"{lng},{lat}",
        "extensions": "base",
        "output": "json"
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "1" and data.get("regeocode"):
            comp = data["regeocode"]["addressComponent"]
            city = comp.get("city", "") or comp.get("province", "")
            return city.replace("市", "") if city else None
    except Exception as e:
        logging.warning(f"逆地理编码获取城市失败：{str(e)}")
    return None


def filter_low_value_poi(poi: Dict, poi_type: str) -> bool:
    """
    过滤无效/低价值POI：返回True表示有效，False表示无效
    校验规则：
    1. 排除命中EXCLUDE_POI_TYPES的POI
    2. 排除命中LOW_VALUE_KEYWORDS的POI
    3. 仅保留高匹配度的有效POI
    """
    poi_name = poi.get("name", "").strip().lower()
    poi_type_str = poi.get("type", "").strip().lower()

    # 规则1：排除完全无效的类型
    for exclude_type in EXCLUDE_POI_TYPES:
        if exclude_type.lower() in poi_type_str or exclude_type.lower() in poi_name:
            logging.debug(f"排除低价值POI（无效类型）：{poi_name}")
            return False

    # 规则2：排除低价值关键词
    for low_key in LOW_VALUE_KEYWORDS:
        if low_key.lower() in poi_name or low_key.lower() in poi_type_str:
            logging.debug(f"排除低价值POI（低价值关键词）：{poi_name}")
            return False

    # 规则3：校验是否命中目标高价值类型
    # 无偏好/无权重配置时，只要通过前两条规则即有效
    if poi_type == "无偏好" or poi_type not in VALID_POI_WEIGHT:
        return True

    target_weights = VALID_POI_WEIGHT.get(poi_type, {})
    if not target_weights:
        return True

    # 匹配高价值关键词（名称/类型任一命中）
    for valid_key in target_weights.keys():
        if valid_key.lower() in poi_name or valid_key.lower() in poi_type_str:
            return True

    logging.debug(f"排除低价值POI（未命中目标类型）：{poi_name}")
    return False


# ==================== 核心逻辑：沿最短路线操作 ====================
def get_shortest_route(start: Tuple[float, float], end: Tuple[float, float]) -> Dict:
    """第一步：获取起点→终点的最短步行路线（高德步行规划，起终点为全国有效坐标即可）"""
    url = "https://restapi.amap.com/v3/direction/walking"
    params = {
        "key": AMAP_KEY,
        "origin": f"{start[0]},{start[1]}",
        "destination": f"{end[0]},{end[1]}",
        "output": "json"
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            raise ValueError(f"获取最短路线失败：{data.get('info', '未知错误')}")

        # 提取最短路线（paths[0]默认是最短）
        path = data["route"]["paths"][0]
        # 解析路线的经纬度点（按顺序）
        route_points = []
        total_distance = int(path["distance"])
        total_duration = int(path["duration"]) // 60

        for step in path["steps"]:
            for point_str in step["polyline"].split(";"):
                lng, lat = map(float, point_str.split(","))
                route_points.append((lng, lat))

        return {
            "route_points": route_points,  # 最短路线的所有经纬度点（按顺序）
            "total_distance": total_distance,  # 最短路线总距离（米）
            "total_duration": total_duration,  # 最短路线总耗时（分钟）
            "original_path": path  # 原始路线数据
        }
    except Exception as e:
        logging.error(f"获取最短路线异常：{str(e)}")
        # 兜底：返回空路线，避免400错误
        return {
            "route_points": [start, end],
            "total_distance": 0,
            "total_duration": 0,
            "original_path": {}
        }


def sample_poi_along_shortest_route(route_points: List[Tuple[float, float]],
                                    poi_type: str, target_city: str = None) -> List[Dict]:
    """第二步：沿最短路线采样POI（严格贴合路线 + 过滤低价值）"""
    # 1. 沿最短路线均匀取采样点（每500米1个，最多8个）
    sample_points = []
    current_distance = 0
    prev_point = route_points[0]

    # 先加起点作为第一个采样点
    sample_points.append(route_points[0])

    for point in route_points[1:]:
        # 计算当前点与上一个点的距离
        dist = haversine(prev_point[0], prev_point[1], point[0], point[1])
        current_distance += dist

        # 每累计500米取一个采样点
        if current_distance >= ROUTE_SAMPLE_INTERVAL and len(sample_points) < MAX_SAMPLE_POINTS:
            sample_points.append(point)
            current_distance = 0  # 重置累计距离

        prev_point = point

    # 确保终点是最后一个采样点
    if sample_points[-1] != route_points[-1] and len(sample_points) < MAX_SAMPLE_POINTS:
        sample_points.append(route_points[-1])

    # 限制最多8个采样点
    sample_points = sample_points[:MAX_SAMPLE_POINTS]
    logging.info(f"沿最短路线生成采样点：{len(sample_points)}个（严格贴合路线）")

    # 2. 每个采样点搜索周边POI（支持全国任意城市+过滤低价值）
    target_keywords = list(VALID_POI_WEIGHT.get(poi_type, {}).keys()) or ["咖啡馆", "甜品店", "公园"]

    all_pois = []
    used_poi_names = set()
    used_poi_locations = set()  # 用于坐标去重（避免同一地点不同名称的重复）

    for idx, (lng, lat) in enumerate(sample_points):
        # 搜索采样点周边800米的POI（支持全国任意城市）
        url = "https://restapi.amap.com/v3/place/around"
        params = {
            "key": AMAP_KEY,
            "location": f"{lng:.6f},{lat:.6f}",
            "radius": POI_SEARCH_RADIUS,
            "keywords": "|".join(target_keywords),
            "offset": 20,
            "page": 1,
            "output": "json",
            "sortrule": "distance"  # 按离采样点的距离排序（最贴合路线）
        }
        # 如果指定了目标城市，添加城市参数
        if target_city:
            params["city"] = target_city

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "1":
                continue

            # 筛选：目标城市 + 未重复 + 过滤低价值 + 匹配类型
            for poi in data.get("pois", []):
                if target_city and not is_poi_in_target_city(poi, target_city):
                    continue
                poi_name = poi.get("name", "").strip()
                if not poi_name:
                    continue

                # 格式化POI（保留离路线的距离，用于排序）
                poi_lng, poi_lat = map(float, poi.get("location", "0,0").split(","))

                # 去重检查1：按名称去重
                if poi_name in used_poi_names:
                    continue

                # 去重检查2：按坐标去重（同一地点50米范围内视为同一POI）
                is_duplicate_location = False
                for used_lng, used_lat in used_poi_locations:
                    if haversine(poi_lng, poi_lat, used_lng, used_lat) < 50:  # 50米内视为同一地点
                        is_duplicate_location = True
                        logging.debug(f"坐标去重：{poi_name} 与已有POI位置重复（<50米）")
                        break
                if is_duplicate_location:
                    continue

                # 核心：过滤无效/低价值POI
                if not filter_low_value_poi(poi, poi_type):
                    continue

                # 计算POI到当前采样点的距离（用于排序，优先选近的）
                dist_to_route = haversine(lng, lat, poi_lng, poi_lat)

                # 获取POI类型图标
                poi_type_str = poi.get("type", "").split(";")[0]
                poi_icon = "📍"  # 默认图标
                for key, icon in POI_TYPE_ICONS.items():
                    if key in poi_name or key in poi_type_str:
                        poi_icon = icon
                        break

                all_pois.append({
                    "name": poi_name,
                    "address": poi.get("address", "暂无地址"),
                    "location": [poi_lng, poi_lat],
                    "type": poi_type_str,
                    "icon": poi_icon,  # POI类型图标
                    "dist_to_route": dist_to_route,  # 离采样点的距离（米）
                    "sample_idx": idx  # 记录属于哪个采样点
                })
                used_poi_names.add(poi_name)
                used_poi_locations.add((poi_lng, poi_lat))

                # 每个采样点最多取3个有效POI
                sample_poi_count = sum(1 for p in all_pois if p.get('sample_idx') == idx)
                if sample_poi_count >= POI_PER_SAMPLE:
                    break
        except Exception as e:
            logging.warning(f"采样点{idx + 1}搜索POI失败：{str(e)}")
            continue

    # 排序：优先离路线近的高价值POI
    all_pois.sort(key=lambda x: x["dist_to_route"])
    return all_pois


def filter_poi_for_route(pois: List[Dict], plan_time: int,
                         original_route_duration: int) -> List[Dict]:
    """第三步：筛选POI（匹配计划时间，仅保留高价值）"""
    # 目标：总耗时 = 原始路线耗时 + POI游览时间 ≈ 计划时间
    target_total_duration = plan_time
    # 可用于POI游览的时间
    available_stay_time = max(0, target_total_duration - original_route_duration)
    # 估算每个POI游览时间约5分钟，计算最多能选的POI数量
    avg_poi_time = 5
    max_poi_count = int(available_stay_time / avg_poi_time)

    # 筛选规则：
    # 1. 优先选离路线最近的（dist_to_route最小）
    # 2. 数量不超过max_poi_count（避免超时）
    # 3. 至少保留1个（如果有），最多12个
    max_poi_count = min(max(max_poi_count, 1), 12)
    filtered_pois = pois[:max_poi_count] if pois else []

    logging.info(f"筛选后高价值POI数量：{len(filtered_pois)}，匹配计划时间{plan_time}分钟")
    return filtered_pois


def generate_new_route(start: Tuple[float, float], end: Tuple[float, float],
                       filtered_pois: List[Dict]) -> Dict:
    """第四步：基于筛选后的高价值POI生成新路线（分段规划后合并）"""
    # 使用贪心算法优化POI访问顺序，减少走回头路
    # 从起点开始，每次选择离当前位置最近的下一个POI
    waypoints = []
    ordered_pois = []

    if filtered_pois:
        remaining_pois = filtered_pois.copy()
        current_pos = start

        while remaining_pois:
            # 找到离当前位置最近的POI
            min_dist = float('inf')
            nearest_poi = None
            nearest_idx = -1

            for i, poi in enumerate(remaining_pois):
                poi_lng, poi_lat = poi["location"]
                dist = haversine(current_pos[0], current_pos[1], poi_lng, poi_lat)
                if dist < min_dist:
                    min_dist = dist
                    nearest_poi = poi
                    nearest_idx = i

            if nearest_poi:
                ordered_pois.append(nearest_poi)
                waypoints.append((nearest_poi["location"][0], nearest_poi["location"][1]))
                current_pos = (nearest_poi["location"][0], nearest_poi["location"][1])
                remaining_pois.pop(nearest_idx)

        # 更新filtered_pois为优化后的顺序
        filtered_pois.clear()
        filtered_pois.extend(ordered_pois)

    # 高德步行路线API不支持waypoints，需要分段规划后合并
    # 构建路线点序列：起点 → POI1 → POI2 → ... → 终点
    route_sequence = [start] + waypoints + [end]

    all_route_points = []
    total_distance = 0
    total_walk_duration = 0
    segment_distances = []  # 记录每段距离，用于判断POI是否在路边

    # 分段调用步行路线规划
    for i in range(len(route_sequence) - 1):
        seg_start = route_sequence[i]
        seg_end = route_sequence[i + 1]

        url = "https://restapi.amap.com/v3/direction/walking"
        params = {
            "key": AMAP_KEY,
            "origin": f"{seg_start[0]},{seg_start[1]}",
            "destination": f"{seg_end[0]},{seg_end[1]}",
            "output": "json"
        }

        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "1":
                logging.warning(f"分段路线{i+1}规划失败：{data.get('info', '未知错误')}")
                continue

            seg_path = data["route"]["paths"][0]
            total_distance += int(seg_path.get("distance", 0))
            total_walk_duration += int(seg_path.get("duration", 0)) // 60

            # 解析该分段的路线坐标
            for step in seg_path.get("steps", []):
                polyline = step.get("polyline", "")
                if polyline:
                    for point_str in polyline.split(";"):
                        try:
                            lng, lat = map(float, point_str.split(","))
                            all_route_points.append((lng, lat))
                        except:
                            continue

            # 添加短暂延迟避免API限流
            time.sleep(0.1)

        except Exception as e:
            logging.warning(f"分段路线{i+1}规划异常：{str(e)}")
            continue

    # 如果分段规划失败，直接返回直线连接
    if not all_route_points:
        logging.error("所有分段路线规划失败，使用直线连接")
        all_route_points = route_sequence

    return {
        "new_route_points": all_route_points,
        "new_total_distance": total_distance,
        "new_walk_duration": total_walk_duration,
        "new_total_duration": total_walk_duration,  # 纯步行时间
        "waypoints": waypoints  # 途经的POI坐标（按顺序）
    }


# ==================== 核心接口：/plan ====================
@app.route('/plan', methods=['POST', 'OPTIONS'])
def plan_route():
    """最短路线 → 沿路采样选 POI → 筛选 → 重规划路线（支持全国城市）。"""
    if request.method == 'OPTIONS':
        return jsonify({"success": True})

    try:
        data = {}
        if request.is_json:
            data = request.get_json(silent=True) or {}  # 静默解析，非法 JSON 不抛异常
        else:
            data = request.form.to_dict() or request.args.to_dict()

        start_raw = data.get("start", "")
        end_raw = data.get("end", "")
        plan_time = int(data.get("plan_time", 60))  # 默认 60 分钟
        poi_type = data.get("poi_type", "无偏好").strip() or "无偏好"
        target_city = data.get("city", "").strip() or None

        start_lng, start_lat = None, None
        if isinstance(start_raw, list) and len(start_raw) == 2:
            start_lng, start_lat = float(start_raw[0]), float(start_raw[1])
            start_address = f"{start_lng},{start_lat}"
        elif isinstance(start_raw, str) and start_raw.strip():
            start_address = start_raw.strip()
        else:
            start_address = "北京市东城区天安门"  # 缺省起点

        end_lng, end_lat = None, None
        if isinstance(end_raw, list) and len(end_raw) == 2:
            end_lng, end_lat = float(end_raw[0]), float(end_raw[1])
            end_address = f"{end_lng},{end_lat}"
        elif isinstance(end_raw, str) and end_raw.strip():
            end_address = end_raw.strip()
        else:
            end_address = "北京市西城区王府井"  # 缺省终点

        # 参数校验：返回明确 message，避免无说明的 400
        error_msg = None
        if plan_time < 10 or plan_time > 240:
            error_msg = "计划时间需在10-240分钟之间"
        elif poi_type not in VALID_POI_WEIGHT.keys():
            error_msg = f"POI类型仅支持：{list(VALID_POI_WEIGHT.keys())}"

        if error_msg:
            return jsonify({"success": False, "message": error_msg}), 400

        # 起点—终点最短步行路线；坐标已给则跳过地理编码
        if start_lng is None or start_lat is None:
            start_lng, start_lat = get_geo_code(start_address, target_city)
        if end_lng is None or end_lat is None:
            end_lng, end_lat = get_geo_code(end_address, target_city)
        start = (start_lng, start_lat)
        end = (end_lng, end_lat)

        # 如果没有指定城市，尝试从起点坐标反推城市
        if not target_city:
            detected_city = get_city_from_location(start_lng, start_lat)
            if detected_city:
                target_city = detected_city
                logging.info(f"自动识别城市：{target_city}")

        shortest_route = get_shortest_route(start, end)
        logging.info(f"最短路线：距离{shortest_route['total_distance']}米，耗时{shortest_route['total_duration']}分钟")

        # 沿路采样 POI（类型与权重见 VALID_POI_WEIGHT）
        route_pois = sample_poi_along_shortest_route(shortest_route["route_points"], poi_type, target_city)
        if not route_pois:
            return jsonify({
                "success": True,
                "message": "沿最短路线未找到符合条件的高价值POI",
                # 与前端约定的空结果字段结构
                "path": shortest_route["route_points"],
                "distance": shortest_route["total_distance"],
                "duration": shortest_route["total_duration"],
                "pois": [],
                "original_route": {
                    "distance": shortest_route["total_distance"],
                    "duration": shortest_route["total_duration"],
                    "route_points": shortest_route["route_points"]
                },
                "filtered_pois": [],
                "new_route": {}
            }), 200

        # 按 plan_time 筛选 POI 子集
        filtered_pois = filter_poi_for_route(route_pois, plan_time, shortest_route["total_duration"])

        # 途经筛选后的 POI 重新规划路线
        new_route = generate_new_route(start, end, filtered_pois)

        # 新路线无效（无点或距离为 0）时回退为原始最短路线
        new_route_valid = (new_route.get("new_route_points") and
                          len(new_route.get("new_route_points", [])) > 0 and
                          new_route.get("new_total_distance", 0) > 0)

        route_points = new_route["new_route_points"] if new_route_valid else shortest_route["route_points"]
        total_distance = new_route["new_total_distance"] if new_route_valid else shortest_route["total_distance"]
        total_duration = new_route["new_total_duration"] if new_route_valid else shortest_route["total_duration"]

        return jsonify({
            "success": True,
            "message": "路线规划成功",
            "path": route_points,
            "distance": total_distance,
            "duration": total_duration,
            "pois": filtered_pois,
            "original_route": {
                "start": start_address,
                "end": end_address,
                "distance_m": shortest_route["total_distance"],
                "duration_min": shortest_route["total_duration"],
                "route_points": shortest_route["route_points"]
            },
            "filtered_pois": filtered_pois,
            "new_route": {
                "waypoints_count": len(new_route["waypoints"]),
                "walk_distance_m": new_route["new_total_distance"],
                "walk_duration_min": new_route["new_walk_duration"],
                "total_duration_min": new_route["new_total_duration"],
                "route_points": new_route["new_route_points"],
                "warning": new_route.get("warning", "")
            }
        }), 200

    except Exception as e:
        logging.error(f"系统异常：{str(e)}", exc_info=True)
        # 捕获所有异常，返回500而非400，方便排查
        return jsonify({
            "success": False,
            "message": f"服务器内部错误：{str(e)}",
            "error_type": type(e).__name__
        }), 500


# ==================== 图片搜索API ====================
def get_district_by_coords(lng: float, lat: float) -> dict:
    """通过坐标逆地理编码，精确到区/县级"""
    url = "https://restapi.amap.com/v3/geocode/regeo"
    params = {
        "key": AMAP_KEY,
        "location": f"{lng},{lat}",
        "extensions": "base",
        "output": "json"
    }
    try:
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "1" and data.get("regeocode"):
            comp = data["regeocode"]["addressComponent"]
            city = comp.get("city", "") or comp.get("province", "")
            district = comp.get("district", "")
            township = comp.get("township", "")
            return {
                "city": city.replace("市", ""),
                "district": district.replace("区", "").replace("县", ""),
                "township": township,
                "raw_district": district,  # 保留原始区/县名
            }
    except Exception as e:
        logging.warning(f"逆地理编码失败：{str(e)}")
    return {}


def get_amap_static_map_url(lng: float, lat: float, zoom: int = 15) -> Optional[str]:
    """生成高德卫星静态地图URL（精确到起点坐标，使用专用静态地图Key）"""
    if not lng or not lat:
        return None
    base_url = "https://restapi.amap.com/v3/staticmap"
    # style=7 卫星图，scale=2 高清，添加中心标记点
    marker = f"mid,,A:{lng:.6f},{lat:.6f}"
    params = (
        f"key={AMAP_STATIC_MAP_KEY}"
        f"&location={lng:.6f},{lat:.6f}"
        f"&zoom={zoom}"
        f"&size=1600*900"
        f"&scale=2"
        f"&style=7"
        f"&markers={marker}"
    )
    return f"{base_url}?{params}"


def smart_image_search(queries: list, lng: float = None, lat: float = None, city: str = "") -> Optional[str]:
    """生成分享图背景：直接使用高德静态地图API"""
    if lng and lat:
        result = get_amap_static_map_url(lng, lat)
        if result:
            logging.info(f"使用高德卫星地图：lng={lng}, lat={lat}")
            return result
    return None


@app.route('/search_image', methods=['POST', 'OPTIONS'])
def search_location_image():
    """搜索地点美图接口（支持坐标精确到区/县级）"""
    if request.method == 'OPTIONS':
        return jsonify({"success": True})

    try:
        data = request.get_json(silent=True) or {}
        city = data.get("city", "").strip()
        poi_name = data.get("poi_name", "").strip()   # 第一个POI名称
        start_lng = data.get("start_lng")              # 起点经度
        start_lat = data.get("start_lat")              # 起点纬度

        # 精确到区/县级：通过坐标逆地理编码
        district_info = {}
        if start_lng and start_lat:
            district_info = get_district_by_coords(float(start_lng), float(start_lat))
            logging.info(f"逆地理编码结果：{district_info}")

        city_name = district_info.get("city") or city or "上海"
        district_name = district_info.get("raw_district") or ""  # 如"浦东新区"

        # 构建多级搜索关键词列表（由精确到宽泛）
        queries = []

        # 最精确：城市+区县+POI
        if district_name and poi_name:
            queries.append(f"{city_name}{district_name} {poi_name}")
        # 城市+POI
        if poi_name:
            queries.append(f"{city_name} {poi_name}")
        # 城市+区县
        if district_name:
            queries.append(f"{city_name}{district_name}")
        # 仅城市
        queries.append(city_name)

        logging.info(f"图片搜索关键词队列：{queries}")

        image_url = smart_image_search(
            queries,
            lng=float(start_lng) if start_lng else None,
            lat=float(start_lat) if start_lat else None,
            city=city_name
        )

        if image_url:
            return jsonify({
                "success": True,
                "image_url": image_url,
                "query_used": queries[0] if queries else "",
                "district": district_name,
                "city": city_name
            })
        else:
            return jsonify({
                "success": False,
                "message": "未找到相关图片，将使用默认背景"
            }), 404

    except Exception as e:
        logging.error(f"图片搜索接口异常：{str(e)}")
        return jsonify({"success": False, "message": f"服务器错误：{str(e)}"}), 500


@app.route('/locate_city', methods=['GET', 'OPTIONS'])
def locate_city():
    """IP定位城市接口 - 优先使用前端传递的坐标进行逆地理编码"""
    if request.method == 'OPTIONS':
        return jsonify({"success": True})

    try:
        # 尝试获取前端传递的坐标参数
        lng = request.args.get('lng', type=float)
        lat = request.args.get('lat', type=float)

        # 如果有坐标，使用逆地理编码获取城市
        if lng and lat:
            url = "https://restapi.amap.com/v3/geocode/regeo"
            params = {
                "key": AMAP_KEY,
                "location": f"{lng},{lat}",
                "extensions": "base",
                "output": "json"
            }

            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") == "1" and data.get("regeocode"):
                address = data["regeocode"]["addressComponent"]
                city = address.get("city", "")
                province = address.get("province", "")

                # 处理直辖市的情况
                if not city and province in ["北京市", "上海市", "天津市", "重庆市"]:
                    city = province.replace("市", "")

                if city:
                    return jsonify({
                        "success": True,
                        "city": city.replace("市", ""),
                        "province": province,
                        "center": [lng, lat],
                        "source": "browser_geolocation"
                    })

        # 降级：使用高德IP定位API（服务器IP）
        url = "https://restapi.amap.com/v3/ip"
        params = {
            "key": AMAP_KEY,
            "output": "json"
        }

        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "1" and data.get("city"):
            city = data.get("city", "").replace("市", "")
            province = data.get("province", "")
            rectangle = data.get("rectangle", "")

            # 解析矩形区域获取中心点坐标
            center_lng, center_lat = 116.4074, 39.9042
            if rectangle:
                try:
                    coords = rectangle.split(";")
                    if len(coords) == 2:
                        lng1, lat1 = map(float, coords[0].split(","))
                        lng2, lat2 = map(float, coords[1].split(","))
                        center_lng = (lng1 + lng2) / 2
                        center_lat = (lat1 + lat2) / 2
                except:
                    pass

            return jsonify({
                "success": True,
                "city": city,
                "province": province,
                "center": [center_lng, center_lat],
                "source": "ip_location"
            })
        else:
            # 返回默认城市（北京）
            return jsonify({
                "success": True,
                "city": "北京",
                "province": "北京市",
                "center": [116.4074, 39.9042],
                "source": "default"
            })

    except Exception as e:
        logging.error(f"定位异常：{str(e)}")
        return jsonify({
            "success": True,
            "city": "北京",
            "province": "北京市",
            "center": [116.4074, 39.9042],
            "source": "default"
        })


# ==================== 启动 ====================
if __name__ == "__main__":
    # 生产环境配置
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    # 关闭debug模式避免生产环境风险，调整端口避免冲突
    app.run(host="0.0.0.0", port=port, debug=debug)