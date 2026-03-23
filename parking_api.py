# -*- coding: utf-8 -*-
"""
停车分配 HTTP 接口（Flask Blueprint）。

路由：
    GET  /api/default   — 返回归一化后的默认场景 JSON。
    POST /api/optimize — 请求体 JSON：可选 scenario、method（exact|pso）、可选 seed（仅 pso）。

与 parking_engine 共用数据模型；供静态前端（如 GitHub Pages）跨域调用。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from parking_engine import default_scenario, normalize_scenario, run_optimize

parking_bp = Blueprint("parking", __name__)


def _coerce_seed(raw: Any) -> Optional[int]:
    """将请求中的 seed 转为 int；无效或缺省则返回 None（PSO 使用随机种子）。"""
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


@parking_bp.route("/api/default")
def api_default():
    """返回内置演示场景，经 normalize_scenario 补齐字段与吸附规则。"""
    return jsonify(normalize_scenario(default_scenario()))


@parking_bp.route("/api/optimize", methods=["POST"])
def api_optimize():
    """对 scenario 运行 exact（匈牙利全局最优）或 pso；错误时 JSON 内带 error 字段，HTTP 400。"""
    data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    raw = data.get("scenario") or default_scenario()
    method = str(data.get("method") or "exact").strip().lower()
    if method not in ("exact", "pso"):
        method = "exact"
    result = run_optimize(
        raw,
        seed=_coerce_seed(data.get("seed")),
        method=method,
    )
    if result.get("error"):
        return jsonify(result), 400
    return jsonify(result)
