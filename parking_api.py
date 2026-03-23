# -*- coding: utf-8 -*-
"""小区停车 PSO：/api/default、/api/optimize（供 GitHub Pages 前端调用）"""
from __future__ import annotations

from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

from parking_engine import default_scenario, normalize_scenario, run_optimize

parking_bp = Blueprint("parking", __name__)


def _coerce_seed(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


@parking_bp.route("/api/default")
def api_default():
    return jsonify(normalize_scenario(default_scenario()))


@parking_bp.route("/api/optimize", methods=["POST"])
def api_optimize():
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
