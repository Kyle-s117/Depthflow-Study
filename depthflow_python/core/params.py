# core/params.py
from dataclasses import dataclass

@dataclass
class DFParams:
    """存放所有 DepthFlow 效果參數的資料類別。"""
    height: float = 0.2; steady: float = 0.0; focus: float = 0.0; quality: float = 0.9
    invert: float = 0.0; mirror: bool  = True; glued: bool   = True; offset_x: float = 0.0
    offset_y: float = 0.0; center_x: float = 0.0; center_y: float = 0.0; origin_x: float = 0.0
    origin_y: float = 0.0; isometric: float = 0.0; dolly: float = 0.0; zoom: float = 1.0
    max_iter: int = 1000