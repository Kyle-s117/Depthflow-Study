# utils/image_utils.py
import cv2 
import numpy as np
from pathlib import Path

def load_image_rgb(path: str) -> np.ndarray:
    """載入彩色圖片並確保其為 RGB 順序。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None: raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_depth(path: str) -> np.ndarray:
    """載入深度圖，支援多種格式，並正規化到 0-1 範圍。"""
    p = Path(path)
    if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}:
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if d is None: raise FileNotFoundError(path)
        if d.ndim == 3: d = d[...,0]
        d = d.astype(np.float32)
        if d.max() > 1.0: d /= 255.0
        return d
    raise ValueError(f"不支援的深度圖格式: {p}")

def preprocess_depth(d: np.ndarray, H: int, W: int,
                     auto_contrast: bool, invert: bool, gamma: float) -> np.ndarray:
    """對深度圖進行預處理，包括縮放、自動對比、反轉和 Gamma 校正。"""
    if d.shape[:2] != (H,W):
        d = cv2.resize(d, (W,H), interpolation=cv2.INTER_LINEAR)
    d = d.astype(np.float32)
    if auto_contrast:
        mn, mx = float(d.min()), float(d.max())
        if mx > mn: d = (d - mn)/(mx - mn)
    d = np.clip(d, 0.0, 1.0)
    if invert: d = 1.0 - d
    if gamma and gamma != 1.0: d = np.power(d, gamma)
    return d