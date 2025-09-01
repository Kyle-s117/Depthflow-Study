# utils/file_utils.py
import io
import base64
import numpy as np
from PIL import Image, PngImagePlugin

PNG_KEY = "OFFSET_PAINT_UV_NPZ_B64"

def save_png_with_npz(path: str, rgb: np.ndarray, **arrays):
    """將結果影像儲存為 PNG，並將額外數據嵌入到元資料中。"""
    img = Image.fromarray(rgb, mode="RGB")
    meta = PngImagePlugin.PngInfo()
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrays)
    meta.add_itxt(PNG_KEY, base64.b64encode(bio.getvalue()).decode("ascii"))
    img.save(path, "PNG", pnginfo=meta)