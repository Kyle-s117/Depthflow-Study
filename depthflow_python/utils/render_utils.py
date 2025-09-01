
# utils/render_utils.py
import torch

def reflect_coord(x: torch.Tensor, size: int) -> torch.Tensor:
    """
    計算鏡像重複的座標。
    等同於 OpenGL 中的 GL_MIRRORED_REPEAT 紋理環繞模式。
    """
    if size <= 1: return torch.zeros_like(x)
    period = 2*(size-1)
    xr = torch.remainder(x, period)
    return torch.where(xr <= (size-1), xr, period - xr)

def bilinear(img: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor, mirror: bool, clamp: bool=True):
    """
    對影像進行雙線性內插採樣。
    支援 mirror (鏡像) 和 clamp (邊緣延伸) 兩種邊界處理模式。
    """
    H,W,C = img.shape
    if mirror:
        sx = reflect_coord(fx, W); sy = reflect_coord(fy, H)
    else:
        sx = fx.clamp(0, W-1) if clamp else fx
        sy = fy.clamp(0, H-1) if clamp else fy
        
    x0 = torch.floor(sx).long(); y0 = torch.floor(sy).long()
    x1 = (x0+1).clamp(0, W-1);  y1 = (y0+1).clamp(0, H-1)
    ax = (sx - x0).unsqueeze(-1); ay = (sy - y0).unsqueeze(-1)
    flat = img.view(-1, C)
    idx00 = (y0*W + x0).view(-1); idx10 = (y0*W + x1).view(-1)
    idx01 = (y1*W + x0).view(-1); idx11 = (y1*W + x1).view(-1)
    p00 = flat.index_select(0, idx00).view(H,W,C)
    p10 = flat.index_select(0, idx10).view(H,W,C)
    p01 = flat.index_select(0, idx01).view(H,W,C)
    p11 = flat.index_select(0, idx11).view(H,W,C)
    top = torch.lerp(p00, p10, ax)
    bot = torch.lerp(p01, p11, ax)
    return torch.lerp(top, bot, ay)

def gluv_to_pixel(gluv_x: torch.Tensor, gluv_y: torch.Tensor, H: int, W: int):
    """
    將 GLUV 座標轉換為像素座標。
    """
    px = gluv_x * (H*0.5) + (W-1)*0.5
    py = -gluv_y * (H*0.5) + (H-1)*0.5
    return px, py