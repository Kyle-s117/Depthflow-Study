# core/renderer.py
import torch
import numpy as np
from tqdm import tqdm
from .params import DFParams
from utils.render_utils import gluv_to_pixel, bilinear

def depthflow_raycast(image_rgb_np: np.ndarray, depth01_np: np.ndarray,
                      params: DFParams, device="cuda", show_progress=True):
    """
    DepthFlow 演算法的核心實作。
    """
    H,W,_ = image_rgb_np.shape
    img = torch.from_numpy(image_rgb_np.astype(np.float32)).to(device)
    dpt = torch.from_numpy(depth01_np.astype(np.float32)).to(device).unsqueeze(-1)
    
    # --- 步驟 1: 設定座標系與相機基礎參數 ---
    X = torch.arange(W, device=device, dtype=torch.float32).view(1, W).expand(H, W)
    Y = torch.arange(H, device=device, dtype=torch.float32).view(H, 1).expand(H, W)
    cx, cy = (W-1)*0.5, (H-1)*0.5
    gluv_x_raw = (X - cx) / (H*0.5)
    gluv_y_raw = -(Y - cy) / (H*0.5)
    cam_pos_xy = torch.tensor([params.offset_x, params.offset_y], device=device).view(1, 1, 2)

    # --- 步驟 2: 計算投影座標 (Projected GLUV) 與鏡頭縮放因子 ---
    focal_length = max(1e-6, 1.0 - params.focus * params.height)
    projection_scale = (params.zoom / focal_length)
    gluv_proj_x = gluv_x_raw * projection_scale + cam_pos_xy[..., 0]
    gluv_proj_y = gluv_y_raw * projection_scale + cam_pos_xy[..., 1]

    # --- 步驟 3: 計算光線目標點 (Intersect) ---
    rel_steady = params.steady * params.height
    glued_scale = (1.0 / (1.0 - rel_steady)) if params.glued else 0.0
    intersect_x = (params.center_x + gluv_proj_x) - (cam_pos_xy[..., 0] * glued_scale)
    intersect_y = (params.center_y + gluv_proj_y) - (cam_pos_xy[..., 1] * glued_scale)
    intersect = torch.stack([intersect_x, intersect_y, torch.ones_like(intersect_x)], dim=-1)
    
    # --- 步驟 4: 計算精確的 3D 光線起始點 (Ray Origin) ---
    iso_scale = params.zoom * params.isometric
    iso_rect_x = gluv_x_raw * iso_scale
    iso_rect_y = gluv_y_raw * iso_scale
    origin_base_x = cam_pos_xy[..., 0] + iso_rect_x
    origin_base_y = cam_pos_xy[..., 1] + iso_rect_y
    origin_base_z = torch.full_like(origin_base_x, -params.dolly)
    final_origin_x = origin_base_x + params.origin_x
    final_origin_y = origin_base_y + params.origin_y
    final_origin_z = origin_base_z
    final_origin = torch.stack([final_origin_x, final_origin_y, final_origin_z], dim=-1)

    # --- 步驟 5: 光線步進 (Ray Marching) ---
    safe = 1.0 - params.height
    walk = torch.zeros((H,W), device=device, dtype=torch.float32)
    hit = torch.zeros_like(walk, dtype=torch.bool)
    map_xy_final = torch.zeros((H,W,2), device=device, dtype=torch.float32)
    q = params.quality
    quality_step = 1.0 / (200.0 + 1800.0 * q)
    probe_step   = 1.0 / (50.0 + 70.0 * q)
    pbar = tqdm(total=H*W, desc="Raymarching", disable=(not show_progress))
    active = torch.ones_like(walk, dtype=torch.bool)
    
    for _ in range(params.max_iter): # Forward
        if not torch.any(active): break
        walk.add_(probe_step * active.float())
        S = (safe + (1.0 - safe) * walk).unsqueeze(-1)
        point = torch.lerp(final_origin, intersect, S)
        gx, gy = gluv_to_pixel(point[..., 0], point[..., 1], H, W)
        value = bilinear(dpt, gx, gy, mirror=params.mirror, clamp=True)[..., 0]
        surface = params.height * torch.lerp(value, 1.0 - value, params.invert)
        ceiling = 1.0 - point[..., 2]
        enter = (ceiling < surface) & active
        if torch.any(enter):
            hit[enter] = True; active[enter] = False
            pbar.update(int(enter.sum().item()))
        walk_limit = (walk > 1.0) & active
        active[walk_limit] = False

    for _ in range(params.max_iter): # Backward
        if not torch.any(hit): break
        walk.sub_(quality_step * hit.float())
        S = (safe + (1.0 - safe) * walk).unsqueeze(-1)
        point = torch.lerp(final_origin, intersect, S)
        gx, gy = gluv_to_pixel(point[..., 0], point[..., 1], H, W)
        value = bilinear(dpt, gx, gy, mirror=params.mirror, clamp=True)[..., 0]
        surface = params.height * torch.lerp(value, 1.0 - value, params.invert)
        ceiling = 1.0 - point[..., 2]
        done = (ceiling >= surface) & hit
        if torch.any(done):
            map_xy_final[done] = point[..., :2][done]
            hit[done] = False
            pbar.update(int(done.sum().item()))
            
    pbar.close()
    
    # --- 步驟 6: 最終採樣 ---
    final_map_x, final_map_y = gluv_to_pixel(map_xy_final[..., 0], map_xy_final[..., 1], H, W)
    out = bilinear(img, final_map_x, final_map_y, mirror=params.mirror, clamp=True)
    out = out.clamp(0, 255).cpu().numpy().astype(np.uint8)
    
    return out, map_xy_final