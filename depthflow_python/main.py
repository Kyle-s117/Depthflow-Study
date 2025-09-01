#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python DepthFlow 2.12版
=============================================================================
  Copyright (c) 2025 JoyVision Kyle. All rights reserved.
  - 整體思路是在一個虛擬的 3D 空間中進行光線步進，空間由 2D 深度圖沿 Z 軸拉伸而成(像浮雕)。
  - 'depthflow_raycast' 為核心函式，模擬從相機(每個像素)發出光線，並計算該光線與深度圖表面交點的過程。
  - 對應 Unity 可以是 :
    1. 一個正交相機(Orthographic Camera)。
    2. 一個代表螢幕的平面(Quad)。
    3. 在 Fragment Shader(或 Unity 的 Shader Graph)中實現 `depthflow_raycast` 的邏輯。
    4. 深度圖作為一張紋理(Texture2D)傳入 Shader。
    5. 所有 DFParams 參數都作為 Uniforms 傳入 Shader。
  - 6. `--embed-data` 命令列參數，選擇是否將額外的採樣座標數據嵌入到輸出的 PNG 檔案中。預設只儲存純粹的 RGB 影像。
"""
import torch
import argparse
from PIL import Image


from utils.image_utils import load_image_rgb, load_depth, preprocess_depth
from utils.file_utils import save_png_with_npz
from core.params import DFParams
from core.renderer import depthflow_raycast

def build_argparser():
    """建立命令列參數解析器。"""
    p = argparse.ArgumentParser("DepthFlow GPU (Final Definitive Version)")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("paintDF")
    q.add_argument("-i","--image", required=True)
    q.add_argument("-d","--depth", required=True)
    q.add_argument("-o","--output", required=True)
    q.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    q.add_argument("--show-progress", action="store_true")
    q.add_argument("--auto-contrast", action="store_true")
    q.add_argument("--invert-depth", action="store_true")
    q.add_argument("--gamma", type=float, default=1.0)
    q.add_argument("--embed-data", action="store_true", help="將採樣座標等額外數據嵌入到輸出的 PNG 中。")

    q.add_argument("--height", type=float, default=0.20)
    q.add_argument("--steady", type=float, default=0.00)
    q.add_argument("--focus",  type=float, default=0.00)
    q.add_argument("--quality", type=float, default=0.90)
    q.add_argument("--invert", type=float, default=0.00)
    q.add_argument("--no-mirror", action="store_true")
    q.add_argument("--no-glued",  action="store_true")
    
    q.add_argument("--offset-x", type=float, default=0.00)
    q.add_argument("--offset-y", type=float, default=0.00)
    q.add_argument("--center-x", type=float, default=0.00)
    q.add_argument("--center-y", type=float, default=0.00)
    q.add_argument("--origin-x", type=float, default=0.00)
    q.add_argument("--origin-y", type=float, default=0.00)
    q.add_argument("--isometric", type=float, default=0.00)
    q.add_argument("--dolly",     type=float, default=0.00)
    q.add_argument("--zoom",      type=float, default=1.00)
    return p

def main():
    """主執行函式，解析參數並呼叫核心處理函式。"""

    p = build_argparser()
    args = p.parse_args()
    if args.cmd == "paintDF":
        # 1. 載入與預處理資料
        img = load_image_rgb(args.image)
        H,W,_ = img.shape
        dp = preprocess_depth(load_depth(args.depth), H, W, args.auto_contrast, args.invert_depth, args.gamma)
        
        # 2. 將所有命令列參數打包進 DFParams 物件
        params = DFParams(
            height=args.height, steady=args.steady, focus=args.focus, quality=args.quality, invert=args.invert,
            mirror=(not args.no_mirror), glued=(not args.no_glued), offset_x=args.offset_x, offset_y=args.offset_y,
            center_x=args.center_x, center_y=args.center_y, origin_x=args.origin_x, origin_y=args.origin_y,
            isometric=args.isometric, dolly=args.dolly, zoom=args.zoom, max_iter=256
        )
        
        # 3. 呼叫核心渲染函式
        out, map_xy_final_tensor = depthflow_raycast(img, dp, params, device=args.device, show_progress=args.show_progress)
        
        # 4. 根據使用者選擇決定如何儲存檔案
        if args.embed_data:
            print("正在將額外數據嵌入 PNG...")
            map_xy_final_np = map_xy_final_tensor.cpu().numpy()
            save_png_with_npz(args.output, out, map_xy=map_xy_final_np)
        else:
            print("正在儲存純影像 PNG...")
            Image.fromarray(out).save(args.output, "PNG")

if __name__ == "__main__":
    main()