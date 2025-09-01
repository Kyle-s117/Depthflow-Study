# DepthFlow 的 Python 實現 (PyTorch 版本)

參考 : 原生 [DepthFlow](https://github.com/BrokenSource/DepthFlow)。

將原始 GLSL 著色器中的複雜邏輯，轉換為一個獨立、可讀性高且模組化的 Python 腳本，以便於理解、二次開發，或作為將此演算法移植到其他平台的參考實現。

## 核心功能

* **演算法重現**：用Python實現原生 DepthFlow 的相機模型與光線步進邏輯，確保所有參數（如 `offset`, `steady`, `dolly` 等）的行為與視覺效果與原始 UI 版本完全一致。
* **GPU 加速**：用 PyTorch 的 CUDA 後端進行大規模並行運算，確保在大尺寸影像上依然能有高效的處理速度。
* **模組化設計**：將程式碼拆分為邏輯清晰的模組，方便維護與擴充。
* **豐富的命令列介面**：完整的命令列參數，可以微調幾乎所有 `DepthFlow` 的效果參數。
* **可選的數據嵌入**：可以選擇是否將計算出的採樣座標等除錯數據，打包嵌入到輸出的 PNG 影像中，方便後續分析。

## 專案結構

```
depthflow_project/
│
├── utils/
│   ├── __init__.py         
│   ├── image_utils.py      # 影像載入與預處理相關函式
│   ├── render_utils.py     # 渲染與採樣相關的底層輔助函式
│   └── file_utils.py       # 檔案儲存相關的輔助函式
│
├── core/
│   ├── __init__.py         
│   ├── params.py           # 存放 DFParams 資料類別
│   └── renderer.py         # 核心 depthflow_raycast 演算法
│
├── main.py                 # 主程式
│
└── Readme.md               # 說明檔案
```

## 安裝與設定

### 1. 環境需求

* Python 3.10 或更高版本
* NVIDIA GPU，並已正確安裝對應的 CUDA Toolkit

### 2. 安裝依賴套件


* `torch`: GPU 計算框架。
* `numpy`: 科學計算基礎庫。
* `opencv-python`: 讀取影像與深度圖。
* `Pillow`: 儲存最終的影像結果。
* `tqdm`: 進度條。


```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install numpy opencv-python Pillow tqdm
```
※根據系統的 CUDA 版本，從 [PyTorch 官網](https://pytorch.org/get-started/locally/) 取得對應的安裝指令。

## 使用方法

透過 `main.py` 腳本在命令列中執行。

### 基本語法

```bash
python main.py paintDF -i <原始圖片路徑> -d <深度圖路徑> -o <輸出圖片路徑> [可選參數...]
```


### 主要參數說明

* `-i`, `--image`: **[必需]** 原始彩色圖片的路徑。
* `-d`, `--depth`: **[必需]** 對應的灰階深度圖路徑（白色為近，黑色為遠）。
* `-o`, `--output`: **[必需]** 輸出結果的檔案路徑。
* `--offset-x`, `--offset-y`: 控制相機的水平和垂直位移，是產生視差效果的主要參數。(要自己調整)
* `--steady`: 視差的「不動點」深度。設為 `0.0` 時，最遠處的背景會保持靜止；設為 `0.5` 時，中間深度的物體會保持靜止。(state時預設0)
* `--height`: 深度圖的整體強度，影響前景與背景的最大位移量。(預設0.2)
* `--zoom`, `--dolly`, `--isometric`: 模擬相機的變焦、推軌和正交投影效果，影響畫面的透視感。
* `--no-mirror`: 關閉預設的鏡像邊界模式，改用邊緣像素延伸模式。
* `--embed-data`: 在輸出的 PNG 中嵌入額外的 NumPy 數據。
* `--show-progress`: 在命令列中顯示渲染進度條。
