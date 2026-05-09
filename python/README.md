# YOLO Python 訓練與推論工具

此目錄包含用於 YOLO 模型的 **模型訓練 (Training)**、**格式導出 (Exporting)** 以及 **物件偵測推論 (Inference)** 的 Python 腳本。所有腳本均基於 [Ultralytics](https://docs.ultralytics.com/) 官方 API 實作。

---

## 🚀 環境要求與安裝

請確保您正在使用專用的 Conda 虛擬環境 `toby`：

```bash
# 1. 啟用 conda 環境
conda activate toby

# 2. 安裝本模組所需依賴項（如果您尚未安裝）
pip install ultralytics torch torchvision
```

---

## 🛠️ 腳本說明與使用方式

### 1. 模型訓練 (`train.py`)
使用 Ultralytics YOLO API 訓練自定義或預訓練模型。預設會下載一個小型的內建 COCO8 數據集進行演示。

* **基本用法**：
  ```bash
  python train.py --epochs 3 --batch 16 --model yolov8n.pt
  ```
* **常用參數列表**：
  * `--model`：預訓練權重或模型配置文件名稱（如 `yolov8n.pt`、`yolo11n.pt`，預設為 `yolov8n.pt`）。
  * `--data`：數據集 YAML 文件路徑（預設為 `coco8.yaml`）。
  * `--epochs`：訓練的總輪數（Epochs，預設為 `3`）。
  * `--batch`：每批次樣本數（Batch Size，預設為 `16`）。
  * `--device`：指定運行設備（如 `'cpu'`、`'0'` 代表第一張 GPU 顯卡，不填則會自動選擇最佳設備）。

---

### 2. 模型導出 (`export.py`)
將 PyTorch 權重檔 (`.pt`) 轉換為其他推論格式（例如：供瀏覽器或網頁端高速推論所需的 ONNX 格式）。

* **基本用法**：
  ```bash
  python export.py --model runs/detect/train_experiment/weights/best.pt --format onnx
  ```
* **常用參數列表**：
  * `--model`：需要導出的 PyTorch `.pt` 權重文件路徑。
  * `--format`：導出的格式名稱（如 `onnx`、`tflite`、`openvino`，預設為 `onnx`）。

---

### 3. 物件偵測推論 (`detect.py`)
使用訓練完成的模型或內建預訓練模型，對本機圖片、影片或網路 URL 進行物件偵測與標註。

* **基本用法**（使用內建 YOLOv8 Nano 對網路圖片進行偵測）：
  ```bash
  python detect.py --model yolov8n.pt --source https://ultralytics.com/images/bus.jpg
  ```
* **常用參數列表**：
  * `--model`：模型權重檔案路徑或官方權重名（預設為 `yolov8n.pt`）。
  * `--source`：推論來源，支援**圖片路徑、影片路徑、網頁 URL 或資料夾路徑**（必填參數）。
  * `--conf`：物件偵測置信度閥值（預設為 `0.25`）。
  * `--save`：是否儲存標註後的視覺化結果（預設為啟用，結果會保存在 `runs/detect/predict_experiment` 中）。

---

## 📂 輸出目錄說明
在運行上述腳本時，會在此儲存庫自動創建 `runs/` 目錄：
* `runs/detect/train_experiment/`：存放訓練時的權重檔（包含 `best.pt` 與 `last.pt`）、損失圖表和評估指標。
* `runs/detect/predict_experiment/`：存放標註有物件邊界框 (Bounding Box) 的預測輸出圖片/影片。

---

## 🌐 網頁端部署與效果呈現 (GitHub Pages)

### 方案 B：本地放置於 `public/` 目錄（當前優先採用 🛠️）
如果您想直接在專案本地載入自定義轉換好的 ONNX 模型：

1. **將訓練好的 `.pt` 轉成 `.onnx`**：
   ```bash
   python python/export.py --model python/models/your_model.pt --format onnx
   ```
2. **放置模型檔案**：
   在前端專案的 `public/models/` 目錄下建立您模型的專屬資料夾，並將匯出的 `.onnx` 命名為 `model.onnx` 放入其中：
   ```bash
   mkdir -p public/models/custom_model
   cp python/models/your_model.onnx public/models/custom_model/model.onnx
   ```
3. **網頁端載入測試**：
   * 在 [index.html](file:///home/toymsi/documents/Yolo/Github/ai-yolo/index.html) 的 `<select id="model-select">` 標籤中，已為您預留好選單選項：
     ```html
     <option value="models/custom_model">Custom Local Model (public/models/)</option>
     ```
   * 我們的高速 Web Worker 會**自動偵測並調用本地模型路徑**。在網頁端直接下拉選擇該項，即可載入您的本地模型！

---

### 方案 A：上傳至 Hugging Face 模型庫（未來擴充方案 ☁️）
為避免將體積過大（數十至數百 MB）的 ONNX 模型 Push 到 GitHub 儲存庫：

1. **轉換模型**：
   同樣執行轉換指令，產生 `your_model.onnx`。
2. **在 Hugging Face 建立模型庫**：
   * 註冊一個免費的 [Hugging Face](https://huggingface.co/) 帳號。
   * 新增一個 **Model Repository**（例如 `toydogcat/my-yolo-model`）。
   * 將 `.onnx` 檔案改名為 **`model.onnx`**，連同必要的配置文件（如 mapping 設定等）上傳到倉庫。
3. **載入與效果預覽**：
   在 [index.html](file:///home/toymsi/documents/Yolo/Github/ai-yolo/index.html) 選項中添加對應的 Hugging Face 模型標識符：
   ```html
   <option value="toydogcat/my-yolo-model">My Hugging Face Model</option>
   ```

---

## 🔗 相關參考文獻
* [Ultralytics YOLO 官方文件](https://docs.ultralytics.com/)
* [YOLOv8 模型架構與詳細介紹](https://docs.ultralytics.com/models/yolov8/)

