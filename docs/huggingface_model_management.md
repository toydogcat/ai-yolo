# Hugging Face 統一模型管理指南 ☁️

本文件說明如何將訓練好的 YOLO 模型（包含 Detection 與 Pose 模型）有條理地部署至 Hugging Face 平台，並讓網頁前端能以極低延遲直接遠端抓取推論。

---

## 📂 1. 目錄架構說明

為了方便分類「官方預設」與「自定義訓練」模型，我們採用多層級分類結構：

### 🚀 本機開發結構 (`/python/models`)
```text
python/models/
├── default/             <-- 官方或預設模型在此
│   ├── yolo26l-pose.pt
│   ├── yolo26l-pose.onnx
│   └── ...
└── custom/              <-- (未來擴充) 自行訓練的模型在此
```

### ☁️ 雲端儲存庫結構 (`huggingface.co/tobytoy/yolo_base_home`)
```text
tobytoy/yolo_base_home/
├── default/
│   ├── yolo26l-pose/
│   │   ├── model.onnx      # 統一名稱
│   │   ├── config.json     # 自動產生的配置檔
│   │   └── original.pt     # 權重檔備份
│   └── ...
└── custom/
    └── ...
```

---

## 🔐 2. 環境變數安全設定 (.env)

為了避免在公開代碼或指令歷史紀錄中洩漏您的 `Hugging Face Write Token`，我們使用 `.env` 檔案進行本地快取。

1. 請確認 `.gitignore` 內已包含 `.env`。
2. 在 `python/.env` 檔案中填入：
   ```env
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

---

## ⚙️ 3. 自動化操作流程

當您在 `python/models/default/` 底下新增了轉換好的 ONNX 與 PT 檔案後，只需兩步即可完成發佈：

### Step 1：自動化打包整理
這會根據模型類型（偵測/姿勢）自動產生 `config.json`，並將檔案重新命名且歸納至暫存區。
```bash
conda run -n toby python python/organize_hf_upload.py
```

### Step 2：批次推送至雲端
這將自動讀取您的 `.env` Token，並利用 `huggingface_hub` 的高併發 API 將整個暫存目錄完整推上您的 HF Repositories。
```bash
conda run -n toby python python/hf_push_all.py
```

---

## 🎨 4. 網頁端對接方式

完成上傳後，您只需在 `index.html` 的選單中加入一筆新的 `<option>`，並使用 `hf:` 的格式前綴，Web Worker 即可自動跳轉至 CDN 位址抓取！

**格式規範：**
`hf:[使用者名稱]/[儲存庫名稱]/[子資料夾路徑]/[模型資料夾]`

**實例：**
```html
<option value="hf:tobytoy/yolo_base_home/default/yolo26l-pose">YOLO26l Pose (HF)</option>
```

---

## ✅ 維運小技巧
- **更新模型**：直接替換本機 `python/models/default/` 中的檔案，重複執行上述 Step 1 & Step 2 即可自動覆蓋雲端舊檔。
- **加入新任務**：未來如果有分割 (Segment) 任務模型，只需在 `organize_hf_upload.py` 中擴增任務偵測條件與對應標籤即可！
