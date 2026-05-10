# 🚀 ANTIGRAVITY OCR 引擎操作與部署手冊

這份文件詳細說明了專案中新增的 OCR 文字辨識模組的運作機制、外部模型管理，以及發布到 GitHub Pages 所需的步驟。

---

## 1. 模型託管與載入機制 (Hugging Face)

目前系統內部的 ONNX 模型**完全不需要**上傳到 Github 倉庫中，也不會佔用專案體積。系統現已內建「模型動態切換」機制：

- **動態型號選擇**：你可以透過介面上的選單，即時在「V3 平衡型」、「V4 高準確度型」以及「V4 伺服器端精準型」之間自由切換。
- **模型來源**：
  - **V3**：從你的 Hugging Face 倉庫 `tobytoy/yolo_base_home` 拉取。
  - **V4 系列**：全自動由社群高優化的 `deepghs/paddleocr` 公開儲存庫中進行低延遲串流載入。
- **載入方式**：主線程執行非同步下載 (`fetch`)，並將 `ArrayBuffer` 零拷貝轉移 (Zero-copy Transfer) 至 Web Worker 中初始化，最大化記憶體利用效率。

> [!NOTE]
> **你現在完全不需要手動上傳檔案到 Hugging Face，也不需要變更程式碼**。我已經驗證過 `tobytoy/yolo_base_home` 上的 PaddleOCR 模型連結都處於正常狀態，可以直接被應用程式抓取。

---

## 2. 建置流程與依賴 (Build & Deploy)

為了讓部署流程盡可能輕量化，我已經將先前的 Python 修復工具**完全改寫為 Node.js 原生腳本 (`patch_wasm_js.js`)**。

### 💡 開發 / 建置時是否需要 Python？
**不需要！** 
現在整個專案的構建完全是 **Pure JavaScript (Node)** 環境。

### 自動化指令週期
我們在 `package.json` 中加入了 `predev` 與 `prebuild` 的生命週期鉤子，當你執行 `npm run dev` 或 `npm run build` 時，系統會自動完成以下魔術操作：
1. 自動偵測 `node_modules` 裡面的 ONNX WebAssembly 原生函式庫。
2. 複製二進制檔案到 `public/` 目錄以便瀏覽器訪問。
3. **自動執行清洗過濾**：將官方 JS 套件中的 Node.js 專用程式碼剃除，並更改副檔名以防止 Vite 的靜態分析器報錯。

---

## 3. 準備發佈至 GitHub Pages 的檢查清單

當你準備好推送到 GitHub 進行測試時，請確認以下事項：

1. **本地建置測試**
   在推代碼前，建議可以在終端機跑一次 `npm run build`，確保打包過程無任何紅字報錯。
   
2. **確認 `vite.config.js` 的 `base` 設定**
   專案目前設定 `base: '/ai-yolo/'`。請確保你的 Github Repository 名字也是 `ai-yolo`。如果名字不同，請記得去 `vite.config.js` 與 `src/ocr-worker.js` 的 `publicBase` 處做相對應修正。

3. **跨域隔離 (COOP / COEP) 標頭設定**
   由於 OCR 引擎使用了多線程 `SharedArrayBuffer` 來加速，如果部署上 Github Pages 後出現功能失效，你需要確保靜態頁面送出的 Header 包含：
   - `Cross-Origin-Embedder-Policy: require-corp`
   - `Cross-Origin-Opener-Policy: same-origin`

   > [!TIP]
   > 如果 Github Pages 原生不支援這兩個標頭，通常可以使用開源的 [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) 指令碼置入首頁，系統就能在瀏覽器端自動掛載這兩個標頭了！

---

祝你測試順利！🎉 現在你可以直接將代碼推送上 Github 享受純瀏覽器端的極速 OCR 了！
