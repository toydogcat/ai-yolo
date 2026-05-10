/**
 * Main Application Script for the ANTIGRAVITY OCR Lab
 */
import OCRWorker from './ocr-worker.js?worker';

const UI = {
  dropZone: document.getElementById('dropzone'),
  fileInput: document.getElementById('file-input'),
  previewContainer: document.getElementById('preview-container'),
  imagePreview: document.getElementById('image-preview'),
  ocrOverlay: document.getElementById('ocr-overlay'),
  statusDisplay: document.getElementById('status-display'),
  engineStateDisplay: document.getElementById('engine-state'),
  timeDisplay: document.getElementById('time-display'),
  sidebarResults: document.getElementById('sidebar-results'),
  runBtn: document.getElementById('run-btn'),
  clearBtn: document.getElementById('clear-btn'),
  copyAllBtn: document.getElementById('copy-all-btn'),
};

let worker = null;
let currentImageData = null;
let ocrDataCache = [];
let isInitializing = false;
let isInitialized = false;

const MODEL_MAP = {
  v3_balanced: {
    base: "https://huggingface.co/tobytoy/yolo_base_home/resolve/main/paddle",
    det: "/ch_PP-OCRv3_det_infer.onnx",
    rec: "/ch_PP-OCRv3_rec_infer.onnx",
    dict: "/ppocr_keys_v1.txt"
  },
  v4_mobile: {
    base: "https://huggingface.co/tobytoy/yolo_base_home/resolve/main/paddle",
    det: "/ch_PP-OCRv4_det_infer.onnx",
    rec: "/ch_PP-OCRv4_rec_infer.onnx",
    dict: "/ppocr_keys_v1.txt"
  },
  v4_server: {
    base: "https://huggingface.co/tobytoy/yolo_base_home/resolve/main/paddle",
    det: "/ch_PP-OCRv4_server_det_infer.onnx",
    rec: "/ch_PP-OCRv4_server_rec_infer.onnx",
    dict: "/ppocr_keys_v1.txt"
  }
};

UI.modelSelect = document.getElementById('ocr-model-select');

// ---------------------------------------------------------
// Core Worker Orchestration
// ---------------------------------------------------------

async function initOCR() {
  if (isInitialized || isInitializing) return;
  isInitializing = true;
  updateStatus('Initializing...');
  UI.runBtn.disabled = true; // ENFORCE: Terminate user race-condition triggers during download

  const selectedModelId = UI.modelSelect.value;
  const cfg = MODEL_MAP[selectedModelId] || MODEL_MAP.v3_balanced;

  try {
    if (worker) {
      worker.terminate();
      worker = null;
    }
    
    worker = new OCRWorker();
    
    worker.onerror = (err) => {
      console.error("Worker crashed on load:", err);
      updateStatus("Worker Crash");
      isInitializing = false;
    };

    worker.onmessage = (e) => {
      const { type, data } = e.data;
      
      if (type === 'status') {
        updateStatus(data);
      } else if (type === 'initialized') {
        isInitialized = true;
        isInitializing = false;
        updateStatus('Ready');
        UI.runBtn.disabled = false;
      } else if (type === 'result') {
        handleResults(data.results, data.duration);
      } else if (type === 'error') {
        console.error("OCR Worker Failed:", data);
        const errBox = document.createElement('div');
        errBox.style = "color:red; font-size:12px; margin-top:5px; word-break:break-all;";
        errBox.innerText = "Err details: " + data;
        UI.statusDisplay.parentNode.appendChild(errBox);
        
        updateStatus('Engine Error');
        isInitializing = false;
      }
    };

    // 1. Fetch model chunks from respective HF endpoint
    updateStatus('Downloading Models...');
    
    const detUrl = cfg.det.startsWith('http') ? cfg.det : cfg.base + cfg.det;
    const recUrl = cfg.rec.startsWith('http') ? cfg.rec : cfg.base + cfg.rec;
    const dictUrl = cfg.dict.startsWith('http') ? cfg.dict : cfg.base + cfg.dict;
    
    const [detRes, recRes, dictRes] = await Promise.all([
      fetch(detUrl),
      fetch(recUrl),
      fetch(dictUrl)
    ]);

    if(!detRes.ok || !recRes.ok || !dictRes.ok) {
      throw new Error("Failed to stream model data from server.");
    }

    const [detBuf, recBuf, dictText] = await Promise.all([
      detRes.arrayBuffer(),
      recRes.arrayBuffer(),
      dictRes.text()
    ]);

    // 2. Pass to worker
    worker.postMessage({
      type: 'init',
      data: {
        detBuffer: detBuf,
        recBuffer: recBuf,
        dictContent: dictText
      }
    }, [detBuf, recBuf]); // Transfer ownership for zero-copy speed!

  } catch (error) {
    console.error(error);
    updateStatus('Download Failed');
    isInitializing = false;
  }
}

// Add trigger to allow reloading upon selection change
UI.modelSelect.addEventListener('change', () => {
  isInitialized = false;
  isInitializing = false;
  UI.runBtn.disabled = true; // Lock immediate clicking
  updateStatus('Reloading Engine...');
  initOCR();
});

async function runOCR() {
  if (!currentImageData || !isInitialized) {
    if (!isInitialized) await initOCR();
    if (!currentImageData) return;
  }

  UI.runBtn.disabled = true;
  UI.ocrOverlay.innerHTML = '';
  updateStatus('Extracting...');
  
  // Extract raw pixel data using a virtual canvas
  const img = UI.imagePreview;
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const idata = ctx.getImageData(0, 0, canvas.width, canvas.height);

  worker.postMessage({
    type: 'recognize',
    data: {
      imageData: idata.data,
      width: canvas.width,
      height: canvas.height
    }
  }, [idata.data.buffer]);
}

// ---------------------------------------------------------
// UI Rendering & Interaction logic
// ---------------------------------------------------------

function handleResults(results, duration) {
  UI.runBtn.disabled = false;
  UI.timeDisplay.textContent = `${duration} ms`;
  ocrDataCache = results;

  const { naturalWidth, naturalHeight } = UI.imagePreview;
  UI.ocrOverlay.innerHTML = '';
  UI.sidebarResults.innerHTML = '';

  if (!results || results.length === 0) {
    UI.sidebarResults.innerHTML = `
      <div class="results-empty-state">
        <div class="empty-icon">📭</div>
        <p>No recognizable text was detected in this image.</p>
      </div>
    `;
    updateStatus('No Text Found');
    return;
  }

  results.forEach((item) => {
    const { text, box, confidence } = item;
    
    // 1. Create visual interactive overlay element
    const div = document.createElement('div');
    div.className = 'ocr-text-box';
    
    // Convert percentages
    const left = (box.x / naturalWidth) * 100;
    const top = (box.y / naturalHeight) * 100;
    const width = (box.width / naturalWidth) * 100;
    const height = (box.height / naturalHeight) * 100;

    div.style.left = `${left}%`;
    div.style.top = `${top}%`;
    div.style.width = `${width}%`;
    div.style.height = `${height}%`;
    
    div.innerText = text;
    div.title = `${text} (${Math.round(confidence*100)}%)`;
    
    div.addEventListener('click', (e) => {
      navigator.clipboard.writeText(text);
      showMiniNotification(`Copied: "${text}"`);
      e.stopPropagation();
    });

    UI.ocrOverlay.appendChild(div);

    // 2. Add sidebar item matching core system
    const card = document.createElement('div');
    card.className = 'sidebar-link-item';
    card.innerHTML = `
      <span class="text-content">${text}</span>
      <span class="copy-hint">Click to copy text</span>
    `;
    card.addEventListener('click', () => {
      navigator.clipboard.writeText(text);
      showMiniNotification(`Copied to clipboard`);
    });
    UI.sidebarResults.appendChild(card);
  });

  updateStatus('Success');
}

function showMiniNotification(msg) {
  const banner = document.createElement('div');
  banner.style = `
    position: fixed; bottom: 20px; right: 20px;
    background: var(--color-emerald); color: white; padding: 0.75rem 1.5rem;
    border-radius: 8px; font-weight: 600; font-family: var(--font-sans);
    box-shadow: 0 10px 25px rgba(0,0,0,0.5); z-index: 9999;
    animation: slideUp 0.3s ease-out;
  `;
  banner.textContent = msg;
  
  const styleTag = document.getElementById('slide-anim') || document.createElement('style');
  if (!document.getElementById('slide-anim')) {
    styleTag.id = 'slide-anim';
    styleTag.innerHTML = `@keyframes slideUp { from { transform: translateY(20px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }`;
    document.head.appendChild(styleTag);
  }
  
  document.body.appendChild(banner);
  setTimeout(() => banner.remove(), 2000);
}

function updateStatus(txt) {
  UI.statusDisplay.textContent = txt;
  if (UI.engineStateDisplay) UI.engineStateDisplay.textContent = txt;
}

// ---------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------

// Dropzone functionality
UI.dropZone.addEventListener('click', (e) => {
  if (e.target.tagName !== 'INPUT') UI.fileInput.click();
});

UI.fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) loadFile(file);
});

UI.dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  UI.dropZone.classList.add('dragover');
});

UI.dropZone.addEventListener('dragleave', () => {
  UI.dropZone.classList.remove('dragover');
});

UI.dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  UI.dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    loadFile(file);
  }
});

function loadFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    UI.imagePreview.src = e.target.result;
    currentImageData = e.target.result;
    UI.dropZone.classList.add('hidden');
    UI.previewContainer.classList.remove('hidden');
    UI.ocrOverlay.innerHTML = '';
    UI.sidebarResults.innerHTML = `
       <div class="results-empty-state">
          <div class="empty-icon">🔥</div>
          <p>Image loaded. Click "Extract Text" to process.</p>
       </div>
    `;
    
    // Auto Trigger Run if already loaded
    if (isInitialized) {
      setTimeout(runOCR, 300);
    }
  };
  reader.readAsDataURL(file);
}

UI.runBtn.addEventListener('click', runOCR);

UI.clearBtn.addEventListener('click', () => {
  currentImageData = null;
  UI.imagePreview.src = '';
  UI.dropZone.classList.remove('hidden');
  UI.previewContainer.classList.add('hidden');
  UI.ocrOverlay.innerHTML = '';
  UI.sidebarResults.innerHTML = `
     <div class="results-empty-state">
       <div class="empty-icon">📄</div>
       <p>No text extracted yet. Drag an image to analyze.</p>
     </div>
  `;
  UI.timeDisplay.textContent = '-- ms';
});

UI.copyAllBtn.addEventListener('click', () => {
  if (!ocrDataCache.length) return;
  const fullText = ocrDataCache.map(d => d.text).join('\n');
  navigator.clipboard.writeText(fullText);
  showMiniNotification('Full batch copied!');
});

// Trigger Lazy Init on page mount
setTimeout(initOCR, 500);
