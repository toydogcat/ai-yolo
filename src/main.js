// YOLO Object Detection Main Controller

// 1. Initialize Web Worker with ESM support
const worker = new Worker(
  new URL('./worker.js', import.meta.url),
  { type: 'module' }
);

// 2. DOM Elements Selection
const dropzone = document.getElementById('dropzone');
const fileUpload = document.getElementById('file-upload');
const previewContainer = document.getElementById('preview-container');
const imageElement = document.getElementById('image-element');
const canvasOverlay = document.getElementById('canvas-overlay');
const btnReset = document.getElementById('btn-reset');
const btnReprocess = document.getElementById('btn-reprocess');

const modelSelect = document.getElementById('model-select');
const thresholdRange = document.getElementById('threshold-range');
const thresholdVal = document.getElementById('threshold-val');

const progressOverlay = document.getElementById('progress-overlay');
const progressFilename = document.getElementById('progress-filename');
const progressBar = document.getElementById('progress-bar');
const progressPercent = document.getElementById('progress-percent');
const progressBytes = document.getElementById('progress-bytes');

const modelState = document.getElementById('model-state');
const inferenceTime = document.getElementById('inference-time');
const detectedCount = document.getElementById('detected-count');
const resultsList = document.getElementById('results-list');
const resultsEmpty = document.getElementById('results-empty');
const searchInput = document.getElementById('search-input');
const backendText = document.getElementById('backend-text');

// 3. Application State Variables
let currentFile = null;
let activeDetections = [];
let hoveredDetectionIndex = null;
let downloadTrackers = {};
let isModelReady = false;

// 4. Utility: Get consistent vibrant distinct colors for each label class
const classColors = {};
const vibrantColors = [
  '#6366f1', // Electric Indigo
  '#10b981', // Emerald Green
  '#ec4899', // Cyber Pink
  '#f59e0b', // Glowing Amber
  '#3b82f6', // Bright Azure
  '#8b5cf6', // Vivid Purple
  '#ef4444', // Intense Red
  '#06b6d4'  // Cyan Neon
];

function getDistinctColor(label) {
  if (classColors[label]) return classColors[label];
  const colorIndex = Object.keys(classColors).length % vibrantColors.length;
  classColors[label] = vibrantColors[colorIndex];
  return classColors[label];
}

// 5. Initialize selected model on startup
function initActiveModel() {
  const modelName = modelSelect.value;
  const localModelPath = window.location.pathname;
  isModelReady = false;
  modelState.textContent = 'Loading...';
  modelState.className = 'stat-value text-glow-indigo';
  
  worker.postMessage({
    type: 'init',
    data: { modelName, localModelPath }
  });
}

// 6. Handle model download progress
function handleProgress(progressData) {
  if (progressOverlay.classList.contains('hidden')) {
    progressOverlay.classList.remove('hidden');
  }

  const { status, file, loaded, total } = progressData;

  if (file) {
    if (status === 'initiate') {
      downloadTrackers[file] = { loaded: 0, total: 0 };
    } else if (status === 'progress') {
      downloadTrackers[file] = { loaded, total };
    } else if (status === 'done') {
      downloadTrackers[file] = { loaded: total || loaded, total: total || loaded };
    }

    // Abbreviate long file paths
    const fileParts = file.split('/');
    progressFilename.textContent = `Fetching: .../${fileParts[fileParts.length - 1]}`;
  }

  // Calculate aggregated progress details
  let totalLoaded = 0;
  let totalBytes = 0;
  for (const key in downloadTrackers) {
    if (downloadTrackers[key]) {
      totalLoaded += downloadTrackers[key].loaded || 0;
      totalBytes += downloadTrackers[key].total || 0;
    }
  }

  if (totalBytes > 0) {
    const percent = Math.round((totalLoaded / totalBytes) * 100);
    progressBar.style.width = `${percent}%`;
    progressPercent.textContent = `${percent}%`;
    progressBytes.textContent = `${(totalLoaded / (1024 * 1024)).toFixed(1)} MB / ${(totalBytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  // Hide overlay if everything has loaded
  if (status === 'ready' || (totalBytes > 0 && totalLoaded === totalBytes)) {
    setTimeout(() => {
      progressOverlay.classList.add('hidden');
      downloadTrackers = {};
    }, 800);
  }
}

// 7. Extract raw pixels on main thread using offscreen canvas
function getImagePixels(img) {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  return {
    pixels: new Uint8Array(imageData.data.buffer),
    width: canvas.width,
    height: canvas.height,
    channels: 4
  };
}

// 8. Run YOLO inference by posting raw pixels to Web Worker
function triggerInference() {
  if (!isModelReady) {
    initActiveModel();
    // Wait until model is ready, re-triggered automatically
    return;
  }

  if (!imageElement.src || imageElement.src === window.location.href) return;

  modelState.textContent = 'Analyzing...';
  modelState.className = 'stat-value text-glow-indigo';

  const imgData = getImagePixels(imageElement);
  const threshold = parseFloat(thresholdRange.value) / 100;
  const modelName = modelSelect.value;
  const localModelPath = window.location.pathname;

  // Use Transferable Object (pixels.buffer) to eliminate memory duplication lag
  worker.postMessage({
    type: 'detect',
    data: {
      modelName,
      pixels: imgData.pixels,
      width: imgData.width,
      height: imgData.height,
      channels: imgData.channels,
      threshold,
      localModelPath
    }
  }, [imgData.pixels.buffer]);
}

// 9. Canvas rendering of bounding boxes & labels (with highlight support)
function renderCanvas() {
  if (!imageElement.naturalWidth) return;

  // Synchronize canvas internal resolution to the actual rendered dimensions
  const rect = imageElement.getBoundingClientRect();
  canvasOverlay.width = rect.width;
  canvasOverlay.height = rect.height;

  const scaleX = rect.width / imageElement.naturalWidth;
  const scaleY = rect.height / imageElement.naturalHeight;

  const ctx = canvasOverlay.getContext('2d');
  ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

  activeDetections.forEach((det, index) => {
    const { label, score, box } = det;
    const { xmin, ymin, xmax, ymax } = box;

    const x = xmin * scaleX;
    const y = ymin * scaleY;
    const w = (xmax - xmin) * scaleX;
    const h = (ymax - ymin) * scaleY;

    const color = getDistinctColor(label);
    const isHovered = hoveredDetectionIndex === index;

    // Glowing stroke effect
    ctx.shadowColor = color;
    ctx.shadowBlur = isHovered ? 20 : 8;
    ctx.strokeStyle = color;
    ctx.lineWidth = isHovered ? 6 : 3;
    ctx.strokeRect(x, y, w, h);

    // Reset shadow blur for crisp text drawing
    ctx.shadowBlur = 0;

    // Capsule title card background
    const labelText = `${label} ${Math.round(score * 100)}%`;
    ctx.font = 'bold 11px "Outfit", sans-serif';
    const textWidth = ctx.measureText(labelText).width;
    
    ctx.fillStyle = color;
    // Render text background bubble
    ctx.fillRect(x, y - 20 > 0 ? y - 20 : y, textWidth + 14, 18);

    // Text label
    ctx.fillStyle = '#ffffff';
    ctx.fillText(labelText, x + 7, y - 20 > 0 ? y - 7 : y + 13);
  });
}

// 10. Render interactive sidebar detection lists
function renderSidebarList(filter = '') {
  resultsList.innerHTML = '';

  const filteredDetections = activeDetections.map((d, index) => ({ ...d, originalIndex: index }))
    .filter(d => d.label.toLowerCase().includes(filter.toLowerCase()));

  if (filteredDetections.length === 0) {
    resultsEmpty.classList.remove('hidden');
    resultsList.appendChild(resultsEmpty);
    return;
  }

  resultsEmpty.classList.add('hidden');

  filteredDetections.forEach((det) => {
    const color = getDistinctColor(det.label);
    const itemCard = document.createElement('div');
    itemCard.style.color = color;
    itemCard.className = `detection-item ${hoveredDetectionIndex === det.originalIndex ? 'active' : ''}`;
    
    itemCard.innerHTML = `
      <div class="detection-item-left">
        <span class="detection-color-badge"></span>
        <span class="detection-name" style="color: #f3f4f6">${det.label}</span>
      </div>
      <span class="detection-score">${Math.round(det.score * 100)}%</span>
    `;

    // Interactive hover triggers drawing synchronizations
    itemCard.addEventListener('mouseenter', () => {
      hoveredDetectionIndex = det.originalIndex;
      renderCanvas();
      itemCard.classList.add('active');
    });

    itemCard.addEventListener('mouseleave', () => {
      hoveredDetectionIndex = null;
      renderCanvas();
      itemCard.classList.remove('active');
    });

    resultsList.appendChild(itemCard);
  });
}

// 11. Handle files inputted via browsing or drag-and-drop
function handleSelectedFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  currentFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    imageElement.src = e.target.result;
    
    imageElement.onload = () => {
      dropzone.classList.add('hidden');
      previewContainer.classList.remove('hidden');
      // Wait for image render layout to trigger inference perfectly
      setTimeout(triggerInference, 100);
    };
  };
  reader.readAsDataURL(file);
}

// 12. Listen to Worker communications
worker.addEventListener('message', (event) => {
  const { type, data } = event.data;

  if (type === 'status') {
    const { data: statusMsg } = event.data;
    if (statusMsg === 'loading') {
      modelState.textContent = 'Loading...';
      modelState.className = 'stat-value text-glow-indigo';
    } else if (statusMsg === 'ready') {
      isModelReady = true;
      progressOverlay.classList.add('hidden');
      modelState.textContent = 'Ready';
      modelState.className = 'stat-value text-glow-green';
      
      // Auto-detect if image is already uploaded
      if (currentFile) {
        triggerInference();
      }
    } else if (statusMsg === 'processing') {
      modelState.textContent = 'Running...';
      modelState.className = 'stat-value text-glow-indigo';
    }
  }

  else if (type === 'progress') {
    handleProgress(data);
  }

  else if (type === 'result') {
    const { results, duration } = data;
    activeDetections = results;
    
    // Update performance details
    modelState.textContent = 'Idle';
    modelState.className = 'stat-value text-glow-green';
    inferenceTime.textContent = `${duration} ms`;
    detectedCount.textContent = results.length;

    // Detect WebGPU capabilities in runtime context
    if (navigator.gpu) {
      backendText.textContent = 'WebGPU Accelerated';
    } else {
      backendText.textContent = 'WebGL / WASM Backend';
    }

    renderCanvas();
    renderSidebarList(searchInput.value);
  }

  else if (type === 'error') {
    modelState.textContent = 'Load Error';
    modelState.className = 'stat-value text-glow-pink';
    alert(data);
  }
});

// 13. DOM Event Listeners Setup
modelSelect.addEventListener('change', () => {
  initActiveModel();
});

thresholdRange.addEventListener('input', (e) => {
  thresholdVal.textContent = `${e.target.value}%`;
});

thresholdRange.addEventListener('change', () => {
  triggerInference();
});

searchInput.addEventListener('input', (e) => {
  renderSidebarList(e.target.value);
});

// Drag and Drop
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length > 0) {
    handleSelectedFile(e.dataTransfer.files[0]);
  }
});

fileUpload.addEventListener('change', (e) => {
  if (e.target.files.length > 0) {
    handleSelectedFile(e.target.files[0]);
  }
});

btnReset.addEventListener('click', () => {
  currentFile = null;
  activeDetections = [];
  hoveredDetectionIndex = null;
  imageElement.src = '';
  
  const ctx = canvasOverlay.getContext('2d');
  ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

  previewContainer.classList.add('hidden');
  dropzone.classList.remove('hidden');

  inferenceTime.textContent = '0 ms';
  detectedCount.textContent = '0';
  renderSidebarList();
});

btnReprocess.addEventListener('click', () => {
  triggerInference();
});

// Sync overlay canvas size on resizing
window.addEventListener('resize', renderCanvas);

// 14. Initialize pipeline on boot
initActiveModel();
