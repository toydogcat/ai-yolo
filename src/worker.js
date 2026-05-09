import { pipeline, RawImage, env } from '@huggingface/transformers';
import * as ort from 'onnxruntime-web';

// Global cache for active models & sessions
let detector = null;
let customSession = null;
let customConfig = null;
let currentModelName = null;

// Configure Hugging Face environments
env.allowLocalModels = true;

/**
 * Singleton pattern for allocating either the Hugging Face pipeline or custom ONNX session.
 */
async function getDetector(modelName, localModelPath = '/ai-yolo/') {
  if (currentModelName === modelName) {
    if (modelName.startsWith('models/')) return { type: 'custom', session: customSession, config: customConfig };
    return { type: 'transformers', detector };
  }

  self.postMessage({ type: 'status', data: 'loading', model: modelName });

  if (modelName.startsWith('models/')) {
    // 1. Custom YOLOv8/YOLOv12 Direct ONNX Inference
    const modelUrl = `${localModelPath}models/custom_model/model.onnx`;
    const configUrl = `${localModelPath}models/custom_model/config.json`;

    try {
      // Fetch labels configuration
      const response = await fetch(configUrl);
      customConfig = await response.json();

      // Configure ONNX Runtime to use CDN for WASM files if needed
      ort.env.wasm.numThreads = 1;
      
      // Load custom session
      customSession = await ort.InferenceSession.create(modelUrl);
      detector = null;
    } catch (err) {
      throw new Error(`Failed to load custom local model: ${err.message}`);
    }

    currentModelName = modelName;
    self.postMessage({ type: 'status', data: 'ready', model: modelName });
    return { type: 'custom', session: customSession, config: customConfig };
  } else {
    // 2. Standard Transformers.js pipeline
    env.allowLocalModels = false;
    detector = await pipeline('object-detection', modelName, {
      progress_callback: (progressData) => {
        self.postMessage({ type: 'progress', data: progressData });
      }
    });
    customSession = null;
    customConfig = null;

    currentModelName = modelName;
    self.postMessage({ type: 'status', data: 'ready', model: modelName });
    return { type: 'transformers', detector };
  }
}

/**
 * Custom Preprocessing for YOLO (640x640, RGB, Channel-first NCHW, normalized to [0, 1])
 */
function preprocessYOLO(pixels, width, height) {
  const targetWidth = 640;
  const targetHeight = 640;
  const float32Data = new Float32Array(1 * 3 * targetWidth * targetHeight);

  // Simple and fast nearest-neighbor resize + RGB normalization
  for (let y = 0; y < targetHeight; y++) {
    for (let x = 0; x < targetWidth; x++) {
      const srcX = Math.floor(x * (width / targetWidth));
      const srcY = Math.floor(y * (height / targetHeight));
      const srcIndex = (srcY * width + srcX) * 4; // RGBA input buffer

      const r = pixels[srcIndex] / 255.0;
      const g = pixels[srcIndex + 1] / 255.0;
      const b = pixels[srcIndex + 2] / 255.0;

      // Planar NCHW format
      float32Data[0 * 640 * 640 + y * 640 + x] = r;
      float32Data[1 * 640 * 640 + y * 640 + x] = g;
      float32Data[2 * 640 * 640 + y * 640 + x] = b;
    }
  }

  return new ort.Tensor('float32', float32Data, [1, 3, targetWidth, targetHeight]);
}

/**
 * Custom Postprocessing for YOLOv8/YOLOv12 with Non-Maximum Suppression (NMS)
 * Outputs shape: [1, 84, 8400]
 */
function postprocessYOLO(outputTensor, threshold, originalWidth, originalHeight, id2label) {
  const data = outputTensor.data; // Float32Array
  const numCandidates = 8400;
  const numClasses = 80;
  const candidates = [];

  for (let i = 0; i < numCandidates; i++) {
    let maxScore = -1;
    let classId = -1;

    // Class scores start from index 4 to 83
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numCandidates + i];
      if (score > maxScore) {
        maxScore = score;
        classId = c;
      }
    }

    if (maxScore > threshold) {
      // Bounding box coordinates at indices 0, 1, 2, 3 [xc, yc, w, h] (normalized to 640)
      const xc = data[0 * numCandidates + i];
      const yc = data[1 * numCandidates + i];
      const w = data[2 * numCandidates + i];
      const h = data[3 * numCandidates + i];

      // Convert center to min/max pixel coordinates
      const xmin = (xc - w / 2) / 640 * originalWidth;
      const ymin = (yc - h / 2) / 640 * originalHeight;
      const xmax = (xc + w / 2) / 640 * originalWidth;
      const ymax = (yc + h / 2) / 640 * originalHeight;

      candidates.push({
        score: maxScore,
        label: id2label[classId] || `class_${classId}`,
        box: {
          xmin: Math.max(0, Math.round(xmin)),
          ymin: Math.max(0, Math.round(ymin)),
          xmax: Math.min(originalWidth, Math.round(xmax)),
          ymax: Math.min(originalHeight, Math.round(ymax))
        }
      });
    }
  }

  // Non-Maximum Suppression (IoU Threshold = 0.45)
  return nms(candidates, 0.45);
}

function nms(candidates, iouThreshold) {
  candidates.sort((a, b) => b.score - a.score);
  const kept = [];
  const active = new Array(candidates.length).fill(true);

  for (let i = 0; i < candidates.length; i++) {
    if (!active[i]) continue;
    const candA = candidates[i];
    kept.push(candA);

    const boxA = candA.box;
    for (let j = i + 1; j < candidates.length; j++) {
      if (!active[j]) continue;
      const boxB = candidates[j].box;

      const x1 = Math.max(boxA.xmin, boxB.xmin);
      const y1 = Math.max(boxA.ymin, boxB.ymin);
      const x2 = Math.min(boxA.xmax, boxB.xmax);
      const y2 = Math.min(boxA.ymax, boxB.ymax);

      const intersectionWidth = Math.max(0, x2 - x1);
      const intersectionHeight = Math.max(0, y2 - y1);
      const intersectionArea = intersectionWidth * intersectionHeight;

      const areaA = (boxA.xmax - boxA.xmin) * (boxA.ymax - boxA.ymin);
      const areaB = (boxB.xmax - boxB.xmin) * (boxB.ymax - boxB.ymin);
      const unionArea = areaA + areaB - intersectionArea;

      const iou = unionArea > 0 ? intersectionArea / unionArea : 0;
      if (iou > iouThreshold) {
        active[j] = false;
      }
    }
  }
  return kept;
}

// Listen for messages from the main thread
self.addEventListener('message', async (event) => {
  const { type, data } = event.data;

  if (type === 'init') {
    const { modelName, localModelPath } = data;
    try {
      await getDetector(modelName, localModelPath);
    } catch (err) {
      self.postMessage({ type: 'error', data: `Model initialization error: ${err.message}` });
    }
  }

  else if (type === 'detect') {
    const { modelName, pixels, width, height, channels, threshold, localModelPath } = data;
    try {
      const activeModel = await getDetector(modelName, localModelPath);

      self.postMessage({ type: 'status', data: 'processing' });

      if (activeModel.type === 'custom') {
        const startTime = performance.now();

        // Run custom preprocessing
        const inputTensor = preprocessYOLO(pixels, width, height);

        // Run direct ONNX Inference
        const inputs = {};
        inputs[activeModel.session.inputNames[0]] = inputTensor;
        const outputs = await activeModel.session.run(inputs);
        const outputTensor = outputs[activeModel.session.outputNames[0]];

        // Run custom postprocessing with NMS
        const results = postprocessYOLO(
          outputTensor,
          threshold,
          width,
          height,
          activeModel.config.id2label
        );

        const endTime = performance.now();

        self.postMessage({
          type: 'result',
          data: {
            results,
            duration: Math.round(endTime - startTime)
          }
        });
      } else {
        // Standard Transformers.js inference
        const rawImage = new RawImage(pixels, width, height, channels);
        const startTime = performance.now();
        const results = await activeModel.detector(rawImage, { threshold: threshold });
        const endTime = performance.now();

        self.postMessage({
          type: 'result',
          data: {
            results,
            duration: Math.round(endTime - startTime)
          }
        });
      }
    } catch (err) {
      self.postMessage({ type: 'error', data: `Inference failed: ${err.message}` });
    }
  }
});
