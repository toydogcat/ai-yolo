import * as ort from 'onnxruntime-web';
import { PaddleOcrService } from 'paddleocr';

// Define explicit absolute public paths
const publicBase = '/ai-yolo/onnx-wasm/';
const loaderUrl = publicBase + 'ort-wasm-simd-threaded.js';

// Fetch the script text directly via static HTTP to entirely bypass Vite's 'import()' interceptor
console.log("[Worker] Jailbreaking loader from Vite via Blob Proxy...");
const response = await fetch(loaderUrl);
const scriptText = await response.text();

// Re-hydrate the script into a fully-autonomous memory module which Vite cannot monitor or audit
const blob = new Blob([scriptText], { type: 'application/javascript' });
const blobUrl = URL.createObjectURL(blob);
console.log("[Worker] Secured standalone execution handle:", blobUrl);

// According to reverse engineering of onnxruntime-web sources, 
// 'wasmPaths' supports specific 'mjs' and 'wasm' keys.
ort.env.wasm.wasmPaths = {
  // Point loading script to our dynamic autonomous BLOB
  mjs: blobUrl,
  // Keep absolute binary destination pointed firmly to public path
  wasm: publicBase + 'ort-wasm-simd-threaded.wasm'
};

let ocrService = null;

self.onmessage = async (event) => {
  const { type, data } = event.data;

  if (type === 'init') {
    const { detBuffer, recBuffer, dictContent } = data;
    try {
      self.postMessage({ type: 'status', data: 'Loading OCR Models...' });
      
      // Use available concurrency now that Server headers are configured
      ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

      // Create array of lines from dictionary string
      let dict = dictContent.split(/\r?\n/).map(l => l.trim());
      
      // ABSOLUTE CRITICAL FIX: PaddleOCR output vectors are 1-based for the dataset matrix.
      // By prepending '<blank>' to element 0, we perfectly phase-shift the remaining 6,600+ characters
      // into their definitive alignment, restoring lossless phonetic extraction.
      dict = ['', ...dict, ' ']; 

      ocrService = await PaddleOcrService.createInstance({
        ort,
        detection: {
          modelBuffer: detBuffer,
          minimumAreaThreshold: 24,
          textPixelThreshold: 0.6,
        },
        recognition: {
          modelBuffer: recBuffer,
          charactersDictionary: dict,
        },
      });

      self.postMessage({ type: 'initialized', data: 'success' });
    } catch (err) {
      self.postMessage({ type: 'error', data: `Initialization failed: ${err.message}` });
    }
  } 

  else if (type === 'recognize') {
    if (!ocrService) {
      self.postMessage({ type: 'error', data: 'Service not initialized' });
      return;
    }

    const { imageData, width, height } = data;
    try {
      self.postMessage({ type: 'status', data: 'Extracting Text...' });
      const startTime = performance.now();

      // Run the PaddleOCR recognition pipeline
      const input = {
        data: imageData,
        width: width,
        height: height,
      };

      const results = await ocrService.recognize(input, {
        onProgress(e) {
           if(e.type === 'det') {
             self.postMessage({ type: 'status', data: `Detecting Text (${e.stage})` });
           } else if(e.type === 'rec') {
             if(e.stage === 'item') {
                // We can stream partial progress back if desired
             } else if (e.stage === 'start') {
                self.postMessage({ type: 'status', data: `Recognizing items...` });
             }
           }
        }
      });
      
      const endTime = performance.now();

      self.postMessage({
        type: 'result',
        data: {
          results,
          duration: Math.round(endTime - startTime)
        }
      });
      
      self.postMessage({ type: 'status', data: 'Idle' });
    } catch (err) {
      self.postMessage({ type: 'error', data: `Inference failed: ${err.message}` });
    }
  }
};
