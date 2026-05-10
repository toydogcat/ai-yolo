import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SRC_DIR = path.join(__dirname, 'node_modules/onnxruntime-web/dist');
const DST_DIR = path.join(__dirname, 'public/onnx-wasm');

console.log(`🚀 Commencing Node-native ONNX WASM stabilization...`);

// 1. Create dest
if (!fs.existsSync(DST_DIR)) {
    fs.mkdirSync(DST_DIR, { recursive: true });
}

// 2. File list
const patterns = [
    "ort-wasm-simd-threaded.wasm",
    "ort-wasm-simd-threaded.jsep.wasm",
    "ort-wasm-simd-threaded.mjs",
    "ort-wasm-simd-threaded.jsep.mjs"
];

patterns.forEach(file => {
    const srcPath = path.join(SRC_DIR, file);
    if (fs.existsSync(srcPath)) {
        const dstPath = path.join(DST_DIR, file);
        fs.copyFileSync(srcPath, dstPath);
        console.log(` -> Copied: ${file}`);
    }
});

// 3. Patch MJS and rename to JS
const filesInDest = fs.readdirSync(DST_DIR);
filesInDest.forEach(file => {
    if (file.endsWith('.mjs')) {
        const filePath = path.join(DST_DIR, file);
        let content = fs.readFileSync(filePath, 'utf8');
        
        // Neutralize Node blocks
        content = content.replace(/if\s*\(isNode\)\s*isPthread\s*=\s*\(await\s*import\('worker_threads'\)\)\.workerData\s*===\s*'em-pthread';/, "if (false) {}");
        content = content.replace(/await\s*import\("module"\)/g, 'null');
        content = content.replace(/import\("module"\)/g, 'null');
        content = content.replace(/import\('worker_threads'\)/g, 'null');
        
        const finalName = file.replace(".mjs", ".js");
        const finalPath = path.join(DST_DIR, finalName);
        
        fs.writeFileSync(finalPath, content, 'utf8');
        fs.unlinkSync(filePath); // remove the .mjs
        console.log(` -> Processed and renamed to: ${finalName}`);
    }
});

console.log("✨ Native Stabilization Complete. Ready for production build.");
