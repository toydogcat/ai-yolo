import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

let handLandmarker = undefined;
const runningMode = "IMAGE";

// MediaPipe skeleton connections
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // Index
  [5, 9], [9, 10], [10, 11], [11, 12], // Middle
  [9, 13], [13, 14], [14, 15], [15, 16], // Ring
  [13, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [0, 17] // Palm baseline
];

// HTML Elements
const imageElement = document.getElementById("uploadedImage");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingOverlay = document.getElementById("loadingOverlay");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const fileInput = document.getElementById("fileInput");
const selectBtn = document.getElementById("selectBtn");
const uploadPrompt = document.getElementById("uploadPrompt");
const dropZone = document.getElementById("dropZone");

// 1. Initialize Hand Landmarker in IMAGE mode
async function createHandLandmarker() {
  try {
    const base = window.location.origin + '/ai-yolo';
    const vision = await FilesetResolver.forVisionTasks(
      `${base}/wasm` 
    );
    
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: `${base}/models/hand_landmarker.task`,
        delegate: "GPU"
      },
      runningMode: runningMode,
      numHands: 4, // Detect up to 4 hands for static photos
      minHandDetectionConfidence: 0.3, // Lower slightly for static images
    });
    
    loadingOverlay.style.opacity = "0";
    setTimeout(() => {
      loadingOverlay.style.display = "none";
    }, 500);
    
    selectBtn.disabled = false;
    statusText.innerText = "Ready";
    statusDot.classList.add("active");
  } catch (error) {
    console.error("Error creating hand landmarker:", error);
    loadingOverlay.innerText = "Failed to load model. Check DevTools.";
  }
}

// 2. Event Bindings
uploadPrompt.addEventListener("click", () => fileInput.click());
selectBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", handleFileSelect);

// Drag & drop
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "#6366f1";
});
dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "rgba(255,255,255,0.05)";
});
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "rgba(255,255,255,0.05)";
  if (e.dataTransfer.files.length > 0) {
    fileInput.files = e.dataTransfer.files;
    handleFileSelect();
  }
});

function handleFileSelect() {
  const file = fileInput.files[0];
  if (!file || !file.type.startsWith("image/")) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    imageElement.src = e.target.result;
    imageElement.style.display = "block";
    uploadPrompt.style.display = "none";
    
    imageElement.onload = () => {
      runDetection();
    };
  };
  reader.readAsDataURL(file);
}

// 3. Drawing Logic
function drawLandmarks(landmarks) {
  canvasCtx.lineWidth = 3;
  
  // Draw Connections
  for (const connection of HAND_CONNECTIONS) {
    const start = landmarks[connection[0]];
    const end = landmarks[connection[1]];
    
    canvasCtx.beginPath();
    canvasCtx.moveTo(start.x * canvasElement.width, start.y * canvasElement.height);
    canvasCtx.lineTo(end.x * canvasElement.width, end.y * canvasElement.height);
    
    const gradient = canvasCtx.createLinearGradient(
      start.x * canvasElement.width, start.y * canvasElement.height,
      end.x * canvasElement.width, end.y * canvasElement.height
    );
    gradient.addColorStop(0, "#06b6d4");
    gradient.addColorStop(1, "#6366f1");
    
    canvasCtx.strokeStyle = gradient;
    canvasCtx.stroke();
  }

  // Draw Joints
  for (const landmark of landmarks) {
    canvasCtx.beginPath();
    canvasCtx.arc(landmark.x * canvasElement.width, landmark.y * canvasElement.height, 5, 0, 2 * Math.PI);
    canvasCtx.fillStyle = "#ffffff";
    canvasCtx.fill();
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = "#6366f1";
    canvasCtx.stroke();
  }
}

// 4. Image Detection
async function runDetection() {
  if (!handLandmarker) return;
  
  statusText.innerText = "Detecting...";

  // Setup canvas sizing to match actual rendered image size
  // We set canvas attribute size to display size
  canvasElement.width = imageElement.clientWidth;
  canvasElement.height = imageElement.clientHeight;
  
  // Keep internal dimensions separate if we want, but let's just map it.
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  // Perform synchronous-feeling detection on single image
  const result = handLandmarker.detect(imageElement);
  
  if (result.landmarks && result.landmarks.length > 0) {
    for (const landmarks of result.landmarks) {
      drawLandmarks(landmarks);
    }
    statusText.innerText = `Found ${result.landmarks.length} Hand(s)`;
  } else {
    statusText.innerText = "No hands found.";
  }
}

// Adjust canvas overlay on resize in case container layout shifts
window.addEventListener("resize", () => {
  if (imageElement.style.display !== "none" && imageElement.complete) {
    // redraw
    canvasElement.width = imageElement.clientWidth;
    canvasElement.height = imageElement.clientHeight;
    runDetection(); 
  }
});

// Initialize
window.addEventListener("DOMContentLoaded", createHandLandmarker);
