import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

let handLandmarker = undefined;
let runningMode = "VIDEO";
let enableWebcamButton;
let webcamRunning = false;

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
const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingOverlay = document.getElementById("loadingOverlay");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");

// 1. Initialize Hand Landmarker
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
      numHands: 2, // Track 2 hands max for real-time speed balance
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    
    loadingOverlay.style.opacity = "0";
    setTimeout(() => {
      loadingOverlay.style.display = "none";
    }, 500);
    
    enableWebcamButton.disabled = false;
    statusText.innerText = "Ready";
    statusDot.classList.add("active");
  } catch (error) {
    console.error("Error creating hand landmarker:", error);
    loadingOverlay.innerText = "Failed to load models. Check console.";
  }
}

// 2. Camera Control
enableWebcamButton = document.getElementById("webcamButton");
enableWebcamButton.addEventListener("click", toggleCam);

function toggleCam() {
  if (!handLandmarker) return;

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "Start Camera Stream";
    statusText.innerText = "Stopped";
    
    const stream = video.srcObject;
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
    }
    video.srcObject = null;
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "Stop Camera Stream";
    statusText.innerText = "Connecting...";
    
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: "user"
      }
    };

    navigator.mediaDevices.getUserMedia(constraints)
      .then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
          statusText.innerText = "Tracking active";
          predictWebcam();
        });
      })
      .catch((err) => {
        console.error("Camera access error: ", err);
        statusText.innerText = "Camera error";
        webcamRunning = false;
        enableWebcamButton.innerText = "Start Camera Stream";
      });
  }
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
    gradient.addColorStop(0, "#10b981");
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

// 4. Main Inference Loop
let lastVideoTime = -1;
let results = undefined;

async function predictWebcam() {
  if (!webcamRunning) return;

  // Auto match canvas viewport to video actual resolution
  if (canvasElement.width !== video.videoWidth) {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
  }
  
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  let startTimeMs = performance.now();
  
  // Detect only if video playback progress has advanced
  if (lastVideoTime !== video.currentTime && video.readyState >= 2) {
    lastVideoTime = video.currentTime;
    results = handLandmarker.detectForVideo(video, startTimeMs);
  }
  
  if (results && results.landmarks) {
    for (const landmarks of results.landmarks) {
      drawLandmarks(landmarks);
    }
  }
  
  window.requestAnimationFrame(predictWebcam);
}

// Init
window.addEventListener("DOMContentLoaded", createHandLandmarker);
