Stone Analysis System (3D + AI + GPU Acceleration)

🚀 Overview

Stone Analysis System v9.3 is a high-performance AI-powered 3D stone matching and analysis engine designed for precision geometric matching, mesh reconstruction, and real-time piece detection.

This system combines:

⚡ GPU Acceleration (CUDA, NVENC, NVFBC)
🤖 Deep Learning (YOLOv8, ResNet18)
🧊 3D Geometry (Point Clouds, Mesh Analysis)
🖥️ High-FPS Capture (DXCAM / NVFBC)
🔍 Hybrid Matching Algorithms (7 advanced methods)

👉 Built for industrial-grade stone analysis workflows like Sarine Advisor systems.

🧩 Key Features
⚡ High-Speed Capture Pipeline
DXCAM → 120 FPS
NVFBC → 144 FPS (zero-copy GPU capture)
MSS fallback → universal support

✔ Frame stability detection
✔ Rotation-synced capture
✔ GPU memory processing

🤖 AI & Deep Learning
YOLOv8 → mesh + object detection
ResNet18 → feature extraction
Automatic piece detection:
🟢 Green
🔵 Blue
🔴 Red

✔ Deep feature comparison
✔ Mesh topology understanding
✔ Texture + boundary analysis


3D Analysis Workflow
STEP 2: A-Stone (Half Stone)
360° rotation (X, Y, Z)
Point cloud + contour extraction
Geometric fingerprint generation
STEP 3: B-Stone (Full Stone)
Full shape capture
Mesh feature extraction
Deep learning analysis
🧠 Matching Engine (7 Methods)
GPU Fingerprint Matching
Contour Matching (Hu Moments)
ICP Alignment (3D registration)
Rotation Matching
YOLO Mesh Matching
Enhanced Piece Matching
OpenGL Visualization

Includes advanced techniques:

Shape Context
Curvature Analysis
Fourier Descriptors
Break-edge detection
Mesh topology comparison
Boundary texture matching

👉 Designed for high-accuracy broken stone matching

🎯 Smart Piece Detection
🎨 Color detection (HSV + fallback)
📍 Position-aware targeting
🔎 OCR label detection (ct / %)
🟡 Highlight detection (active piece)
🧠 Contour-based matching

✔ Auto-select + auto-click system
✔ Real-time matching logic

⚙️ Configuration

Fully customizable via config.py:

Window targeting
Detection thresholds
Rotation cycles
Speed tuning
Safety settings
🖥️ Visualization

OpenGL 3D viewer
Interactive rotation
Real-time feedback
📦 Installation
pip install pyautogui opencv-python numpy pywin32 pillow mss scipy torch PyOpenGL PyOpenGL_accelerate glfw pyrr ultralytics torchvision --break-system-packages
CUDA (Recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
YOLO
pip install ultralytics
▶️ Usage
python 3d.py
🧪 Performance
⚡ 120–144 FPS capture
🧠 +10–25% accuracy boost (confidence boosting)
🎯 Multi-algorithm fusion scoring
🔬 High precision matching


💡 Use Cases
💎 Diamond / gemstone analysis
🧩 Fragment reconstruction
🏭 Industrial inspection
🤖 AI-based CAD analysis
🔍 Reverse engineering
⚠️ Requirements
NVIDIA GPU (recommended)
Windows OS
Python 3.9+
📌 Notes
Automatic fallback if GPU unavailable
Hybrid CPU + GPU pipeline
Designed for precision workflows
⭐ Future Improvements
Cloud-based processing
Batch stone analysis
Better YOLO training datasets
Neural mesh matching
