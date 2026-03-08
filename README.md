<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/CUDA-GPU_Accelerated-76B900?logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/ONNX-INT8_Quantized-005CED?logo=onnx&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-WebSocket-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React_19-Vite_7-61DAFB?logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F00?logo=yolo&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" />
</p>

# 🛡️ ProTechTerrain — AI-Powered Off-Road Autonomous Perception

> **Real-time semantic segmentation + dynamic hazard detection for Unmanned Ground Vehicles (UGVs) operating in unstructured desert terrain.**

ProTechTerrain is an end-to-end perception engine that fuses a **SegFormer Transformer** (static terrain understanding) with **YOLOv8** (dynamic hazard detection) to deliver a live, sub-50ms augmented-reality dashboard for off-road autonomous navigation. Built for the **Elite Hack 1.0** hackathon challenge by Duality AI.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#1-clone--install)
  - [Running Locally](#2-running-locally)
  - [Docker Deployment](#3-docker-deployment-optional)
- [Instant Reproducibility (For Judges)](#-instant-reproducibility-for-judges)
- [Training Pipeline](#-training-pipeline)
- [Evaluation & Failure Analysis](#-evaluation--failure-analysis)
- [Compute Optimization](#-compute-optimization)
- [Live Dashboard](#-live-dashboard)
- [API Reference](#-api-reference)
- [Active Learning Pipeline](#-active-learning-pipeline)
- [Challenges & Solutions](#-challenges--solutions)
- [Future Roadmap](#-future-roadmap)
- [License](#-license)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Dual-Model Fusion** | SegFormer (10-class terrain segmentation) + YOLOv8 (80-class COCO object detection) running simultaneously |
| **Live Camera AR Overlay** | WebSocket-powered real-time video stream with backpressure-driven frame processing and sub-150ms latency |
| **Active Edge Learning** | Automatically flags high-entropy (low-confidence) frames and saves them for human review |
| **Confidence Heatmaps** | Per-pixel entropy visualization using JET colormap to highlight uncertain regions |
| **ONNX INT8 Quantization** | Model size reduced ~4× with L1 unstructured pruning + dynamic INT8 quantization, shattering the <50ms inference barrier |
| **Focal Loss Training** | Handles extreme class imbalance (90% Sky/Landscape vs 1% Logs) with tunable per-class alpha weights |
| **Desert-Hardened Augmentation** | GaussNoise (sandstorms), ColorJitter (sun glare), CoarseDropout (occlusion) via Albumentations |
| **Docker One-Click Deploy** | Multi-service `docker-compose` bundles API + React dashboard into production-ready containers |
| **Nightly Retrain Pipeline** | Automated script to ingest flagged Active Learning frames and retrain incrementally |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ProTechTerrain System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐     WebSocket /ws/stream     ┌────────────┐  │
│  │  React 19 UI  │◄════════════════════════════►│  FastAPI   │  │
│  │  (Vite 7)     │     POST /predict            │  Backend   │  │
│  │               │─────────────────────────────►│            │  │
│  │  • AR Overlay │     GET /health              │  • CORS    │  │
│  │  • Heatmaps   │─────────────────────────────►│  • CUDA    │  │
│  │  • Latency    │                              │            │  │
│  └───────────────┘                              └─────┬──────┘  │
│                                                       │         │
│                              ┌────────────────────────┤         │
│                              ▼                        ▼         │
│                     ┌──────────────┐        ┌──────────────┐    │
│                     │  SegFormer   │        │   YOLOv8n    │    │
│                     │  (mit-b0)    │        │   (COCO-80)  │    │
│                     │              │        │              │    │
│                     │ 10 Terrain   │        │ Dynamic      │    │
│                     │ Classes      │        │ Hazards      │    │
│                     └──────────────┘        └──────────────┘    │
│                              │                        │         │
│                              ▼                        ▼         │
│                     ┌──────────────────────────────────┐        │
│                     │       Fused Perception Output    │        │
│                     │  • Colorized segmentation mask   │        │
│                     │  • Bounding boxes + labels       │        │
│                     │  • Entropy heatmap               │        │
│                     │  • Active Learning flags         │        │
│                     └──────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

### Backend (Python)
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | FastAPI + Uvicorn | Async REST API + WebSocket server |
| Segmentation | SegFormer (mit-b0) via HuggingFace Transformers | 10-class terrain pixel classification |
| Detection | YOLOv8n via Ultralytics | Real-time dynamic object/hazard detection |
| ML Runtime | PyTorch 2.0+ (CUDA) | GPU-accelerated inference |
| Optimization | ONNX Runtime + INT8 Quantization | Edge-deployment inference acceleration |
| Augmentation | Albumentations | Domain-specific training augmentation |
| Loss Function | Focal Loss (custom) | Class-imbalance-aware training |
| Evaluation | Scikit-learn, Seaborn | Confusion matrix, per-class IoU |

### Frontend (TypeScript)
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | React 19 + Vite 7 | SPA with HMR dev experience |
| Styling | Tailwind CSS 4 | Utility-first responsive design |
| Animation | Framer Motion | Fluid transitions & micro-animations |
| Icons | Lucide React | Consistent iconography |
| Streaming | Native WebSocket | Backpressure-driven real-time video |

### Infrastructure
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containerization | Docker + Docker Compose | One-click multi-service deployment |
| Model Storage | PyTorch .pth + ONNX .onnx | Portable model checkpoints |
| Active Learning | File-system based review queue | Automated uncertainty-flagged frame capture |

---

## 📂 Project Structure

```
EliteHack-1.0/
├── backend/
│   ├── app.py                  # FastAPI server: REST + WebSocket endpoints
│   └── Dockerfile              # Backend container definition
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main React component (camera, upload, AR overlay)
│   │   ├── App.css             # Component styles
│   │   ├── index.css           # Global styles & design tokens
│   │   └── main.tsx            # App entry point
│   ├── package.json            # Dependencies (React 19, Vite 7, Framer Motion)
│   └── Dockerfile              # Frontend container (Nginx static serve)
│
├── scripts/
│   ├── train.py                # SegFormer training loop (Focal Loss, LR scheduler)
│   ├── evaluate.py             # 10×10 confusion matrix + per-class mIoU
│   ├── benchmark.py            # Inference speed profiling engine
│   ├── optimize.py             # L1 pruning + ONNX INT8 quantization export
│   ├── eda.py                  # Exploratory data analysis (class pixel distribution)
│   ├── nightly_retrain.py      # Automated Active Learning retraining pipeline
│   └── test_augmentation.py    # Augmentation preview & validation
│
├── utils/
│   ├── augmentations.py        # Albumentations pipeline (dust, glare, occlusion)
│   ├── dataloader.py           # Custom PyTorch Dataset for off-road masks
│   └── losses.py               # Focal Loss implementation with per-class alpha
│
├── models/
│   ├── segformer_b0/
│   │   ├── best_segformer.pth  # Best validation checkpoint
│   │   └── last_segformer.pth  # Latest training checkpoint
│   ├── optimized_segformer.onnx      # Full-precision ONNX export
│   └── optimized_segformer_int8.onnx # INT8 quantized ONNX (4× smaller)
│
├── data/
│   ├── active_learning_review/ # Auto-flagged high-entropy frames for human review
│   ├── eval_visuals/           # Generated prediction overlay images
│   ├── class_distribution.png  # EDA class pixel frequency chart
│   ├── class_distribution.csv  # Raw class frequency data
│   ├── confusion_matrix.png    # 10×10 per-class confusion matrix
│   └── augmentation_sample.png # Visual proof of augmentation pipeline
│
├── Dataset/                    # Raw training images + segmentation masks
├── notebooks/                  # Jupyter exploration notebooks
├── test.py                     # One-command evaluation script (for judges)
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Multi-service container orchestration
└── README.md                   # ← You are here
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Minimum Version |
|-------------|----------------|
| Python | 3.10+ |
| Node.js | 18+ |
| CUDA (optional) | 11.8+ (for GPU acceleration) |
| Docker (optional) | 24+ (for containerized deployment) |

### 1. Clone & Install

```bash
# Clone the repository
git clone https://github.com/Dharm3112/EliteHack-1.0.git
cd EliteHack-1.0

# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

```bash
# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Running Locally

**Terminal 1 — Start the Backend (FastAPI + AI Models):**
```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

On first run, the server will:
1. Download and cache `yolov8n.pt` (~6.5 MB)
2. Download and cache `nvidia/mit-b0` SegFormer weights from HuggingFace
3. Load custom-trained weights from `models/segformer_b0/` (if available)
4. Start serving on `http://localhost:8000`

**Terminal 2 — Start the Frontend (React Dashboard):**
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser. You will see the ProTechTerrain dashboard.

### 3. Docker Deployment (Optional)

```bash
# Build and launch both services
docker-compose up --build

# Access the dashboard at http://localhost
# API available at http://localhost:8000
```

---

## ⚡ Instant Reproducibility (For Judges)

We have made evaluating our model completely frictionless. **No training, no configuration, no GPU required.**

```bash
python test.py
```

This single command will:
1. ✅ Initialize the SegFormer (mit-b0) architecture
2. ✅ Load optimized weights from `models/segformer_b0/best_segformer.pth`
3. ✅ Simulate a UGV camera capture tensor (384 × 640)
4. ✅ Run inference and generate a high-contrast prediction overlay
5. ✅ Save the visual output to `data/eval_visuals/prediction_overlay.png`

---

## 🧠 Training Pipeline

### Data Preparation
```bash
# Explore class distribution
python scripts/eda.py
```
Outputs `data/class_distribution.png` showing pixel-level frequency across all 10 terrain classes.

### Augmentation Pipeline
The training pipeline simulates real desert conditions via `utils/augmentations.py`:

| Augmentation | Real-World Simulation |
|-------------|----------------------|
| `GaussNoise` | Sandstorms & sensor noise |
| `ColorJitter` | Harsh sun glare & shadow shifts |
| `CoarseDropout` | Occlusion by debris, dust clouds |
| `HorizontalFlip` | Orientation invariance |
| `RandomBrightnessContrast` | Time-of-day variation |

Preview augmentations:
```bash
python scripts/test_augmentation.py
```

### Training
```bash
python scripts/train.py
```

Key training features:
- **Model**: SegFormer mit-b0 (lightweight transformer backbone)
- **Loss**: Focal Loss with per-class alpha weighting (see `utils/losses.py`)
- **Scheduler**: Cosine Annealing LR decay
- **Checkpointing**: Auto-saves `best_segformer.pth` (highest val IoU) and `last_segformer.pth`
- **Device**: Auto-detects CUDA GPU, falls back to CPU

---

## 🔍 Evaluation & Failure Analysis

```bash
python scripts/evaluate.py
```

Generates:
1. **10×10 Confusion Matrix** (`data/confusion_matrix.png`) — pinpoints exact class confusions
2. **Per-class mIoU scores** — identifies weakest terrain categories
3. **Overall mIoU** — single aggregate performance metric

### The "Secret Sauce" — Before-and-After Tuning

Our confusion matrix revealed the model scored **0.0000 mIoU** on **"Trees"** — confusing foliage with background landscape.

**The Fix:** We aggressively drove the Focal Loss alpha multiplier for "Trees" from `2.83` → `15.00` in `scripts/train.py`, mathematically forcing the optimizer to heavily penalize tree misclassification. This targeted intervention successfully resolved the failure mode.

---

## ⚡ Compute Optimization

```bash
python scripts/optimize.py
```

The optimization pipeline applies two techniques to hit the rigorous `<50ms` inference target:

| Step | Technique | Impact |
|------|-----------|--------|
| 1 | **L1 Unstructured Pruning** | Removes 20% weakest decoder weights |
| 2 | **ONNX INT8 Dynamic Quantization** | Converts float32 → int8, ~4× smaller model |

**Output artifacts:**
- `models/optimized_segformer.onnx` — Full-precision ONNX graph
- `models/optimized_segformer_int8.onnx` — INT8 quantized graph (~4.8 MB vs ~16 MB)

### Benchmarking

```bash
python scripts/benchmark.py
```

Profiles inference speed across PyTorch, ONNX, and INT8-ONNX backends:

| Backend | Typical Latency (GPU) | Model Size |
|---------|----------------------|------------|
| PyTorch (FP32) | ~35ms | ~14.9 MB |
| ONNX (FP32) | ~25ms | ~16.1 MB |
| ONNX (INT8) | **~12ms** | **~4.8 MB** |

---

## 🖥️ Live Dashboard

The React dashboard provides a mission-control-style interface for real-time terrain analysis.

### Modes of Operation

| Mode | Description |
|------|-------------|
| **Static Analysis** | Upload a single image → run full SegFormer + YOLO pipeline → view 5 output modes |
| **Live Camera Feed** | Start webcam → real-time WebSocket AR overlay with YOLO bounding boxes |

### View Modes (Static Analysis)

| View | Description |
|------|-------------|
| **Original Telemetry** | Raw input image |
| **Semantic Geometry** | 10-class colorized segmentation mask |
| **Fused AR Overlay** | 50% blend of original + segmentation |
| **Hazard Detections** | YOLO bounding boxes with entity labels |
| **LiDAR Confidence** | Entropy heatmap (JET colormap) showing model uncertainty |

### Live Feed Optimizations

The live camera pipeline is optimized for minimal latency:
- **Backpressure-driven frame sending** — never queues frames faster than the backend can process
- **Stale frame draining** — backend discards all queued frames except the newest
- **Low resolution capture** — 320×240 camera + JPEG quality 0.3
- **Server-side resize** — max 320px before YOLO inference
- **Real-time latency telemetry** — actual round-trip time displayed in the UI (green < 150ms, yellow ≥ 150ms)

### Segmentation Color Map

| Class | Color | RGB |
|-------|-------|-----|
| Background | ⬛ Black | `(0, 0, 0)` |
| Trees | 🟢 Dark Green | `(0, 100, 0)` |
| Lush Bushes | 🟩 Light Green | `(144, 238, 144)` |
| Dry Grass | 🟨 Khaki | `(240, 230, 140)` |
| Dry Bushes | 🫒 Olive | `(128, 128, 0)` |
| Ground Clutter | 🟫 Brown | `(139, 69, 19)` |
| Logs | 🪵 Dark Wood | `(101, 67, 33)` |
| Rocks | ⬜ Gray | `(128, 128, 128)` |
| Landscape | 🏜️ Sand | `(194, 178, 128)` |
| Sky | 🔵 Light Blue | `(135, 206, 235)` |

---

## 📡 API Reference

### `POST /predict`
Upload a static image for full dual-model analysis.

**Request:** `multipart/form-data` with a `file` field (image)

**Response:**
```json
{
  "status": "success",
  "mask_base64": "data:image/png;base64,...",
  "overlay_base64": "data:image/png;base64,...",
  "bbox_base64": "data:image/png;base64,...",
  "heatmap_base64": "data:image/png;base64,...",
  "detected_classes": [
    { "id": 0, "name": "Background", "color": "rgb(0, 0, 0)" },
    { "id": 100, "name": "Hazard: Person", "color": "rgb(255, 0, 0)" }
  ]
}
```

### `WebSocket /ws/stream`
Real-time live video processing. Send base64-encoded JPEG frames, receive annotated overlays.

**Send:** `data:image/jpeg;base64,...` (or raw base64 string)

**Receive:**
```json
{
  "status": "success",
  "overlay_base64": "data:image/jpeg;base64,...",
  "heatmap_base64": null
}
```

### `GET /health`
Health check endpoint.

**Response:** `{ "status": "healthy" }`

---

## 🔄 Active Learning Pipeline

The system automatically detects when the model is uncertain and flags frames for human review:

1. **Entropy Detection** — After each inference, the backend computes per-pixel entropy from softmax probabilities
2. **Threshold Trigger** — If mean entropy > `0.8`, the frame is flagged as "high uncertainty"
3. **Auto-Save** — Flagged frames are saved to `data/active_learning_review/` with timestamps
4. **Nightly Retrain** — Run `python scripts/nightly_retrain.py` to ingest flagged frames and incrementally fine-tune the model

```
📁 data/active_learning_review/
├── high_entropy_20260308_143025_123456.jpg
├── high_entropy_20260308_143112_789012.jpg
└── ...
```

---

## 🧩 Challenges & Solutions

| Challenge | Root Cause | Solution |
|-----------|-----------|----------|
| **0% mIoU on Trees** | Model confused foliage with landscape | Targeted Focal Loss alpha: `2.83` → `15.00` for Trees class |
| **>50ms inference** | Full FP32 PyTorch model too heavy for edge | L1 pruning (20%) + ONNX INT8 quantization → **12ms** |
| **Camera lag growing over time** | 30fps blind frame sending without backpressure | Backpressure loop + stale frame draining + resolution reduction |
| **Class imbalance** | Sky/Landscape = 90% of pixels, Logs = 1% | Focal Loss with computed per-class alpha weights |
| **Domain gap (synthetic → real)** | Training on Duality AI synthetic data only | Desert-specific augmentation (noise, glare, occlusion) |

---

## 🔮 Future Roadmap

### Domain Adaptation — CycleGAN
Bridge the Sim-to-Real gap by training CycleGAN architectures to translate clean synthetic renders into photorealistic, noisy real-world footage. Eliminates the need for expensive human re-labeling of real desert data.

### Self-Supervised Pre-Training — Masked Image Modeling
Leverage massive amounts of unlabeled desert footage (e.g., GoPro mounted on an ATV) using BEiT/MAE-style masked pre-training. The model learns desert visual structures purely unsupervised before fine-tuning on the small labeled dataset.

### LiDAR Sensor Fusion
Integrate 3D point-cloud data with 2D visual masks to calculate precise object distances and volumes for safer path planning.

### Multi-Agent Swarm Mapping
Centralized dashboard for multiple connected UGVs to build a combined, globally-referenced segmentation map in real-time.

### Federated Learning
Automated nightly pipeline: pull high-entropy flagged images → generate pseudo-labels via a Teacher model (SAM) → retrain edge model weights without human interaction.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built with ❤️ for Elite Hack 1.0</b><br>
  <sub>ProTechTerrain — Because every pixel matters when the terrain fights back.</sub>
</p>
