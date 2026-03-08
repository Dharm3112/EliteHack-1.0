# EliteHack-1.0: Advanced Off-Road Semantic Segmentation

EliteHack-1.0 is an AI-driven vision system designed to provide high-precision semantic segmentation for unstructured off-road environments. By leveraging deep learning, the system transforms raw visual data into detailed terrain maps, enabling autonomous systems to navigate complex natural landscapes where traditional road rules do not apply.

## 🚧 Project Status: 75% Complete

* **Current State:** The core engine is fully functional and successfully segments off-road terrain using demo test images.
* **The Road Ahead (Remaining 25%):** Development is currently focused on expanding the model's capabilities to detect dynamic objects both living (humans, wildlife) and non-living (cars, ATVs, equipment) to ensure safety in shared environments.

## 🚩 Problem Statement

Autonomous navigation in the wild is significantly more difficult than city driving because:

* **Lack of Structure:** There are no lane markings, traffic signs, or paved boundaries to guide the system.
* **Critical Hazards:** Natural obstacles like fallen logs and sharp rocks are often small but can cause catastrophic damage if not identified.
* **Terrain Ambiguity:** Distinguishing between safe "drivable" surfaces like dry grass and "non-drivable" hazards like thick lush bushes requires pixel-level accuracy.

## 💡 Solution

The project addresses these challenges through a specialized vision and training pipeline:

* **SegFormer Architecture:** Uses the lightweight **nvidia/mit-b0** model, optimized for real-time performance and low-latency inference.
* **Intelligent Hazard Prioritization:** Implements **Focal Loss** and heavy class-weighting (weighting "Logs" 128x higher) to ensure the AI never misses rare but dangerous obstacles.
* **10-Class Segmentation:** Identifies and maps: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, and Sky.
* **Seamless Visualization:** A web-based dashboard provides a raw segmentation mask and a 50% transparent overlay for real-time terrain verification.

## 🛠 Tech Stack

* **AI/ML:** PyTorch, Hugging Face Transformers (SegFormer), Albumentations.
* **Backend:** FastAPI (Python), Uvicorn, Pillow (PIL), NumPy.
* **Frontend:** React 18, TypeScript, Vite, Tailwind CSS.
* **Optimization:** ONNX for high-speed model deployment.

## 🌍 Use Cases & Applications

* **Autonomous Off-Road Vehicles:** Safe navigation for ATVs, scouting units, and rovers in unmapped wild terrain.
* **Search & Rescue (SAR):** Helping rescue robots navigate debris-heavy disaster zones where standard infrastructure has been destroyed.
* **Precision Agriculture:** Navigating automated farming equipment through unpaved fields and monitoring crop/vegetation health.
* **Infrastructure Inspection:** Monitoring power line corridors or remote pipelines for vegetation encroachment.
* **Military Reconnaissance:** Identifying natural cover (trees) and navigating hazardous terrain during scouting missions.
* **Wildlife & Forestry:** Tracking changes in forest density or identifying erosion patterns in remote landscapes.
* **Drone Navigation:** Enabling low-altitude flights to avoid collisions with natural obstacles like rocks or logs.

## 📂 Project Structure

* **`backend/`**: FastAPI application and inference engine.
* **`frontend/`**: React-based dashboard for image analysis.
* **`scripts/`**: Training pipelines, Exploratory Data Analysis (EDA), and ONNX optimization.
* **`utils/`**: Core utilities for dataloading, custom loss functions, and image augmentations.

## 🎨 Terrain Color Map

| Class | Color | Class | Color |
| --- | --- | --- | --- |
| **Background** | Black | **Ground Clutter** | Brown |
| **Trees** | Dark Green | **Logs** | Dark Wood |
| **Lush Bushes** | Light Green | **Rocks** | Gray |
| **Dry Grass** | Khaki | **Landscape** | Sand |
| **Dry Bushes** | Olive | **Sky** | Light Blue |

