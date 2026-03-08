# EliteHack-1.0: Off-Road Semantic Segmentation

EliteHack-1.0 is an AI-powered project designed to provide real-time semantic segmentation for unstructured off-road environments. Using advanced deep learning, it enables machines to understand complex natural terrains by identifying specific elements like rocks, logs, and vegetation.

## 🚩 Problem Statement

Autonomous navigation in off-road environments presents unique challenges that standard self-driving systems (designed for city streets) cannot handle:

* **Lack of Structure:** There are no lane lines, traffic signs, or paved roads to follow.
* **Critical Obstacles:** Rare but dangerous hazards like fallen logs and sharp rocks are often overlooked by general AI models.
* **Visual Ambiguity:** Distinguishing between "drivable" dry grass and "non-drivable" thick bushes requires high-precision pixel-level understanding.

## 💡 Solution

The project implements a specialized vision system to solve these challenges:

* **SegFormer Architecture:** Utilizes the lightweight **nvidia/mit-b0** model, optimized for high-speed inference (under 50ms) without sacrificing accuracy.
* **Hard-Class Prioritization:** Employs **Focal Loss** and extreme class weighting (e.g., weighting "Logs" 128x higher) to ensure the AI detects rare, critical hazards.
* **10-Class Terrain Mapping:** Segments images into 10 distinct categories: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, and Sky.
* **Interactive Visualization:** A web-based interface that provides both raw segmentation masks and 50% transparent overlays for immediate terrain assessment.

## 🚀 Key Features

* **Real-time API:** High-performance backend powered by FastAPI for quick image processing.
* **Visual Insights:** Returns detected class names, IDs, and their corresponding colors alongside processed images.
* **Advanced Training:** Features a robust training pipeline with Cosine Annealing learning rate schedules and automated model checkpointing.
* **Modern Frontend:** A responsive user interface built with React, TypeScript, and Tailwind CSS.

## 🛠 Tech Stack

* **AI/ML:** PyTorch, Hugging Face Transformers (SegFormer), Albumentations.
* **Backend:** FastAPI, Uvicorn, Pillow (PIL), NumPy.
* **Frontend:** React 18, TypeScript, Vite, Tailwind CSS.
* **Optimization:** ONNX for model deployment and speed.

## 📂 Project Structure

```text
EliteHack-1.0/
├── backend/            # FastAPI server and inference logic
├── frontend/           # React + Vite web application
├── scripts/            # Training, EDA, and model optimization scripts
├── utils/              # Dataloaders, loss functions, and augmentations
├── models/             # Saved model checkpoints and optimized ONNX files
└── Dataset/            # Scripts for dataset management and processing

```

## 🌍 Use Cases

* **Autonomous Off-Road Vehicles:** Safe navigation for ATVs and scouting units in wild terrain.
* **Search & Rescue Robots:** Identifying clear paths through debris and "ground clutter" in disaster zones.
* **Precision Agriculture:** Monitoring forest density and navigating unpaved fields for automated farming.
* **Drone Navigation:** Avoiding low-level obstacles like trees and rocks during low-altitude flights.

## ⚙️ Setup and Installation

### Backend Setup

1. Navigate to the `backend` directory.
2. Install requirements: `pip install -r ../requirements.txt`.
3. Start the API server: `uvicorn app:app --reload`.

### Frontend Setup

1. Navigate to the `frontend` directory.
2. Install dependencies: `npm install`.
3. Start the development server: `npm run dev`.

## 🎨 Color Map Reference

| Class | Color | Class | Color |
| --- | --- | --- | --- |
| **Background** | Black | **Ground Clutter** | Brown |
| **Trees** | Dark Green | **Logs** | Dark Wood |
| **Lush Bushes** | Light Green | **Rocks** | Gray |
| **Dry Grass** | Khaki | **Landscape** | Sand |
| **Dry Bushes** | Olive | **Sky** | Light Blue |
| *(Source:)* |  |  |  |
