# AI Emergency Severity Detection System

## 📌 Overview

This project presents a hybrid deep learning framework for emergency severity estimation from visual scenes.

Unlike traditional systems that only detect events (fire, accident, flood), this system:

Detects emergency-related objects

Classifies scene context

Estimates severity level (LOW / MEDIUM / HIGH)

Provides confidence scores

Uses hybrid ML + rule-based reasoning

The system is designed for intelligent emergency prioritization in real-world applications such as CCTV monitoring, traffic management, and disaster response.

## 🧠 System Architecture
```text
Input Image
     ↓
Object Detection (YOLOv8)
     ↓
Scene Classification (ResNet-18)
     ↓
Feature Fusion
     ↓
Severity Estimation (Neural Network + Rule Engine)
     ↓
Severity Level + Confidence Score
```

## 🔍 Modules Used

### 1️⃣ Object Detection — YOLOv8

Detects:

Person

Vehicle

Fire

Smoke

Why YOLOv8?

Real-time capable

High detection accuracy

Lightweight architecture

### 2️⃣ Scene Classification — ResNet-18

Classifies scenes into:

Accident

Fire

Flood

Normal

Features:

Transfer learning (ImageNet pretrained)

Backbone freezing for efficiency

~99% validation accuracy

### 3️⃣ Severity Estimation Model

Input feature vector (8 dimensions):

[people_count,
 vehicle_count,
 fire_presence,
 smoke_presence,
 scene_accident,
 scene_fire,
 scene_flood,
 scene_normal]

Outputs:

LOW

MEDIUM

HIGH

Confidence score

### 4️⃣ Hybrid Rule-Based Override

Domain rules improve reliability:

Examples:

Flood + People → HIGH

Accident + Vehicle → HIGH

Smoke → HIGH

Normal + No objects → LOW

This ensures safety in edge cases.

## 📊 Performance
| Metric | Result |
| :--- | :--- |
| **Scene Classification Accuracy** | ~99% |
| **Object Detection Speed** | Real-time Capable |
| **Inference Latency** | Optimized for Edge Deployment |

## 📁 Project Structure
```text
emergency_ai/
├── detection/        # YOLO training configs and weights
├── scene/            # ResNet-18 classification models
├── severity/         # Neural Network severity logic
├── inference/        # Unified prediction pipeline
├── utils/            # Image processing & logging
├── requirements.txt  # Dependencies
└── README.md
```

## 📂 Dataset

Due to GitHub storage limits, datasets are not included.

Scene Dataset Structure
```text
datasets/

└── scene/
    ├── train/
    │   ├── accident/
    │   ├── fire/
    │   ├── flood/
    │   └── normal/
    └── val/
        ├── accident/
        ├── fire/
        ├── flood/
        └── normal/
```
Detection dataset follows YOLO format.

Dataset available upon request.


## ⚙️ Installation

Clone repository:

git clone https://github.com/Yeshwanth-develops/Emergify.ai.git

cd emergency-ai

Install dependencies:

pip install -r requirements.txt

## ▶️ Running Inference

python inference/predict.py path_to_image.jpg

Example:

python inference/predict.py test4.png

Output:

Scene Prediction: accident (Confidence: 0.98)

Objects: {person:1, smoke:1}

Decision Path: Rule-based override

Final Severity: HIGH (Confidence: 0.95)

## 🚀 Future Improvements

Multi-label scene classification

Video-based temporal severity estimation

Attention-based feature fusion

Explainable AI (Grad-CAM visualization)

Cloud deployment for real-time alerts
