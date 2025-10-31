Workout-Performance-Analysis
 ---

A deep learning project that automatically classifies workout exercises in real-time using computer vision and pose estimation. The system extracts 33 body landmarks from video frames and uses a neural network to identify the type of exercise being performed.

---
 Table of Contents

- [Overview](#Overview)

- [Dataset](#Dataset)

- [Technologies Used](#Technologie-Used)

- [Model Architecture](#Model-Architecture)

- [Installation](#Installation)

- [Usage](#Usage)

- [Results](#Results)

- [Project Structure](#Project-Structure)

- [Challenges & Learnings](#Challenges-&-Learnings)

- [Future Improvements](#Future-Improvements)

- [Contributing](#Contributing)

- [License](#License)
 
- [Contact](#Contact)
  
-----
## Overview


This project addresses the challenge of automated exercise form detection using computer vision and deep learning. The system:

- Processes workout videos frame-by-frame
- Detects 33 body keypoints (landmarks) using MediaPipe Pose
- Classifies exercises using a custom PyTorch neural network
- Works in real-time with new video inputs


##  Dataset

Source: Workout Fitness Video Dataset from Kaggle
Dataset Details:

Multiple exercise categories (e.g., squats, push-ups, lunges, etc.)
Video format: MP4
Each video contains demonstrations of a single exercise type
Videos vary in duration, angle, and lighting conditions

Preprocessing:

Each video frame is processed using MediaPipe Pose
33 landmarks are extracted per frame (x, y, z coordinates)
Total features per frame: 99 (33 landmarks × 3 coordinates)
Labels are assigned based on the exercise folder name


##  Technologies Used


| Category | Tools & Libraries |
|----------|-------------------|
| **Programming Language** | Python 3.8+ |
| **Deep Learning Framework** | PyTorch |
| **Computer Vision** | MediaPipe, OpenCV |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Evaluation** | Scikit-learn |
| **Environment** | Google Colab |


## Model Architecture

### Neural Network Design

> **Input Layer:** 99 features (33 landmarks × 3 coordinates)  
> ↓  
> **Hidden Layer 1:** 512 neurons + ReLU + Dropout(0.4)  
> ↓  
> **Hidden Layer 2:** 256 neurons + ReLU + Dropout(0.4)  
> ↓  
> **Hidden Layer 3:** 128 neurons + ReLU  
> ↓  
> **Output Layer:** N classes (number of exercise types)


## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 (initial) |
| **Scheduler** | StepLR (step_size=10, gamma=0.5) |
| **Loss Function** | CrossEntropyLoss |
| **Batch Size** | 32 |
| **Max Epochs** | 80 |
| **Early Stopping** | Patience = 5 |
| **Regularization** | Dropout (0.4) |


Key Features

- Dropout regularization to prevent overfitting
-Learning rate scheduling for better convergence
- Early stopping to avoid unnecessary training
- Data standardization using StandardScaler
