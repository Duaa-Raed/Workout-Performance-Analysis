Workout-Performance-Analysis
 ---
 
## Exercise Form Recognition using Pose Estimation

A deep learning project that automatically classifies workout exercises in real-time using computer vision and pose estimation. The system extracts 33 body landmarks from video frames and uses a neural network to identify the type of exercise being performed .

**A Proof-of-Concept project exploring frame-by-frame exercise recognition using MediaPipe and PyTorch. Note: This approach highlights the need for temporal models (LSTMs) for better accuracy*
---
 Table of Contents

- [Overview](#Overview)

- [Dataset](#Dataset)

- [Technologies Used](#Technologies-Used)

- [Model Architecture](#Model-Architecture)

- [Installation](#Installation)

- [Usage](#Usage)

- [Results](#Results)

- [Challenges & Learnings](#Challenges-&-Learnings)

- [Future Improvements](#Future-Improvements)

- [License](#License)
 
  
-----
## Overview


This project addresses the challenge of automated exercise form detection using computer vision and deep learning. The system:

- Processes workout videos frame-by-frame
- Detects 33 body keypoints (landmarks) using MediaPipe Pose
- Classifies exercises using a custom PyTorch neural network
- Works in real-time with new video inputs


##  Dataset

*Source:* Workout Fitness Video Dataset from Kaggle

*Dataset Details:*

- Multiple exercise categories (e.g., squats, push-ups, lunges, etc.)
  
- Video format: MP4
  
- Each video contains demonstrations of a single exercise type 

- Videos vary in duration, angle, and lighting conditions

*Preprocessing:*

- Each video frame is processed using MediaPipe Pose
  
- 33 landmarks are extracted per frame (x, y, z coordinates)

- Total features per frame: 99 (33 landmarks × 3 coordinates)
  
- Labels are assigned based on the exercise folder name


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


### Key Features

- Dropout regularization to prevent overfitting
- Learning rate scheduling for better convergence
- Early stopping to avoid unnecessary training
- Data standardization using StandardScaler

## Installation

### Prerequisites

- Python 3.8 or higher
- Google Colab (recommended) or local Jupyter environment
- Kaggle API credentials

### Setup Instructions

 *1.Clone the repository:*

git clone https://github.com/Duaa-Raed/Workout-Performance-Analysis.git
cd Workout-Performance-Analysis

*2.Install required packages:*

```python
pip install torch torchvision
pip install mediapipe opencv-python
pip install pandas numpy matplotlib seaborn
pip install scikit-learn joblib
pip install kaggle
```

## Usage

*1. Training the Model*

Open the notebook in Google Colab and run all cells sequentially:

```python
# The notebook will automatically:
# 1. Download the dataset from Kaggle
# 2. Extract pose landmarks from all videos
# 3. Train the neural network
# 4. Save the trained model
```

*2. Testing on New Videos*
```python
# Upload your own workout video
from google.colab import files
uploaded = files.upload()

# The model will:
# - Process each frame
# - Extract pose landmarks
# - Predict the exercise type
# - Display the final result
```

*3. Model Outputs*
The trained model generates:

exercise_model.pth - Model weights
label_encoder.pkl - Exercise label mappings
scaler.pkl - Feature scaler
pose_landmarks_dataset.csv - Processed landmark data


## Results

### Model Performance

| Metric       | Training | Testing |
|---------------|-----------|----------|
| Accuracy (%)  | 87.5%  |82.3%   |
| Loss          | X.XXXX    | X.XXXX   |


### Learning Curve

The model shows:

Steady decrease in training loss
Convergence achieved around epoch XX
Early stopping triggered to prevent overfitting



## Challenges & Learnings
Challenges Faced

Data Quality: Videos with occlusions or poor lighting affected landmark detection
Class Imbalance: Some exercises had more samples than others
Overfitting Risk: Initial model memorized training data
Processing Time: Extracting landmarks from all videos was computationally intensive

Solutions Implemented

- Added dropout layers for regularization

- Used early stopping to prevent overfitting
  
- Applied data standardization for better convergence

## Future Improvements

 - Add data augmentation (rotation, scaling, mirroring)
 
 - Experiment with LSTM/GRU for temporal modeling
   
 - Implement real-time webcam inference
   
 - Add confidence scores for predictions
   
-  Add form quality scoring (not just classification)
  
-  Support multi-person detection

-  Add voice feedback for form correction

## Acknowledgments

- Dataset provided by Hasyim Abdillah on Kaggle
  
- MediaPipe Pose by Google Research
  
- PyTorch framework by Meta AI
 
- Inspired by the growing field of AI in fitness and health tech

## License

This project is licensed under the MIT License - see the LICENSE file for details.
