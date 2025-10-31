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
---

This project addresses the challenge of automated exercise form detection using computer vision and deep learning. The system:

- Processes workout videos frame-by-frame
- Detects 33 body keypoints (landmarks) using MediaPipe Pose
- Classifies exercises using a custom PyTorch neural network
- Works in real-time with new video inputs


