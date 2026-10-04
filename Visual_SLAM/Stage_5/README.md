# Stage 5 — Monocular Visual Odometry

## Overview

Implemented a complete **Monocular Visual Odometry (VO)** pipeline using a real video sequence.

The pipeline estimates camera motion from consecutive image frames using feature matching, epipolar geometry, and camera pose recovery.

## Tasks

### Task 1 — Video to Frames
- Extracted frames from the input video.
- Video: 701 frames, 848×478 resolution.
- Sampled every 5th frame.
- Generated 141 images.

### Task 2 — Feature Detection & Matching
- Detected ORB features.
- Used BFMatcher with Hamming distance.
- Applied KNN matching and Lowe's Ratio Test.
- Average good matches: **152.40**.

### Task 3 — Essential Matrix
- Estimated the Essential Matrix using matched feature points.
- Used RANSAC for outlier rejection.
- Example result:
  - Matches: 468
  - Inliers: 308
  - Outliers: 160.

### Task 4 — Camera Pose Recovery
- Recovered relative camera rotation and translation using `cv2.recoverPose()`.
- Estimated camera motion between consecutive frames.

### Task 5 — Camera Trajectory
- Accumulated relative poses to estimate the global camera trajectory.
- Initial camera position: `[0, 0, 0]`.
- Generated 125 camera positions.

### Task 6 — Trajectory Analysis
Calculated:
- Camera displacement
- Total path length
- Frame-to-frame motion
- Coordinate ranges
- Trajectory directness

Final trajectory analysis:
- Displacement: **22.77 arbitrary units**
- Path length: **124 arbitrary units**
- Directness: **0.1836**

### Task 7 — Drift & Deviation Analysis
- Analyzed trajectory deviation and motion consistency.
- Demonstrated accumulated drift in monocular visual odometry.
- Investigated feature-tracking failures and unreliable frame pairs.

### Mini Project — End-to-End Visual Odometry
Integrated the complete pipeline into one program:

```text
Video
 ↓
Image Frames
 ↓
ORB Features
 ↓
Feature Matching
 ↓
Lowe Ratio Test
 ↓
Essential Matrix + RANSAC
 ↓
Recover Camera Pose
 ↓
Pose Accumulation
 ↓
Camera Trajectory
 ↓
Trajectory Analysis


Mini Project Results
Input images: 141
Image pairs: 140
Successful pose estimations: 124
Failed pairs: 16
Average good matches: 364.25
Average pose inliers: 167.91
Final camera position:
[ -4.04, -10.48, 11.89 ]
Displacement: 16.36 arbitrary units
Path length: 124 arbitrary units
Trajectory directness: 0.1319
Technologies
Python
OpenCV
NumPy
Matplotlib
Key Concepts
ORB Feature Detection
Feature Matching
Lowe Ratio Test
RANSAC
Epipolar Geometry
Essential Matrix
Camera Pose Recovery
Rotation & Translation
Monocular Visual Odometry
Camera Trajectory
Drift Analysis
Limitations
Monocular VO has unknown absolute scale.
Trajectory distances are therefore in arbitrary units.
Feature tracking can fail in low-texture regions.
Accumulated pose errors cause trajectory drift.
No ground-truth trajectory is available.
Results

Generated outputs include:

Camera trajectory
Top-view trajectory
Frame-to-frame motion
Feature matching statistics
Pose inlier statistics
Trajectory data
VO statistics
Status

Stage 5 — Completed ✅