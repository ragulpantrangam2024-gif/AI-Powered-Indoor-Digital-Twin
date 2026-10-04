# Stage 3: Camera Geometry and Motion Estimation for Visual SLAM

## Overview

Stage 3 focuses on the mathematical and geometric foundations of **Visual SLAM** by implementing the complete **Two-View Geometry Pipeline**. In this stage, a calibrated camera is used to estimate its relative motion from two images. The workflow follows the same principles used in classical feature-based Visual SLAM systems such as **ORB-SLAM**.

The implementation begins with understanding camera geometry and calibration, followed by feature detection, feature matching, geometric verification using the Fundamental and Essential Matrices, camera pose estimation, and finally epipolar geometry visualization.

---

# Objectives

The primary objectives of this stage are:

- Understand camera coordinate systems
- Learn camera intrinsic and extrinsic parameters
- Calibrate a real camera using a chessboard pattern
- Detect and match ORB features
- Estimate the Fundamental Matrix using RANSAC
- Estimate the Essential Matrix using camera calibration
- Recover camera rotation and translation
- Visualize epipolar geometry
- Build a complete camera motion estimation pipeline

---

# Project Workflow

```text
Input Images
      │
      ▼
Camera Calibration
      │
      ▼
ORB Feature Detection
      │
      ▼
Feature Matching
      │
      ▼
Fundamental Matrix (RANSAC)
      │
      ▼
Essential Matrix
      │
      ▼
Recover Camera Pose (R, t)
      │
      ▼
Epipolar Geometry Visualization
      │
      ▼
Camera Motion Estimation
```

---

# Task 1 – Camera Coordinate Systems

## Objective

Understand the coordinate systems used in computer vision.

## Topics Covered

- World Coordinate System
- Camera Coordinate System
- Image Coordinate System
- Pixel Coordinate System
- Pinhole Camera Model

## Learning Outcome

Learn how a 3D point in the world is transformed into a 2D image pixel.

---

# Task 2 – Camera Intrinsic Matrix

## Objective

Understand the internal parameters of a camera.

## Topics Covered

- Focal Length
- Principal Point
- Camera Intrinsic Matrix

### Intrinsic Matrix

\[
K=
\begin{bmatrix}
f_x & 0 & c_x\\
0 & f_y & c_y\\
0 & 0 & 1
\end{bmatrix}
\]

## Learning Outcome

Understand how camera coordinates are projected into image coordinates.

---

# Task 3 – Camera Extrinsic Parameters

## Objective

Understand the position and orientation of the camera.

## Topics Covered

- Rotation Matrix
- Translation Vector

### Transformation

\[
P_c = RP_w+t
\]

## Learning Outcome

Understand how camera motion is represented mathematically.

---

# Task 4 – Camera Calibration

## Objective

Estimate the intrinsic parameters of a real camera.

## Dataset

- Real chessboard images
- OpenCV calibration toolbox

## OpenCV Functions Used

- `cv2.findChessboardCornersSB()`
- `cv2.calibrateCamera()`

## Outputs

- Camera Matrix
- Distortion Coefficients
- Rotation Vectors
- Translation Vectors
- Reprojection Error

## Result

Successfully calibrated the camera with a low reprojection error, providing accurate intrinsic parameters for subsequent pose estimation tasks.

---

# Task 5 – Image Formation and Projection

## Objective

Understand the mathematical relationship between 3D world points and 2D image pixels.

## Topics Covered

- Perspective Projection
- Homogeneous Coordinates
- Camera Projection Matrix

### Projection Equation

\[
x = K[R|t]X
\]

## Learning Outcome

Understand how cameras form images using projective geometry.

---

# Task 6 – ORB Feature Detection and Matching

## Objective

Detect robust feature points and establish correspondences between two images.

## Techniques Used

- ORB Feature Detector
- ORB Binary Descriptor
- Brute Force Matcher
- Hamming Distance
- CrossCheck Matching

## Results

| Metric | Value |
|---------|------:|
| Image 1 Keypoints | 1000 |
| Image 2 Keypoints | 1000 |
| Total Matches | 468 |

## Learning Outcome

Learn how feature correspondences are established between multiple views.

---

# Task 7 – Fundamental Matrix Estimation

## Objective

Estimate the geometric relationship between two uncalibrated images.

## OpenCV Function

```python
cv2.findFundamentalMat()
```

## Method

- RANSAC
- Epipolar Geometry

## Results

| Metric | Value |
|---------|------:|
| Total Matches | 468 |
| Inliers | 260 |
| Outliers | 208 |

## Learning Outcome

Understand robust outlier rejection and epipolar constraints.

---

# Task 8 – Essential Matrix Estimation

## Objective

Estimate the geometric relationship between two calibrated cameras.

## OpenCV Function

```python
cv2.findEssentialMat()
```

## Results

| Metric | Value |
|---------|------:|
| Total Matches | 468 |
| Inliers | 308 |
| Outliers | 160 |

## Learning Outcome

Understand the role of camera calibration in motion estimation.

---

# Task 9 – Recover Camera Pose

## Objective

Estimate the relative camera motion.

## OpenCV Function

```python
cv2.recoverPose()
```

## Results

### Pose Inliers

```text
391
```

### Rotation Matrix

```text
[[ 0.98160859  0.09988626 -0.16268780]
 [-0.09604339  0.99488373  0.03133728]
 [ 0.16498561 -0.01513586  0.98617983]]
```

### Translation Vector

```text
[[-0.37761659]
 [ 0.11517791]
 [-0.91877079]]
```

## Learning Outcome

Estimate the camera's rotation and translation between two views.

---

# Task 10 – Epipolar Geometry Visualization

## Objective

Visualize the epipolar geometry between two images.

## OpenCV Function

```python
cv2.computeCorrespondEpilines()
```

## Outputs

- Epipolar Lines
- Matched Feature Points

## Learning Outcome

Understand how the Fundamental Matrix constrains feature correspondences between images.

---

# Mini Project – Camera Motion Estimation

## Objective

Integrate all Stage 3 components into a complete camera motion estimation pipeline.

## Pipeline

```text
Input Images
      │
      ▼
ORB Feature Detection
      │
      ▼
Feature Matching
      │
      ▼
Fundamental Matrix (RANSAC)
      │
      ▼
Essential Matrix
      │
      ▼
Recover Camera Pose
      │
      ▼
Epipolar Geometry Visualization
      │
      ▼
Camera Motion Estimation
```

## Outputs

- Camera Calibration
- ORB Features
- Feature Matches
- Fundamental Matrix
- Essential Matrix
- Rotation Matrix
- Translation Vector
- Epipolar Geometry Visualization

---

# Technologies Used

- Python
- OpenCV
- NumPy
- ORB Feature Detector
- Brute Force Matcher
- RANSAC
- Multiple View Geometry
- Camera Calibration
- Epipolar Geometry

---

# Results Summary

| Task | Result |
|------|--------|
| Camera Calibration | Successfully estimated intrinsic parameters |
| ORB Feature Detection | 1000 keypoints detected in each image |
| Feature Matching | 468 reliable feature correspondences |
| Fundamental Matrix | 260 inliers after RANSAC |
| Essential Matrix | 308 inliers |
| Recover Camera Pose | 391 pose-consistent correspondences |
| Epipolar Geometry | Successfully visualized epipolar constraints |

---

# Applications

The techniques implemented in this stage are fundamental to:

- Visual SLAM
- Stereo Vision
- Structure from Motion (SfM)
- Autonomous Driving
- Mobile Robotics
- Drone Navigation
- Camera Tracking
- 3D Reconstruction
- Augmented Reality (AR)

---

# Key Learning Outcomes

By completing Stage 3, I gained practical experience in:

- Camera modeling
- Camera calibration
- Coordinate transformations
- Feature detection and matching
- Robust geometric estimation using RANSAC
- Fundamental Matrix estimation
- Essential Matrix estimation
- Camera pose recovery
- Epipolar geometry
- Two-view geometry for Visual SLAM

This stage provides the mathematical and algorithmic foundation for **Stage 4**, where the recovered camera poses will be used to triangulate feature correspondences and reconstruct a sparse 3D representation of the scene.