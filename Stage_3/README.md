Stage 3: Camera Geometry and Motion Estimation for Visual SLAM
Overview

Stage 3 focuses on the mathematical foundations of Visual SLAM by implementing the complete two-view geometry pipeline. The objective is to estimate the relative motion of a calibrated camera using two images and prepare the data for 3D reconstruction.

The implementation follows the same sequence used in classical feature-based Visual SLAM systems such as ORB-SLAM, where image correspondences are used to estimate camera motion before reconstructing the environment.

Objectives

The goals of this stage are:

Understand camera geometry.
Learn intrinsic and extrinsic camera parameters.
Calibrate a real camera using a chessboard.
Detect and match ORB features.
Estimate the Fundamental Matrix.
Estimate the Essential Matrix.
Recover camera rotation and translation.
Visualize epipolar geometry.
Build a complete camera motion estimation pipeline.
Topics Covered
Task 1 – Camera Coordinate Systems
Objective

Understand the different coordinate systems used in computer vision.

Concepts
World Coordinate System
Camera Coordinate System
Image Coordinate System
Pixel Coordinate System
Pinhole Camera Model
Learning Outcome

Understand how a 3D point is transformed from the real world into a pixel in an image.

Task 2 – Camera Intrinsic Matrix
Objective

Study the internal parameters of a camera.

Concepts
Focal Length
Principal Point
Intrinsic Matrix
K=
	​

f
x
	​

0
0
	​

0
f
y
	​

0
	​

c
x
	​

c
y
	​

1
	​

	​

Learning Outcome

Learn how pixel coordinates are generated from camera coordinates.

Task 3 – Camera Extrinsic Parameters
Objective

Understand the position and orientation of the camera.

Concepts
Rotation Matrix
Translation Vector

Transformation:

P
c
	​

=RP
w
	​

+t
Learning Outcome

Understand how the camera moves in 3D space.

Task 4 – Camera Calibration
Objective

Estimate the intrinsic parameters of a real camera.

Dataset

Real chessboard images captured using a mobile phone.

OpenCV Functions
cv2.findChessboardCornersSB()
cv2.calibrateCamera()
Outputs
Camera Matrix
Distortion Coefficients
Rotation Vectors
Translation Vectors
RMS Reprojection Error
Results

The camera was successfully calibrated with a low reprojection error, indicating accurate calibration suitable for Visual SLAM applications.

Task 5 – Image Formation and Projection
Objective

Understand how a 3D point projects onto the image plane.

Concepts
Perspective Projection
Homogeneous Coordinates
Camera Projection

Projection Equation

x=K[R∣t]X
Learning Outcome

Understand the mathematical relationship between 3D points and image pixels.

Task 6 – ORB Feature Detection and Feature Matching
Objective

Detect robust feature points and establish correspondences between two images.

Techniques
ORB Feature Detector
ORB Descriptor
Brute Force Matcher
Hamming Distance
CrossCheck Matching
Results
Detected Keypoints

Image 1 : 1000

Image 2 : 1000

Total Matches : 468
Learning Outcome

Understand how Visual SLAM establishes feature correspondences between images.

Task 7 – Fundamental Matrix Estimation
Objective

Estimate the geometric relationship between two images.

OpenCV Function
cv2.findFundamentalMat()
Method

RANSAC

Results
Total Matches : 468

Inliers : 260

Outliers : 208
Learning Outcome

Understand epipolar constraints and robust outlier rejection.

Task 8 – Essential Matrix Estimation
Objective

Estimate the relative geometry between two calibrated cameras.

OpenCV Function
cv2.findEssentialMat()
Results
Total Matches : 468

Inliers : 308

Outliers : 160
Learning Outcome

Understand the importance of camera calibration in pose estimation.

Task 9 – Recover Camera Pose
Objective

Estimate the camera motion between two images.

OpenCV Function
cv2.recoverPose()
Results
Recovered Pose Inliers : 391

Rotation Matrix

[[ 0.98160859  0.09988626 -0.16268780]
 [-0.09604339  0.99488373  0.03133728]
 [ 0.16498561 -0.01513586  0.98617983]]

Translation Vector

[[-0.37761659]
 [ 0.11517791]
 [-0.91877079]]
Learning Outcome

Estimate the relative rotation and translation of a moving camera.

Task 10 – Epipolar Geometry Visualization
Objective

Visualize the epipolar geometry between two calibrated images.

OpenCV Function
cv2.computeCorrespondEpilines()
Outputs
Epipolar Lines
Feature Correspondences
Learning Outcome

Understand how the Fundamental Matrix constrains feature matching between images.

Mini Project – Camera Motion Estimation
Objective

Integrate all Stage 3 components into a complete two-view geometry pipeline.

Pipeline
Input Images
      │
      ▼
ORB Feature Detection
      │
      ▼
Descriptor Matching
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
Technologies Used
Python
OpenCV
NumPy
ORB Features
RANSAC
Epipolar Geometry
Camera Calibration
Multiple View Geometry
Results Summary
Task	Result
Camera Calibration	Successfully estimated intrinsic parameters
ORB Feature Detection	1000 keypoints per image
Descriptor Matching	468 feature matches
Fundamental Matrix	260 inliers
Essential Matrix	308 inliers
Recover Pose	391 pose-consistent correspondences
Epipolar Geometry	Successfully visualized epipolar lines
Applications

The techniques implemented in this stage form the foundation of numerous computer vision and robotics applications:

Visual SLAM
Stereo Vision
Structure from Motion (SfM)
Autonomous Navigation
Mobile Robotics
Augmented Reality (AR)
Drone Navigation
Camera Tracking
3D Reconstruction
Key Learning Outcomes

By completing Stage 3, I gained practical experience in:

Camera modeling and calibration
Coordinate system transformations
Feature detection and descriptor matching
Robust geometric estimation using RANSAC
Fundamental and Essential Matrix estimation
Camera pose recovery
Epipolar geometry
Two-view geometry for Visual SLAM

These concepts provide the mathematical and algorithmic foundation required for the next stage of the project, where the estimated camera motion will be used to reconstruct a sparse 3D representation of the environment through triangulation.