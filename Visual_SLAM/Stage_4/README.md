# Stage 4 – 3D Reconstruction & Sparse Point Cloud Generation

## Overview

Stage 4 focuses on reconstructing a sparse 3D representation of a scene using two monocular images. Starting from matched feature points obtained in Stage 3, the camera projection matrices are constructed, corresponding 3D points are triangulated, converted to Cartesian coordinates, visualized as a sparse point cloud, and finally exported in the PLY format for use in professional 3D visualization software.

This stage demonstrates the complete two-view geometry pipeline that forms the basis of Structure-from-Motion (SfM) and feature-based Visual SLAM systems such as ORB-SLAM.

---

# Learning Objectives

- Understand camera projection matrices
- Perform 3D point triangulation
- Convert homogeneous coordinates to Cartesian coordinates
- Generate and visualize sparse 3D point clouds
- Export reconstructed point clouds in PLY format
- Build a complete two-view 3D reconstruction pipeline

---

# Stage 4 Workflow

```
Two Images
      │
      ▼
Feature Matches
      │
      ▼
Essential Matrix
      │
      ▼
Recover Camera Pose
      │
      ▼
Projection Matrices
      │
      ▼
Triangulation
      │
      ▼
3D Points
      │
      ▼
Sparse Point Cloud
      │
      ▼
PLY Export
```

---

# Task 1 – Camera Projection Matrices

## Objective

Construct the camera projection matrices for the reference camera and the second camera using the intrinsic matrix and the recovered camera pose.

### Formula

```
P1 = K [ I | 0 ]

P2 = K [ R | t ]
```

### Output

- Projection Matrix of Camera 1
- Projection Matrix of Camera 2

Saved Files

```
results/
│
├── projection_matrix_camera1.txt
└── projection_matrix_camera2.txt
```

---

# Task 2 – Triangulation

## Objective

Reconstruct the 3D position of matched feature points using stereo triangulation.

### OpenCV Function

```python
cv2.triangulatePoints()
```

### Output

Homogeneous 3D coordinates

```
(X,Y,Z,W)
```

Saved File

```
results/
└── points4D.txt
```

---

# Task 3 – Homogeneous to Cartesian Coordinates

## Objective

Convert homogeneous coordinates into standard Cartesian coordinates.

### Formula

```
X = X/W
Y = Y/W
Z = Z/W
```

Result

```
(X,Y,Z)
```

Saved File

```
results/
└── points3D.txt
```

---

# Task 3.5 – Triangulation Using Pose Inliers

## Objective

Improve the reconstruction by triangulating only the geometrically verified pose inliers instead of all descriptor matches.

Pipeline

```
Descriptor Matches
        │
        ▼
Recover Pose
        │
        ▼
Pose Inliers
        │
        ▼
Triangulation
```

### Results

```
Total BF Matches : 468

Recover Pose Inliers : 391

Triangulated Points : 391
```

Saved Files

```
results/
│
├── pose_inlier_matches.jpg
└── points3D_pose_inliers.txt
```

This produces a cleaner and more reliable sparse point cloud similar to the approach used in ORB-SLAM.

---

# Task 4 – Sparse Point Cloud Visualization

## Objective

Visualize the reconstructed 3D points using Matplotlib.

The generated scatter plot represents a sparse reconstruction of the scene, where every point corresponds to a successfully triangulated ORB feature.

Saved File

```
results/
└── sparse_point_cloud.png
```

---

# Task 5 – Export Point Cloud

## Objective

Export the reconstructed 3D points into the Polygon File Format (PLY).

The exported point cloud can be opened using:

- MeshLab
- CloudCompare
- Open3D
- Blender

Saved File

```
results/
└── point_cloud.ply
```

---

# Results

The complete Stage 4 pipeline successfully reconstructed a sparse 3D representation of the scene using only two monocular images.

### Summary

| Metric | Value |
|---------|------:|
| ORB Features | 1000 |
| Descriptor Matches | 468 |
| Pose Inliers | 391 |
| Reconstructed 3D Points | 391 |

Generated Outputs

```
results/
│
├── projection_matrix_camera1.txt
├── projection_matrix_camera2.txt
├── points4D.txt
├── points3D.txt
├── points3D_pose_inliers.txt
├── sparse_point_cloud.png
├── point_cloud.ply
└── pose_inlier_matches.jpg
```

---

# Concepts Learned

During Stage 4, the following concepts were implemented:

- Camera Projection Matrix
- Perspective Projection
- Stereo Triangulation
- Homogeneous Coordinates
- Cartesian Coordinates
- Sparse 3D Reconstruction
- Point Cloud Generation
- Point Cloud Visualization
- PLY File Export

---

# Applications

The techniques implemented in this stage are fundamental to:

- Visual SLAM
- Structure from Motion (SfM)
- Autonomous Robots
- Drone Navigation
- AR/VR Systems
- Indoor Mapping
- Digital Twin Generation
- Robotics and Computer Vision

---

# Conclusion

Stage 4 successfully reconstructs a sparse three-dimensional representation of a scene from two monocular images. By estimating the camera pose, triangulating feature correspondences, and exporting the resulting point cloud, this stage establishes the core geometric framework required for large-scale mapping and Visual SLAM systems.

The generated sparse map serves as the foundation for the next stage, where multiple image frames will be processed to estimate continuous camera motion through Visual Odometry.