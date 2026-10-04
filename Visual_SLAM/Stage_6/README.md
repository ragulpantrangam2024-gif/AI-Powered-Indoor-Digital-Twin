Overview

Stage 6 extends the monocular visual odometry pipeline from Stage 5 into a keyframe-based sparse SLAM pipeline.

The stage selects important frames as keyframes, tracks visual features between keyframes, reconstructs 3D landmarks through triangulation, builds a global sparse map, refines camera poses and 3D points using Bundle Adjustment, and finally evaluates the reconstructed system.

Pipeline
141 Video Frames
       ↓
Keyframe Selection
       ↓
Feature Tracking
       ↓
3D Triangulation
       ↓
Sparse 3D Map
       ↓
Global Map Construction
       ↓
Bundle Adjustment
       ↓
Final Evaluation
Dataset
Input: 141 sampled frames from the Stage 5 video
Image resolution: 848 × 478
Keyframes selected: 42
Camera model: Monocular
Feature detector: ORB
Camera Intrinsic Matrix
[[879.03578678   0.         438.89954652]
 [  0.         896.60912854 557.52366461]
 [  0.           0.           1.        ]]
Task 1 — Keyframe Selection

Selected important frames from the 141 input frames based on feature overlap and frame interval.

Results
Total Frames       : 141
Selected Keyframes : 42

Minimum Interval   : 1
Maximum Interval   : 10
Average Interval   : 3.29

Generated:

keyframe_indices.npy
keyframes.txt
keyframe_statistics.txt
keyframe_selection.png
Task 2 — Feature Tracking

ORB features were extracted and matched between consecutive keyframes.

A Lowe ratio test and geometric verification were used to identify reliable correspondences.

Results
Keyframes Processed : 42
Keyframe Pairs      : 41

Good Matches
Minimum : 0
Maximum : 296
Average : 44.61

Geometric Inliers
Minimum : 0
Maximum : 192
Average : 30.39

Tracking Ratio
Minimum : 0.000
Maximum : 0.959
Average : 0.416

Generated:

task2_good_matches.npy
task2_inliers.npy
task2_outliers.npy
task2_tracking_ratios.npy
task2_tracking_statistics.txt
task2_good_matches.png
task2_inliers.png
task2_tracking_ratio.png
Task 3 — Sparse 3D Mapping

Reliable feature correspondences were triangulated using the estimated camera poses to reconstruct 3D landmarks.

The resulting map is a sparse point cloud, meaning it contains only successfully reconstructed visual feature points rather than every pixel or surface point in the scene.

Results
Keyframes            : 42
Keyframe Pairs       : 41
Successful Pairs     : 21
Failed/Skipped Pairs : 20

Final 3D Map Points  : 1701

3D coordinate ranges:

X : -7.033 to 13.233
Y : -14.330 to -0.092
Z : 0.640 to 33.770

Generated:

task3_map_points.npy
task3_map_points.txt
task3_3d_map_statistics.txt
task3_sparse_3d_map.png
task3_map_top_view.png
Task 4 — Global Map Construction

The individual triangulated points were merged into a global map by identifying observations corresponding to the same visual landmarks.

This produces a more consistent sparse 3D representation across the keyframes.

Results
Total Images       : 141
Keyframes          : 42
Keyframe Pairs     : 41
Successful Pairs   : 21
Failed Pairs       : 20

Global Map Points  : 639

Map Point Observations
Minimum : 2
Maximum : 4
Average : 2.013

Generated:

task4_global_map_points.npy
task4_global_map_points.txt
task4_keyframe_camera_positions.npy
task4_map_point_observations.npy
task4_keyframe_poses.txt
task4_global_map_statistics.txt
task4_global_3d_map.png
task4_global_map_top_view.png
task4_observation_distribution.png
Task 5 — Bundle Adjustment

Bundle Adjustment was used to refine the camera poses and 3D landmarks by minimizing reprojection error.

The first camera pose was fixed while the remaining camera poses and 3D points were optimized.

Optimization
Cameras              : 42
Fixed Camera         : 0
Optimized Cameras    : 41
3D Points            : 2236
Observations         : 4472
Robust Loss          : soft_l1
Sparse Jacobian      : Enabled
Reprojection Error
Initial RMS Error : 13.584638 pixels
Final RMS Error   : 2.344250 pixels

Initial Mean Error : 0.943618 pixels
Final Mean Error   : 0.398535 pixels

RMS Improvement : 82.74 %

The optimizer reached the configured function-evaluation limit, but the reprojection error was reduced substantially.

Interpretation

Bundle Adjustment significantly improved the consistency between:

3D landmarks
     ↕
Camera poses
     ↕
2D image observations

Generated:

task5_optimized_map_points.npy
task5_optimized_map_points.txt
task5_optimized_camera_poses.txt
task5_optimized_rotations.npy
task5_optimized_translations.npy
task5_bundle_adjustment_statistics.txt
task5_initial_reprojection_errors.npy
task5_final_reprojection_errors.npy
task5_reprojection_error.png
task5_observation_errors.png
task5_optimized_trajectory.png
Task 6 — Final Evaluation

The final task compared the original reconstruction with the Bundle Adjustment result.

Reprojection Error
Initial RMS   : 13.584638 pixels
Final RMS     : 2.344250 pixels

Initial Mean  : 0.943618 pixels
Final Mean    : 0.398535 pixels

Initial Median : 0.296111 pixels
Final Median   : 0.257438 pixels

RMS Improvement : 82.74 %
Final Reconstruction
Final 3D Map       : 2236 points
Final Camera Poses : 42 keyframes
Camera Trajectory

Original Task 4:

Displacement : 4.692602
Path Length  : 21.000000
Directness   : 0.223457

After Bundle Adjustment:

Displacement : 4.712669
Path Length  : 22.104688
Directness   : 0.213198

Camera position refinement:

Mean Difference    : 0.201305
Maximum Difference : 1.177940
Final Outputs
task6_map_comparison.png
task6_trajectory_comparison.png
task6_reprojection_distribution.png
task6_reprojection_comparison.png
task6_camera_position_change.png
task6_final_evaluation.txt
task6_final_camera_trajectory.npy
task6_final_camera_trajectory.txt
Final Stage 6 Result

Stage 6 successfully demonstrates a complete keyframe-based monocular sparse SLAM workflow:

Video
 ↓
Frame Sampling
 ↓
ORB Feature Extraction
 ↓
Keyframe Selection
 ↓
Feature Matching
 ↓
Geometric Verification
 ↓
Camera Pose Estimation
 ↓
Triangulation
 ↓
Sparse 3D Mapping
 ↓
Global Map Construction
 ↓
Bundle Adjustment
 ↓
Final Evaluation
Key Achievement

Bundle Adjustment reduced the RMS reprojection error from 13.58 pixels to 2.34 pixels, an improvement of 82.74%.

This demonstrates that the reconstructed camera poses and 3D landmarks were successfully refined using multi-view geometric optimization.