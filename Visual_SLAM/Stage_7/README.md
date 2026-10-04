Stage 7 – Semantic 3D Digital Twin
Overview

Stage 7 adds semantic understanding to the reconstructed 3D environment. YOLO object detection is combined with the optimized 3D map and camera poses from Stage 6 to create an interactive semantic indoor Digital Twin.

The pipeline is:

Stage 6 Optimized 3D Reconstruction
              ↓
        YOLO Detection
              ↓
     Room-Specific Filtering
              ↓
       2D → 3D Association
              ↓
      Semantic Object Fusion
              ↓
   Interactive 3D Digital Twin
Tasks
Task 1 – Load and Validate Stage 6 Reconstruction

Loaded and validated the Stage 6 reconstruction.

Results:

3D map points: 2236
Camera poses: 42
Camera trajectory positions: 42
Mean pose difference: 0
Maximum pose difference: 0
Task 2 – Interactive 3D Reconstruction

Created an interactive visualization of the reconstructed environment.

Results:

3D map: 2236 points
Camera keyframes: 42
Trajectory displacement: 4.712669
Path length: 22.104688
Directness: 0.213198

Output:

task2_interactive_digital_twin.html
Task 3 – Semantic Object Detection

Used YOLO11n to detect objects in the 141 video frames.

Room-specific filtering was applied to remove irrelevant detections.

Allowed semantic classes included:

backpack
bed
book
bottle
chair
dining table
keyboard
laptop

After filtering:

Total raw detections: 48
Filtered detections: 27
Frames with detections: 26

Detected classes:

laptop     : 14
bed        : 9
keyboard   : 2
chair      : 2
Task 4 – 2D → 3D Semantic Association

Associated YOLO detections with the reconstructed 3D environment using the nearest available camera/keyframe information.

Final results:

YOLO detections: 27
Successful associations: 23
Failed associations: 4
Association success rate: 85.19%

Semantic observations:

bed        : 9
chair      : 2
keyboard   : 2
laptop     : 10
Task 5 – Semantic Object Fusion

Repeated observations of the same semantic class were clustered in 3D using DBSCAN.

Results:

Input observations : 23
Fused objects      : 5


bed                : 1
chair              : 1
keyboard           : 1
laptop             : 2

The two laptop clusters represent two spatial clusters produced by the 3D observations.

Generated files:

task5_fused_semantic_objects.json
task5_fused_semantic_objects.npy
task5_semantic_object_statistics.txt
Task 6 – Final Semantic 3D Digital Twin

Combined:

Optimized 3D map
Camera trajectory
Camera keyframes
Fused semantic objects

Final results:

Optimized 3D Map Points : 2236
Camera Keyframes        : 42
Semantic Objects        : 5
Trajectory Displacement : 4.712669
Trajectory Path Length  : 22.104688
Trajectory Directness   : 0.213198

Main output:

task6_final_semantic_digital_twin.html

Open the HTML file in Google Chrome to interact with the Digital Twin.

Final Stage 7 Structure
Stage_7/
│
├── task_1/
│   └── load_stage6.py
│
├── task_2/
│   └── interactive_visualization.py
│
├── task_3/
│   └── object_detection.py
│
├── task_4/
│   └── semantic_3d_association.py
│
├── task_5/
│   └── semantic_object_fusion.py
│
├── task_6/
│   └── semantic_digital_twin.py
│
└── results/
    │
    ├── task1_stage6_3d_map.png
    ├── task1_map_and_trajectory.png
    ├── task1_camera_trajectory_3d.png
    ├── task1_validation_statistics.txt
    ├── stage6_final_evaluation.txt
    │
    ├── task2_interactive_digital_twin.html
    ├── task2_visualization_statistics.txt
    │
    ├── task3_filtered_detections.json
    ├── task3_filtered_detections.npy
    ├── task3_filtered_statistics.txt
    ├── task3_filtered_detections/
    │
    ├── task4_nearest_keyframe_semantic_3d.json
    ├── task4_nearest_keyframe_semantic_3d.npy
    ├── task4_nearest_keyframe_statistics.txt
    ├── task4_nearest_keyframe_object_centers.json
    │
    ├── task5_fused_semantic_objects.json
    ├── task5_fused_semantic_objects.npy
    ├── task5_semantic_object_statistics.txt
    │
    ├── task6_final_semantic_digital_twin.html
    └── task6_semantic_digital_twin_statistics.txt

    
Final Outcome

Stage 7 transforms the geometric reconstruction from Stage 6 into a semantic 3D Digital Twin by combining visual SLAM, 3D reconstruction, object detection, 2D-to-3D association, and semantic object fusion.

Final Digital Twin
2236 3D map points
        +
42 camera keyframes
        +
Camera trajectory
        +
5 semantic 3D objects
        ↓
Interactive Semantic Indoor Digital Twin