import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# STAGE 7 - TASK 1
# LOAD AND VALIDATE STAGE 6 RECONSTRUCTION
# ============================================================

print("=" * 70)
print("STAGE 7 - TASK 1")
print("LOAD AND VALIDATE STAGE 6 RECONSTRUCTION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STAGE_7_DIR = os.path.dirname(
    SCRIPT_DIR
)

PROJECT_DIR = os.path.dirname(
    STAGE_7_DIR
)

STAGE_6_RESULTS = os.path.join(
    PROJECT_DIR,
    "Stage_6",
    "results"
)

RESULTS_DIR = os.path.join(
    STAGE_7_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


print("\nProject directory:")
print(PROJECT_DIR)

print("\nStage 6 results:")
print(STAGE_6_RESULTS)

print("\nStage 7 results:")
print(RESULTS_DIR)


# ============================================================
# REQUIRED FILES
# ============================================================

FILES = {

    "map_points":
        "task5_optimized_map_points.npy",

    "rotations":
        "task5_optimized_rotations.npy",

    "translations":
        "task5_optimized_translations.npy",

    "trajectory":
        "task6_final_camera_trajectory.npy",

    "evaluation":
        "task6_final_evaluation.txt"
}


# ============================================================
# CHECK FILES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CHECKING STAGE 6 FILES"
)

print(
    "=" * 70
)


for name, filename in FILES.items():

    path = os.path.join(
        STAGE_6_RESULTS,
        filename
    )

    if os.path.isfile(path):

        print(
            f"[OK] {filename}"
        )

    else:

        print(
            f"[ERROR] Missing: {filename}"
        )

        raise SystemExit


# ============================================================
# LOAD 3D MAP
# ============================================================

map_points = np.load(
    os.path.join(
        STAGE_6_RESULTS,
        FILES["map_points"]
    )
)


print(
    "\n" + "=" * 70
)

print(
    "3D MAP"
)

print(
    "=" * 70
)

print(
    "Number of points:",
    len(map_points)
)

print(
    "Shape:",
    map_points.shape
)


if map_points.ndim != 2:

    print(
        "ERROR: Invalid 3D map shape."
    )

    raise SystemExit


if map_points.shape[1] != 3:

    print(
        "ERROR: Expected XYZ coordinates."
    )

    raise SystemExit


# ============================================================
# MAP STATISTICS
# ============================================================

map_min = np.min(
    map_points,
    axis=0
)

map_max = np.max(
    map_points,
    axis=0
)

map_mean = np.mean(
    map_points,
    axis=0
)

map_std = np.std(
    map_points,
    axis=0
)

map_range = (
    map_max - map_min
)


print(
    "\nMinimum XYZ:"
)

print(
    map_min
)

print(
    "\nMaximum XYZ:"
)

print(
    map_max
)

print(
    "\nMean XYZ:"
)

print(
    map_mean
)

print(
    "\nStandard Deviation XYZ:"
)

print(
    map_std
)

print(
    "\nMap Range:"
)

print(
    map_range
)


# ============================================================
# LOAD ROTATIONS
# ============================================================

rotations = np.load(
    os.path.join(
        STAGE_6_RESULTS,
        FILES["rotations"]
    )
)


print(
    "\n" + "=" * 70
)

print(
    "CAMERA ROTATIONS"
)

print(
    "=" * 70
)

print(
    "Shape:",
    rotations.shape
)


if rotations.shape != (
    42,
    3,
    3
):

    print(
        "WARNING: Expected shape (42, 3, 3)"
    )


# ============================================================
# LOAD TRANSLATIONS
# ============================================================

translations = np.load(
    os.path.join(
        STAGE_6_RESULTS,
        FILES["translations"]
    )
)


print(
    "\n" + "=" * 70
)

print(
    "CAMERA TRANSLATIONS"
)

print(
    "=" * 70
)

print(
    "Shape:",
    translations.shape
)


if translations.shape != (
    42,
    3
):

    print(
        "WARNING: Expected shape (42, 3)"
    )


# ============================================================
# LOAD FINAL TRAJECTORY
# ============================================================

trajectory = np.load(
    os.path.join(
        STAGE_6_RESULTS,
        FILES["trajectory"]
    )
)


print(
    "\n" + "=" * 70
)

print(
    "FINAL CAMERA TRAJECTORY"
)

print(
    "=" * 70
)

print(
    "Shape:",
    trajectory.shape
)


if trajectory.shape[1] != 3:

    print(
        "ERROR: Trajectory must contain XYZ."
    )

    raise SystemExit


print(
    "\nInitial camera position:"
)

print(
    trajectory[0]
)


print(
    "\nFinal camera position:"
)

print(
    trajectory[-1]
)


# ============================================================
# TRAJECTORY STATISTICS
# ============================================================

if len(trajectory) > 1:

    frame_motion = np.linalg.norm(
        np.diff(
            trajectory,
            axis=0
        ),
        axis=1
    )

    path_length = np.sum(
        frame_motion
    )

else:

    frame_motion = np.array([])

    path_length = 0.0


displacement_vector = (
    trajectory[-1]
    -
    trajectory[0]
)

displacement = np.linalg.norm(
    displacement_vector
)


if path_length > 0:

    directness = (
        displacement
        /
        path_length
    )

else:

    directness = 0.0


print(
    "\nStart-to-end displacement:"
)

print(
    displacement_vector
)

print(
    "\nDisplacement magnitude:",
    f"{displacement:.6f}"
)

print(
    "\nPath length:",
    f"{path_length:.6f}"
)

print(
    "\nTrajectory directness:",
    f"{directness:.6f}"
)


# ============================================================
# CHECK CAMERA POSE CONSISTENCY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CAMERA POSE VALIDATION"
)

print(
    "=" * 70
)


calculated_positions = []


for i in range(
    len(rotations)
):

    R = rotations[i]

    t = translations[i]

    camera_position = (
        -R.T @ t
    )

    calculated_positions.append(
        camera_position
    )


calculated_positions = np.asarray(
    calculated_positions
)


print(
    "Calculated camera positions:",
    calculated_positions.shape
)


# Compare against stored trajectory

num_compare = min(
    len(calculated_positions),
    len(trajectory)
)


pose_difference = (
    calculated_positions[:num_compare]
    -
    trajectory[:num_compare]
)


pose_difference_magnitude = (
    np.linalg.norm(
        pose_difference,
        axis=1
    )
)


mean_pose_difference = np.mean(
    pose_difference_magnitude
)

max_pose_difference = np.max(
    pose_difference_magnitude
)


print(
    "\nMean pose difference:",
    f"{mean_pose_difference:.10f}"
)

print(
    "Maximum pose difference:",
    f"{max_pose_difference:.10f}"
)


# ============================================================
# PLOT 1 - 3D MAP
# ============================================================

print(
    "\nGenerating 3D map visualization..."
)


fig = plt.figure(
    figsize=(12, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


ax.scatter(
    map_points[:, 0],
    map_points[:, 1],
    map_points[:, 2],
    s=3,
    alpha=0.6
)


ax.set_xlabel(
    "X"
)

ax.set_ylabel(
    "Y"
)

ax.set_zlabel(
    "Z"
)

ax.set_title(
    "Stage 7 - Task 1\n"
    "Stage 6 Optimized Sparse 3D Map"
)


plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task1_stage6_3d_map.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2 - TOP VIEW + TRAJECTORY
# ============================================================

print(
    "Generating top-view visualization..."
)


plt.figure(
    figsize=(10, 8)
)


plt.scatter(
    map_points[:, 0],
    map_points[:, 2],
    s=3,
    alpha=0.35,
    label="3D Map"
)


plt.plot(
    trajectory[:, 0],
    trajectory[:, 2],
    marker="o",
    markersize=3,
    linewidth=1.5,
    label="Camera Trajectory"
)


plt.scatter(
    trajectory[0, 0],
    trajectory[0, 2],
    s=100,
    label="Start"
)


plt.scatter(
    trajectory[-1, 0],
    trajectory[-1, 2],
    s=100,
    label="End"
)


plt.xlabel(
    "X"
)

plt.ylabel(
    "Z"
)

plt.title(
    "Stage 7 - Task 1\n"
    "3D Map and Camera Trajectory"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.axis(
    "equal"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task1_map_and_trajectory.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 3 - CAMERA TRAJECTORY
# ============================================================

print(
    "Generating camera trajectory..."
)


fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


ax.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    trajectory[:, 2],
    marker="o",
    markersize=3,
    linewidth=1.5
)


ax.scatter(
    trajectory[0, 0],
    trajectory[0, 1],
    trajectory[0, 2],
    s=100,
    label="Start"
)


ax.scatter(
    trajectory[-1, 0],
    trajectory[-1, 1],
    trajectory[-1, 2],
    s=100,
    label="End"
)


ax.set_xlabel(
    "X"
)

ax.set_ylabel(
    "Y"
)

ax.set_zlabel(
    "Z"
)

ax.set_title(
    "Stage 7 - Task 1\n"
    "Optimized Camera Trajectory"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task1_camera_trajectory_3d.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# SAVE VALIDATION STATISTICS
# ============================================================

statistics_file = os.path.join(
    RESULTS_DIR,
    "task1_validation_statistics.txt"
)


with open(
    statistics_file,
    "w"
) as f:

    f.write(
        "STAGE 7 - TASK 1\n"
    )

    f.write(
        "LOAD AND VALIDATE STAGE 6 RECONSTRUCTION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )


    f.write(
        "3D MAP\n"
    )

    f.write(
        f"Number of Points : "
        f"{len(map_points)}\n"
    )

    f.write(
        f"Shape : "
        f"{map_points.shape}\n"
    )

    f.write(
        f"Minimum XYZ : "
        f"{map_min}\n"
    )

    f.write(
        f"Maximum XYZ : "
        f"{map_max}\n"
    )

    f.write(
        f"Mean XYZ : "
        f"{map_mean}\n"
    )

    f.write(
        f"Standard Deviation : "
        f"{map_std}\n"
    )

    f.write(
        f"Map Range : "
        f"{map_range}\n"
    )


    f.write(
        "\nCAMERA POSES\n"
    )

    f.write(
        f"Rotations : "
        f"{rotations.shape}\n"
    )

    f.write(
        f"Translations : "
        f"{translations.shape}\n"
    )


    f.write(
        "\nCAMERA TRAJECTORY\n"
    )

    f.write(
        f"Number of Positions : "
        f"{len(trajectory)}\n"
    )

    f.write(
        f"Initial Position : "
        f"{trajectory[0]}\n"
    )

    f.write(
        f"Final Position : "
        f"{trajectory[-1]}\n"
    )

    f.write(
        f"Displacement Vector : "
        f"{displacement_vector}\n"
    )

    f.write(
        f"Displacement : "
        f"{displacement:.8f}\n"
    )

    f.write(
        f"Path Length : "
        f"{path_length:.8f}\n"
    )

    f.write(
        f"Directness : "
        f"{directness:.8f}\n"
    )


    f.write(
        "\nPOSE VALIDATION\n"
    )

    f.write(
        f"Mean Pose Difference : "
        f"{mean_pose_difference:.10f}\n"
    )

    f.write(
        f"Maximum Pose Difference : "
        f"{max_pose_difference:.10f}\n"
    )


## ============================================================
# COPY STAGE 6 EVALUATION
# ============================================================

STAGE6_EVALUATION_FILE = os.path.join(
    STAGE_6_RESULTS,
    "task6_final_evaluation.txt"
)

evaluation_copy = os.path.join(
    RESULTS_DIR,
    "stage6_final_evaluation.txt"
)


if os.path.isfile(STAGE6_EVALUATION_FILE):

    with open(
        STAGE6_EVALUATION_FILE,
        "r"
    ) as source:

        evaluation_text = source.read()


    with open(
        evaluation_copy,
        "w"
    ) as destination:

        destination.write(
            evaluation_text
        )

    print(
        "\nStage 6 evaluation copied successfully."
    )

else:

    print(
        "\nWarning: Stage 6 evaluation file not found."
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TASK 1 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)

print(
    "\nStage 6 reconstruction successfully loaded."
)

print(
    f"3D Map Points : {len(map_points)}"
)

print(
    f"Camera Poses  : {len(rotations)}"
)

print(
    f"Trajectory Positions : {len(trajectory)}"
)

print(
    f"Mean Pose Difference : "
    f"{mean_pose_difference:.10f}"
)

print(
    f"Maximum Pose Difference : "
    f"{max_pose_difference:.10f}"
)

print(
    "\nResults saved in:"
)

print(
    RESULTS_DIR
)

print(
    "\nGenerated files:"
)

print(
    "1. task1_stage6_3d_map.png"
)

print(
    "2. task1_map_and_trajectory.png"
)

print(
    "3. task1_camera_trajectory_3d.png"
)

print(
    "4. task1_validation_statistics.txt"
)

print(
    "5. stage6_final_evaluation.txt"
)

print(
    "\nNo plot windows were opened."
)

print(
    "\nStage 7 Task 1 completed."
)