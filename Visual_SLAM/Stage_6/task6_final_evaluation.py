import os
import re
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ============================================================
# STAGE 6 - TASK 6
# FINAL RECONSTRUCTION EVALUATION
# ============================================================

print("=" * 70)
print("STAGE 6 - TASK 6")
print("FINAL RECONSTRUCTION EVALUATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STAGE_DIR = os.path.dirname(
    SCRIPT_DIR
)

RESULTS_DIR = os.path.join(
    STAGE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

TASK4_MAP = os.path.join(
    RESULTS_DIR,
    "task4_global_map_points.npy"
)

TASK4_POSITIONS = os.path.join(
    RESULTS_DIR,
    "task4_keyframe_camera_positions.npy"
)

TASK5_MAP = os.path.join(
    RESULTS_DIR,
    "task5_optimized_map_points.npy"
)

TASK5_ROTATIONS = os.path.join(
    RESULTS_DIR,
    "task5_optimized_rotations.npy"
)

TASK5_TRANSLATIONS = os.path.join(
    RESULTS_DIR,
    "task5_optimized_translations.npy"
)

TASK5_STATS = os.path.join(
    RESULTS_DIR,
    "task5_bundle_adjustment_statistics.txt"
)

TASK5_INITIAL_ERRORS = os.path.join(
    RESULTS_DIR,
    "task5_initial_reprojection_errors.npy"
)

TASK5_FINAL_ERRORS = os.path.join(
    RESULTS_DIR,
    "task5_final_reprojection_errors.npy"
)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking input files...")


required_files = [
    TASK4_MAP,
    TASK4_POSITIONS,
    TASK5_MAP,
    TASK5_ROTATIONS,
    TASK5_TRANSLATIONS,
    TASK5_STATS,
    TASK5_INITIAL_ERRORS,
    TASK5_FINAL_ERRORS
]


for file_path in required_files:

    if not os.path.isfile(file_path):

        print(
            "\nERROR: Required file not found:"
        )

        print(file_path)

        raise SystemExit


print(
    "All required files found."
)


# ============================================================
# LOAD TASK 4 MAP
# ============================================================

task4_map = np.load(
    TASK4_MAP
)


print(
    "\nTask 4 Global Map"
)

print(
    "Points:",
    len(task4_map)
)

print(
    "Shape:",
    task4_map.shape
)


# ============================================================
# LOAD TASK 4 CAMERA POSITIONS
# ============================================================

task4_positions = np.load(
    TASK4_POSITIONS
)


print(
    "\nTask 4 Camera Positions"
)

print(
    "Positions:",
    len(task4_positions)
)

print(
    "Shape:",
    task4_positions.shape
)


# ============================================================
# LOAD TASK 5 OPTIMIZED MAP
# ============================================================

task5_map = np.load(
    TASK5_MAP
)


print(
    "\nTask 5 Optimized Map"
)

print(
    "Points:",
    len(task5_map)
)

print(
    "Shape:",
    task5_map.shape
)


# ============================================================
# LOAD TASK 5 POSES
# ============================================================

task5_rotations = np.load(
    TASK5_ROTATIONS
)

task5_translations = np.load(
    TASK5_TRANSLATIONS
)


print(
    "\nTask 5 Optimized Poses"
)

print(
    "Rotations:",
    task5_rotations.shape
)

print(
    "Translations:",
    task5_translations.shape
)


# ============================================================
# CALCULATE TASK 5 CAMERA POSITIONS
# ============================================================

task5_positions = []


for i in range(
    len(task5_rotations)
):

    R = task5_rotations[i]

    t = task5_translations[i]

    camera_center = (
        -R.T @ t
    )

    task5_positions.append(
        camera_center
    )


task5_positions = np.asarray(
    task5_positions
)


print(
    "\nTask 5 Camera Positions"
)

print(
    "Positions:",
    len(task5_positions)
)


# ============================================================
# LOAD REPROJECTION ERRORS
# ============================================================

initial_errors = np.load(
    TASK5_INITIAL_ERRORS
)

final_errors = np.load(
    TASK5_FINAL_ERRORS
)


initial_errors = initial_errors.reshape(
    -1,
    2
)

final_errors = final_errors.reshape(
    -1,
    2
)


initial_magnitude = np.linalg.norm(
    initial_errors,
    axis=1
)

final_magnitude = np.linalg.norm(
    final_errors,
    axis=1
)


# ============================================================
# ERROR STATISTICS
# ============================================================

initial_rms = np.sqrt(
    np.mean(
        initial_errors ** 2
    )
)

final_rms = np.sqrt(
    np.mean(
        final_errors ** 2
    )
)


initial_mean = np.mean(
    initial_magnitude
)

final_mean = np.mean(
    final_magnitude
)


initial_median = np.median(
    initial_magnitude
)

final_median = np.median(
    final_magnitude
)


improvement = (
    (
        initial_rms
        -
        final_rms
    )
    /
    max(
        initial_rms,
        1e-12
    )
) * 100.0


print(
    "\n" + "=" * 70
)

print(
    "REPROJECTION ERROR COMPARISON"
)

print(
    "=" * 70
)

print(
    f"Initial RMS   : "
    f"{initial_rms:.6f} pixels"
)

print(
    f"Final RMS     : "
    f"{final_rms:.6f} pixels"
)

print(
    f"Initial Mean  : "
    f"{initial_mean:.6f} pixels"
)

print(
    f"Final Mean    : "
    f"{final_mean:.6f} pixels"
)

print(
    f"Initial Median: "
    f"{initial_median:.6f} pixels"
)

print(
    f"Final Median  : "
    f"{final_median:.6f} pixels"
)

print(
    f"RMS Improvement: "
    f"{improvement:.2f} %"
)


# ============================================================
# MAP STATISTICS FUNCTION
# ============================================================

def map_statistics(points):

    minimum = np.min(
        points,
        axis=0
    )

    maximum = np.max(
        points,
        axis=0
    )

    mean = np.mean(
        points,
        axis=0
    )

    std = np.std(
        points,
        axis=0
    )

    ranges = (
        maximum - minimum
    )

    return (
        minimum,
        maximum,
        mean,
        std,
        ranges
    )


# ============================================================
# TASK 4 MAP STATISTICS
# ============================================================

(
    task4_min,
    task4_max,
    task4_mean,
    task4_std,
    task4_ranges
) = map_statistics(
    task4_map
)


# ============================================================
# TASK 5 MAP STATISTICS
# ============================================================

(
    task5_min,
    task5_max,
    task5_mean,
    task5_std,
    task5_ranges
) = map_statistics(
    task5_map
)


# ============================================================
# MAP COMPARISON
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "3D MAP COMPARISON"
)

print(
    "=" * 70
)

print(
    "\nTask 4 Map"
)

print(
    "X range:",
    f"{task4_ranges[0]:.4f}"
)

print(
    "Y range:",
    f"{task4_ranges[1]:.4f}"
)

print(
    "Z range:",
    f"{task4_ranges[2]:.4f}"
)


print(
    "\nTask 5 Optimized Map"
)

print(
    "X range:",
    f"{task5_ranges[0]:.4f}"
)

print(
    "Y range:",
    f"{task5_ranges[1]:.4f}"
)

print(
    "Z range:",
    f"{task5_ranges[2]:.4f}"
)


# ============================================================
# CAMERA TRAJECTORY STATISTICS
# ============================================================

def trajectory_statistics(
    positions
):

    start = positions[0]

    end = positions[-1]

    displacement_vector = (
        end - start
    )

    displacement = np.linalg.norm(
        displacement_vector
    )


    if len(positions) > 1:

        frame_motion = np.linalg.norm(
            np.diff(
                positions,
                axis=0
            ),
            axis=1
        )

        path_length = np.sum(
            frame_motion
        )

        average_motion = np.mean(
            frame_motion
        )

        median_motion = np.median(
            frame_motion
        )

    else:

        frame_motion = np.array([])

        path_length = 0.0

        average_motion = 0.0

        median_motion = 0.0


    if path_length > 1e-12:

        directness = (
            displacement
            /
            path_length
        )

    else:

        directness = 0.0


    return (
        start,
        end,
        displacement_vector,
        displacement,
        path_length,
        average_motion,
        median_motion,
        directness,
        frame_motion
    )


# ============================================================
# TASK 4 TRAJECTORY
# ============================================================

task4_trajectory = trajectory_statistics(
    task4_positions
)


# ============================================================
# TASK 5 TRAJECTORY
# ============================================================

task5_trajectory = trajectory_statistics(
    task5_positions
)


# ============================================================
# PRINT TRAJECTORY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CAMERA TRAJECTORY COMPARISON"
)

print(
    "=" * 70
)


print(
    "\nTASK 4 ORIGINAL TRAJECTORY"
)

print(
    "Start:",
    task4_trajectory[0]
)

print(
    "End:",
    task4_trajectory[1]
)

print(
    "Displacement:",
    f"{task4_trajectory[3]:.6f}"
)

print(
    "Path Length:",
    f"{task4_trajectory[4]:.6f}"
)

print(
    "Directness:",
    f"{task4_trajectory[7]:.6f}"
)


print(
    "\nTASK 5 OPTIMIZED TRAJECTORY"
)

print(
    "Start:",
    task5_trajectory[0]
)

print(
    "End:",
    task5_trajectory[1]
)

print(
    "Displacement:",
    f"{task5_trajectory[3]:.6f}"
)

print(
    "Path Length:",
    f"{task5_trajectory[4]:.6f}"
)

print(
    "Directness:",
    f"{task5_trajectory[7]:.6f}"
)


# ============================================================
# CAMERA POSITION DIFFERENCE
# ============================================================

num_compare = min(
    len(task4_positions),
    len(task5_positions)
)


position_difference = (
    task5_positions[:num_compare]
    -
    task4_positions[:num_compare]
)


position_difference_magnitude = (
    np.linalg.norm(
        position_difference,
        axis=1
    )
)


mean_position_difference = (
    np.mean(
        position_difference_magnitude
    )
)

max_position_difference = (
    np.max(
        position_difference_magnitude
    )
)


print(
    "\nCamera position refinement:"
)

print(
    "Mean difference:",
    f"{mean_position_difference:.6f}"
)

print(
    "Maximum difference:",
    f"{max_position_difference:.6f}"
)


# ============================================================
# PLOT 1
# TASK 4 vs TASK 5 MAP
# ============================================================

print(
    "\nGenerating map comparison..."
)


fig = plt.figure(
    figsize=(12, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


ax.scatter(
    task4_map[:, 0],
    task4_map[:, 1],
    task4_map[:, 2],
    s=3,
    alpha=0.35,
    label="Task 4 Global Map"
)


ax.scatter(
    task5_map[:, 0],
    task5_map[:, 1],
    task5_map[:, 2],
    s=3,
    alpha=0.35,
    label="Task 5 Optimized Map"
)


ax.set_xlabel("X")

ax.set_ylabel("Y")

ax.set_zlabel("Z")

ax.set_title(
    "Stage 6 - Task 6\n"
    "Original vs Optimized 3D Map"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task6_map_comparison.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2
# TRAJECTORY COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 8)
)


plt.plot(
    task4_positions[:, 0],
    task4_positions[:, 2],
    marker="o",
    markersize=3,
    label="Task 4"
)


plt.plot(
    task5_positions[:, 0],
    task5_positions[:, 2],
    marker="o",
    markersize=3,
    label="Task 5 Optimized"
)


plt.scatter(
    task4_positions[0, 0],
    task4_positions[0, 2],
    s=100,
    label="Start"
)


plt.xlabel("X")

plt.ylabel("Z")

plt.title(
    "Stage 6 - Task 6\n"
    "Camera Trajectory Before vs After Bundle Adjustment"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.axis("equal")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task6_trajectory_comparison.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 3
# REPROJECTION ERROR DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 7)
)


plt.hist(
    initial_magnitude,
    bins=50,
    alpha=0.6,
    label="Initial"
)


plt.hist(
    final_magnitude,
    bins=50,
    alpha=0.6,
    label="After Bundle Adjustment"
)


plt.xlabel(
    "Reprojection Error (pixels)"
)

plt.ylabel(
    "Number of Observations"
)

plt.title(
    "Stage 6 - Task 6\n"
    "Reprojection Error Distribution"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task6_reprojection_distribution.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 4
# REPROJECTION ERROR PER OBSERVATION
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    initial_magnitude,
    label="Initial"
)


plt.plot(
    final_magnitude,
    label="After Bundle Adjustment"
)


plt.xlabel(
    "Observation"
)

plt.ylabel(
    "Error (pixels)"
)

plt.title(
    "Stage 6 - Task 6\n"
    "Reprojection Error Before vs After"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task6_reprojection_comparison.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 5
# CAMERA POSITION REFINEMENT
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    position_difference_magnitude,
    marker="o",
    markersize=2
)


plt.xlabel(
    "Keyframe"
)

plt.ylabel(
    "Position Change"
)

plt.title(
    "Stage 6 - Task 6\n"
    "Camera Position Change After Bundle Adjustment"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task6_camera_position_change.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FINAL STATISTICS FILE
# ============================================================

statistics_file = os.path.join(
    RESULTS_DIR,
    "task6_final_evaluation.txt"
)


with open(
    statistics_file,
    "w"
) as f:

    f.write(
        "STAGE 6 - TASK 6\n"
    )

    f.write(
        "FINAL RECONSTRUCTION EVALUATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    f.write(
        "DATASET\n"
    )

    f.write(
        f"Total Images : 141\n"
    )

    f.write(
        f"Keyframes : "
        f"{len(task5_positions)}\n"
    )

    f.write(
        f"Task 4 Map Points : "
        f"{len(task4_map)}\n"
    )

    f.write(
        f"Task 5 Map Points : "
        f"{len(task5_map)}\n"
    )


    # --------------------------------------------------------
    # Reprojection
    # --------------------------------------------------------

    f.write(
        "\nREPROJECTION ERROR\n"
    )

    f.write(
        f"Initial RMS : "
        f"{initial_rms:.8f} pixels\n"
    )

    f.write(
        f"Final RMS : "
        f"{final_rms:.8f} pixels\n"
    )

    f.write(
        f"Initial Mean : "
        f"{initial_mean:.8f} pixels\n"
    )

    f.write(
        f"Final Mean : "
        f"{final_mean:.8f} pixels\n"
    )

    f.write(
        f"Initial Median : "
        f"{initial_median:.8f} pixels\n"
    )

    f.write(
        f"Final Median : "
        f"{final_median:.8f} pixels\n"
    )

    f.write(
        f"RMS Improvement : "
        f"{improvement:.4f} %\n"
    )


    # --------------------------------------------------------
    # Task 4 Map
    # --------------------------------------------------------

    f.write(
        "\nTASK 4 MAP\n"
    )

    f.write(
        f"Minimum : "
        f"{task4_min}\n"
    )

    f.write(
        f"Maximum : "
        f"{task4_max}\n"
    )

    f.write(
        f"Mean : "
        f"{task4_mean}\n"
    )

    f.write(
        f"Standard Deviation : "
        f"{task4_std}\n"
    )

    f.write(
        f"Axis Ranges : "
        f"{task4_ranges}\n"
    )


    # --------------------------------------------------------
    # Task 5 Map
    # --------------------------------------------------------

    f.write(
        "\nTASK 5 OPTIMIZED MAP\n"
    )

    f.write(
        f"Minimum : "
        f"{task5_min}\n"
    )

    f.write(
        f"Maximum : "
        f"{task5_max}\n"
    )

    f.write(
        f"Mean : "
        f"{task5_mean}\n"
    )

    f.write(
        f"Standard Deviation : "
        f"{task5_std}\n"
    )

    f.write(
        f"Axis Ranges : "
        f"{task5_ranges}\n"
    )


    # --------------------------------------------------------
    # Task 4 trajectory
    # --------------------------------------------------------

    f.write(
        "\nTASK 4 TRAJECTORY\n"
    )

    f.write(
        f"Start : "
        f"{task4_trajectory[0]}\n"
    )

    f.write(
        f"End : "
        f"{task4_trajectory[1]}\n"
    )

    f.write(
        f"Displacement : "
        f"{task4_trajectory[3]:.8f}\n"
    )

    f.write(
        f"Path Length : "
        f"{task4_trajectory[4]:.8f}\n"
    )

    f.write(
        f"Directness : "
        f"{task4_trajectory[7]:.8f}\n"
    )


    # --------------------------------------------------------
    # Task 5 trajectory
    # --------------------------------------------------------

    f.write(
        "\nTASK 5 OPTIMIZED TRAJECTORY\n"
    )

    f.write(
        f"Start : "
        f"{task5_trajectory[0]}\n"
    )

    f.write(
        f"End : "
        f"{task5_trajectory[1]}\n"
    )

    f.write(
        f"Displacement : "
        f"{task5_trajectory[3]:.8f}\n"
    )

    f.write(
        f"Path Length : "
        f"{task5_trajectory[4]:.8f}\n"
    )

    f.write(
        f"Directness : "
        f"{task5_trajectory[7]:.8f}\n"
    )


    # --------------------------------------------------------
    # Camera refinement
    # --------------------------------------------------------

    f.write(
        "\nCAMERA POSE REFINEMENT\n"
    )

    f.write(
        f"Mean Position Difference : "
        f"{mean_position_difference:.8f}\n"
    )

    f.write(
        f"Maximum Position Difference : "
        f"{max_position_difference:.8f}\n"
    )


# ============================================================
# SAVE OPTIMIZED TRAJECTORY
# ============================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "task6_final_camera_trajectory.npy"
    ),
    task5_positions
)


np.savetxt(
    os.path.join(
        RESULTS_DIR,
        "task6_final_camera_trajectory.txt"
    ),
    task5_positions,
    fmt="%.8f"
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TASK 6 COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nFinal RMS Error:"
)

print(
    f"{final_rms:.6f} pixels"
)

print(
    "\nRMS Improvement:"
)

print(
    f"{improvement:.2f} %"
)

print(
    "\nFinal 3D Map:"
)

print(
    f"{len(task5_map)} points"
)

print(
    "\nFinal Camera Poses:"
)

print(
    f"{len(task5_positions)} keyframes"
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
    "1. task6_map_comparison.png"
)

print(
    "2. task6_trajectory_comparison.png"
)

print(
    "3. task6_reprojection_distribution.png"
)

print(
    "4. task6_reprojection_comparison.png"
)

print(
    "5. task6_camera_position_change.png"
)

print(
    "6. task6_final_evaluation.txt"
)

print(
    "7. task6_final_camera_trajectory.npy"
)

print(
    "8. task6_final_camera_trajectory.txt"
)

print(
    "\nNo plot windows were opened."
)

print(
    "\nStage 6 final evaluation completed successfully."
)