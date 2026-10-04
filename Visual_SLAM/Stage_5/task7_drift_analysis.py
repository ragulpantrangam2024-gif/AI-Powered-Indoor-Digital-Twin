import numpy as np
import os
import matplotlib.pyplot as plt

# =====================================================
# STAGE 5 - TASK 7
# MONOCULAR VISUAL ODOMETRY DRIFT ANALYSIS
# =====================================================

# -----------------------------------------------------
# BASE DIRECTORY
# -----------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

# -----------------------------------------------------
# INPUT
# -----------------------------------------------------

TRAJECTORY_FILE = os.path.join(
    RESULT_DIR,
    "camera_trajectory_corrected.npy"
)

# -----------------------------------------------------
# CHECK INPUT
# -----------------------------------------------------

if not os.path.exists(
    TRAJECTORY_FILE
):

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print(
        "\nCorrected trajectory file not found:"
    )

    print(
        TRAJECTORY_FILE
    )

    print(
        "\nRun Task 5 first."
    )

    exit()

# -----------------------------------------------------
# LOAD TRAJECTORY
# -----------------------------------------------------

trajectory = np.load(
    TRAJECTORY_FILE
)

print("=" * 60)
print("STAGE 5 - TASK 7")
print("MONOCULAR VISUAL ODOMETRY DRIFT ANALYSIS")
print("=" * 60)

print(
    f"\nTrajectory Shape : "
    f"{trajectory.shape}"
)

num_positions = len(
    trajectory
)

print(
    f"Camera Positions : "
    f"{num_positions}"
)

# =====================================================
# 1. FRAME-TO-FRAME MOTION
# =====================================================

motion_vectors = (
    trajectory[1:]
    -
    trajectory[:-1]
)

motion_magnitudes = np.linalg.norm(
    motion_vectors,
    axis=1
)

print("\n")
print("=" * 60)
print("FRAME-TO-FRAME MOTION")
print("=" * 60)

print(
    f"\nMean Motion   : "
    f"{np.mean(motion_magnitudes):.6f}"
)

print(
    f"Std Motion    : "
    f"{np.std(motion_magnitudes):.6f}"
)

print(
    f"Min Motion    : "
    f"{np.min(motion_magnitudes):.6f}"
)

print(
    f"Max Motion    : "
    f"{np.max(motion_magnitudes):.6f}"
)

# =====================================================
# 2. MOTION CONSISTENCY
# =====================================================

motion_mean = np.mean(
    motion_magnitudes
)

motion_std = np.std(
    motion_magnitudes
)

if motion_mean > 0:

    motion_coefficient_variation = (
        motion_std
        /
        motion_mean
    )

else:

    motion_coefficient_variation = 0.0

print("\n")
print("=" * 60)
print("MOTION CONSISTENCY")
print("=" * 60)

print(
    f"\nCoefficient of Variation : "
    f"{motion_coefficient_variation:.6f}"
)

# =====================================================
# 3. CUMULATIVE DISTANCE
# =====================================================

cumulative_distance = np.concatenate(
    (
        [0.0],
        np.cumsum(
            motion_magnitudes
        )
    )
)

# =====================================================
# 4. STRAIGHT-LINE REFERENCE
# =====================================================

start = trajectory[0]

end = trajectory[-1]

overall_vector = (
    end - start
)

overall_distance = np.linalg.norm(
    overall_vector
)

print("\n")
print("=" * 60)
print("OVERALL TRAJECTORY")
print("=" * 60)

print(
    f"\nStart Position : "
    f"{start}"
)

print(
    f"\nEnd Position   : "
    f"{end}"
)

print(
    f"\nOverall Displacement : "
    f"{overall_distance:.6f}"
)

# =====================================================
# 5. DEVIATION FROM STRAIGHT LINE
# =====================================================

# Direction from start to end

if overall_distance > 0:

    overall_direction = (
        overall_vector
        /
        overall_distance
    )

else:

    overall_direction = np.zeros(
        3
    )

straight_line_points = []

deviations = []

for point in trajectory:

    relative_point = (
        point - start
    )

    projection_length = np.dot(
        relative_point,
        overall_direction
    )

    projection_point = (
        start
        +
        projection_length
        *
        overall_direction
    )

    deviation = np.linalg.norm(
        point - projection_point
    )

    straight_line_points.append(
        projection_point
    )

    deviations.append(
        deviation
    )

deviations = np.array(
    deviations
)

straight_line_points = np.array(
    straight_line_points
)

# =====================================================
# 6. DEVIATION STATISTICS
# =====================================================

mean_deviation = np.mean(
    deviations
)

median_deviation = np.median(
    deviations
)

maximum_deviation = np.max(
    deviations
)

final_deviation = deviations[-1]

print("\n")
print("=" * 60)
print("DEVIATION FROM START-END LINE")
print("=" * 60)

print(
    f"\nMean Deviation   : "
    f"{mean_deviation:.6f}"
)

print(
    f"Median Deviation : "
    f"{median_deviation:.6f}"
)

print(
    f"Maximum Deviation: "
    f"{maximum_deviation:.6f}"
)

print(
    f"Final Deviation  : "
    f"{final_deviation:.6f}"
)

# =====================================================
# 7. NORMALIZED DEVIATION
# =====================================================

if overall_distance > 0:

    normalized_deviation = (
        deviations
        /
        overall_distance
    )

else:

    normalized_deviation = np.zeros(
        len(deviations)
    )

mean_normalized_deviation = np.mean(
    normalized_deviation
)

max_normalized_deviation = np.max(
    normalized_deviation
)

print("\n")
print("=" * 60)
print("NORMALIZED DEVIATION")
print("=" * 60)

print(
    f"\nMean Normalized Deviation : "
    f"{mean_normalized_deviation:.6f}"
)

print(
    f"Maximum Normalized Deviation : "
    f"{max_normalized_deviation:.6f}"
)

# =====================================================
# 8. MOTION CHANGE / INSTABILITY
# =====================================================

# Difference between consecutive motion vectors

motion_changes = (
    motion_vectors[1:]
    -
    motion_vectors[:-1]
)

motion_change_magnitude = np.linalg.norm(
    motion_changes,
    axis=1
)

print("\n")
print("=" * 60)
print("MOTION CHANGE / INSTABILITY")
print("=" * 60)

print(
    f"\nMean Motion Change : "
    f"{np.mean(motion_change_magnitude):.6f}"
)

print(
    f"Maximum Motion Change : "
    f"{np.max(motion_change_magnitude):.6f}"
)

# =====================================================
# 9. IDENTIFY HIGH-DRIFT REGIONS
# =====================================================

# Use the 90th percentile as an adaptive threshold

drift_threshold = np.percentile(
    deviations,
    90
)

high_deviation_indices = np.where(
    deviations >= drift_threshold
)[0]

print("\n")
print("=" * 60)
print("HIGH-DEVIATION REGIONS")
print("=" * 60)

print(
    f"\n90th Percentile Threshold : "
    f"{drift_threshold:.6f}"
)

print(
    f"Number of High-Deviation "
    f"Positions : "
    f"{len(high_deviation_indices)}"
)

if len(high_deviation_indices) > 0:

    print(
        "\nFirst High-Deviation Positions:"
    )

    for index in (
        high_deviation_indices[:10]
    ):

        print(
            f"Frame {index:03d} : "
            f"{deviations[index]:.6f}"
        )

# =====================================================
# 10. SAVE NUMERICAL RESULTS
# =====================================================

statistics_file = os.path.join(
    RESULT_DIR,
    "task7_drift_statistics.txt"
)

with open(
    statistics_file,
    "w"
) as file:

    file.write(
        "STAGE 5 - TASK 7\n"
    )

    file.write(
        "MONOCULAR VISUAL ODOMETRY DRIFT ANALYSIS\n"
    )

    file.write(
        "=" * 60
        +
        "\n\n"
    )

    file.write(
        f"Trajectory Shape : "
        f"{trajectory.shape}\n"
    )

    file.write(
        f"Camera Positions : "
        f"{num_positions}\n\n"
    )

    file.write(
        "FRAME-TO-FRAME MOTION\n"
    )

    file.write(
        f"Mean Motion : "
        f"{np.mean(motion_magnitudes):.6f}\n"
    )

    file.write(
        f"Std Motion : "
        f"{np.std(motion_magnitudes):.6f}\n"
    )

    file.write(
        f"Min Motion : "
        f"{np.min(motion_magnitudes):.6f}\n"
    )

    file.write(
        f"Max Motion : "
        f"{np.max(motion_magnitudes):.6f}\n\n"
    )

    file.write(
        "MOTION CONSISTENCY\n"
    )

    file.write(
        f"Coefficient of Variation : "
        f"{motion_coefficient_variation:.6f}\n\n"
    )

    file.write(
        "OVERALL TRAJECTORY\n"
    )

    file.write(
        f"Overall Displacement : "
        f"{overall_distance:.6f}\n\n"
    )

    file.write(
        "DEVIATION FROM STRAIGHT LINE\n"
    )

    file.write(
        f"Mean Deviation : "
        f"{mean_deviation:.6f}\n"
    )

    file.write(
        f"Median Deviation : "
        f"{median_deviation:.6f}\n"
    )

    file.write(
        f"Maximum Deviation : "
        f"{maximum_deviation:.6f}\n"
    )

    file.write(
        f"Final Deviation : "
        f"{final_deviation:.6f}\n\n"
    )

    file.write(
        "NORMALIZED DEVIATION\n"
    )

    file.write(
        f"Mean Normalized Deviation : "
        f"{mean_normalized_deviation:.6f}\n"
    )

    file.write(
        f"Maximum Normalized Deviation : "
        f"{max_normalized_deviation:.6f}\n\n"
    )

    file.write(
        "MOTION CHANGE / INSTABILITY\n"
    )

    file.write(
        f"Mean Motion Change : "
        f"{np.mean(motion_change_magnitude):.6f}\n"
    )

    file.write(
        f"Maximum Motion Change : "
        f"{np.max(motion_change_magnitude):.6f}\n\n"
    )

    file.write(
        "HIGH-DEVIATION REGION\n"
    )

    file.write(
        f"90th Percentile Threshold : "
        f"{drift_threshold:.6f}\n"
    )

    file.write(
        f"High-Deviation Positions : "
        f"{len(high_deviation_indices)}\n\n"
    )

    file.write(
        "IMPORTANT LIMITATION\n"
    )

    file.write(
        "No ground-truth trajectory is available "
        "for this video.\n"
    )

    file.write(
        "Therefore true ATE/RPE cannot be computed.\n"
    )

    file.write(
        "The reported values represent internal "
        "trajectory instability and deviation.\n"
    )

    file.write(
        "Translation scale is unknown because "
        "the system is monocular.\n"
    )

print(
    "\nStatistics saved to:"
)

print(
    "results/task7_drift_statistics.txt"
)

# =====================================================
# PLOT 1
# FRAME-TO-FRAME MOTION
# =====================================================

frames_motion = np.arange(
    len(motion_magnitudes)
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    frames_motion,
    motion_magnitudes,
    linewidth=1.5
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Motion Magnitude"
)

plt.title(
    "Frame-to-Frame Camera Motion"
)

plt.grid(
    True
)

plt.tight_layout()

motion_plot = os.path.join(
    RESULT_DIR,
    "task7_frame_motion.png"
)

plt.savefig(
    motion_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# PLOT 2
# DEVIATION FROM STRAIGHT LINE
# =====================================================

frames_deviation = np.arange(
    len(deviations)
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    frames_deviation,
    deviations,
    linewidth=2
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Deviation"
)

plt.title(
    "Trajectory Deviation from Start-End Direction"
)

plt.grid(
    True
)

plt.tight_layout()

deviation_plot = os.path.join(
    RESULT_DIR,
    "task7_trajectory_deviation.png"
)

plt.savefig(
    deviation_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# PLOT 3
# CUMULATIVE DISTANCE
# =====================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    np.arange(
        len(cumulative_distance)
    ),
    cumulative_distance,
    linewidth=2
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Cumulative Distance"
)

plt.title(
    "Cumulative Camera Motion"
)

plt.grid(
    True
)

plt.tight_layout()

cumulative_plot = os.path.join(
    RESULT_DIR,
    "task7_cumulative_distance.png"
)

plt.savefig(
    cumulative_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# PLOT 4
# TRAJECTORY WITH STRAIGHT REFERENCE
# =====================================================

fig = plt.figure(
    figsize=(10, 7)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    trajectory[:, 2],
    linewidth=2,
    label="Estimated Trajectory"
)

ax.plot(
    straight_line_points[:, 0],
    straight_line_points[:, 1],
    straight_line_points[:, 2],
    linestyle="--",
    label="Start-End Reference"
)

ax.scatter(
    trajectory[0, 0],
    trajectory[0, 1],
    trajectory[0, 2],
    s=80,
    label="Start"
)

ax.scatter(
    trajectory[-1, 0],
    trajectory[-1, 1],
    trajectory[-1, 2],
    s=80,
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
    "Trajectory Drift / Deviation Analysis"
)

ax.legend()

plt.tight_layout()

trajectory_drift_plot = os.path.join(
    RESULT_DIR,
    "task7_trajectory_drift_3D.png"
)

plt.savefig(
    trajectory_drift_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# FINAL
# =====================================================

print("\n")
print("=" * 60)
print("TASK 7 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nResults saved in:")

print(
    "results/task7_drift_statistics.txt"
)

print(
    "results/task7_frame_motion.png"
)

print(
    "results/task7_trajectory_deviation.png"
)

print(
    "results/task7_cumulative_distance.png"
)

print(
    "results/task7_trajectory_drift_3D.png"
)

print(
    "\nNo plot windows were opened."
)

print(
    "\nTask 7 Completed Successfully!"
)