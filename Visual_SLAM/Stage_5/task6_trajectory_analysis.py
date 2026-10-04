import numpy as np
import os
import matplotlib.pyplot as plt

# =====================================================
# STAGE 5 - TASK 6
# CORRECTED CAMERA TRAJECTORY ANALYSIS
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
# INPUT FILE
# -----------------------------------------------------

TRAJECTORY_FILE = os.path.join(
    RESULT_DIR,
    "camera_trajectory_corrected.npy"
)

# -----------------------------------------------------
# CHECK INPUT FILE
# -----------------------------------------------------

if not os.path.exists(TRAJECTORY_FILE):

    print("=" * 60)
    print("ERROR")
    print("=" * 60)

    print("\nCorrected trajectory file not found:")

    print(
        TRAJECTORY_FILE
    )

    print(
        "\nPlease run Task 5 first."
    )

    exit()

# -----------------------------------------------------
# LOAD TRAJECTORY
# -----------------------------------------------------

trajectory = np.load(
    TRAJECTORY_FILE
)

print("=" * 60)
print("STAGE 5 - TASK 6")
print("CORRECTED CAMERA TRAJECTORY ANALYSIS")
print("=" * 60)

print(
    f"\nTrajectory Shape : "
    f"{trajectory.shape}"
)

# -----------------------------------------------------
# NUMBER OF POSITIONS
# -----------------------------------------------------

num_positions = len(
    trajectory
)

print(
    f"Number of Camera Positions : "
    f"{num_positions}"
)

# -----------------------------------------------------
# INITIAL POSITION
# -----------------------------------------------------

start_position = trajectory[0]

print("\nInitial Position")
print("----------------")

print(
    start_position
)

# -----------------------------------------------------
# FINAL POSITION
# -----------------------------------------------------

end_position = trajectory[-1]

print("\nFinal Position")
print("--------------")

print(
    end_position
)

# -----------------------------------------------------
# START-TO-END DISPLACEMENT
# -----------------------------------------------------

displacement_vector = (
    end_position
    -
    start_position
)

displacement_magnitude = np.linalg.norm(
    displacement_vector
)

print("\nStart-to-End Displacement")
print("-------------------------")

print(
    "Displacement Vector:"
)

print(
    displacement_vector
)

print(
    f"\nDisplacement Magnitude "
    f"(arbitrary scale): "
    f"{displacement_magnitude:.6f}"
)

# -----------------------------------------------------
# FRAME-TO-FRAME DIFFERENCES
# -----------------------------------------------------

frame_differences = (
    trajectory[1:]
    -
    trajectory[:-1]
)

# -----------------------------------------------------
# FRAME-TO-FRAME DISTANCES
# -----------------------------------------------------

frame_distances = np.linalg.norm(
    frame_differences,
    axis=1
)

# -----------------------------------------------------
# TOTAL PATH LENGTH
# -----------------------------------------------------

total_path_length = np.sum(
    frame_distances
)

print("\nTotal Estimated Path Length")
print("---------------------------")

print(
    f"{total_path_length:.6f} "
    "(arbitrary scale)"
)

# -----------------------------------------------------
# FRAME-TO-FRAME MOTION
# -----------------------------------------------------

average_motion = np.mean(
    frame_distances
)

median_motion = np.median(
    frame_distances
)

minimum_motion = np.min(
    frame_distances
)

maximum_motion = np.max(
    frame_distances
)

print("\nFrame-to-Frame Motion")
print("---------------------")

print(
    f"Average : {average_motion:.6f}"
)

print(
    f"Median  : {median_motion:.6f}"
)

print(
    f"Minimum : {minimum_motion:.6f}"
)

print(
    f"Maximum : {maximum_motion:.6f}"
)

# -----------------------------------------------------
# X / Y / Z
# -----------------------------------------------------

x_values = trajectory[:, 0]

y_values = trajectory[:, 1]

z_values = trajectory[:, 2]

# -----------------------------------------------------
# COORDINATE RANGES
# -----------------------------------------------------

x_min = np.min(x_values)
x_max = np.max(x_values)

y_min = np.min(y_values)
y_max = np.max(y_values)

z_min = np.min(z_values)
z_max = np.max(z_values)

print("\nTrajectory Coordinate Ranges")
print("----------------------------")

print(
    f"X Range : "
    f"{x_min:.6f} "
    f"to "
    f"{x_max:.6f}"
)

print(
    f"Y Range : "
    f"{y_min:.6f} "
    f"to "
    f"{y_max:.6f}"
)

print(
    f"Z Range : "
    f"{z_min:.6f} "
    f"to "
    f"{z_max:.6f}"
)

# -----------------------------------------------------
# AXIS RANGES
# -----------------------------------------------------

x_range = (
    x_max - x_min
)

y_range = (
    y_max - y_min
)

z_range = (
    z_max - z_min
)

print("\nAxis Ranges")
print("-----------")

print(
    f"X : {x_range:.6f}"
)

print(
    f"Y : {y_range:.6f}"
)

print(
    f"Z : {z_range:.6f}"
)

# -----------------------------------------------------
# TRAJECTORY DIRECTNESS
# -----------------------------------------------------

if total_path_length > 0:

    trajectory_directness = (
        displacement_magnitude
        /
        total_path_length
    )

else:

    trajectory_directness = 0.0

print("\nTrajectory Directness")
print("---------------------")

print(
    f"{trajectory_directness:.6f}"
)

# =====================================================
# SAVE NUMERICAL STATISTICS
# =====================================================

statistics_file = os.path.join(
    RESULT_DIR,
    "task6_corrected_statistics.txt"
)

with open(
    statistics_file,
    "w"
) as file:

    file.write(
        "STAGE 5 - TASK 6\n"
    )

    file.write(
        "CORRECTED CAMERA TRAJECTORY ANALYSIS\n"
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
        f"Number of Camera Positions : "
        f"{num_positions}\n\n"
    )

    file.write(
        "Initial Position\n"
    )

    file.write(
        str(start_position)
        +
        "\n\n"
    )

    file.write(
        "Final Position\n"
    )

    file.write(
        str(end_position)
        +
        "\n\n"
    )

    file.write(
        "Start-to-End Displacement\n"
    )

    file.write(
        "Displacement Vector:\n"
    )

    file.write(
        str(displacement_vector)
        +
        "\n"
    )

    file.write(
        f"Displacement Magnitude "
        f"(arbitrary scale): "
        f"{displacement_magnitude:.6f}\n\n"
    )

    file.write(
        "Total Estimated Path Length\n"
    )

    file.write(
        f"{total_path_length:.6f} "
        f"(arbitrary scale)\n\n"
    )

    file.write(
        "Frame-to-Frame Motion\n"
    )

    file.write(
        f"Average : "
        f"{average_motion:.6f}\n"
    )

    file.write(
        f"Median  : "
        f"{median_motion:.6f}\n"
    )

    file.write(
        f"Minimum : "
        f"{minimum_motion:.6f}\n"
    )

    file.write(
        f"Maximum : "
        f"{maximum_motion:.6f}\n\n"
    )

    file.write(
        "Trajectory Coordinate Ranges\n"
    )

    file.write(
        f"X Range : "
        f"{x_min:.6f} "
        f"to "
        f"{x_max:.6f}\n"
    )

    file.write(
        f"Y Range : "
        f"{y_min:.6f} "
        f"to "
        f"{y_max:.6f}\n"
    )

    file.write(
        f"Z Range : "
        f"{z_min:.6f} "
        f"to "
        f"{z_max:.6f}\n\n"
    )

    file.write(
        "Axis Ranges\n"
    )

    file.write(
        f"X : {x_range:.6f}\n"
    )

    file.write(
        f"Y : {y_range:.6f}\n"
    )

    file.write(
        f"Z : {z_range:.6f}\n\n"
    )

    file.write(
        f"Trajectory Directness : "
        f"{trajectory_directness:.6f}\n\n"
    )

    file.write(
        "NOTE:\n"
    )

    file.write(
        "The trajectory is from monocular Visual "
        "Odometry.\n"
    )

    file.write(
        "Translation scale is unknown.\n"
    )

    file.write(
        "Therefore, distances are expressed in "
        "arbitrary scale units.\n"
    )

print(
    "\nStatistics saved to:"
)

print(
    "results/task6_corrected_statistics.txt"
)

# =====================================================
# PLOT 1
# TOP VIEW TRAJECTORY
# =====================================================

plt.figure(
    figsize=(9, 7)
)

plt.plot(
    x_values,
    y_values,
    linewidth=2
)

plt.scatter(
    x_values[0],
    y_values[0],
    s=80,
    label="Start"
)

plt.scatter(
    x_values[-1],
    y_values[-1],
    s=80,
    label="End"
)

plt.xlabel(
    "X Position"
)

plt.ylabel(
    "Y Position"
)

plt.title(
    "Corrected Camera Trajectory - Top View"
)

plt.grid(
    True
)

plt.axis(
    "equal"
)

plt.legend()

plt.tight_layout()

top_view_file = os.path.join(
    RESULT_DIR,
    "task6_corrected_top_view.png"
)

plt.savefig(
    top_view_file,
    dpi=300,
    bbox_inches="tight"
)

# IMPORTANT:
# Close instead of plt.show()
# so the script does not block.

plt.close()

print(
    "\nTop-view trajectory saved to:"
)

print(
    "results/task6_corrected_top_view.png"
)

# =====================================================
# PLOT 2
# X / Y / Z POSITION VS FRAME
# =====================================================

frames = np.arange(
    num_positions
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    frames,
    x_values,
    label="X"
)

plt.plot(
    frames,
    y_values,
    label="Y"
)

plt.plot(
    frames,
    z_values,
    label="Z"
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Position (arbitrary scale)"
)

plt.title(
    "Camera Position vs Frame"
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()

position_file = os.path.join(
    RESULT_DIR,
    "task6_corrected_position_vs_frame.png"
)

plt.savefig(
    position_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Position-vs-frame plot saved to:"
)

print(
    "results/task6_corrected_position_vs_frame.png"
)

# =====================================================
# PLOT 3
# CUMULATIVE PATH LENGTH
# =====================================================

cumulative_distance = np.concatenate(
    (
        [0.0],
        np.cumsum(
            frame_distances
        )
    )
)

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    frames,
    cumulative_distance,
    linewidth=2
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Cumulative Distance "
    "(arbitrary scale)"
)

plt.title(
    "Cumulative Camera Motion"
)

plt.grid(
    True
)

plt.tight_layout()

cumulative_file = os.path.join(
    RESULT_DIR,
    "task6_corrected_cumulative_distance.png"
)

plt.savefig(
    cumulative_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Cumulative-distance plot saved to:"
)

print(
    "results/task6_corrected_cumulative_distance.png"
)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n")
print("=" * 60)
print("TASK 6 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nAll calculations completed."
)

print(
    "\nAll plots were saved directly to the"
)

print(
    "results folder."
)

print(
    "\nNo plot windows were opened."
)

print(
    "\nResults:"
)

print(
    "1. task6_corrected_statistics.txt"
)

print(
    "2. task6_corrected_top_view.png"
)

print(
    "3. task6_corrected_position_vs_frame.png"
)

print(
    "4. task6_corrected_cumulative_distance.png"
)

print(
    "\nTask 6 Completed Successfully!"
)