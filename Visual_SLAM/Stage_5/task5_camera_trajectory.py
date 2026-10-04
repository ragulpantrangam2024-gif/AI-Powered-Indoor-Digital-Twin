import numpy as np
import os
import matplotlib.pyplot as plt

# =====================================================
# STAGE 5 - TASK 5
# CORRECTED CAMERA TRAJECTORY ESTIMATION
# =====================================================

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
# INPUT FILES
# -----------------------------------------------------

ROTATION_FILE = os.path.join(
    RESULT_DIR,
    "rotation_matrices.npy"
)

TRANSLATION_FILE = os.path.join(
    RESULT_DIR,
    "translation_vectors.npy"
)

# -----------------------------------------------------
# CHECK FILES
# -----------------------------------------------------

if not os.path.exists(ROTATION_FILE):

    print("ERROR: Rotation matrices not found!")
    print(ROTATION_FILE)
    exit()

if not os.path.exists(TRANSLATION_FILE):

    print("ERROR: Translation vectors not found!")
    print(TRANSLATION_FILE)
    exit()

# -----------------------------------------------------
# LOAD RELATIVE POSES
# -----------------------------------------------------

R_all = np.load(
    ROTATION_FILE
)

t_all = np.load(
    TRANSLATION_FILE
)

print("=" * 60)
print("STAGE 5 - TASK 5")
print("CORRECTED CAMERA TRAJECTORY")
print("=" * 60)

print(
    f"\nRotation Matrices    : {len(R_all)}"
)

print(
    f"Translation Vectors : {len(t_all)}"
)

# -----------------------------------------------------
# INITIAL CAMERA POSE
#
# World coordinate system = first camera
#
# R_wc = rotation from camera coordinates
#        into world coordinates
#
# C_wc = camera position in world coordinates
# -----------------------------------------------------

R_wc = np.eye(3)

C_wc = np.zeros(
    (3, 1)
)

trajectory = []

trajectory.append(
    C_wc.flatten()
)

# -----------------------------------------------------
# ACCUMULATE POSES
# -----------------------------------------------------

for i in range(
    len(R_all)
):

    # Relative transformation:
    #
    # X_next = R_rel * X_current + t_rel
    #
    # This transformation describes the new
    # camera coordinate system relative to
    # the previous camera.

    R_rel = R_all[i]

    t_rel = t_all[i].reshape(
        3, 1
    )

    # -------------------------------------------------
    # Convert relative camera motion into camera
    # center movement in the world coordinate system.
    #
    # Camera center:
    #
    # C = -R^T t
    # -------------------------------------------------

    R_step = R_rel.T

    t_step = -R_rel.T @ t_rel

    # -------------------------------------------------
    # Transform the relative movement into the
    # current world coordinate system.
    # -------------------------------------------------

    C_wc = (
        C_wc
        +
        R_wc @ t_step
    )

    # -------------------------------------------------
    # Update global camera orientation
    # -------------------------------------------------

    R_wc = (
        R_wc @ R_step
    )

    # -------------------------------------------------
    # Save camera position
    # -------------------------------------------------

    trajectory.append(
        C_wc.flatten()
    )

# -----------------------------------------------------
# CONVERT TO ARRAY
# -----------------------------------------------------

trajectory = np.array(
    trajectory
)

print("\nTrajectory Shape")
print("----------------")

print(
    trajectory.shape
)

# -----------------------------------------------------
# INITIAL POSITION
# -----------------------------------------------------

print("\nInitial Camera Position")
print("----------------------")

print(
    trajectory[0]
)

# -----------------------------------------------------
# FINAL POSITION
# -----------------------------------------------------

print("\nFinal Camera Position")
print("--------------------")

print(
    trajectory[-1]
)

# -----------------------------------------------------
# SAVE TRAJECTORY
# -----------------------------------------------------

trajectory_npy = os.path.join(
    RESULT_DIR,
    "camera_trajectory_corrected.npy"
)

np.save(
    trajectory_npy,
    trajectory
)

# -----------------------------------------------------
# SAVE TXT
# -----------------------------------------------------

trajectory_txt = os.path.join(
    RESULT_DIR,
    "camera_trajectory_corrected.txt"
)

np.savetxt(
    trajectory_txt,
    trajectory,
    fmt="%.6f",
    header="X Y Z"
)

# -----------------------------------------------------
# 3D TRAJECTORY
# -----------------------------------------------------

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
    linewidth=2
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

ax.set_title(
    "Corrected Monocular Camera Trajectory"
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

ax.legend()

plt.tight_layout()

plot_file = os.path.join(
    RESULT_DIR,
    "camera_trajectory_corrected_3D.png"
)

plt.savefig(
    plot_file,
    dpi=300
)

plt.show()

# -----------------------------------------------------
# FINAL OUTPUT
# -----------------------------------------------------

print("\n")
print("=" * 60)
print("TASK 5 COMPLETED")
print("=" * 60)

print("\nResults saved in:")

print(
    "results/camera_trajectory_corrected.npy"
)

print(
    "results/camera_trajectory_corrected.txt"
)

print(
    "results/camera_trajectory_corrected_3D.png"
)

print(
    "\nImportant:"
)

print(
    "Translation scale is unknown because this is"
)

print(
    "monocular Visual Odometry."
)