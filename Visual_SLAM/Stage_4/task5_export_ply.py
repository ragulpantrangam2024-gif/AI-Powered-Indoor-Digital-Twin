import numpy as np
import os

# =====================================================
# TASK 5 - EXPORT POINT CLOUD TO PLY
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

# -----------------------------------------------------
# Load 3D Points
# -----------------------------------------------------

points = np.loadtxt(
    os.path.join(
        RESULT_DIR,
        "points3D_pose_inliers.txt"
    )
)

print("=" * 60)
print("EXPORT POINT CLOUD")
print("=" * 60)

print(f"\nLoaded {len(points)} points.")

# -----------------------------------------------------
# Output File
# -----------------------------------------------------

ply_file = os.path.join(
    RESULT_DIR,
    "point_cloud.ply"
)

# -----------------------------------------------------
# Write PLY Header
# -----------------------------------------------------

with open(ply_file, "w") as f:

    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {len(points)}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")

    # ---------------------------------------------
    # Write Points
    # ---------------------------------------------

    for p in points:

        f.write(f"{p[0]} {p[1]} {p[2]}\n")

print("\nPoint cloud exported successfully!")

print("\nSaved File:")
print("results/point_cloud.ply")