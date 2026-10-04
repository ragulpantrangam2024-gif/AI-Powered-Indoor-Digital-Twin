import numpy as np
import matplotlib.pyplot as plt
import os

# =====================================================
# TASK 4 - SPARSE POINT CLOUD VISUALIZATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

# -----------------------------------------------------
# Load 3D Points
# -----------------------------------------------------

points3D = np.loadtxt(
    os.path.join(
        RESULT_DIR,
        "points3D_pose_inliers.txt"
    )
)

print("=" * 60)
print("SPARSE POINT CLOUD VISUALIZATION")
print("=" * 60)

print("\nNumber of Points :", len(points3D))

# -----------------------------------------------------
# Remove Extreme Outliers
# -----------------------------------------------------

distance = np.linalg.norm(points3D, axis=1)

threshold = np.percentile(distance, 95)

points3D = points3D[distance < threshold]

print("Points after filtering :", len(points3D))

# -----------------------------------------------------
# Split Coordinates
# -----------------------------------------------------

X = points3D[:,0]
Y = points3D[:,1]
Z = points3D[:,2]

# -----------------------------------------------------
# Plot
# -----------------------------------------------------

fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    X,
    Y,
    Z,
    s=8
)

ax.set_title("Sparse 3D Point Cloud")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()

# -----------------------------------------------------
# Save Figure
# -----------------------------------------------------

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "sparse_point_cloud.png"
    ),
    dpi=300
)

plt.show()

print("\nSaved Successfully!")

print("results/sparse_point_cloud.png")