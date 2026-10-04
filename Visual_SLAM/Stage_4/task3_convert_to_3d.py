import numpy as np
import os

# =====================================================
# TASK 3 - HOMOGENEOUS TO CARTESIAN COORDINATES
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RESULT_DIR = os.path.join(BASE_DIR, "results")

# -----------------------------------------------------
# Load Homogeneous Points
# -----------------------------------------------------

points4D = np.loadtxt(
    os.path.join(RESULT_DIR, "points4D.txt")
)

# -----------------------------------------------------
# Convert to Cartesian Coordinates
# -----------------------------------------------------

points3D = points4D[:3] / points4D[3]

# Shape becomes (3,N)

# Convert to (N,3)

points3D = points3D.T

# -----------------------------------------------------
# Remove Invalid Points
# -----------------------------------------------------

mask = np.isfinite(points3D).all(axis=1)

points3D = points3D[mask]

# -----------------------------------------------------
# Save
# -----------------------------------------------------

np.savetxt(
    os.path.join(
        RESULT_DIR,
        "points3D.txt"
    ),
    points3D,
    fmt="%.6f"
)

# -----------------------------------------------------
# Console
# -----------------------------------------------------

print("=" * 60)
print("3D POINT CONVERSION")
print("=" * 60)

print("\nNumber of 3D Points")

print(len(points3D))

print("\nShape")

print(points3D.shape)

print("\nFirst Five 3D Points\n")

print(points3D[:5])

print("\nSaved Successfully!")

print("results/points3D.txt")