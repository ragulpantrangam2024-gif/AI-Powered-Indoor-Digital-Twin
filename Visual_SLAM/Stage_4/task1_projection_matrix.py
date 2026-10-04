import numpy as np
import os

# =====================================================
# TASK 1 - CAMERA PROJECTION MATRICES
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stage 3 results folder
STAGE3_RESULTS = os.path.join(BASE_DIR, "..", "Stage_3", "results")

# -----------------------------------------------------
# Load Camera Calibration
# -----------------------------------------------------

calib = np.load(
    os.path.join(STAGE3_RESULTS, "camera_calibration.npz")
)

K = calib["cameraMatrix"]

# -----------------------------------------------------
# Load Rotation and Translation
# -----------------------------------------------------

R = np.loadtxt(
    os.path.join(STAGE3_RESULTS, "rotation_matrix.txt")
)

t = np.loadtxt(
    os.path.join(STAGE3_RESULTS, "translation_vector.txt")
)

t = t.reshape(3,1)

# -----------------------------------------------------
# Camera 1 Projection Matrix
# -----------------------------------------------------

R1 = np.eye(3)

t1 = np.zeros((3,1))

P1 = K @ np.hstack((R1,t1))

# -----------------------------------------------------
# Camera 2 Projection Matrix
# -----------------------------------------------------

P2 = K @ np.hstack((R,t))

# -----------------------------------------------------
# Save Results
# -----------------------------------------------------

RESULT_DIR = os.path.join(BASE_DIR,"results")

os.makedirs(RESULT_DIR,exist_ok=True)

np.savetxt(
    os.path.join(RESULT_DIR,"projection_matrix_camera1.txt"),
    P1,
    fmt="%.8f"
)

np.savetxt(
    os.path.join(RESULT_DIR,"projection_matrix_camera2.txt"),
    P2,
    fmt="%.8f"
)

# -----------------------------------------------------
# Console Output
# -----------------------------------------------------

print("="*60)
print("CAMERA PROJECTION MATRICES")
print("="*60)

print("\nIntrinsic Matrix (K)")
print(K)

print("\nProjection Matrix P1")
print(P1)

print("\nProjection Matrix P2")
print(P2)

print("\nSaved Successfully!")

print("results/projection_matrix_camera1.txt")
print("results/projection_matrix_camera2.txt")