import cv2
import numpy as np
import glob
import os

# =====================================================
# CAMERA CALIBRATION USING A REAL CHESSBOARD
# =====================================================

CHECKERBOARD = (7, 7)
SQUARE_SIZE = 1.0

# -----------------------------------------------------
# Base directories (everything saved inside Stage_3)
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "images")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CORNER_DIR = os.path.join(RESULTS_DIR, "detected_corners")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CORNER_DIR, exist_ok=True)

# -----------------------------------------------------
# Prepare object points
# -----------------------------------------------------

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []

# -----------------------------------------------------
# Load calibration images
# -----------------------------------------------------

images = []
images.extend(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
images.extend(glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")))
images.extend(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
images = sorted(images)

print("=" * 60)
print("CAMERA CALIBRATION")
print("=" * 60)
print(f"\nFound {len(images)} calibration images.\n")

if len(images) == 0:
    raise FileNotFoundError(f"No calibration images found in:\n{IMAGE_DIR}")

# -----------------------------------------------------
# Detect chessboard corners
# -----------------------------------------------------

gray = None

for image_path in images:

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD)

    print(f"{os.path.basename(image_path):25s} -> {ret}")

    if ret:

        objpoints.append(objp)
        imgpoints.append(corners)

        cv2.drawChessboardCorners(
            image,
            CHECKERBOARD,
            corners,
            ret
        )

        save_path = os.path.join(
            CORNER_DIR,
            os.path.basename(image_path)
        )

        cv2.imwrite(save_path, image)

        cv2.imshow("Detected Corners", image)
        cv2.waitKey(400)

cv2.destroyAllWindows()

if len(objpoints) == 0:
    raise RuntimeError("No chessboard corners were detected.")

# -----------------------------------------------------
# Camera calibration
# -----------------------------------------------------

print("\nPerforming Camera Calibration...\n")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

# -----------------------------------------------------
# Results
# -----------------------------------------------------

print("=" * 60)
print("CALIBRATION RESULTS")
print("=" * 60)

print("\nRMS Reprojection Error")
print(ret)

print("\nIntrinsic Matrix (K)")
print(K)

print("\nDistortion Coefficients")
print(dist)

print("\n" + "=" * 60)
print("EXTRINSIC PARAMETERS")
print("=" * 60)

for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs), start=1):

    R, _ = cv2.Rodrigues(rvec)

    print(f"\nImage {i}")
    print("-" * 30)
    print("Rotation Matrix (R)")
    print(R)
    print("\nTranslation Vector (t)")
    print(tvec)

# -----------------------------------------------------
# Mean reprojection error
# -----------------------------------------------------

total_error = 0

for i in range(len(objpoints)):

    projected_points, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        K,
        dist
    )

    error = cv2.norm(
        imgpoints[i],
        projected_points,
        cv2.NORM_L2
    ) / len(projected_points)

    total_error += error

mean_error = total_error / len(objpoints)

print("\n" + "=" * 60)
print("REPROJECTION ERROR")
print("=" * 60)
print(f"\nMean Reprojection Error : {mean_error:.6f}")

# -----------------------------------------------------
# Save calibration
# -----------------------------------------------------

calibration_file = os.path.join(
    RESULTS_DIR,
    "camera_calibration.npz"
)

np.savez(
    calibration_file,
    cameraMatrix=K,
    distortion=dist,
    rotationVectors=rvecs,
    translationVectors=tvecs
)

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(f"\nCalibration file:\n{calibration_file}")
print(f"\nDetected corner images:\n{CORNER_DIR}")

print(f"\nSaved {len(objpoints)} detected corner images.")
print("\nCalibration Completed Successfully!")
