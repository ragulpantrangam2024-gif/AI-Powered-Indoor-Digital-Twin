import cv2
import numpy as np
import os
import glob

# =====================================================
# TASK 3.5 - TRIANGULATION USING POSE INLIERS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "images")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------------------------------
# Load Images
# -----------------------------------------------------

image_files = []

image_files.extend(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
image_files.extend(glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")))
image_files.extend(glob.glob(os.path.join(IMAGE_DIR, "*.png")))

image_files = sorted(image_files)

if len(image_files) < 2:
    raise ValueError("At least two images are required.")

img1 = cv2.imread(image_files[0])
img2 = cv2.imread(image_files[1])

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# -----------------------------------------------------
# Load Camera Calibration
# -----------------------------------------------------

calib = np.load(
    os.path.join(
        BASE_DIR,
        "..",
        "Stage_3",
        "results",
        "camera_calibration.npz"
    )
)

K = calib["cameraMatrix"]

# -----------------------------------------------------
# ORB Feature Detection
# -----------------------------------------------------

orb = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# -----------------------------------------------------
# Feature Matching
# -----------------------------------------------------

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)

print("Total BF Matches :", len(matches))

# -----------------------------------------------------
# Extract Matched Points
# -----------------------------------------------------

pts1 = np.float32(
    [kp1[m.queryIdx].pt for m in matches]
)

pts2 = np.float32(
    [kp2[m.trainIdx].pt for m in matches]
)

# -----------------------------------------------------
# Essential Matrix
# -----------------------------------------------------

E, mask = cv2.findEssentialMat(
    pts1,
    pts2,
    K,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0
)

# -----------------------------------------------------
# Recover Pose
# -----------------------------------------------------

num_inliers, R, t, pose_mask = cv2.recoverPose(
    E,
    pts1,
    pts2,
    K
)

print("Recover Pose Inliers :", num_inliers)

# -----------------------------------------------------
# Keep Only Pose Inliers
# -----------------------------------------------------

pose_matches = []

pts1_inliers = []
pts2_inliers = []

for i, m in enumerate(matches):

    if pose_mask[i]:

        pose_matches.append(m)

        pts1_inliers.append(kp1[m.queryIdx].pt)

        pts2_inliers.append(kp2[m.trainIdx].pt)

pts1_inliers = np.float32(pts1_inliers).T
pts2_inliers = np.float32(pts2_inliers).T

print("Triangulated Matches :", pts1_inliers.shape[1])

# -----------------------------------------------------
# Projection Matrices
# -----------------------------------------------------

P1 = np.loadtxt(
    os.path.join(
        RESULT_DIR,
        "projection_matrix_camera1.txt"
    )
)

P2 = np.loadtxt(
    os.path.join(
        RESULT_DIR,
        "projection_matrix_camera2.txt"
    )
)

# -----------------------------------------------------
# Triangulation
# -----------------------------------------------------

points4D = cv2.triangulatePoints(
    P1,
    P2,
    pts1_inliers,
    pts2_inliers
)

# -----------------------------------------------------
# Convert to Cartesian Coordinates
# -----------------------------------------------------

points3D = points4D[:3] / points4D[3]

points3D = points3D.T

# -----------------------------------------------------
# Remove Invalid Points
# -----------------------------------------------------

valid = np.isfinite(points3D).all(axis=1)

points3D = points3D[valid]

# -----------------------------------------------------
# Save Results
# -----------------------------------------------------

np.savetxt(
    os.path.join(
        RESULT_DIR,
        "points3D_pose_inliers.txt"
    ),
    points3D,
    fmt="%.6f"
)

# -----------------------------------------------------
# Draw Inlier Matches
# -----------------------------------------------------

match_img = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    pose_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imwrite(
    os.path.join(
        RESULT_DIR,
        "pose_inlier_matches.jpg"
    ),
    match_img
)

# -----------------------------------------------------
# Statistics
# -----------------------------------------------------

print("\n")
print("=" * 60)
print("TRIANGULATION USING POSE INLIERS")
print("=" * 60)

print("\nTotal BF Matches :", len(matches))
print("Pose Inliers     :", num_inliers)
print("3D Points        :", len(points3D))

print("\nPoint Cloud Shape")

print(points3D.shape)

print("\nFirst Five Points\n")

print(points3D[:5])

print("\nSaved Files")

print("results/points3D_pose_inliers.txt")
print("results/pose_inlier_matches.jpg")