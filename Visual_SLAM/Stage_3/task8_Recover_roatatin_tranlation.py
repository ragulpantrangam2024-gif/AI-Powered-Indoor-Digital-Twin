import cv2
import numpy as np
import os

# =====================================================
# TASK 8 - RECOVER CAMERA POSE
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "images")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------------------------------
# Load Images
# -----------------------------------------------------

img1 = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_3\images\Image1.jpeg")
img2 = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_3\images\Image2.jpeg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# -----------------------------------------------------
# Load Camera Calibration
# -----------------------------------------------------

calib = np.load(os.path.join(RESULT_DIR, "camera_calibration.npz"))

K = calib["cameraMatrix"]
dist = calib["distortion"]

# -----------------------------------------------------
# ORB Feature Detection
# -----------------------------------------------------

orb = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# -----------------------------------------------------
# BF Matcher
# -----------------------------------------------------

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# -----------------------------------------------------
# Extract Points
# -----------------------------------------------------

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

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
# Recover Camera Pose
# -----------------------------------------------------

num_inliers, R, t, pose_mask = cv2.recoverPose(
    E,
    pts1,
    pts2,
    K
)

# -----------------------------------------------------
# Collect Pose Inlier Matches
# -----------------------------------------------------

pose_matches = []

for i, m in enumerate(matches):
    if pose_mask[i]:
        pose_matches.append(m)

# -----------------------------------------------------
# Draw Pose Inliers
# -----------------------------------------------------

pose_img = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    pose_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# -----------------------------------------------------
# Save Outputs
# -----------------------------------------------------

cv2.imwrite(
    os.path.join(RESULT_DIR, "pose_inlier_matches.jpg"),
    pose_img
)

np.savetxt(
    os.path.join(RESULT_DIR, "rotation_matrix.txt"),
    R,
    fmt="%.8f"
)

np.savetxt(
    os.path.join(RESULT_DIR, "translation_vector.txt"),
    t,
    fmt="%.8f"
)

with open(os.path.join(RESULT_DIR, "pose_statistics.txt"), "w") as f:

    f.write("Recover Camera Pose\n")
    f.write("="*40 + "\n\n")

    f.write(f"Total Matches : {len(matches)}\n")
    f.write(f"Pose Inliers  : {num_inliers}\n\n")

    f.write("Rotation Matrix\n")
    f.write(np.array2string(R))

    f.write("\n\nTranslation Vector\n")
    f.write(np.array2string(t))

# -----------------------------------------------------
# Console
# -----------------------------------------------------

print("="*60)
print("RECOVER CAMERA POSE")
print("="*60)

print("\nRecovered Pose Inliers :", num_inliers)

print("\nRotation Matrix\n")
print(R)

print("\nTranslation Vector\n")
print(t)

print("\nResults saved to Stage_3/results/")