import cv2
import numpy as np
import os

# =====================================================
# TASK 7 - ESSENTIAL MATRIX ESTIMATION
# =====================================================

# Base Directory
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

print("=" * 60)
print("CAMERA INTRINSIC MATRIX")
print("=" * 60)
print(K)

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

print("\nTotal Raw Matches :", len(matches))

# -----------------------------------------------------
# Extract Matched Points
# -----------------------------------------------------

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# -----------------------------------------------------
# Compute Essential Matrix
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
# Keep Inliers
# -----------------------------------------------------

inlier_matches = []

for i, m in enumerate(matches):
    if mask[i]:
        inlier_matches.append(m)

# -----------------------------------------------------
# Draw Inlier Matches
# -----------------------------------------------------

img_matches = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# -----------------------------------------------------
# Save Outputs
# -----------------------------------------------------

cv2.imwrite(
    os.path.join(RESULT_DIR, "essential_inlier_matches.jpg"),
    img_matches
)

np.savetxt(
    os.path.join(RESULT_DIR, "essential_matrix.txt"),
    E,
    fmt="%.8f"
)

with open(os.path.join(RESULT_DIR, "essential_statistics.txt"), "w") as f:

    f.write("Essential Matrix Estimation\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Total Matches : {len(matches)}\n")
    f.write(f"Inliers       : {len(inlier_matches)}\n")
    f.write(f"Outliers      : {len(matches)-len(inlier_matches)}\n\n")
    f.write("Essential Matrix:\n")
    f.write(np.array2string(E))

# -----------------------------------------------------
# Display
# -----------------------------------------------------

cv2.imshow("Essential Matrix Inliers", img_matches)

cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------------------
# Console Output
# -----------------------------------------------------

print("\n")
print("=" * 60)
print("ESSENTIAL MATRIX ESTIMATION")
print("=" * 60)

print("\nEssential Matrix\n")
print(E)

print("\nStatistics")
print("-" * 30)

print("Total Matches :", len(matches))
print("Inliers       :", len(inlier_matches))
print("Outliers      :", len(matches) - len(inlier_matches))

print("\nResults saved in:")
print("results/essential_matrix.txt")
print("results/essential_statistics.txt")
print("results/essential_inlier_matches.jpg")