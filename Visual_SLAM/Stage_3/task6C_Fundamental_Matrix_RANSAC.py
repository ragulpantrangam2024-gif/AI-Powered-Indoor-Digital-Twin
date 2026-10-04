import cv2
import numpy as np
import os

# =====================================================
# TASK 6.3 - FUNDAMENTAL MATRIX ESTIMATION
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
# Extract Matched Points
# -----------------------------------------------------

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# -----------------------------------------------------
# Compute Fundamental Matrix
# -----------------------------------------------------

F, mask = cv2.findFundamentalMat(
    pts1,
    pts2,
    cv2.FM_RANSAC,
    ransacReprojThreshold=1.0,
    confidence=0.99
)

# -----------------------------------------------------
# Keep Only Inliers
# -----------------------------------------------------

inlier_matches = []

pts1_inliers = []
pts2_inliers = []

for i, m in enumerate(matches):
    if mask[i]:
        inlier_matches.append(m)
        pts1_inliers.append(pts1[i])
        pts2_inliers.append(pts2[i])

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
# Save Image
# -----------------------------------------------------

cv2.imwrite(
    os.path.join(RESULT_DIR, "inlier_matches.jpg"),
    img_matches
)

# -----------------------------------------------------
# Save Fundamental Matrix
# -----------------------------------------------------

np.savetxt(
    os.path.join(RESULT_DIR, "fundamental_matrix.txt"),
    F,
    fmt="%.8f"
)

# -----------------------------------------------------
# Save Statistics
# -----------------------------------------------------

with open(os.path.join(RESULT_DIR, "match_statistics.txt"), "w") as f:

    f.write("Fundamental Matrix Estimation\n")
    f.write("="*40 + "\n\n")

    f.write(f"Total Matches : {len(matches)}\n")
    f.write(f"Inliers       : {len(inlier_matches)}\n")
    f.write(f"Outliers      : {len(matches)-len(inlier_matches)}\n")

# -----------------------------------------------------
# Display
# -----------------------------------------------------

cv2.imshow("Inlier Matches", img_matches)

cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------------------
# Console Output
# -----------------------------------------------------

print("="*60)
print("FUNDAMENTAL MATRIX ESTIMATION")
print("="*60)

print("\nFundamental Matrix\n")
print(F)

print("\nStatistics")
print("----------------------")

print("Total Matches :", len(matches))
print("Inliers       :", len(inlier_matches))
print("Outliers      :", len(matches)-len(inlier_matches))

print("\nResults saved inside Stage_3/results/")