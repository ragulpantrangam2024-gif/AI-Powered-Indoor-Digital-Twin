import cv2
import numpy as np
import os

# =====================================================
# TASK 2 - TRIANGULATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "images")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------------------------------
# Load Images
# -----------------------------------------------------

img1 = cv2.imread(os.path.join(IMAGE_DIR, "image1.jpeg"))
img2 = cv2.imread(os.path.join(IMAGE_DIR, "image2.jpeg"))

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# -----------------------------------------------------
# Load Projection Matrices
# -----------------------------------------------------

P1 = np.loadtxt(
    os.path.join(RESULT_DIR,
                 "projection_matrix_camera1.txt")
)

P2 = np.loadtxt(
    os.path.join(RESULT_DIR,
                 "projection_matrix_camera2.txt")
)

# -----------------------------------------------------
# ORB Feature Detection
# -----------------------------------------------------

orb = cv2.ORB_create(1000)

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

pts1 = np.float32(
    [kp1[m.queryIdx].pt for m in matches]
).T

pts2 = np.float32(
    [kp2[m.trainIdx].pt for m in matches]
).T

# -----------------------------------------------------
# Triangulation
# -----------------------------------------------------

points4D = cv2.triangulatePoints(
    P1,
    P2,
    pts1,
    pts2
)

# -----------------------------------------------------
# Save Homogeneous Coordinates
# -----------------------------------------------------

np.savetxt(
    os.path.join(
        RESULT_DIR,
        "points4D.txt"
    ),
    points4D,
    fmt="%.6f"
)

# -----------------------------------------------------
# Console
# -----------------------------------------------------

print("="*60)
print("TRIANGULATION")
print("="*60)

print("\nNumber of Matched Points")

print(points4D.shape[1])

print("\nOutput Shape")

print(points4D.shape)

print("\nFirst Five Homogeneous Points\n")

print(points4D[:,0:5])

print("\nSaved Successfully!")

print("results/points4D.txt")