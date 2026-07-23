import cv2
import numpy as np
import os

# =====================================================
# TASK 9 - EPIPOLAR GEOMETRY VISUALIZATION
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
# ORB Features
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
# Matched Points
# -----------------------------------------------------

pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# -----------------------------------------------------
# Fundamental Matrix
# -----------------------------------------------------

F, mask = cv2.findFundamentalMat(
    pts1,
    pts2,
    cv2.FM_RANSAC
)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# -----------------------------------------------------
# Compute Epilines
# -----------------------------------------------------

lines1 = cv2.computeCorrespondEpilines(
    pts2.reshape(-1,1,2),
    2,
    F
)

lines1 = lines1.reshape(-1,3)

lines2 = cv2.computeCorrespondEpilines(
    pts1.reshape(-1,1,2),
    1,
    F
)

lines2 = lines2.reshape(-1,3)

# -----------------------------------------------------
# Draw Function
# -----------------------------------------------------

def draw_lines(img, lines, pts):

    img = img.copy()

    h, w = img.shape[:2]

    np.random.seed(42)

    for r, pt in zip(lines[:30], pts[:30]):

        color = tuple(np.random.randint(0,255,3).tolist())

        x0 = 0
        y0 = int(-r[2]/r[1])

        x1 = w
        y1 = int(-(r[2] + r[0]*w)/r[1])

        cv2.line(img, (x0,y0), (x1,y1), color, 1)

        cv2.circle(
            img,
            tuple(np.int32(pt)),
            5,
            color,
            -1
        )

    return img

# -----------------------------------------------------
# Draw Images
# -----------------------------------------------------

epi1 = draw_lines(img1, lines1, pts1)
epi2 = draw_lines(img2, lines2, pts2)

# -----------------------------------------------------
# Save
# -----------------------------------------------------

cv2.imwrite(
    os.path.join(RESULT_DIR, "epipolar_image1.jpg"),
    epi1
)

cv2.imwrite(
    os.path.join(RESULT_DIR, "epipolar_image2.jpg"),
    epi2
)

# -----------------------------------------------------
# Display
# -----------------------------------------------------

cv2.imshow("Epipolar Image 1", epi1)
cv2.imshow("Epipolar Image 2", epi2)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("="*60)
print("EPIPOLAR GEOMETRY")
print("="*60)

print("Fundamental Matrix")
print(F)

print("\nImages Saved")

print("results/epipolar_image1.jpg")
print("results/epipolar_image2.jpg")