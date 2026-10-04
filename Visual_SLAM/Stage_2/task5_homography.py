import cv2
import numpy as np

# -------------------------------------
# Read Images
# -------------------------------------
image1 = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_2\images\Image1.jpeg")
image2 = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_2\images\Image2.jpeg")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# -------------------------------------
# ORB Feature Detection
# -------------------------------------
orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# -------------------------------------
# KNN Matching
# -------------------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# -------------------------------------
# Lowe Ratio Test
# -------------------------------------
good_matches = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# -------------------------------------
# Extract Matched Points
# -------------------------------------
src_pts = np.float32(
    [keypoints1[m.queryIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

dst_pts = np.float32(
    [keypoints2[m.trainIdx].pt for m in good_matches]
).reshape(-1, 1, 2)

# -------------------------------------
# Homography Estimation
# -------------------------------------
H, mask = cv2.findHomography(
    src_pts,
    dst_pts,
    cv2.RANSAC,
    5.0
)

print("Homography Matrix:\n")
print(H)

# -------------------------------------
# Transform Image 1
# -------------------------------------
height, width = image2.shape[:2]

warped = cv2.warpPerspective(
    image1,
    H,
    (width, height)
)

# -------------------------------------
# Display
# -------------------------------------
cv2.imshow("Image 1", image1)
cv2.imshow("Image 2", image2)
cv2.imshow("Warped Image", warped)

cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_2\results\warped_image.jpg", warped)

cv2.waitKey(0)
cv2.destroyAllWindows()