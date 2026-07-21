import cv2

# -------------------------------------
# Read images
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
# BFMatcher without Cross Check
# -------------------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Find two nearest neighbors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# -------------------------------------
# Lowe's Ratio Test
# -------------------------------------
good_matches = []

ratio = 0.75

for m, n in matches:
    if m.distance < ratio * n.distance:
        good_matches.append(m)

# -------------------------------------
# Draw Matches
# -------------------------------------
matched_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# -------------------------------------
# Display
# -------------------------------------
cv2.imshow("KNN Matching + Lowe's Ratio Test", matched_image)

cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_2\results\knn_ratio_test.jpg", matched_image)

print("Keypoints Image 1:", len(keypoints1))
print("Keypoints Image 2:", len(keypoints2))
print("KNN Matches:", len(matches))
print("Good Matches:", len(good_matches))

cv2.waitKey(0)
cv2.destroyAllWindows()