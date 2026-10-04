import cv2

# -------------------------------
# Read the two images
# -------------------------------
image1 = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_2\images\Image1.jpeg")
image2 = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_2\images\Image2.jpeg")

# Convert to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# -------------------------------
# Create ORB detector
# -------------------------------
orb = cv2.ORB_create()

# Detect keypoints and descriptors
keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# -------------------------------
# Create Brute Force Matcher
# -------------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 50 matches
matched_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    matches[:50],
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Display
cv2.imshow("ORB Feature Matching", matched_image)

# Save
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_2\results\orb_keypoints.jpg", matched_image)

print("Number of keypoints in Image 1:", len(keypoints1))
print("Number of keypoints in Image 2:", len(keypoints2))
print("Number of matches:", len(matches))

cv2.waitKey(0)
cv2.destroyAllWindows()