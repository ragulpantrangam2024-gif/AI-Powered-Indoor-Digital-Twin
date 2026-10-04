import cv2

# Read image
image = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_1\images\room.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create FAST detector
fast = cv2.FastFeatureDetector_create()

fast.setThreshold(60)

fast.setNonmaxSuppression(True)

# Detect keypoints
keypoints = fast.detect(gray, None)

# Draw keypoints
output = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    color=(0, 0, 255)
)

# Display
cv2.imshow("Original", image)
cv2.imshow("FAST Keypoints", output)

# Save
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\fast_keypoints.jpg", output)

cv2.waitKey(0)
cv2.destroyAllWindows()