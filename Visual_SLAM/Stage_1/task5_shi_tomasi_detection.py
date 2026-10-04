import cv2
import numpy as np

# Read image
image = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_1\images\room.jpg")


# Copy image
output = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect Shi-Tomasi corners
corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=10
)

# Draw corners
corners = np.int32(corners)

for corner in corners:
    x, y = corner.ravel()
    cv2.circle(output, (x, y), 4, (0, 0, 255), -1)

# Display
cv2.imshow("Original", image)
cv2.imshow("Shi-Tomasi Corners", output)

# Save
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\shi_tomasi_corners.jpg", output)

cv2.waitKey(0)
cv2.destroyAllWindows()