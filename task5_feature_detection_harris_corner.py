import cv2
import numpy as np

# Read image
image = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_1\images\room.jpg")

# Create a copy for visualization
output = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert to float32 (required by cornerHarris)
gray = np.float32(gray)

# Harris Corner Detection
corners = cv2.cornerHarris(gray,
                           blockSize=2,
                           ksize=3,
                           k=0.04)

# Dilate the corner image to make corners more visible
corners = cv2.dilate(corners, None)

# Mark detected corners
output[corners > 0.01 * corners.max()] = [0, 0, 255]

# Display results
cv2.imshow("Original Image", image)
cv2.imshow("Harris Corners", output)

# Save result
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\harris_corners.jpg", output)

cv2.waitKey(0)
cv2.destroyAllWindows()