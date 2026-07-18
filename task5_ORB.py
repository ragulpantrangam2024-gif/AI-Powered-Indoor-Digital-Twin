import cv2

# Read image
image = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_1\images\room.jpg")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create ORB detector
orb = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Draw keypoints
output = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    color=(0, 0, 255),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# Display
cv2.imshow("ORB Keypoints", output)

# Save
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\orb_keypoints.jpg", output)

print("Number of ORB keypoints:", len(keypoints))
print("Descriptor shape:", descriptors.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()