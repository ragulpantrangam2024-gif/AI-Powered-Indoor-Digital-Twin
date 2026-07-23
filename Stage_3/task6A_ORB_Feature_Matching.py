import cv2
import os

# =====================================================
# TASK 6.1 - ORB FEATURE DETECTION
# =====================================================

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "images")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------------------------------
# Load Images
# -----------------------------------------------------

image1 = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_2\images\Image1.jpeg")
image2 = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_2\images\Image2.jpeg")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# -----------------------------------------------------
# Create ORB Detector
# -----------------------------------------------------

orb = cv2.ORB_create(nfeatures=1000)

# -----------------------------------------------------
# Detect Keypoints and Compute Descriptors
# -----------------------------------------------------

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# -----------------------------------------------------
# Draw Keypoints
# -----------------------------------------------------

img1_kp = cv2.drawKeypoints(
    image1,
    kp1,
    None,
    color=(0,255,0),
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

img2_kp = cv2.drawKeypoints(
    image2,
    kp2,
    None,
    color=(0,255,0),
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

# -----------------------------------------------------
# Save Results
# -----------------------------------------------------

cv2.imwrite(
    os.path.join(RESULT_DIR, "orb_keypoints_image1.jpg"),
    img1_kp
)

cv2.imwrite(
    os.path.join(RESULT_DIR, "orb_keypoints_image2.jpg"),
    img2_kp
)

# -----------------------------------------------------
# Display
# -----------------------------------------------------

cv2.imshow("ORB Features - Image 1", img1_kp)
cv2.imshow("ORB Features - Image 2", img2_kp)

cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------------------
# Statistics
# -----------------------------------------------------

print("="*60)
print("ORB FEATURE DETECTION")
print("="*60)

print(f"\nImage 1 Keypoints : {len(kp1)}")
print(f"Image 2 Keypoints : {len(kp2)}")

print("\nDescriptor Shape")
print("----------------------------")

print("Image 1 :", des1.shape)
print("Image 2 :", des2.shape)

print("\nResults Saved Successfully!")