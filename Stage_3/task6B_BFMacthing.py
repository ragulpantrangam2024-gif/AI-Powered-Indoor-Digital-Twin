import cv2
import os

# =====================================================
# TASK 6.2 - DESCRIPTOR MATCHING
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
# ORB Detector
# -----------------------------------------------------

orb = cv2.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# -----------------------------------------------------
# BF Matcher
# -----------------------------------------------------

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=True
)

matches = bf.match(des1, des2)

# -----------------------------------------------------
# Sort Matches
# -----------------------------------------------------

matches = sorted(matches, key=lambda x: x.distance)

# -----------------------------------------------------
# Draw Best Matches
# -----------------------------------------------------

match_image = cv2.drawMatches(
    img1,
    kp1,
    img2,
    kp2,
    matches[:100],     # Show best 100 matches
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# -----------------------------------------------------
# Save Result
# -----------------------------------------------------

cv2.imwrite(
    os.path.join(RESULT_DIR, "raw_matches.jpg"),
    match_image
)

# -----------------------------------------------------
# Display
# -----------------------------------------------------

cv2.imshow("ORB Matches", match_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------------------
# Statistics
# -----------------------------------------------------

print("=" * 60)
print("DESCRIPTOR MATCHING")
print("=" * 60)

print(f"\nImage 1 Keypoints : {len(kp1)}")
print(f"Image 2 Keypoints : {len(kp2)}")

print(f"\nTotal Matches : {len(matches)}")

print("\nTop 10 Match Distances")

for i in range(10):
    print(f"Match {i+1}: {matches[i].distance:.2f}")

print("\nResults saved to:")
print("results/raw_matches.jpg")