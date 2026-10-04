import cv2
import numpy as np
import os
import glob

# =====================================================
# STAGE 5 - TASK 2
# ORB FEATURE TRACKING BETWEEN CONSECUTIVE FRAMES
# =====================================================

# -----------------------------------------------------
# Paths
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "images")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------------------------------
# Load Image Sequence
# -----------------------------------------------------

image_files = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
)

if len(image_files) < 2:
    raise ValueError(
        "At least two images are required."
    )

print("=" * 60)
print("STAGE 5 - TASK 2")
print("ORB FEATURE TRACKING")
print("=" * 60)

print(f"\nTotal Frames : {len(image_files)}")

# -----------------------------------------------------
# ORB Detector
# -----------------------------------------------------

orb = cv2.ORB_create(
    nfeatures=1000
)

# -----------------------------------------------------
# BF Matcher
# -----------------------------------------------------

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

# -----------------------------------------------------
# Test Consecutive Frames
# -----------------------------------------------------

NUM_TEST_PAIRS = min(10, len(image_files) - 1)

match_counts = []

for i in range(NUM_TEST_PAIRS):

    # ---------------------------------------------
    # Load Frames
    # ---------------------------------------------

    img1 = cv2.imread(image_files[i])
    img2 = cv2.imread(image_files[i + 1])

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------------------
    # Detect ORB Features
    # ---------------------------------------------

    kp1, des1 = orb.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray2,
        None
    )

    # ---------------------------------------------
    # Check Descriptors
    # ---------------------------------------------

    if des1 is None or des2 is None:

        print(
            f"\nFrame {i:03d} → {i+1:03d}"
        )

        print("No descriptors detected.")

        continue

    # ---------------------------------------------
    # KNN Matching
    # ---------------------------------------------

    knn_matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    # ---------------------------------------------
    # Lowe Ratio Test
    # ---------------------------------------------

    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:

            good_matches.append(m)

    # ---------------------------------------------
    # Statistics
    # ---------------------------------------------

    match_counts.append(
        len(good_matches)
    )

    print(
        f"\nFrame {i:03d} → {i+1:03d}"
    )

    print(
        f"Keypoints Image 1 : {len(kp1)}"
    )

    print(
        f"Keypoints Image 2 : {len(kp2)}"
    )

    print(
        f"KNN Matches        : {len(knn_matches)}"
    )

    print(
        f"Good Matches       : {len(good_matches)}"
    )

    # ---------------------------------------------
    # Visualize First Pair
    # ---------------------------------------------

    if i == 0:

        match_image = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        output_path = os.path.join(
            RESULT_DIR,
            "task2_frame0_frame1_matches.jpg"
        )

        cv2.imwrite(
            output_path,
            match_image
        )

# -----------------------------------------------------
# Summary
# -----------------------------------------------------

print("\n")
print("=" * 60)
print("TASK 2 SUMMARY")
print("=" * 60)

if len(match_counts) > 0:

    print(
        f"\nMinimum Good Matches : "
        f"{min(match_counts)}"
    )

    print(
        f"Maximum Good Matches : "
        f"{max(match_counts)}"
    )

    print(
        f"Average Good Matches : "
        f"{np.mean(match_counts):.2f}"
    )

print(
    "\nFirst pair visualization saved to:"
)

print(
    "results/task2_frame0_frame1_matches.jpg"
)

print(
    "\nTask 2 Completed Successfully!"
)