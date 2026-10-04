import cv2
import numpy as np
import os
import glob

# =====================================================
# STAGE 5 - TASK 3
# ESSENTIAL MATRIX BETWEEN CONSECUTIVE FRAMES
# =====================================================

# -----------------------------------------------------
# BASE DIRECTORY
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "images"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

# Stage 3 directory
STAGE3_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "Stage_3"
)

CALIBRATION_FILE = os.path.join(
    STAGE3_DIR,
    "results",
    "camera_calibration.npz"
)

os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------------------------------
# SETTINGS
# -----------------------------------------------------

ORB_FEATURES = 1000

LOWE_RATIO = 0.75

RANSAC_PROBABILITY = 0.999

RANSAC_THRESHOLD = 1.0

# -----------------------------------------------------
# LOAD IMAGE SEQUENCE
# -----------------------------------------------------

image_files = sorted(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.jpg"
        )
    )
)

print("=" * 60)
print("STAGE 5 - TASK 3")
print("ESSENTIAL MATRIX ESTIMATION")
print("=" * 60)

print(
    f"\nTotal Image Frames : {len(image_files)}"
)

if len(image_files) < 2:

    print(
        "\nERROR: At least two images are required."
    )

    exit()

# -----------------------------------------------------
# LOAD FIRST VIDEO FRAME
# -----------------------------------------------------

first_image = cv2.imread(
    image_files[0]
)

if first_image is None:

    print(
        "\nERROR: Could not read first image."
    )

    exit()

video_height, video_width = (
    first_image.shape[:2]
)

print("\nVideo Image Resolution")
print("----------------------")

print(
    f"Width  : {video_width}"
)

print(
    f"Height : {video_height}"
)

# -----------------------------------------------------
# LOAD STAGE 3 CALIBRATION
# -----------------------------------------------------

if not os.path.exists(CALIBRATION_FILE):

    print("\nERROR: Stage 3 calibration file not found.")

    print(
        "\nExpected:"
    )

    print(
        CALIBRATION_FILE
    )

    exit()

calib = np.load(
    CALIBRATION_FILE
)

# IMPORTANT:
# Our Stage 3 code saved the matrix using
# the key "cameraMatrix"

if "cameraMatrix" not in calib:

    print(
        "\nERROR: cameraMatrix not found "
        "inside calibration file."
    )

    print(
        "Available keys:"
    )

    print(
        calib.files
    )

    exit()

K_original = calib["cameraMatrix"]

# -----------------------------------------------------
# FIND ORIGINAL CALIBRATION IMAGE SIZE
# -----------------------------------------------------

calibration_images = []

for extension in [
    "*.jpg",
    "*.jpeg",
    "*.png"
]:

    calibration_images.extend(
        glob.glob(
            os.path.join(
                STAGE3_DIR,
                "images",
                extension
            )
        )
    )

if len(calibration_images) == 0:

    print(
        "\nERROR: Could not find Stage 3 "
        "calibration images."
    )

    exit()

calibration_image = cv2.imread(
    calibration_images[0]
)

if calibration_image is None:

    print(
        "\nERROR: Could not read Stage 3 "
        "calibration image."
    )

    exit()

calib_height, calib_width = (
    calibration_image.shape[:2]
)

# -----------------------------------------------------
# SCALE INTRINSIC MATRIX
# -----------------------------------------------------

scale_x = video_width / calib_width
scale_y = video_height / calib_height

K = K_original.copy().astype(
    np.float64
)

K[0, 0] *= scale_x
K[0, 2] *= scale_x

K[1, 1] *= scale_y
K[1, 2] *= scale_y

# -----------------------------------------------------
# DISPLAY CAMERA MATRIX
# -----------------------------------------------------

print("\nStage 3 Calibration Resolution")
print("-------------------------------")

print(
    f"Width  : {calib_width}"
)

print(
    f"Height : {calib_height}"
)

print("\nOriginal Intrinsic Matrix")
print("-------------------------")

print(K_original)

print("\nScaled Intrinsic Matrix")
print("-----------------------")

print(K)

# -----------------------------------------------------
# ORB
# -----------------------------------------------------

orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES
)

# -----------------------------------------------------
# BF MATCHER
# -----------------------------------------------------

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

# -----------------------------------------------------
# STORAGE
# -----------------------------------------------------

essential_matrices = []

total_matches_all = []

inliers_all = []

outliers_all = []

# -----------------------------------------------------
# PROCESS CONSECUTIVE FRAMES
# -----------------------------------------------------

for i in range(
    len(image_files) - 1
):

    # ---------------------------------------------
    # Load Images
    # ---------------------------------------------

    img1 = cv2.imread(
        image_files[i]
    )

    img2 = cv2.imread(
        image_files[i + 1]
    )

    if img1 is None or img2 is None:

        print(
            f"\nCould not read frame pair {i}"
        )

        continue

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------------------
    # ORB FEATURES
    # ---------------------------------------------

    kp1, des1 = orb.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray2,
        None
    )

    if des1 is None or des2 is None:

        print(
            f"\nFrame {i:03d} → {i+1:03d}"
        )

        print(
            "No descriptors found."
        )

        continue

    # ---------------------------------------------
    # KNN MATCHING
    # ---------------------------------------------

    knn_matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    # ---------------------------------------------
    # LOWE RATIO TEST
    # ---------------------------------------------

    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:

            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:

            good_matches.append(m)

    total_good = len(
        good_matches
    )

    if total_good < 8:

        print(
            f"\nFrame {i:03d} → {i+1:03d}"
        )

        print(
            f"Good Matches : {total_good}"
        )

        print(
            "Not enough matches "
            "for Essential Matrix."
        )

        continue

    # ---------------------------------------------
    # MATCH POINTS
    # ---------------------------------------------

    pts1 = np.float32(
        [
            kp1[m.queryIdx].pt
            for m in good_matches
        ]
    )

    pts2 = np.float32(
        [
            kp2[m.trainIdx].pt
            for m in good_matches
        ]
    )

    # ---------------------------------------------
    # ESSENTIAL MATRIX
    # ---------------------------------------------

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=RANSAC_PROBABILITY,
        threshold=RANSAC_THRESHOLD
    )

    if E is None:

        print(
            f"\nFrame {i:03d} → {i+1:03d}"
        )

        print(
            "Essential Matrix could not be estimated."
        )

        continue

    # ------------------------------------------------
    # OpenCV can theoretically return multiple
    # 3x3 matrices. We take the first one if needed.
    # ------------------------------------------------

    if E.shape[0] > 3:

        E = E[
            :3,
            :3
        ]

    # ---------------------------------------------
    # RANSAC MASK
    # ---------------------------------------------

    inlier_mask = mask.ravel().astype(
        bool
    )

    inlier_count = int(
        np.sum(inlier_mask)
    )

    outlier_count = (
        total_good - inlier_count
    )

    # ---------------------------------------------
    # STORE
    # ---------------------------------------------

    essential_matrices.append(
        E
    )

    total_matches_all.append(
        total_good
    )

    inliers_all.append(
        inlier_count
    )

    outliers_all.append(
        outlier_count
    )

    # ---------------------------------------------
    # PRINT
    # ---------------------------------------------

    print(
        f"\nFrame {i:03d} → {i+1:03d}"
    )

    print(
        f"Good Matches : {total_good}"
    )

    print(
        f"Inliers      : {inlier_count}"
    )

    print(
        f"Outliers     : {outlier_count}"
    )

    # ---------------------------------------------
    # SAVE FIRST PAIR VISUALIZATION
    # ---------------------------------------------

    if i == 0:

        inlier_matches = [
            m
            for j, m in enumerate(good_matches)
            if inlier_mask[j]
        ]

        inlier_image = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            inlier_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        output_image = os.path.join(
            RESULT_DIR,
            "task3_essential_inlier_matches.jpg"
        )

        cv2.imwrite(
            output_image,
            inlier_image
        )

# -----------------------------------------------------
# SUMMARY
# -----------------------------------------------------

print("\n")
print("=" * 60)
print("TASK 3 SUMMARY")
print("=" * 60)

if len(essential_matrices) == 0:

    print(
        "\nNo Essential Matrices were successfully estimated."
    )

    exit()

average_matches = np.mean(
    total_matches_all
)

average_inliers = np.mean(
    inliers_all
)

average_outliers = np.mean(
    outliers_all
)

print(
    f"\nProcessed Frame Pairs : "
    f"{len(essential_matrices)}"
)

print(
    f"Average Good Matches  : "
    f"{average_matches:.2f}"
)

print(
    f"Average Inliers       : "
    f"{average_inliers:.2f}"
)

print(
    f"Average Outliers      : "
    f"{average_outliers:.2f}"
)

# -----------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------

# Save all Essential Matrices
E_array = np.array(
    essential_matrices
)

np.save(
    os.path.join(
        RESULT_DIR,
        "essential_matrices.npy"
    ),
    E_array
)

# Save readable statistics
statistics_file = os.path.join(
    RESULT_DIR,
    "task3_essential_statistics.txt"
)

with open(
    statistics_file,
    "w"
) as f:

    f.write(
        "STAGE 5 - TASK 3\n"
    )

    f.write(
        "ESSENTIAL MATRIX ESTIMATION\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Video Resolution : "
        f"{video_width} x {video_height}\n"
    )

    f.write(
        f"Calibration Resolution : "
        f"{calib_width} x {calib_height}\n\n"
    )

    f.write(
        f"Processed Pairs : "
        f"{len(essential_matrices)}\n"
    )

    f.write(
        f"Average Matches : "
        f"{average_matches:.2f}\n"
    )

    f.write(
        f"Average Inliers : "
        f"{average_inliers:.2f}\n"
    )

    f.write(
        f"Average Outliers : "
        f"{average_outliers:.2f}\n"
    )

# -----------------------------------------------------
# SAVE FIRST ESSENTIAL MATRIX
# -----------------------------------------------------

first_E_file = os.path.join(
    RESULT_DIR,
    "task3_first_essential_matrix.txt"
)

np.savetxt(
    first_E_file,
    essential_matrices[0],
    fmt="%.8e"
)

# -----------------------------------------------------
# FINAL MESSAGE
# -----------------------------------------------------

print("\nResults saved in:")

print(
    "results/essential_matrices.npy"
)

print(
    "results/task3_essential_statistics.txt"
)

print(
    "results/task3_first_essential_matrix.txt"
)

print(
    "results/task3_essential_inlier_matches.jpg"
)

print(
    "\nTask 3 Completed Successfully!"
)