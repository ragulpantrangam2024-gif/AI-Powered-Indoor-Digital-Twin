import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


# ============================================================
# STAGE 6 - TASK 2
# FEATURE TRACKING BETWEEN KEYFRAMES
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STAGE6_DIR = SCRIPT_DIR

IMAGE_DIR = os.path.join(
    STAGE6_DIR,
    "images"
)

RESULT_DIR = os.path.join(
    STAGE6_DIR,
    "results"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


print("=" * 70)
print("STAGE 6 - TASK 2")
print("FEATURE TRACKING BETWEEN KEYFRAMES")
print("=" * 70)

print("\nStage 6 directory:")
print(STAGE6_DIR)

print("\nImage directory:")
print(IMAGE_DIR)

print("\nResults directory:")
print(RESULT_DIR)


# ============================================================
# 2. CHECK DIRECTORIES
# ============================================================

if not os.path.exists(IMAGE_DIR):

    print("\nERROR: Image directory not found!")

    print(
        "Expected:"
    )

    print(
        IMAGE_DIR
    )

    exit()


# ============================================================
# 3. LOAD IMAGE FILES
# ============================================================

image_files = []

image_files.extend(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.jpg"
        )
    )
)

image_files.extend(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.jpeg"
        )
    )
)

image_files.extend(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.png"
        )
    )
)

image_files = sorted(
    image_files
)


print("\nImages found:")
print(
    len(image_files)
)


if len(image_files) < 2:

    print(
        "\nERROR: Not enough images."
    )

    exit()


# ============================================================
# 4. LOAD KEYFRAME INDICES
# ============================================================

keyframe_file = os.path.join(
    RESULT_DIR,
    "keyframe_indices.npy"
)


if not os.path.exists(
    keyframe_file
):

    print(
        "\nERROR: Keyframe file not found!"
    )

    print(
        keyframe_file
    )

    print(
        "\nRun Task 1 first."
    )

    exit()


keyframe_indices = np.load(
    keyframe_file
)


keyframe_indices = (
    keyframe_indices
    .astype(int)
    .tolist()
)


print("\nKeyframes loaded:")
print(
    len(keyframe_indices)
)

print(
    "\nKeyframe indices:"
)

print(
    keyframe_indices
)


# ============================================================
# 5. ORB FEATURE DETECTOR
# ============================================================

orb = cv2.ORB_create(
    nfeatures=1000
)


# ============================================================
# 6. BRUTE FORCE MATCHER
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# 7. PARAMETERS
# ============================================================

LOWE_RATIO = 0.75

MIN_MATCHES_FOR_TRACKING = 8

RANSAC_THRESHOLD = 1.0

RANSAC_CONFIDENCE = 0.999


# ============================================================
# 8. STORAGE
# ============================================================

pair_results = []

good_match_counts = []

inlier_counts = []

outlier_counts = []

tracking_ratios = []

tracked_points = []


# ============================================================
# 9. PROCESS KEYFRAME PAIRS
# ============================================================

print("\n")
print("=" * 70)
print("STARTING FEATURE TRACKING")
print("=" * 70)


for pair_index in range(
    len(keyframe_indices) - 1
):

    idx1 = keyframe_indices[
        pair_index
    ]

    idx2 = keyframe_indices[
        pair_index + 1
    ]


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    img1 = cv2.imread(
        image_files[idx1]
    )

    img2 = cv2.imread(
        image_files[idx2]
    )


    if img1 is None or img2 is None:

        print(
            f"\nPair {pair_index + 1:03d}: "
            f"Could not load images."
        )

        continue


    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Detect ORB features
    # --------------------------------------------------------

    kp1, des1 = (
        orb.detectAndCompute(
            gray1,
            None
        )
    )

    kp2, des2 = (
        orb.detectAndCompute(
            gray2,
            None
        )
    )


    if (
        des1 is None
        or
        des2 is None
    ):

        print(
            f"Keyframe "
            f"{idx1:03d} -> "
            f"{idx2:03d} | "
            f"Descriptors unavailable"
        )

        pair_results.append({
            "frame1": idx1,
            "frame2": idx2,
            "good_matches": 0,
            "inliers": 0,
            "outliers": 0,
            "tracking_ratio": 0.0
        })

        good_match_counts.append(
            0
        )

        inlier_counts.append(
            0
        )

        outlier_counts.append(
            0
        )

        tracking_ratios.append(
            0.0
        )

        tracked_points.append(
            0
        )

        continue


    # --------------------------------------------------------
    # KNN matching
    # --------------------------------------------------------

    knn_matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )


    # --------------------------------------------------------
    # Lowe Ratio Test
    # --------------------------------------------------------

    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:

            continue

        m, n = pair

        if (
            m.distance
            <
            LOWE_RATIO * n.distance
        ):

            good_matches.append(
                m
            )


    good_count = len(
        good_matches
    )


    # --------------------------------------------------------
    # Default values
    # --------------------------------------------------------

    inlier_count = 0

    outlier_count = good_count

    tracking_ratio = 0.0


    # --------------------------------------------------------
    # Geometric verification
    # --------------------------------------------------------

    if (
        good_count
        >=
        MIN_MATCHES_FOR_TRACKING
    ):

        pts1 = np.float32([
            kp1[m.queryIdx].pt
            for m in good_matches
        ])

        pts2 = np.float32([
            kp2[m.trainIdx].pt
            for m in good_matches
        ])


        # ----------------------------------------------------
        # Fundamental Matrix
        # ----------------------------------------------------

        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD,
            confidence=RANSAC_CONFIDENCE
        )


        if (
            F is not None
            and
            mask is not None
        ):

            mask = mask.ravel()

            inlier_count = int(
                np.sum(
                    mask.astype(bool)
                )
            )

            outlier_count = (
                good_count
                -
                inlier_count
            )

            tracking_ratio = (
                inlier_count
                /
                good_count
            )


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    good_match_counts.append(
        good_count
    )

    inlier_counts.append(
        inlier_count
    )

    outlier_counts.append(
        outlier_count
    )

    tracking_ratios.append(
        tracking_ratio
    )

    tracked_points.append(
        inlier_count
    )


    pair_results.append({
        "frame1": idx1,
        "frame2": idx2,
        "good_matches": good_count,
        "inliers": inlier_count,
        "outliers": outlier_count,
        "tracking_ratio": tracking_ratio
    })


    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        f"Keyframe "
        f"{idx1:03d} -> "
        f"{idx2:03d} | "
        f"Good Matches: "
        f"{good_count:3d} | "
        f"Inliers: "
        f"{inlier_count:3d} | "
        f"Outliers: "
        f"{outlier_count:3d} | "
        f"Tracking Ratio: "
        f"{tracking_ratio:.3f}"
    )


# ============================================================
# 10. CONVERT RESULTS
# ============================================================

good_match_counts = np.array(
    good_match_counts,
    dtype=np.float64
)

inlier_counts = np.array(
    inlier_counts,
    dtype=np.float64
)

outlier_counts = np.array(
    outlier_counts,
    dtype=np.float64
)

tracking_ratios = np.array(
    tracking_ratios,
    dtype=np.float64
)


# ============================================================
# 11. SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FEATURE TRACKING SUMMARY")
print("=" * 70)

print(
    f"\nKeyframes Processed : "
    f"{len(keyframe_indices)}"
)

print(
    f"Keyframe Pairs      : "
    f"{len(pair_results)}"
)


if len(good_match_counts) > 0:

    print(
        f"\nGood Matches"
    )

    print(
        f"Minimum : "
        f"{np.min(good_match_counts):.0f}"
    )

    print(
        f"Maximum : "
        f"{np.max(good_match_counts):.0f}"
    )

    print(
        f"Average : "
        f"{np.mean(good_match_counts):.2f}"
    )


if len(inlier_counts) > 0:

    print(
        f"\nGeometric Inliers"
    )

    print(
        f"Minimum : "
        f"{np.min(inlier_counts):.0f}"
    )

    print(
        f"Maximum : "
        f"{np.max(inlier_counts):.0f}"
    )

    print(
        f"Average : "
        f"{np.mean(inlier_counts):.2f}"
    )


if len(tracking_ratios) > 0:

    print(
        f"\nTracking Ratio"
    )

    print(
        f"Minimum : "
        f"{np.min(tracking_ratios):.3f}"
    )

    print(
        f"Maximum : "
        f"{np.max(tracking_ratios):.3f}"
    )

    print(
        f"Average : "
        f"{np.mean(tracking_ratios):.3f}"
    )


# ============================================================
# 12. SAVE NUMERICAL RESULTS
# ============================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "task2_good_matches.npy"
    ),
    good_match_counts
)

np.save(
    os.path.join(
        RESULT_DIR,
        "task2_inliers.npy"
    ),
    inlier_counts
)

np.save(
    os.path.join(
        RESULT_DIR,
        "task2_outliers.npy"
    ),
    outlier_counts
)

np.save(
    os.path.join(
        RESULT_DIR,
        "task2_tracking_ratios.npy"
    ),
    tracking_ratios
)


# ============================================================
# 13. SAVE TEXT RESULTS
# ============================================================

results_file = os.path.join(
    RESULT_DIR,
    "task2_tracking_statistics.txt"
)


with open(
    results_file,
    "w"
) as file:

    file.write(
        "STAGE 6 - TASK 2\n"
    )

    file.write(
        "FEATURE TRACKING BETWEEN KEYFRAMES\n"
    )

    file.write(
        "=" * 70
        +
        "\n\n"
    )

    file.write(
        f"Keyframes Processed : "
        f"{len(keyframe_indices)}\n"
    )

    file.write(
        f"Keyframe Pairs      : "
        f"{len(pair_results)}\n\n"
    )


    if len(good_match_counts) > 0:

        file.write(
            "GOOD MATCHES\n"
        )

        file.write(
            f"Minimum : "
            f"{np.min(good_match_counts):.0f}\n"
        )

        file.write(
            f"Maximum : "
            f"{np.max(good_match_counts):.0f}\n"
        )

        file.write(
            f"Average : "
            f"{np.mean(good_match_counts):.2f}\n\n"
        )


    if len(inlier_counts) > 0:

        file.write(
            "GEOMETRIC INLIERS\n"
        )

        file.write(
            f"Minimum : "
            f"{np.min(inlier_counts):.0f}\n"
        )

        file.write(
            f"Maximum : "
            f"{np.max(inlier_counts):.0f}\n"
        )

        file.write(
            f"Average : "
            f"{np.mean(inlier_counts):.2f}\n\n"
        )


    if len(tracking_ratios) > 0:

        file.write(
            "TRACKING RATIO\n"
        )

        file.write(
            f"Minimum : "
            f"{np.min(tracking_ratios):.3f}\n"
        )

        file.write(
            f"Maximum : "
            f"{np.max(tracking_ratios):.3f}\n"
        )

        file.write(
            f"Average : "
            f"{np.mean(tracking_ratios):.3f}\n\n"
        )


    file.write(
        "PAIR RESULTS\n"
    )

    file.write(
        "Frame1 Frame2 GoodMatches "
        "Inliers Outliers TrackingRatio\n"
    )

    for result in pair_results:

        file.write(
            f"{result['frame1']} "
            f"{result['frame2']} "
            f"{result['good_matches']} "
            f"{result['inliers']} "
            f"{result['outliers']} "
            f"{result['tracking_ratio']:.4f}\n"
        )


# ============================================================
# 14. GOOD MATCHES PLOT
# ============================================================

pair_numbers = np.arange(
    len(good_match_counts)
)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    pair_numbers,
    good_match_counts,
    linewidth=1.5
)

plt.xlabel(
    "Keyframe Pair"
)

plt.ylabel(
    "Good Matches"
)

plt.title(
    "Feature Matches Between Keyframes"
)

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULT_DIR,
        "task2_good_matches.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. INLIER PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    pair_numbers,
    inlier_counts,
    linewidth=1.5
)

plt.xlabel(
    "Keyframe Pair"
)

plt.ylabel(
    "Geometric Inliers"
)

plt.title(
    "Geometric Feature Tracking Inliers"
)

plt.grid(
    True
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULT_DIR,
        "task2_inliers.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. TRACKING RATIO PLOT
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    pair_numbers,
    tracking_ratios,
    linewidth=1.5
)

plt.axhline(
    0.5,
    linestyle="--",
    label="50% Reference"
)

plt.xlabel(
    "Keyframe Pair"
)

plt.ylabel(
    "Tracking Ratio"
)

plt.title(
    "Feature Tracking Quality"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    True
)

plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULT_DIR,
        "task2_tracking_ratio.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("TASK 2 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nResults saved in:")
print(
    RESULT_DIR
)

print("\nGenerated files:")

print(
    "1. task2_good_matches.npy"
)

print(
    "2. task2_inliers.npy"
)

print(
    "3. task2_outliers.npy"
)

print(
    "4. task2_tracking_ratios.npy"
)

print(
    "5. task2_tracking_statistics.txt"
)

print(
    "6. task2_good_matches.png"
)

print(
    "7. task2_inliers.png"
)

print(
    "8. task2_tracking_ratio.png"
)

print(
    "\nNo plot windows were opened."
)