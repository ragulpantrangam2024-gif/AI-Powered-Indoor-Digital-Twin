import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


# ============================================================
# STAGE 6 - TASK 3
# 3D MAP POINT RECONSTRUCTION
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
print("STAGE 6 - TASK 3")
print("3D MAP POINT RECONSTRUCTION")
print("=" * 70)

print("\nStage 6 directory:")
print(STAGE6_DIR)

print("\nImage directory:")
print(IMAGE_DIR)

print("\nResults directory:")
print(RESULT_DIR)


# ============================================================
# 2. CAMERA INTRINSIC MATRIX
# ============================================================
#
# Obtained from Stage 3 camera calibration.
#
# K =
# [[879.03578678,   0.        , 438.89954652],
#  [  0.        , 896.60912854, 557.52366461],
#  [  0.        ,   0.        ,   1.        ]]
#
# ============================================================

K = np.array([
    [879.03578678, 0.0, 438.89954652],
    [0.0, 896.60912854, 557.52366461],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


print("\nCamera Intrinsic Matrix K:")
print(K)


# ============================================================
# 3. CHECK IMAGE DIRECTORY
# ============================================================

if not os.path.exists(IMAGE_DIR):

    print("\nERROR: Image directory not found!")

    print(
        IMAGE_DIR
    )

    exit()


# ============================================================
# 4. FIND IMAGES
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
# 5. LOAD KEYFRAMES
# ============================================================

keyframe_file = os.path.join(
    RESULT_DIR,
    "keyframe_indices.npy"
)


if not os.path.exists(
    keyframe_file
):

    print(
        "\nERROR: keyframe_indices.npy not found."
    )

    print(
        "Run Task 1 first."
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


# ============================================================
# 6. ORB FEATURE DETECTOR
# ============================================================

orb = cv2.ORB_create(
    nfeatures=1500
)


# ============================================================
# 7. BF MATCHER
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# 8. PARAMETERS
# ============================================================

LOWE_RATIO = 0.75

MIN_GOOD_MATCHES = 15

MIN_INLIERS = 8

RANSAC_THRESHOLD = 1.0

RANSAC_CONFIDENCE = 0.999


# ============================================================
# 9. STORAGE
# ============================================================

all_map_points = []

pair_statistics = []

successful_pairs = 0

failed_pairs = 0

total_candidate_points = 0


# ============================================================
# 10. PROCESS KEYFRAME PAIRS
# ============================================================

print("\n")
print("=" * 70)
print("3D RECONSTRUCTION STARTED")
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


    print(
        f"\nProcessing pair "
        f"{idx1:03d} -> {idx2:03d}"
    )


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
            "  Images could not be loaded."
        )

        failed_pairs += 1

        continue


    # --------------------------------------------------------
    # Grayscale
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
    # ORB features
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
            "  Descriptors unavailable."
        )

        failed_pairs += 1

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
    # Lowe ratio test
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


    if good_count < MIN_GOOD_MATCHES:

        print(
            f"  Skipped: only "
            f"{good_count} good matches."
        )

        failed_pairs += 1

        pair_statistics.append({
            "frame1": idx1,
            "frame2": idx2,
            "good_matches": good_count,
            "inliers": 0,
            "points_3d": 0,
            "status": "SKIPPED"
        })

        continue


    # --------------------------------------------------------
    # Matched 2D points
    # --------------------------------------------------------

    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in good_matches
    ])

    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in good_matches
    ])


    # --------------------------------------------------------
    # Essential Matrix
    # --------------------------------------------------------

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=RANSAC_CONFIDENCE,
        threshold=RANSAC_THRESHOLD
    )


    if (
        E is None
        or
        mask is None
    ):

        print(
            "  Essential matrix failed."
        )

        failed_pairs += 1

        continue


    # --------------------------------------------------------
    # Recover camera pose
    # --------------------------------------------------------

    inlier_mask = mask.ravel().astype(
        np.uint8
    )

    inlier_indices = np.where(
        inlier_mask == 1
    )[0]


    if len(inlier_indices) < MIN_INLIERS:

        print(
            f"  Skipped: only "
            f"{len(inlier_indices)} pose inliers."
        )

        failed_pairs += 1

        pair_statistics.append({
            "frame1": idx1,
            "frame2": idx2,
            "good_matches": good_count,
            "inliers": len(inlier_indices),
            "points_3d": 0,
            "status": "INSUFFICIENT_INLIERS"
        })

        continue


    inlier_pts1 = pts1[
        inlier_indices
    ]

    inlier_pts2 = pts2[
        inlier_indices
    ]


    pose_inliers, R, t, pose_mask = (
        cv2.recoverPose(
            E,
            inlier_pts1,
            inlier_pts2,
            K
        )
    )


    if pose_inliers < MIN_INLIERS:

        print(
            f"  Pose recovery failed: "
            f"{pose_inliers} inliers."
        )

        failed_pairs += 1

        continue


    # --------------------------------------------------------
    # Select pose inliers
    # --------------------------------------------------------

    pose_mask = pose_mask.ravel()

    valid_pose_indices = np.where(
        pose_mask > 0
    )[0]


    final_pts1 = inlier_pts1[
        valid_pose_indices
    ]

    final_pts2 = inlier_pts2[
        valid_pose_indices
    ]


    if len(final_pts1) < MIN_INLIERS:

        print(
            "  Not enough pose points "
            "for triangulation."
        )

        failed_pairs += 1

        continue


    # --------------------------------------------------------
    # Projection matrices
    # --------------------------------------------------------
    #
    # Camera 1:
    #
    # P1 = K [I | 0]
    #
    # Camera 2:
    #
    # P2 = K [R | t]
    #
    # --------------------------------------------------------

    P1 = K @ np.hstack(
        (
            np.eye(3),
            np.zeros(
                (3, 1)
            )
        )
    )


    P2 = K @ np.hstack(
        (
            R,
            t
        )
    )


    # --------------------------------------------------------
    # Triangulation
    # --------------------------------------------------------

    points_4d = cv2.triangulatePoints(
        P1,
        P2,
        final_pts1.T,
        final_pts2.T
    )


    # --------------------------------------------------------
    # Convert homogeneous coordinates
    # --------------------------------------------------------

    points_3d = (
        points_4d[:3]
        /
        points_4d[3]
    ).T


    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    valid_mask = np.isfinite(
        points_3d
    ).all(
        axis=1
    )


    points_3d = points_3d[
        valid_mask
    ]


    # --------------------------------------------------------
    # Remove points too close / behind camera
    # --------------------------------------------------------

    if len(points_3d) > 0:

        positive_depth = (
            points_3d[:, 2] > 0
        )

        points_3d = points_3d[
            positive_depth
        ]


    point_count = len(
        points_3d
    )


    total_candidate_points += (
        point_count
    )


    # --------------------------------------------------------
    # Store points
    # --------------------------------------------------------

    if point_count > 0:

        all_map_points.append(
            points_3d
        )


    successful_pairs += 1


    pair_statistics.append({
        "frame1": idx1,
        "frame2": idx2,
        "good_matches": good_count,
        "inliers": int(pose_inliers),
        "points_3d": point_count,
        "status": "SUCCESS"
    })


    print(
        f"  Good Matches : "
        f"{good_count}"
    )

    print(
        f"  Pose Inliers : "
        f"{pose_inliers}"
    )

    print(
        f"  3D Points    : "
        f"{point_count}"
    )


# ============================================================
# 11. COMBINE ALL MAP POINTS
# ============================================================

if len(all_map_points) > 0:

    map_points = np.vstack(
        all_map_points
    )

else:

    map_points = np.empty(
        (0, 3),
        dtype=np.float64
    )


# ============================================================
# 12. REMOVE EXTREME OUTLIERS
# ============================================================

if len(map_points) > 0:

    # Calculate distance from origin

    distances = np.linalg.norm(
        map_points,
        axis=1
    )


    # Robust percentile filtering

    upper_limit = np.percentile(
        distances,
        99
    )


    valid_distance_mask = (
        distances
        <=
        upper_limit
    )


    map_points = map_points[
        valid_distance_mask
    ]


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("3D MAP RECONSTRUCTION SUMMARY")
print("=" * 70)

print(
    f"\nKeyframes            : "
    f"{len(keyframe_indices)}"
)

print(
    f"Keyframe Pairs       : "
    f"{len(keyframe_indices) - 1}"
)

print(
    f"Successful Pairs     : "
    f"{successful_pairs}"
)

print(
    f"Failed/Skipped Pairs : "
    f"{failed_pairs}"
)

print(
    f"Final 3D Map Points  : "
    f"{len(map_points)}"
)


if len(map_points) > 0:

    print(
        "\n3D Coordinate Ranges"
    )

    print(
        f"X : "
        f"{np.min(map_points[:, 0]):.3f}"
        f" to "
        f"{np.max(map_points[:, 0]):.3f}"
    )

    print(
        f"Y : "
        f"{np.min(map_points[:, 1]):.3f}"
        f" to "
        f"{np.max(map_points[:, 1]):.3f}"
    )

    print(
        f"Z : "
        f"{np.min(map_points[:, 2]):.3f}"
        f" to "
        f"{np.max(map_points[:, 2]):.3f}"
    )


# ============================================================
# 14. SAVE MAP POINTS
# ============================================================

map_npy = os.path.join(
    RESULT_DIR,
    "task3_map_points.npy"
)

np.save(
    map_npy,
    map_points
)


# ============================================================
# 15. SAVE TEXT POINT CLOUD
# ============================================================

map_txt = os.path.join(
    RESULT_DIR,
    "task3_map_points.txt"
)

np.savetxt(
    map_txt,
    map_points,
    fmt="%.6f",
    header="X Y Z"
)


# ============================================================
# 16. SAVE STATISTICS
# ============================================================

statistics_file = os.path.join(
    RESULT_DIR,
    "task3_3d_map_statistics.txt"
)


with open(
    statistics_file,
    "w"
) as file:

    file.write(
        "STAGE 6 - TASK 3\n"
    )

    file.write(
        "3D MAP POINT RECONSTRUCTION\n"
    )

    file.write(
        "=" * 70
        +
        "\n\n"
    )

    file.write(
        f"Keyframes            : "
        f"{len(keyframe_indices)}\n"
    )

    file.write(
        f"Keyframe Pairs       : "
        f"{len(keyframe_indices) - 1}\n"
    )

    file.write(
        f"Successful Pairs     : "
        f"{successful_pairs}\n"
    )

    file.write(
        f"Failed/Skipped Pairs : "
        f"{failed_pairs}\n"
    )

    file.write(
        f"Final 3D Map Points  : "
        f"{len(map_points)}\n"
    )


    if len(map_points) > 0:

        file.write(
            "\n3D Coordinate Ranges\n"
        )

        file.write(
            f"X : "
            f"{np.min(map_points[:, 0]):.6f}"
            f" to "
            f"{np.max(map_points[:, 0]):.6f}\n"
        )

        file.write(
            f"Y : "
            f"{np.min(map_points[:, 1]):.6f}"
            f" to "
            f"{np.max(map_points[:, 1]):.6f}\n"
        )

        file.write(
            f"Z : "
            f"{np.min(map_points[:, 2]):.6f}"
            f" to "
            f"{np.max(map_points[:, 2]):.6f}\n"
        )


    file.write(
        "\nPAIR STATISTICS\n"
    )

    file.write(
        "Frame1 Frame2 "
        "GoodMatches Inliers "
        "Points3D Status\n"
    )


    for result in pair_statistics:

        file.write(
            f"{result['frame1']} "
            f"{result['frame2']} "
            f"{result['good_matches']} "
            f"{result['inliers']} "
            f"{result['points_3d']} "
            f"{result['status']}\n"
        )


# ============================================================
# 17. 3D MAP VISUALIZATION
# ============================================================

if len(map_points) > 0:

    fig = plt.figure(
        figsize=(12, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    # Downsample for faster plotting

    max_plot_points = 5000

    if len(map_points) > max_plot_points:

        plot_indices = np.linspace(
            0,
            len(map_points) - 1,
            max_plot_points,
            dtype=int
        )

        plot_points = map_points[
            plot_indices
        ]

    else:

        plot_points = map_points


    ax.scatter(
        plot_points[:, 0],
        plot_points[:, 1],
        plot_points[:, 2],
        s=4,
        alpha=0.7
    )


    ax.set_xlabel(
        "X"
    )

    ax.set_ylabel(
        "Y"
    )

    ax.set_zlabel(
        "Z"
    )

    ax.set_title(
        "Stage 6 - Sparse 3D Map"
    )


    plt.tight_layout()


    map_plot = os.path.join(
        RESULT_DIR,
        "task3_sparse_3d_map.png"
    )


    plt.savefig(
        map_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 18. TOP VIEW
# ============================================================

if len(map_points) > 0:

    plt.figure(
        figsize=(10, 8)
    )

    plt.scatter(
        map_points[:, 0],
        map_points[:, 2],
        s=4,
        alpha=0.7
    )

    plt.xlabel(
        "X"
    )

    plt.ylabel(
        "Z"
    )

    plt.title(
        "Stage 6 - 3D Map Top View"
    )

    plt.grid(
        True
    )

    plt.axis(
        "equal"
    )

    plt.tight_layout()


    top_view_path = os.path.join(
        RESULT_DIR,
        "task3_map_top_view.png"
    )


    plt.savefig(
        top_view_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 19. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("TASK 3 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nResults saved in:")
print(
    RESULT_DIR
)

print("\nGenerated files:")

print(
    "1. task3_map_points.npy"
)

print(
    "2. task3_map_points.txt"
)

print(
    "3. task3_3d_map_statistics.txt"
)

if len(map_points) > 0:

    print(
        "4. task3_sparse_3d_map.png"
    )

    print(
        "5. task3_map_top_view.png"
    )

print(
    "\nNo plot windows were opened."
)