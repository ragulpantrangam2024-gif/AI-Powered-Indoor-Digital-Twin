import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


# ============================================================
# STAGE 6 - TASK 4
# GLOBAL KEYFRAME-BASED 3D MAP
# ============================================================


# ============================================================
# 1. PATHS
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
print("STAGE 6 - TASK 4")
print("GLOBAL KEYFRAME-BASED 3D MAP")
print("=" * 70)

print("\nStage 6 directory:")
print(STAGE6_DIR)

print("\nImage directory:")
print(IMAGE_DIR)

print("\nResults directory:")
print(RESULT_DIR)


# ============================================================
# 2. CAMERA MATRIX
# ============================================================

K = np.array([
    [879.03578678, 0.0, 438.89954652],
    [0.0, 896.60912854, 557.52366461],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


print("\nCamera Intrinsic Matrix:")
print(K)


# ============================================================
# 3. PARAMETERS
# ============================================================

ORB_FEATURES = 1500

LOWE_RATIO = 0.75

MIN_GOOD_MATCHES = 15

MIN_POSE_INLIERS = 8

RANSAC_THRESHOLD = 1.0

RANSAC_CONFIDENCE = 0.999

# Distance used when merging nearby map points.
# This is in the arbitrary monocular reconstruction scale.

MERGE_DISTANCE = 0.20


# ============================================================
# 4. CHECK IMAGE DIRECTORY
# ============================================================

if not os.path.exists(IMAGE_DIR):

    print("\nERROR: Image directory not found:")
    print(IMAGE_DIR)

    exit()


# ============================================================
# 5. LOAD IMAGES
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

image_files = sorted(image_files)


print("\nImages found:")
print(len(image_files))


if len(image_files) < 2:

    print("\nERROR: Not enough images.")

    exit()


# ============================================================
# 6. LOAD KEYFRAME INDICES
# ============================================================

keyframe_file = os.path.join(
    RESULT_DIR,
    "keyframe_indices.npy"
)


if not os.path.exists(keyframe_file):

    print(
        "\nERROR: keyframe_indices.npy not found."
    )

    print(
        "Please run Task 1 first."
    )

    exit()


keyframe_indices = np.load(
    keyframe_file
).astype(int).tolist()


print("\nKeyframes loaded:")
print(len(keyframe_indices))

print("\nKeyframe indices:")
print(keyframe_indices)


# ============================================================
# 7. ORB + BF MATCHER
# ============================================================

orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES
)

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# 8. CAMERA POSE STORAGE
# ============================================================
#
# Rcw and tcw represent WORLD -> CAMERA transformation:
#
# X_camera = Rcw * X_world + tcw
#
# Projection:
#
# P = K [Rcw | tcw]
#
# ============================================================

camera_rotations = []

camera_translations = []

camera_positions = []


# Initial camera pose
# World coordinate system starts at keyframe 0.

R_world = np.eye(
    3,
    dtype=np.float64
)

t_world = np.zeros(
    (3, 1),
    dtype=np.float64
)


camera_rotations.append(
    R_world.copy()
)

camera_translations.append(
    t_world.copy()
)


camera_center = (
    -R_world.T @ t_world
).reshape(3)

camera_positions.append(
    camera_center
)


# ============================================================
# 9. MAP STORAGE
# ============================================================

map_points = []

map_observations = []

pair_statistics = []


successful_pairs = 0

failed_pairs = 0


# ============================================================
# 10. MERGE MAP POINT
# ============================================================

def find_existing_map_point(
    point,
    points,
    threshold
):

    if len(points) == 0:

        return -1

    points_array = np.asarray(
        points
    )

    distances = np.linalg.norm(
        points_array - point,
        axis=1
    )

    nearest_index = np.argmin(
        distances
    )

    if distances[
        nearest_index
    ] < threshold:

        return int(
            nearest_index
        )

    return -1


# ============================================================
# 11. PROCESS CONSECUTIVE KEYFRAME PAIRS
# ============================================================

print("\n")
print("=" * 70)
print("GLOBAL POSE ESTIMATION + TRIANGULATION")
print("=" * 70)


for pair_id in range(
    len(keyframe_indices) - 1
):

    idx1 = keyframe_indices[
        pair_id
    ]

    idx2 = keyframe_indices[
        pair_id + 1
    ]


    print(
        f"\nKeyframe "
        f"{idx1:03d} -> "
        f"{idx2:03d}"
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
            "  Images unavailable."
        )

        failed_pairs += 1

        # Keep same pose for failed pair

        camera_rotations.append(
            R_world.copy()
        )

        camera_translations.append(
            t_world.copy()
        )

        camera_positions.append(
            camera_center.copy()
        )

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
    # ORB
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

        camera_rotations.append(
            R_world.copy()
        )

        camera_translations.append(
            t_world.copy()
        )

        camera_positions.append(
            camera_center.copy()
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
    # Lowe ratio
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
            f"  Skipped: "
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


        camera_rotations.append(
            R_world.copy()
        )

        camera_translations.append(
            t_world.copy()
        )

        camera_positions.append(
            camera_center.copy()
        )

        continue


    # --------------------------------------------------------
    # Matched points
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
    # Essential matrix
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

        camera_rotations.append(
            R_world.copy()
        )

        camera_translations.append(
            t_world.copy()
        )

        camera_positions.append(
            camera_center.copy()
        )

        continue


    # --------------------------------------------------------
    # Essential matrix inliers
    # --------------------------------------------------------

    E_mask = mask.ravel().astype(
        np.uint8
    )

    E_indices = np.where(
        E_mask == 1
    )[0]


    if len(E_indices) < MIN_POSE_INLIERS:

        print(
            f"  Not enough inliers: "
            f"{len(E_indices)}"
        )

        failed_pairs += 1

        camera_rotations.append(
            R_world.copy()
        )

        camera_translations.append(
            t_world.copy()
        )

        camera_positions.append(
            camera_center.copy()
        )

        continue


    E_pts1 = pts1[
        E_indices
    ]

    E_pts2 = pts2[
        E_indices
    ]


    # --------------------------------------------------------
    # Recover relative pose
    # --------------------------------------------------------

    pose_inliers, R_rel, t_rel, pose_mask = (
        cv2.recoverPose(
            E,
            E_pts1,
            E_pts2,
            K
        )
    )


    if pose_inliers < MIN_POSE_INLIERS:

        print(
            f"  Pose recovery failed: "
            f"{pose_inliers} inliers."
        )

        failed_pairs += 1

        camera_rotations.append(
            R_world.copy()
        )

        camera_translations.append(
            t_world.copy()
        )

        camera_positions.append(
            camera_center.copy()
        )

        continue


    # --------------------------------------------------------
    # Current global pose is camera 1.
    #
    # Relative pose:
    #
    # X2 = R_rel X1 + t_rel
    #
    # Therefore:
    #
    # R2 = R_rel R1
    #
    # t2 = R_rel t1 + t_rel
    #
    # --------------------------------------------------------

    R_previous = R_world.copy()

    t_previous = t_world.copy()


    R_world = (
        R_rel
        @
        R_previous
    )


    t_world = (
        R_rel
        @
        t_previous
        +
        t_rel
    )


    # --------------------------------------------------------
    # Camera center
    # --------------------------------------------------------

    camera_center = (
        -R_world.T
        @
        t_world
    ).reshape(3)


    camera_rotations.append(
        R_world.copy()
    )

    camera_translations.append(
        t_world.copy()
    )

    camera_positions.append(
        camera_center.copy()
    )


    # --------------------------------------------------------
    # Pose inlier points
    # --------------------------------------------------------

    pose_mask = pose_mask.ravel()

    valid_pose = np.where(
        pose_mask > 0
    )[0]


    tri_pts1 = E_pts1[
        valid_pose
    ]

    tri_pts2 = E_pts2[
        valid_pose
    ]


    if len(tri_pts1) < MIN_POSE_INLIERS:

        print(
            "  Not enough points for triangulation."
        )

        failed_pairs += 1

        continue


    # ========================================================
    # GLOBAL PROJECTION MATRICES
    # ========================================================

    # Camera 1
    #
    # Use previous global pose.

    P1 = K @ np.hstack(
        (
            R_previous,
            t_previous
        )
    )


    # Camera 2
    #
    # Use current global pose.

    P2 = K @ np.hstack(
        (
            R_world,
            t_world
        )
    )


    # ========================================================
    # TRIANGULATION
    # ========================================================

    points_4d = cv2.triangulatePoints(
        P1,
        P2,
        tri_pts1.T,
        tri_pts2.T
    )


    # Convert homogeneous coordinates

    points_3d = (
        points_4d[:3]
        /
        points_4d[3]
    ).T


    # --------------------------------------------------------
    # Remove NaN / Inf
    # --------------------------------------------------------

    valid = np.isfinite(
        points_3d
    ).all(
        axis=1
    )


    points_3d = points_3d[
        valid
    ]


    # --------------------------------------------------------
    # Positive depth check
    # --------------------------------------------------------

    if len(points_3d) > 0:

        # Depth in camera 1

        depth1 = (
            R_previous
            @
            points_3d.T
            +
            t_previous
        )[2]


        # Depth in camera 2

        depth2 = (
            R_world
            @
            points_3d.T
            +
            t_world
        )[2]


        positive_depth = (
            (depth1 > 0)
            &
            (depth2 > 0)
        )


        points_3d = points_3d[
            positive_depth
        ]


    point_count = len(
        points_3d
    )


    # ========================================================
    # MERGE MAP POINTS
    # ========================================================

    new_points = 0

    merged_points = 0


    for point in points_3d:

        existing_index = (
            find_existing_map_point(
                point,
                map_points,
                MERGE_DISTANCE
            )
        )


        if existing_index >= 0:

            # Existing landmark

            map_observations[
                existing_index
            ].add(
                pair_id
            )

            map_observations[
                existing_index
            ].add(
                pair_id + 1
            )

            merged_points += 1


        else:

            # New landmark

            map_points.append(
                point.copy()
            )

            map_observations.append(
                {
                    pair_id,
                    pair_id + 1
                }
            )

            new_points += 1


    successful_pairs += 1


    pair_statistics.append({
        "frame1": idx1,
        "frame2": idx2,
        "good_matches": good_count,
        "inliers": int(pose_inliers),
        "points_3d": point_count,
        "new_points": new_points,
        "merged_points": merged_points,
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

    print(
        f"  New Map      : "
        f"{new_points}"
    )

    print(
        f"  Merged       : "
        f"{merged_points}"
    )


# ============================================================
# 12. CONVERT TO NUMPY
# ============================================================

if len(map_points) > 0:

    map_points_array = np.asarray(
        map_points,
        dtype=np.float64
    )

else:

    map_points_array = np.empty(
        (0, 3),
        dtype=np.float64
    )


# ============================================================
# 13. CAMERA POSITIONS
# ============================================================

camera_positions_array = np.asarray(
    camera_positions,
    dtype=np.float64
)


# ============================================================
# 14. OBSERVATION STATISTICS
# ============================================================

if len(map_observations) > 0:

    observation_counts = np.array([
        len(obs)
        for obs in map_observations
    ])

else:

    observation_counts = np.empty(
        0,
        dtype=int
    )


# ============================================================
# 15. SAVE GLOBAL MAP
# ============================================================

map_path = os.path.join(
    RESULT_DIR,
    "task4_global_map_points.npy"
)

np.save(
    map_path,
    map_points_array
)


# ============================================================
# 16. SAVE CAMERA POSITIONS
# ============================================================

camera_path = os.path.join(
    RESULT_DIR,
    "task4_keyframe_camera_positions.npy"
)

np.save(
    camera_path,
    camera_positions_array
)


# ============================================================
# 17. SAVE OBSERVATION COUNTS
# ============================================================

observation_path = os.path.join(
    RESULT_DIR,
    "task4_map_point_observations.npy"
)

np.save(
    observation_path,
    observation_counts
)


# ============================================================
# 18. SAVE KEYFRAME POSES
# ============================================================

pose_path = os.path.join(
    RESULT_DIR,
    "task4_keyframe_poses.txt"
)


with open(
    pose_path,
    "w"
) as file:

    for i in range(
        len(camera_rotations)
    ):

        file.write(
            f"\nKeyframe {i}\n"
        )

        file.write(
            "Rotation:\n"
        )

        file.write(
            np.array2string(
                camera_rotations[i],
                precision=6
            )
        )

        file.write(
            "\nTranslation:\n"
        )

        file.write(
            np.array2string(
                camera_translations[i],
                precision=6
            )
        )

        file.write(
            "\nCamera Position:\n"
        )

        file.write(
            np.array2string(
                camera_positions_array[i],
                precision=6
            )
        )

        file.write(
            "\n"
        )


# ============================================================
# 19. SAVE MAP TEXT
# ============================================================

map_txt_path = os.path.join(
    RESULT_DIR,
    "task4_global_map_points.txt"
)


with open(
    map_txt_path,
    "w"
) as file:

    file.write(
        "MapPointID X Y Z Observations\n"
    )

    for i, point in enumerate(
        map_points_array
    ):

        observations = (
            observation_counts[i]
            if i < len(
                observation_counts
            )
            else 0
        )

        file.write(
            f"{i} "
            f"{point[0]:.6f} "
            f"{point[1]:.6f} "
            f"{point[2]:.6f} "
            f"{observations}\n"
        )


# ============================================================
# 20. SAVE STATISTICS
# ============================================================

statistics_path = os.path.join(
    RESULT_DIR,
    "task4_global_map_statistics.txt"
)


with open(
    statistics_path,
    "w"
) as file:

    file.write(
        "STAGE 6 - TASK 4\n"
    )

    file.write(
        "GLOBAL KEYFRAME-BASED 3D MAP\n"
    )

    file.write(
        "=" * 70
        +
        "\n\n"
    )

    file.write(
        f"Total images       : "
        f"{len(image_files)}\n"
    )

    file.write(
        f"Keyframes          : "
        f"{len(keyframe_indices)}\n"
    )

    file.write(
        f"Keyframe pairs     : "
        f"{len(keyframe_indices) - 1}\n"
    )

    file.write(
        f"Successful pairs   : "
        f"{successful_pairs}\n"
    )

    file.write(
        f"Failed pairs       : "
        f"{failed_pairs}\n"
    )

    file.write(
        f"Global map points  : "
        f"{len(map_points_array)}\n"
    )


    if len(
        map_points_array
    ) > 0:

        file.write(
            "\nGlobal 3D Ranges\n"
        )

        file.write(
            f"X : "
            f"{np.min(map_points_array[:,0]):.6f}"
            f" to "
            f"{np.max(map_points_array[:,0]):.6f}\n"
        )

        file.write(
            f"Y : "
            f"{np.min(map_points_array[:,1]):.6f}"
            f" to "
            f"{np.max(map_points_array[:,1]):.6f}\n"
        )

        file.write(
            f"Z : "
            f"{np.min(map_points_array[:,2]):.6f}"
            f" to "
            f"{np.max(map_points_array[:,2]):.6f}\n"
        )


        file.write(
            "\nObservation Statistics\n"
        )

        file.write(
            f"Minimum observations : "
            f"{np.min(observation_counts)}\n"
        )

        file.write(
            f"Maximum observations : "
            f"{np.max(observation_counts)}\n"
        )

        file.write(
            f"Average observations : "
            f"{np.mean(observation_counts):.3f}\n"
        )


    file.write(
        "\nPAIR STATISTICS\n"
    )

    file.write(
        "Frame1 Frame2 "
        "GoodMatches Inliers "
        "Points3D NewPoints "
        "Merged Status\n"
    )


    for result in pair_statistics:

        file.write(
            f"{result.get('frame1', -1)} "
            f"{result.get('frame2', -1)} "
            f"{result.get('good_matches', 0)} "
            f"{result.get('inliers', 0)} "
            f"{result.get('points_3d', 0)} "
            f"{result.get('new_points', 0)} "
            f"{result.get('merged_points', 0)} "
            f"{result.get('status', 'UNKNOWN')}\n"
        )


# ============================================================
# 21. GLOBAL 3D MAP VISUALIZATION
# ============================================================

if len(
    map_points_array
) > 0:

    fig = plt.figure(
        figsize=(12, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    # Plot map points

    max_points_to_plot = 6000


    if len(
        map_points_array
    ) > max_points_to_plot:

        indices = np.linspace(
            0,
            len(map_points_array) - 1,
            max_points_to_plot,
            dtype=int
        )

        plot_points = (
            map_points_array[
                indices
            ]
        )

    else:

        plot_points = (
            map_points_array
        )


    ax.scatter(
        plot_points[:,0],
        plot_points[:,1],
        plot_points[:,2],
        s=4,
        alpha=0.6,
        label="Map Points"
    )


    # Plot camera trajectory

    ax.plot(
        camera_positions_array[:,0],
        camera_positions_array[:,1],
        camera_positions_array[:,2],
        linewidth=2,
        label="Keyframe Trajectory"
    )


    # Start

    ax.scatter(
        camera_positions_array[0,0],
        camera_positions_array[0,1],
        camera_positions_array[0,2],
        s=70,
        label="Start"
    )


    # End

    ax.scatter(
        camera_positions_array[-1,0],
        camera_positions_array[-1,1],
        camera_positions_array[-1,2],
        s=70,
        label="End"
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
        "Stage 6 - Global Keyframe-Based 3D Map"
    )

    ax.legend()


    plt.tight_layout()


    global_map_plot = os.path.join(
        RESULT_DIR,
        "task4_global_3d_map.png"
    )


    plt.savefig(
        global_map_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 22. TOP VIEW
# ============================================================

if len(
    map_points_array
) > 0:

    plt.figure(
        figsize=(10, 8)
    )


    plt.scatter(
        map_points_array[:,0],
        map_points_array[:,2],
        s=4,
        alpha=0.6,
        label="Map Points"
    )


    plt.plot(
        camera_positions_array[:,0],
        camera_positions_array[:,2],
        linewidth=2,
        label="Keyframe Trajectory"
    )


    plt.scatter(
        camera_positions_array[0,0],
        camera_positions_array[0,2],
        s=70,
        label="Start"
    )


    plt.scatter(
        camera_positions_array[-1,0],
        camera_positions_array[-1,2],
        s=70,
        label="End"
    )


    plt.xlabel(
        "X"
    )

    plt.ylabel(
        "Z"
    )

    plt.title(
        "Stage 6 - Global Map Top View"
    )

    plt.grid(
        True
    )

    plt.axis(
        "equal"
    )

    plt.legend()


    plt.tight_layout()


    top_view_path = os.path.join(
        RESULT_DIR,
        "task4_global_map_top_view.png"
    )


    plt.savefig(
        top_view_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 23. OBSERVATION DISTRIBUTION
# ============================================================

if len(
    observation_counts
) > 0:

    plt.figure(
        figsize=(10, 6)
    )


    plt.hist(
        observation_counts,
        bins=np.arange(
            0,
            np.max(
                observation_counts
            ) + 2
        ) - 0.5
    )


    plt.xlabel(
        "Number of Keyframe Observations"
    )

    plt.ylabel(
        "Number of Map Points"
    )

    plt.title(
        "Stage 6 - Map Point Observation Distribution"
    )

    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    observation_plot = os.path.join(
        RESULT_DIR,
        "task4_observation_distribution.png"
    )


    plt.savefig(
        observation_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("TASK 4 COMPLETED SUCCESSFULLY")
print("=" * 70)


print(
    "\nTotal Images:"
)

print(
    len(image_files)
)


print(
    "\nKeyframes:"
)

print(
    len(keyframe_indices)
)


print(
    "\nKeyframe Pairs:"
)

print(
    len(keyframe_indices) - 1
)


print(
    "\nSuccessful Pairs:"
)

print(
    successful_pairs
)


print(
    "\nFailed Pairs:"
)

print(
    failed_pairs
)


print(
    "\nGlobal Map Points:"
)

print(
    len(map_points_array)
)


if len(
    observation_counts
) > 0:

    print(
        "\nMap Point Observations:"
    )

    print(
        f"Minimum : "
        f"{np.min(observation_counts)}"
    )

    print(
        f"Maximum : "
        f"{np.max(observation_counts)}"
    )

    print(
        f"Average : "
        f"{np.mean(observation_counts):.3f}"
    )


print(
    "\nResults saved in:"
)

print(
    RESULT_DIR
)


print(
    "\nGenerated files:"
)

print(
    "1. task4_global_map_points.npy"
)

print(
    "2. task4_global_map_points.txt"
)

print(
    "3. task4_keyframe_camera_positions.npy"
)

print(
    "4. task4_map_point_observations.npy"
)

print(
    "5. task4_keyframe_poses.txt"
)

print(
    "6. task4_global_map_statistics.txt"
)

print(
    "7. task4_global_3d_map.png"
)

print(
    "8. task4_global_map_top_view.png"
)

print(
    "9. task4_observation_distribution.png"
)

print(
    "\nNo plot windows were opened."
)