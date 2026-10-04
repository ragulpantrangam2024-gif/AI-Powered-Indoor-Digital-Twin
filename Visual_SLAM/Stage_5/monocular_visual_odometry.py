import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


# ============================================================
# STAGE 5 MINI PROJECT
# MONOCULAR VISUAL ODOMETRY
# ============================================================


# ============================================================
# 1. PROJECT PATH DETECTION
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Your current structure is:
#
# TwinCube/
# └── Stage_5/
#     ├── monocular_visual_odometry.py
#     ├── images/
#     └── results/
#
# Therefore, if "images" exists next to this script,
# SCRIPT_DIR is already Stage_5.

if os.path.exists(
    os.path.join(
        SCRIPT_DIR,
        "images"
    )
):

    STAGE5_DIR = SCRIPT_DIR

else:

    # Fallback in case the script is placed inside
    # Stage_5/mini_project/

    possible_stage5 = os.path.dirname(
        SCRIPT_DIR
    )

    if os.path.exists(
        os.path.join(
            possible_stage5,
            "images"
        )
    ):

        STAGE5_DIR = possible_stage5

    else:

        STAGE5_DIR = SCRIPT_DIR


# ============================================================
# 2. INPUT / OUTPUT PATHS
# ============================================================

IMAGE_DIR = os.path.join(
    STAGE5_DIR,
    "images"
)

RESULT_DIR = os.path.join(
    STAGE5_DIR,
    "results",
    "mini_project"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# 3. PROJECT INFORMATION
# ============================================================

print("=" * 70)
print("MONOCULAR VISUAL ODOMETRY")
print("STAGE 5 MINI PROJECT")
print("=" * 70)

print("\nScript location:")
print(
    SCRIPT_DIR
)

print("\nStage 5 directory:")
print(
    STAGE5_DIR
)

print("\nImage directory:")
print(
    IMAGE_DIR
)

print("\nResults directory:")
print(
    RESULT_DIR
)


# ============================================================
# 4. CHECK IMAGE DIRECTORY
# ============================================================

if not os.path.exists(
    IMAGE_DIR
):

    print("\n")
    print("=" * 70)
    print("ERROR: IMAGE DIRECTORY NOT FOUND")
    print("=" * 70)

    print("\nPython expected:")
    print(
        IMAGE_DIR
    )

    print("\nPlease check your folder structure.")

    exit()


# ============================================================
# 5. FIND IMAGE FILES
# ============================================================

image_files = []

# JPG
image_files.extend(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.jpg"
        )
    )
)

# JPEG
image_files.extend(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.jpeg"
        )
    )
)

# PNG
image_files.extend(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.png"
        )
    )
)

# Sort images
image_files = sorted(
    image_files
)


# ============================================================
# 6. CHECK IMAGE COUNT
# ============================================================

print("\nImages found:")
print(
    len(image_files)
)

if len(image_files) < 2:

    print("\n")
    print("=" * 70)
    print("ERROR: NOT ENOUGH IMAGES")
    print("=" * 70)

    print(
        "\nExpected images in:"
    )

    print(
        IMAGE_DIR
    )

    print(
        "\nSupported formats:"
    )

    print(
        "JPG / JPEG / PNG"
    )

    exit()


# ============================================================
# 7. DISPLAY IMAGE INFORMATION
# ============================================================

print("\nFirst five images:")

for filename in image_files[:5]:

    print(
        os.path.basename(
            filename
        )
    )

print("\nLast image:")

print(
    os.path.basename(
        image_files[-1]
    )
)

print(
    f"\nTotal images: "
    f"{len(image_files)}"
)


# ============================================================
# 8. CAMERA INTRINSIC MATRIX
# ============================================================

# Obtained from Stage 3 camera calibration

K = np.array([
    [879.03578678, 0.0, 438.89954652],
    [0.0, 896.60912854, 557.52366461],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


print("\n")
print("=" * 70)
print("CAMERA INTRINSIC MATRIX")
print("=" * 70)

print(K)


# ============================================================
# 9. ORB FEATURE DETECTOR
# ============================================================

orb = cv2.ORB_create(
    nfeatures=1000
)


# ============================================================
# 10. BRUTE-FORCE MATCHER
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# 11. INITIAL CAMERA POSE
# ============================================================

# First camera defines the world coordinate system.
#
# Camera position:
#
# C = [0, 0, 0]
#
# Orientation:
#
# R = Identity

R_global = np.eye(
    3,
    dtype=np.float64
)

C_global = np.zeros(
    (3, 1),
    dtype=np.float64
)


# ============================================================
# 12. TRAJECTORY STORAGE
# ============================================================

trajectory = [
    C_global.flatten()
]

rotation_history = [
    R_global.copy()
]

relative_translation_history = []


# ============================================================
# 13. MATCH / POSE STATISTICS
# ============================================================

good_match_counts = []

pose_inlier_counts = []

pose_outlier_counts = []

processed_pairs = 0

successful_pairs = 0

failed_pairs = 0


# ============================================================
# 14. FIRST MATCH IMAGE
# ============================================================

first_match_saved = False


# ============================================================
# 15. START VISUAL ODOMETRY
# ============================================================

print("\n")
print("=" * 70)
print("STARTING VISUAL ODOMETRY")
print("=" * 70)


# ============================================================
# 16. PROCESS CONSECUTIVE IMAGE PAIRS
# ============================================================

for i in range(
    len(image_files) - 1
):

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    image_path_1 = image_files[i]

    image_path_2 = image_files[i + 1]


    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    img1 = cv2.imread(
        image_path_1
    )

    img2 = cv2.imread(
        image_path_2
    )


    # --------------------------------------------------------
    # Check images
    # --------------------------------------------------------

    if img1 is None or img2 is None:

        print(
            f"Pair {i + 1:03d}: "
            f"Could not load images."
        )

        failed_pairs += 1

        trajectory.append(
            C_global.flatten()
        )

        rotation_history.append(
            R_global.copy()
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
    # ORB feature detection
    # --------------------------------------------------------

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
            f"Pair {i + 1:03d}: "
            f"Descriptors unavailable."
        )

        failed_pairs += 1

        trajectory.append(
            C_global.flatten()
        )

        rotation_history.append(
            R_global.copy()
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

        if m.distance < (
            0.75 * n.distance
        ):

            good_matches.append(
                m
            )


    good_match_count = len(
        good_matches
    )

    good_match_counts.append(
        good_match_count
    )


    # --------------------------------------------------------
    # Save first match visualization
    # --------------------------------------------------------

    if (
        not first_match_saved
        and
        good_match_count >= 8
    ):

        match_image = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        match_output = os.path.join(
            RESULT_DIR,
            "first_pair_matches.jpg"
        )

        cv2.imwrite(
            match_output,
            match_image
        )

        first_match_saved = True


    # --------------------------------------------------------
    # Minimum match requirement
    # --------------------------------------------------------

    if good_match_count < 8:

        print(
            f"Pair {i + 1:03d}: "
            f"Insufficient matches = "
            f"{good_match_count}"
        )

        failed_pairs += 1

        trajectory.append(
            C_global.flatten()
        )

        rotation_history.append(
            R_global.copy()
        )

        continue


    # --------------------------------------------------------
    # Extract matched points
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

    E, essential_mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )


    if E is None:

        print(
            f"Pair {i + 1:03d}: "
            f"Essential Matrix failed."
        )

        failed_pairs += 1

        trajectory.append(
            C_global.flatten()
        )

        rotation_history.append(
            R_global.copy()
        )

        continue


    # --------------------------------------------------------
    # Recover relative pose
    # --------------------------------------------------------

    pose_inliers, R_relative, t_relative, pose_mask = (
        cv2.recoverPose(
            E,
            pts1,
            pts2,
            K
        )
    )


    # --------------------------------------------------------
    # Inlier / outlier statistics
    # --------------------------------------------------------

    inliers = int(
        pose_inliers
    )

    outliers = (
        good_match_count
        -
        inliers
    )

    pose_inlier_counts.append(
        inliers
    )

    pose_outlier_counts.append(
        outliers
    )


    # --------------------------------------------------------
    # CAMERA CENTER UPDATE
    # --------------------------------------------------------
    #
    # recoverPose:
    #
    # X2 = R * X1 + t
    #
    # Camera center displacement:
    #
    # C_relative = -R.T @ t
    #
    # Transform relative displacement into
    # the existing world coordinate system.
    #


    # Relative camera displacement

    C_relative = (
        -R_relative.T
        @
        t_relative
    )


    # Save previous global rotation

    R_old = R_global.copy()


    # Update camera center

    C_global = (
        C_global
        +
        R_old @ C_relative
    )


    # Update camera orientation

    R_global = (
        R_old
        @
        R_relative.T
    )


    # --------------------------------------------------------
    # Save relative translation
    # --------------------------------------------------------

    relative_translation_history.append(
        t_relative.flatten()
    )


    # --------------------------------------------------------
    # Store trajectory
    # --------------------------------------------------------

    trajectory.append(
        C_global.flatten()
    )

    rotation_history.append(
        R_global.copy()
    )


    successful_pairs += 1

    processed_pairs += 1


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        (i + 1) % 10 == 0
        or
        i == 0
    ):

        print(
            f"Pair {i + 1:03d} | "
            f"Good Matches: "
            f"{good_match_count:3d} | "
            f"Inliers: "
            f"{inliers:3d} | "
            f"Outliers: "
            f"{outliers:3d}"
        )


# ============================================================
# 17. CONVERT TRAJECTORY
# ============================================================

trajectory = np.array(
    trajectory,
    dtype=np.float64
)


# ============================================================
# 18. SAVE CAMERA TRAJECTORY
# ============================================================

trajectory_file = os.path.join(
    RESULT_DIR,
    "camera_trajectory.npy"
)

np.save(
    trajectory_file,
    trajectory
)


trajectory_txt = os.path.join(
    RESULT_DIR,
    "camera_trajectory.txt"
)

np.savetxt(
    trajectory_txt,
    trajectory,
    fmt="%.6f",
    header="X Y Z"
)


# ============================================================
# 19. TRAJECTORY STATISTICS
# ============================================================

start_position = trajectory[0]

end_position = trajectory[-1]


# ------------------------------------------------------------
# Displacement
# ------------------------------------------------------------

displacement_vector = (
    end_position
    -
    start_position
)

displacement = np.linalg.norm(
    displacement_vector
)


# ------------------------------------------------------------
# Frame-to-frame movement
# ------------------------------------------------------------

if len(trajectory) > 1:

    frame_differences = (
        trajectory[1:]
        -
        trajectory[:-1]
    )

    frame_distances = np.linalg.norm(
        frame_differences,
        axis=1
    )

else:

    frame_distances = np.array(
        []
    )


# ------------------------------------------------------------
# Path length
# ------------------------------------------------------------

if len(frame_distances) > 0:

    total_path_length = np.sum(
        frame_distances
    )

    average_motion = np.mean(
        frame_distances
    )

    median_motion = np.median(
        frame_distances
    )

    minimum_motion = np.min(
        frame_distances
    )

    maximum_motion = np.max(
        frame_distances
    )

else:

    total_path_length = 0.0

    average_motion = 0.0

    median_motion = 0.0

    minimum_motion = 0.0

    maximum_motion = 0.0


# ------------------------------------------------------------
# Directness
# ------------------------------------------------------------

if total_path_length > 0:

    directness = (
        displacement
        /
        total_path_length
    )

else:

    directness = 0.0


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("VISUAL ODOMETRY SUMMARY")
print("=" * 70)

print(
    f"\nTotal Image Pairs : "
    f"{len(image_files) - 1}"
)

print(
    f"Processed Pairs   : "
    f"{processed_pairs}"
)

print(
    f"Successful Pairs  : "
    f"{successful_pairs}"
)

print(
    f"Failed Pairs      : "
    f"{failed_pairs}"
)

print(
    f"\nTrajectory Shape : "
    f"{trajectory.shape}"
)


print("\nInitial Camera Position")
print("-----------------------")

print(
    start_position
)


print("\nFinal Camera Position")
print("---------------------")

print(
    end_position
)


print("\nStart-to-End Displacement")
print("-------------------------")

print(
    "Displacement Vector:"
)

print(
    displacement_vector
)

print(
    f"\nDisplacement Magnitude "
    f"(arbitrary scale): "
    f"{displacement:.6f}"
)


print("\nTotal Estimated Path Length")
print("---------------------------")

print(
    f"{total_path_length:.6f} "
    f"(arbitrary scale)"
)


print("\nFrame-to-Frame Motion")
print("---------------------")

print(
    f"Average : "
    f"{average_motion:.6f}"
)

print(
    f"Median  : "
    f"{median_motion:.6f}"
)

print(
    f"Minimum : "
    f"{minimum_motion:.6f}"
)

print(
    f"Maximum : "
    f"{maximum_motion:.6f}"
)


print("\nTrajectory Directness")
print("---------------------")

print(
    f"{directness:.6f}"
)


# ============================================================
# 21. MATCH STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("MATCH STATISTICS")
print("=" * 70)

if len(good_match_counts) > 0:

    print(
        f"\nMinimum Good Matches : "
        f"{np.min(good_match_counts)}"
    )

    print(
        f"Maximum Good Matches : "
        f"{np.max(good_match_counts)}"
    )

    print(
        f"Average Good Matches : "
        f"{np.mean(good_match_counts):.2f}"
    )

else:

    print(
        "\nNo valid match statistics."
    )


# ============================================================
# 22. POSE STATISTICS
# ============================================================

print("\n")
print("=" * 70)
print("POSE STATISTICS")
print("=" * 70)

if len(pose_inlier_counts) > 0:

    print(
        f"\nMinimum Pose Inliers : "
        f"{np.min(pose_inlier_counts)}"
    )

    print(
        f"Maximum Pose Inliers : "
        f"{np.max(pose_inlier_counts)}"
    )

    print(
        f"Average Pose Inliers : "
        f"{np.mean(pose_inlier_counts):.2f}"
    )

    print(
        f"Average Pose Outliers : "
        f"{np.mean(pose_outlier_counts):.2f}"
    )

else:

    print(
        "\nNo pose statistics."
    )


# ============================================================
# 23. SAVE STATISTICS
# ============================================================

statistics_file = os.path.join(
    RESULT_DIR,
    "vo_statistics.txt"
)

with open(
    statistics_file,
    "w"
) as file:

    file.write(
        "STAGE 5 MINI PROJECT\n"
    )

    file.write(
        "MONOCULAR VISUAL ODOMETRY\n"
    )

    file.write(
        "=" * 70
        +
        "\n\n"
    )

    file.write(
        f"Total Image Pairs : "
        f"{len(image_files) - 1}\n"
    )

    file.write(
        f"Processed Pairs : "
        f"{processed_pairs}\n"
    )

    file.write(
        f"Successful Pairs : "
        f"{successful_pairs}\n"
    )

    file.write(
        f"Failed Pairs : "
        f"{failed_pairs}\n\n"
    )

    file.write(
        f"Trajectory Shape : "
        f"{trajectory.shape}\n\n"
    )

    file.write(
        "Initial Camera Position\n"
    )

    file.write(
        str(start_position)
        +
        "\n\n"
    )

    file.write(
        "Final Camera Position\n"
    )

    file.write(
        str(end_position)
        +
        "\n\n"
    )

    file.write(
        "Displacement Vector\n"
    )

    file.write(
        str(displacement_vector)
        +
        "\n"
    )

    file.write(
        f"Displacement Magnitude "
        f"(arbitrary scale): "
        f"{displacement:.6f}\n\n"
    )

    file.write(
        f"Total Path Length "
        f"(arbitrary scale): "
        f"{total_path_length:.6f}\n"
    )

    file.write(
        f"Average Frame Motion: "
        f"{average_motion:.6f}\n"
    )

    file.write(
        f"Median Frame Motion: "
        f"{median_motion:.6f}\n"
    )

    file.write(
        f"Minimum Frame Motion: "
        f"{minimum_motion:.6f}\n"
    )

    file.write(
        f"Maximum Frame Motion: "
        f"{maximum_motion:.6f}\n\n"
    )

    file.write(
        f"Trajectory Directness: "
        f"{directness:.6f}\n\n"
    )


    if len(good_match_counts) > 0:

        file.write(
            "MATCH STATISTICS\n"
        )

        file.write(
            f"Minimum Good Matches : "
            f"{np.min(good_match_counts)}\n"
        )

        file.write(
            f"Maximum Good Matches : "
            f"{np.max(good_match_counts)}\n"
        )

        file.write(
            f"Average Good Matches : "
            f"{np.mean(good_match_counts):.2f}\n\n"
        )


    if len(pose_inlier_counts) > 0:

        file.write(
            "POSE STATISTICS\n"
        )

        file.write(
            f"Minimum Pose Inliers : "
            f"{np.min(pose_inlier_counts)}\n"
        )

        file.write(
            f"Maximum Pose Inliers : "
            f"{np.max(pose_inlier_counts)}\n"
        )

        file.write(
            f"Average Pose Inliers : "
            f"{np.mean(pose_inlier_counts):.2f}\n"
        )

        file.write(
            f"Average Pose Outliers : "
            f"{np.mean(pose_outlier_counts):.2f}\n\n"
        )


    file.write(
        "IMPORTANT LIMITATION\n"
    )

    file.write(
        "This is monocular visual odometry.\n"
    )

    file.write(
        "Absolute translation scale is unknown.\n"
    )

    file.write(
        "Therefore trajectory distances are "
        "reported in arbitrary scale units.\n"
    )


# ============================================================
# 24. SAVE ROTATION HISTORY
# ============================================================

rotation_array = np.array(
    rotation_history
)

rotation_file = os.path.join(
    RESULT_DIR,
    "rotation_history.npy"
)

np.save(
    rotation_file,
    rotation_array
)


# ============================================================
# 25. 3D CAMERA TRAJECTORY PLOT
# ============================================================

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    trajectory[:, 2],
    linewidth=2,
    label="Camera Trajectory"
)

ax.scatter(
    trajectory[0, 0],
    trajectory[0, 1],
    trajectory[0, 2],
    s=100,
    label="Start"
)

ax.scatter(
    trajectory[-1, 0],
    trajectory[-1, 1],
    trajectory[-1, 2],
    s=100,
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
    "Monocular Visual Odometry - Camera Trajectory"
)

ax.legend()

plt.tight_layout()

trajectory_plot = os.path.join(
    RESULT_DIR,
    "camera_trajectory.png"
)

plt.savefig(
    trajectory_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 26. TOP VIEW TRAJECTORY
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.plot(
    trajectory[:, 0],
    trajectory[:, 1],
    linewidth=2
)

plt.scatter(
    trajectory[0, 0],
    trajectory[0, 1],
    s=100,
    label="Start"
)

plt.scatter(
    trajectory[-1, 0],
    trajectory[-1, 1],
    s=100,
    label="End"
)

plt.xlabel(
    "X"
)

plt.ylabel(
    "Y"
)

plt.title(
    "Monocular Visual Odometry - Top View"
)

plt.grid(
    True
)

plt.axis(
    "equal"
)

plt.legend()

plt.tight_layout()

top_view_plot = os.path.join(
    RESULT_DIR,
    "camera_trajectory_top_view.png"
)

plt.savefig(
    top_view_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 27. FRAME-TO-FRAME MOTION PLOT
# ============================================================

if len(frame_distances) > 0:

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        frame_distances,
        linewidth=1.5
    )

    plt.xlabel(
        "Frame Pair"
    )

    plt.ylabel(
        "Motion Magnitude"
    )

    plt.title(
        "Frame-to-Frame Camera Motion"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    motion_plot = os.path.join(
        RESULT_DIR,
        "frame_motion.png"
    )

    plt.savefig(
        motion_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 28. GOOD MATCHES PLOT
# ============================================================

if len(good_match_counts) > 0:

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        good_match_counts,
        linewidth=1.5
    )

    plt.xlabel(
        "Frame Pair"
    )

    plt.ylabel(
        "Good Matches"
    )

    plt.title(
        "Good Feature Matches per Frame Pair"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    match_plot = os.path.join(
        RESULT_DIR,
        "good_matches_per_pair.png"
    )

    plt.savefig(
        match_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 29. POSE INLIERS PLOT
# ============================================================

if len(pose_inlier_counts) > 0:

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        pose_inlier_counts,
        linewidth=1.5
    )

    plt.xlabel(
        "Frame Pair"
    )

    plt.ylabel(
        "Pose Inliers"
    )

    plt.title(
        "Pose Inliers per Frame Pair"
    )

    plt.grid(
        True
    )

    plt.tight_layout()

    inlier_plot = os.path.join(
        RESULT_DIR,
        "pose_inliers_per_pair.png"
    )

    plt.savefig(
        inlier_plot,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 30. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("MONOCULAR VISUAL ODOMETRY COMPLETED")
print("=" * 70)

print("\nResults saved in:")

print(
    RESULT_DIR
)

print("\nGenerated files:")

print(
    "1. camera_trajectory.npy"
)

print(
    "2. camera_trajectory.txt"
)

print(
    "3. rotation_history.npy"
)

print(
    "4. vo_statistics.txt"
)

print(
    "5. first_pair_matches.jpg"
)

print(
    "6. camera_trajectory.png"
)

print(
    "7. camera_trajectory_top_view.png"
)

print(
    "8. frame_motion.png"
)

print(
    "9. good_matches_per_pair.png"
)

print(
    "10. pose_inliers_per_pair.png"
)

print(
    "\nNo plot windows were opened."
)

print(
    "\nMini Project Completed Successfully!"
)