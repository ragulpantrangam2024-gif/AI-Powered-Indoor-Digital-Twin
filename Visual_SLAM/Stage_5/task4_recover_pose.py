import cv2
import numpy as np
import os
import glob

# =====================================================
# STAGE 5 - TASK 4
# RECOVER CAMERA POSE
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

ESSENTIAL_FILE = os.path.join(
    RESULT_DIR,
    "essential_matrices.npy"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

# -----------------------------------------------------
# SETTINGS
# -----------------------------------------------------

ORB_FEATURES = 1000
LOWE_RATIO = 0.75

# -----------------------------------------------------
# LOAD IMAGES
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
print("STAGE 5 - TASK 4")
print("RECOVER CAMERA POSE")
print("=" * 60)

print(
    f"\nTotal Image Frames : {len(image_files)}"
)

# -----------------------------------------------------
# CHECK ESSENTIAL MATRICES
# -----------------------------------------------------

if not os.path.exists(ESSENTIAL_FILE):

    print(
        "\nERROR: Essential Matrix file not found."
    )

    print(
        ESSENTIAL_FILE
    )

    exit()

E_all = np.load(
    ESSENTIAL_FILE
)

print(
    f"Essential Matrices Loaded : {len(E_all)}"
)

# -----------------------------------------------------
# LOAD FIRST IMAGE
# -----------------------------------------------------

first_image = cv2.imread(
    image_files[0]
)

if first_image is None:

    print(
        "\nERROR: Could not read first image."
    )

    exit()

height, width = first_image.shape[:2]

# -----------------------------------------------------
# LOAD STAGE 3 CALIBRATION
# -----------------------------------------------------

STAGE3_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "Stage_3"
)

CALIBRATION_FILE = os.path.join(
    STAGE3_DIR,
    "results",
    "camera_calibration.npz"
)

if not os.path.exists(
    CALIBRATION_FILE
):

    print(
        "\nERROR: Stage 3 calibration "
        "file not found."
    )

    print(
        CALIBRATION_FILE
    )

    exit()

calib = np.load(
    CALIBRATION_FILE
)

K_original = calib[
    "cameraMatrix"
]

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
        "\nERROR: Stage 3 calibration "
        "images not found."
    )

    exit()

calibration_image = cv2.imread(
    calibration_images[0]
)

calib_height, calib_width = (
    calibration_image.shape[:2]
)

# -----------------------------------------------------
# SCALE INTRINSIC MATRIX
# -----------------------------------------------------

scale_x = width / calib_width
scale_y = height / calib_height

K = K_original.copy().astype(
    np.float64
)

K[0, 0] *= scale_x
K[0, 2] *= scale_x

K[1, 1] *= scale_y
K[1, 2] *= scale_y

print("\nCamera Matrix Used")
print("------------------")

print(K)

# -----------------------------------------------------
# LOAD ORB
# -----------------------------------------------------

orb = cv2.ORB_create(
    nfeatures=ORB_FEATURES
)

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)

# -----------------------------------------------------
# RESULT STORAGE
# -----------------------------------------------------

rotation_matrices = []

translation_vectors = []

pose_inliers = []

# -----------------------------------------------------
# PROCESS ESSENTIAL MATRICES
# -----------------------------------------------------

E_index = 0

for frame_index in range(
    len(image_files) - 1
):

    # -------------------------------------------------
    # We only have Essential Matrices for pairs
    # that successfully passed Task 3.
    # -------------------------------------------------

    if E_index >= len(E_all):

        break

    img1 = cv2.imread(
        image_files[frame_index]
    )

    img2 = cv2.imread(
        image_files[frame_index + 1]
    )

    if img1 is None or img2 is None:

        continue

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    # -------------------------------------------------
    # ORB FEATURES
    # -------------------------------------------------

    kp1, des1 = orb.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray2,
        None
    )

    if des1 is None or des2 is None:

        continue

    # -------------------------------------------------
    # MATCHING
    # -------------------------------------------------

    knn_matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:

            good_matches.append(m)

    if len(good_matches) < 5:

        continue

    # -------------------------------------------------
    # MATCH POINTS
    # -------------------------------------------------

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

    # -------------------------------------------------
    # RECOVER POSE
    # -------------------------------------------------

    E = E_all[E_index]

    try:

        inliers, R, t, mask = cv2.recoverPose(
            E,
            pts1,
            pts2,
            K
        )

    except cv2.error:

        print(
            f"\nPose recovery failed "
            f"for pair {frame_index}"
        )

        E_index += 1

        continue

    # -------------------------------------------------
    # STORE RESULTS
    # -------------------------------------------------

    rotation_matrices.append(
        R
    )

    translation_vectors.append(
        t
    )

    pose_inliers.append(
        inliers
    )

    # -------------------------------------------------
    # PRINT
    # -------------------------------------------------

    print(
        f"\nFrame "
        f"{frame_index:03d} → "
        f"{frame_index + 1:03d}"
    )

    print(
        f"Pose Inliers : {inliers}"
    )

    print("\nRotation Matrix")

    print(R)

    print("\nTranslation Direction")

    print(t.ravel())

    E_index += 1

# -----------------------------------------------------
# SUMMARY
# -----------------------------------------------------

print("\n")
print("=" * 60)
print("TASK 4 SUMMARY")
print("=" * 60)

print(
    f"\nRecovered Poses : "
    f"{len(rotation_matrices)}"
)

if len(pose_inliers) > 0:

    print(
        f"Average Pose Inliers : "
        f"{np.mean(pose_inliers):.2f}"
    )

# -----------------------------------------------------
# SAVE ROTATION MATRICES
# -----------------------------------------------------

np.save(
    os.path.join(
        RESULT_DIR,
        "rotation_matrices.npy"
    ),
    np.array(
        rotation_matrices
    )
)

# -----------------------------------------------------
# SAVE TRANSLATION VECTORS
# -----------------------------------------------------

np.save(
    os.path.join(
        RESULT_DIR,
        "translation_vectors.npy"
    ),
    np.array(
        translation_vectors
    )
)

# -----------------------------------------------------
# SAVE POSE INLIERS
# -----------------------------------------------------

np.save(
    os.path.join(
        RESULT_DIR,
        "pose_inliers.npy"
    ),
    np.array(
        pose_inliers
    )
)

# -----------------------------------------------------
# SAVE READABLE RESULTS
# -----------------------------------------------------

pose_file = os.path.join(
    RESULT_DIR,
    "task4_camera_poses.txt"
)

with open(
    pose_file,
    "w"
) as f:

    f.write(
        "STAGE 5 - TASK 4\n"
    )

    f.write(
        "RECOVER CAMERA POSE\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    for i in range(
        len(rotation_matrices)
    ):

        f.write(
            f"Pose {i + 1}\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            "\nRotation Matrix R:\n"
        )

        f.write(
            str(
                rotation_matrices[i]
            )
        )

        f.write(
            "\n\nTranslation Vector t:\n"
        )

        f.write(
            str(
                translation_vectors[i].ravel()
            )
        )

        f.write(
            "\n\nPose Inliers: "
            + str(
                pose_inliers[i]
            )
        )

        f.write(
            "\n\n"
        )

# -----------------------------------------------------
# SAVE FIRST POSE
# -----------------------------------------------------

if len(rotation_matrices) > 0:

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            "task4_first_rotation_matrix.txt"
        ),
        rotation_matrices[0],
        fmt="%.8f"
    )

    np.savetxt(
        os.path.join(
            RESULT_DIR,
            "task4_first_translation_vector.txt"
        ),
        translation_vectors[0],
        fmt="%.8f"
    )

# -----------------------------------------------------
# FINAL MESSAGE
# -----------------------------------------------------

print("\nResults saved in:")

print(
    "results/rotation_matrices.npy"
)

print(
    "results/translation_vectors.npy"
)

print(
    "results/pose_inliers.npy"
)

print(
    "results/task4_camera_poses.txt"
)

print(
    "\nTask 4 Completed Successfully!"
)