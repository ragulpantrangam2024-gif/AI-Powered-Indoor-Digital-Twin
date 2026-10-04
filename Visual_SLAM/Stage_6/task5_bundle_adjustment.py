import os
import re
import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


# ============================================================
# STAGE 6 - TASK 5
# FAST BUNDLE ADJUSTMENT
# ============================================================

print("=" * 70)
print("STAGE 6 - TASK 5")
print("FAST BUNDLE ADJUSTMENT")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STAGE_DIR = os.path.dirname(
    SCRIPT_DIR
)

IMAGE_DIR = os.path.join(
    STAGE_DIR,
    "images"
)

RESULTS_DIR = os.path.join(
    STAGE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CAMERA INTRINSICS
# ============================================================

K = np.array([
    [879.03578678, 0.0, 438.89954652],
    [0.0, 896.60912854, 557.52366461],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

print("\nCamera Intrinsic Matrix:")
print(K)


# ============================================================
# INPUT FILES
# ============================================================

KEYFRAME_FILE = os.path.join(
    RESULTS_DIR,
    "keyframes.txt"
)

POSE_FILE = os.path.join(
    RESULTS_DIR,
    "task4_keyframe_poses.txt"
)


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.isdir(IMAGE_DIR):

    print("\nERROR: Image directory not found:")
    print(IMAGE_DIR)

    raise SystemExit


if not os.path.isfile(KEYFRAME_FILE):

    print("\nERROR: Keyframe file not found:")
    print(KEYFRAME_FILE)

    raise SystemExit


if not os.path.isfile(POSE_FILE):

    print("\nERROR: Pose file not found:")
    print(POSE_FILE)

    raise SystemExit


# ============================================================
# LOAD IMAGES
# ============================================================

image_files = []

for filename in os.listdir(IMAGE_DIR):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):

        image_files.append(filename)


def extract_frame_number(filename):

    numbers = re.findall(
        r"\d+",
        filename
    )

    if not numbers:
        return -1

    return int(numbers[-1])


image_files = sorted(
    image_files,
    key=extract_frame_number
)


image_paths = {}

for filename in image_files:

    frame_id = extract_frame_number(
        filename
    )

    image_paths[frame_id] = os.path.join(
        IMAGE_DIR,
        filename
    )


print("\nImages found:", len(image_files))


# ============================================================
# LOAD KEYFRAME INDICES
# ============================================================

def load_keyframes(filename):

    indices = []

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if re.fullmatch(
                r"\d+",
                line
            ):

                indices.append(
                    int(line)
                )

    return list(
        dict.fromkeys(indices)
    )


keyframe_indices = load_keyframes(
    KEYFRAME_FILE
)


print(
    "\nKeyframes loaded:",
    len(keyframe_indices)
)

print(
    "Keyframe indices:"
)

print(
    keyframe_indices
)


if len(keyframe_indices) != 42:

    print(
        "\nERROR: Expected exactly 42 keyframes."
    )

    raise SystemExit


# ============================================================
# LOAD TASK 4 POSES
# ============================================================

def parse_pose_file(filename):

    rotations = []
    translations = []

    with open(filename, "r") as f:

        lines = f.readlines()

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # ----------------------------------------------------
        # Rotation
        # ----------------------------------------------------

        if line.startswith("Rotation:"):

            rows = []

            i += 1

            while i < len(lines):

                s = lines[i].strip()

                if s.startswith("["):

                    nums = re.findall(
                        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                        s
                    )

                    if len(nums) >= 3:

                        rows.append([
                            float(nums[0]),
                            float(nums[1]),
                            float(nums[2])
                        ])

                    if len(rows) == 3:
                        break

                i += 1

            if len(rows) == 3:

                current_R = np.array(
                    rows,
                    dtype=np.float64
                )

            else:

                current_R = None


        # ----------------------------------------------------
        # Translation
        # ----------------------------------------------------

        elif line.startswith(
            "Translation:"
        ):

            values = []

            i += 1

            while i < len(lines):

                s = lines[i].strip()

                if s.startswith("["):

                    nums = re.findall(
                        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
                        s
                    )

                    values.extend(
                        float(x)
                        for x in nums
                    )

                    if len(values) >= 3:
                        break

                i += 1

            if len(values) >= 3:

                current_t = np.array(
                    values[:3],
                    dtype=np.float64
                )

            else:

                current_t = None


            if (
                current_R is not None
                and current_t is not None
            ):

                rotations.append(
                    current_R
                )

                translations.append(
                    current_t
                )

                current_R = None
                current_t = None

        i += 1


    return (
        np.asarray(rotations),
        np.asarray(translations)
    )


rotations, translations = parse_pose_file(
    POSE_FILE
)


print(
    "\nCamera poses loaded:",
    len(rotations)
)

print(
    "Rotation shape:",
    rotations.shape
)

print(
    "Translation shape:",
    translations.shape
)


if len(rotations) != 42:

    print(
        "\nERROR: Expected 42 camera poses."
    )

    raise SystemExit


# ============================================================
# ORB
# ============================================================

orb = cv2.ORB_create(
    nfeatures=2000,
    scaleFactor=1.2,
    nlevels=8,
    fastThreshold=10
)

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

print(
    "\nExtracting ORB features..."
)


keypoints = {}
descriptors = {}


for frame_id in keyframe_indices:

    image = cv2.imread(
        image_paths[frame_id],
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        keypoints[frame_id] = []
        descriptors[frame_id] = None

        continue


    kp, des = orb.detectAndCompute(
        image,
        None
    )

    keypoints[frame_id] = kp
    descriptors[frame_id] = des

    print(
        f"Frame {frame_id:03d} | "
        f"Keypoints: {len(kp)}"
    )


# ============================================================
# TRIANGULATION
# ============================================================

def triangulate(
    pts1,
    pts2,
    R1,
    t1,
    R2,
    t2
):

    P1 = K @ np.hstack(
        (
            R1,
            t1.reshape(3, 1)
        )
    )

    P2 = K @ np.hstack(
        (
            R2,
            t2.reshape(3, 1)
        )
    )

    points_4d = cv2.triangulatePoints(
        P1,
        P2,
        pts1.T,
        pts2.T
    )

    w = points_4d[3]

    w = np.where(
        np.abs(w) < 1e-10,
        1e-10,
        w
    )

    points_3d = (
        points_4d[:3] / w
    ).T

    return points_3d


# ============================================================
# BUILD OBSERVATIONS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "BUILDING OBSERVATIONS"
)

print(
    "=" * 70
)


all_points = []

obs_camera = []
obs_point = []
obs_pixels = []


successful_pairs = 0


for pair_id in range(
    len(keyframe_indices) - 1
):

    frame1 = keyframe_indices[
        pair_id
    ]

    frame2 = keyframe_indices[
        pair_id + 1
    ]


    print(
        f"\nPair {pair_id:02d}: "
        f"{frame1:03d} -> {frame2:03d}"
    )


    des1 = descriptors[
        frame1
    ]

    des2 = descriptors[
        frame2
    ]


    if (
        des1 is None
        or des2 is None
        or len(des1) < 2
        or len(des2) < 2
    ):

        print(
            "Skipped: descriptors unavailable."
        )

        continue


    matches = bf.knnMatch(
        des1,
        des2,
        k=2
    )


    good = []

    for pair in matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < (
            0.75 * n.distance
        ):

            good.append(m)


    print(
        "Good Matches:",
        len(good)
    )


    if len(good) < 8:

        print(
            "Skipped: insufficient matches."
        )

        continue


    pts1 = np.float64([
        keypoints[frame1][
            m.queryIdx
        ].pt
        for m in good
    ])

    pts2 = np.float64([
        keypoints[frame2][
            m.trainIdx
        ].pt
        for m in good
    ])


    # --------------------------------------------------------
    # Fundamental matrix
    # --------------------------------------------------------

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        1.0,
        0.99
    )


    if mask is None:

        print(
            "Skipped: F estimation failed."
        )

        continue


    mask = mask.ravel().astype(
        bool
    )


    pts1 = pts1[mask]
    pts2 = pts2[mask]


    print(
        "Geometric Inliers:",
        len(pts1)
    )


    if len(pts1) < 8:

        print(
            "Skipped: insufficient inliers."
        )

        continue


    # --------------------------------------------------------
    # Pose
    # --------------------------------------------------------

    R1 = rotations[pair_id]
    t1 = translations[pair_id]

    R2 = rotations[pair_id + 1]
    t2 = translations[pair_id + 1]


    # --------------------------------------------------------
    # Triangulate
    # --------------------------------------------------------

    points = triangulate(
        pts1,
        pts2,
        R1,
        t1,
        R2,
        t2
    )


    cam1 = (
        R1 @ points.T
    ).T + t1

    cam2 = (
        R2 @ points.T
    ).T + t2


    valid = (
        np.isfinite(points).all(axis=1)
        &
        (cam1[:, 2] > 0.05)
        &
        (cam2[:, 2] > 0.05)
        &
        (
            np.linalg.norm(
                points,
                axis=1
            ) < 1000
        )
    )


    points = points[valid]
    pts1 = pts1[valid]
    pts2 = pts2[valid]


    print(
        "Valid 3D Points:",
        len(points)
    )


    if len(points) < 5:

        print(
            "Skipped: insufficient valid points."
        )

        continue


    successful_pairs += 1


    start_id = len(
        all_points
    )


    for i in range(
        len(points)
    ):

        all_points.append(
            points[i]
        )


        point_id = (
            start_id + i
        )


        # First camera observation

        obs_camera.append(
            pair_id
        )

        obs_point.append(
            point_id
        )

        obs_pixels.append(
            pts1[i]
        )


        # Second camera observation

        obs_camera.append(
            pair_id + 1
        )

        obs_point.append(
            point_id
        )

        obs_pixels.append(
            pts2[i]
        )


# ============================================================
# ARRAYS
# ============================================================

points_3d = np.asarray(
    all_points,
    dtype=np.float64
)

obs_camera = np.asarray(
    obs_camera,
    dtype=np.int32
)

obs_point = np.asarray(
    obs_point,
    dtype=np.int32
)

obs_pixels = np.asarray(
    obs_pixels,
    dtype=np.float64
)


print(
    "\n" + "=" * 70
)

print(
    "OBSERVATION SUMMARY"
)

print(
    "=" * 70
)

print(
    "Successful pairs:",
    successful_pairs
)

print(
    "3D points:",
    len(points_3d)
)

print(
    "Observations:",
    len(obs_pixels)
)


if len(points_3d) < 20:

    print(
        "\nERROR: Not enough 3D points."
    )

    raise SystemExit


# ============================================================
# VECTORISED PROJECTION
# ============================================================

def project_all(
    camera_params,
    points,
    camera_ids,
    point_ids
):

    cams = camera_params[
        camera_ids
    ]

    rvecs = cams[:, :3]

    translations_local = cams[:, 3:6]


    # Convert rotation vectors to matrices

    rotations_local = np.empty(
        (
            len(rvecs),
            3,
            3
        )
    )


    for i in range(
        len(rvecs)
    ):

        rotations_local[i] = (
            Rotation.from_rotvec(
                rvecs[i]
            ).as_matrix()
        )


    selected_points = points[
        point_ids
    ]


    camera_points = np.einsum(
        "nij,nj->ni",
        rotations_local,
        selected_points
    )

    camera_points += (
        translations_local
    )


    z = camera_points[:, 2]

    z = np.where(
        np.abs(z) < 1e-8,
        1e-8,
        z
    )


    u = (
        K[0, 0]
        * camera_points[:, 0]
        / z
        + K[0, 2]
    )

    v = (
        K[1, 1]
        * camera_points[:, 1]
        / z
        + K[1, 2]
    )


    return np.column_stack(
        (u, v)
    )


# ============================================================
# INITIAL CAMERA PARAMETERS
# ============================================================

initial_camera_params = np.zeros(
    (
        len(rotations),
        6
    ),
    dtype=np.float64
)


for i in range(
    len(rotations)
):

    initial_camera_params[
        i,
        :3
    ] = Rotation.from_matrix(
        rotations[i]
    ).as_rotvec()


    initial_camera_params[
        i,
        3:6
    ] = translations[i]


# ============================================================
# REMOVE FIXED CAMERA
# ============================================================

# Camera 0 remains fixed.
#
# Optimize cameras 1...41.

variable_camera_params = (
    initial_camera_params[1:].copy()
)


x0 = np.hstack(
    (
        variable_camera_params.ravel(),
        points_3d.ravel()
    )
)


num_variable_cameras = (
    len(rotations) - 1
)

num_points = len(
    points_3d
)


print(
    "\n" + "=" * 70
)

print(
    "OPTIMIZATION SETUP"
)

print(
    "=" * 70
)

print(
    "Total cameras:",
    len(rotations)
)

print(
    "Fixed camera:",
    0
)

print(
    "Variable cameras:",
    num_variable_cameras
)

print(
    "3D points:",
    num_points
)

print(
    "Observations:",
    len(obs_pixels)
)

print(
    "Parameters:",
    len(x0)
)


# ============================================================
# RESIDUAL FUNCTION
# ============================================================

def residual_function(x):

    camera_size = (
        num_variable_cameras * 6
    )


    variable_cameras = x[
        :camera_size
    ].reshape(
        num_variable_cameras,
        6
    )


    points = x[
        camera_size:
    ].reshape(
        num_points,
        3
    )


    # Full camera parameter array

    cameras = np.vstack(
        (
            initial_camera_params[
                0:1
            ],
            variable_cameras
        )
    )


    projected = project_all(
        cameras,
        points,
        obs_camera,
        obs_point
    )


    residuals = (
        projected
        - obs_pixels
    )


    return residuals.ravel()


# ============================================================
# INITIAL ERROR
# ============================================================

print(
    "\nCalculating initial reprojection error..."
)


initial_residuals = (
    residual_function(x0)
)


initial_rms = np.sqrt(
    np.mean(
        initial_residuals ** 2
    )
)


initial_mean = np.mean(
    np.linalg.norm(
        initial_residuals.reshape(
            -1,
            2
        ),
        axis=1
    )
)


print(
    f"Initial RMS Error : "
    f"{initial_rms:.6f} pixels"
)

print(
    f"Initial Mean Error : "
    f"{initial_mean:.6f} pixels"
)


# ============================================================
# SPARSE JACOBIAN STRUCTURE
# ============================================================

print(
    "\nBuilding sparse Jacobian structure..."
)


num_residuals = (
    len(obs_pixels) * 2
)

num_parameters = len(
    x0
)


sparsity = lil_matrix(
    (
        num_residuals,
        num_parameters
    ),
    dtype=int
)


camera_parameter_size = (
    num_variable_cameras * 6
)


for observation_id in range(
    len(obs_pixels)
):

    cam_id = obs_camera[
        observation_id
    ]

    point_id = obs_point[
        observation_id
    ]


    row1 = (
        observation_id * 2
    )

    row2 = row1 + 1


    # --------------------------------------------------------
    # Camera parameters
    # --------------------------------------------------------

    if cam_id > 0:

        camera_start = (
            (cam_id - 1) * 6
        )


        sparsity[
            row1,
            camera_start:
            camera_start + 6
        ] = 1


        sparsity[
            row2,
            camera_start:
            camera_start + 6
        ] = 1


    # --------------------------------------------------------
    # 3D point parameters
    # --------------------------------------------------------

    point_start = (
        camera_parameter_size
        +
        point_id * 3
    )


    sparsity[
        row1,
        point_start:
        point_start + 3
    ] = 1


    sparsity[
        row2,
        point_start:
        point_start + 3
    ] = 1


sparsity = sparsity.tocsr()


print(
    "Sparse Jacobian structure ready."
)


# ============================================================
# BUNDLE ADJUSTMENT
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STARTING FAST BUNDLE ADJUSTMENT"
)

print(
    "=" * 70
)

print(
    "Camera 0 is fixed."
)

print(
    "Robust loss: soft_l1"
)

print(
    "Sparse Jacobian enabled."
)

print(
    "Maximum iterations: 50"
)

print(
    "\nPlease wait..."
)


result = least_squares(
    residual_function,
    x0,
    jac_sparsity=sparsity,
    method="trf",
    loss="soft_l1",
    f_scale=3.0,
    x_scale="jac",
    max_nfev=100,
    verbose=2
)


# ============================================================
# EXTRACT RESULTS
# ============================================================

camera_size = (
    num_variable_cameras * 6
)


optimized_variable_cameras = (
    result.x[
        :camera_size
    ].reshape(
        num_variable_cameras,
        6
    )
)


optimized_points = (
    result.x[
        camera_size:
    ].reshape(
        num_points,
        3
    )
)


# Full optimized camera array

optimized_camera_params = np.vstack(
    (
        initial_camera_params[
            0:1
        ],
        optimized_variable_cameras
    )
)


optimized_rotations = []
optimized_translations = []


for i in range(
    len(rotations)
):

    rvec = (
        optimized_camera_params[
            i,
            :3
        ]
    )

    tvec = (
        optimized_camera_params[
            i,
            3:6
        ]
    )


    R = Rotation.from_rotvec(
        rvec
    ).as_matrix()


    optimized_rotations.append(
        R
    )

    optimized_translations.append(
        tvec
    )


optimized_rotations = np.asarray(
    optimized_rotations
)

optimized_translations = np.asarray(
    optimized_translations
)


# ============================================================
# FINAL ERROR
# ============================================================

final_residuals = (
    residual_function(
        result.x
    )
)


final_rms = np.sqrt(
    np.mean(
        final_residuals ** 2
    )
)


final_mean = np.mean(
    np.linalg.norm(
        final_residuals.reshape(
            -1,
            2
        ),
        axis=1
    )
)


improvement = (
    (
        initial_rms
        -
        final_rms
    )
    /
    max(
        initial_rms,
        1e-12
    )
) * 100.0


# ============================================================
# RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "BUNDLE ADJUSTMENT COMPLETED"
)

print(
    "=" * 70
)

print(
    f"Initial RMS Error : "
    f"{initial_rms:.6f} pixels"
)

print(
    f"Final RMS Error   : "
    f"{final_rms:.6f} pixels"
)

print(
    f"Initial Mean Error : "
    f"{initial_mean:.6f} pixels"
)

print(
    f"Final Mean Error   : "
    f"{final_mean:.6f} pixels"
)

print(
    f"RMS Improvement   : "
    f"{improvement:.2f} %"
)

print(
    "Optimizer Success :",
    result.success
)

print(
    "Function Evaluations:",
    result.nfev
)

print(
    "Optimizer Message:",
    result.message
)


# ============================================================
# SAVE MAP
# ============================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "task5_optimized_map_points.npy"
    ),
    optimized_points
)


np.savetxt(
    os.path.join(
        RESULTS_DIR,
        "task5_optimized_map_points.txt"
    ),
    optimized_points,
    fmt="%.8f"
)


# ============================================================
# SAVE CAMERA POSES
# ============================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "task5_optimized_rotations.npy"
    ),
    optimized_rotations
)


np.save(
    os.path.join(
        RESULTS_DIR,
        "task5_optimized_translations.npy"
    ),
    optimized_translations
)


pose_file = os.path.join(
    RESULTS_DIR,
    "task5_optimized_camera_poses.txt"
)


with open(
    pose_file,
    "w"
) as f:

    for i in range(
        len(rotations)
    ):

        frame_id = (
            keyframe_indices[i]
        )

        R = (
            optimized_rotations[i]
        )

        t = (
            optimized_translations[i]
        )

        camera_position = (
            -R.T @ t
        )


        f.write(
            f"Pose {i:02d} "
            f"(Image Frame {frame_id:03d})\n"
        )

        f.write(
            "Rotation:\n"
        )

        f.write(
            np.array2string(
                R,
                precision=8
            )
        )

        f.write("\n")

        f.write(
            "Translation:\n"
        )

        f.write(
            np.array2string(
                t.reshape(3, 1),
                precision=8
            )
        )

        f.write("\n")

        f.write(
            "Camera Position:\n"
        )

        f.write(
            np.array2string(
                camera_position,
                precision=8
            )
        )

        f.write(
            "\n\n"
        )


# ============================================================
# SAVE STATISTICS
# ============================================================

statistics_file = os.path.join(
    RESULTS_DIR,
    "task5_bundle_adjustment_statistics.txt"
)


with open(
    statistics_file,
    "w"
) as f:

    f.write(
        "STAGE 6 - TASK 5\n"
    )

    f.write(
        "FAST BUNDLE ADJUSTMENT\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Images : {len(image_files)}\n"
    )

    f.write(
        f"Keyframes : {len(keyframe_indices)}\n"
    )

    f.write(
        f"Fixed Camera : 0\n"
    )

    f.write(
        f"Optimized Cameras : "
        f"{num_variable_cameras}\n"
    )

    f.write(
        f"3D Points : {num_points}\n"
    )

    f.write(
        f"Observations : "
        f"{len(obs_pixels)}\n"
    )

    f.write(
        f"Successful Pairs : "
        f"{successful_pairs}\n"
    )

    f.write(
        f"\nInitial RMS : "
        f"{initial_rms:.8f} pixels\n"
    )

    f.write(
        f"Final RMS : "
        f"{final_rms:.8f} pixels\n"
    )

    f.write(
        f"Initial Mean : "
        f"{initial_mean:.8f} pixels\n"
    )

    f.write(
        f"Final Mean : "
        f"{final_mean:.8f} pixels\n"
    )

    f.write(
        f"RMS Improvement : "
        f"{improvement:.4f} %\n"
    )

    f.write(
        f"\nOptimizer Success : "
        f"{result.success}\n"
    )

    f.write(
        f"Function Evaluations : "
        f"{result.nfev}\n"
    )

    f.write(
        f"Optimizer Message : "
        f"{result.message}\n"
    )


# ============================================================
# SAVE ERROR ARRAYS
# ============================================================

np.save(
    os.path.join(
        RESULTS_DIR,
        "task5_initial_reprojection_errors.npy"
    ),
    initial_residuals
)


np.save(
    os.path.join(
        RESULTS_DIR,
        "task5_final_reprojection_errors.npy"
    ),
    final_residuals
)


# ============================================================
# PLOT 1
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.bar(
    ["Initial", "Final"],
    [
        initial_rms,
        final_rms
    ]
)

plt.ylabel(
    "RMS Reprojection Error (pixels)"
)

plt.title(
    "Task 5 - Reprojection Error"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task5_reprojection_error.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 2
# ============================================================

initial_magnitude = np.linalg.norm(
    initial_residuals.reshape(
        -1,
        2
    ),
    axis=1
)

final_magnitude = np.linalg.norm(
    final_residuals.reshape(
        -1,
        2
    ),
    axis=1
)


plt.figure(
    figsize=(12, 6)
)

plt.plot(
    initial_magnitude,
    label="Initial"
)

plt.plot(
    final_magnitude,
    label="After BA"
)

plt.xlabel(
    "Observation"
)

plt.ylabel(
    "Reprojection Error (pixels)"
)

plt.title(
    "Task 5 - Observation Reprojection Errors"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task5_observation_errors.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PLOT 3
# OPTIMIZED TRAJECTORY
# ============================================================

camera_positions = []


for i in range(
    len(optimized_rotations)
):

    R = optimized_rotations[i]

    t = optimized_translations[i]

    C = -R.T @ t

    camera_positions.append(
        C
    )


camera_positions = np.asarray(
    camera_positions
)


plt.figure(
    figsize=(10, 7)
)

plt.plot(
    camera_positions[:, 0],
    camera_positions[:, 2],
    marker="o",
    markersize=3
)

plt.scatter(
    camera_positions[0, 0],
    camera_positions[0, 2],
    s=100,
    label="Start"
)

plt.scatter(
    camera_positions[-1, 0],
    camera_positions[-1, 2],
    s=100,
    label="End"
)

plt.xlabel("X")

plt.ylabel("Z")

plt.title(
    "Task 5 - Optimized Camera Trajectory"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.axis("equal")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "task5_optimized_trajectory.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TASK 5 COMPLETED"
)

print(
    "=" * 70
)

print(
    "\nResults saved in:"
)

print(
    RESULTS_DIR
)

print(
    "\nGenerated files:"
)

print(
    "1. task5_optimized_map_points.npy"
)

print(
    "2. task5_optimized_map_points.txt"
)

print(
    "3. task5_optimized_camera_poses.txt"
)

print(
    "4. task5_optimized_rotations.npy"
)

print(
    "5. task5_optimized_translations.npy"
)

print(
    "6. task5_bundle_adjustment_statistics.txt"
)

print(
    "7. task5_initial_reprojection_errors.npy"
)

print(
    "8. task5_final_reprojection_errors.npy"
)

print(
    "9. task5_reprojection_error.png"
)

print(
    "10. task5_observation_errors.png"
)

print(
    "11. task5_optimized_trajectory.png"
)

print(
    "\nNo plot windows were opened."
)