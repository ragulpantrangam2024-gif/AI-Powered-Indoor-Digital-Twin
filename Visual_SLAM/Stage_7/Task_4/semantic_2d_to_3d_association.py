import os
import json
import cv2
import numpy as np


# ============================================================
# STAGE 7 - TASK 4
# NEAREST-KEYFRAME 2D TO 3D SEMANTIC ASSOCIATION
# ============================================================

print("=" * 70)
print("STAGE 7 - TASK 4")
print("NEAREST-KEYFRAME 2D TO 3D SEMANTIC ASSOCIATION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_7_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(STAGE_7_DIR)

STAGE_6_RESULTS = os.path.join(
    PROJECT_DIR,
    "Stage_6",
    "results"
)

STAGE_7_RESULTS = os.path.join(
    STAGE_7_DIR,
    "results"
)

DETECTION_FILE = os.path.join(
    STAGE_7_RESULTS,
    "task3_filtered_detections.json"
)

MAP_FILE = os.path.join(
    STAGE_6_RESULTS,
    "task5_optimized_map_points.npy"
)

ROTATION_FILE = os.path.join(
    STAGE_6_RESULTS,
    "task5_optimized_rotations.npy"
)

TRANSLATION_FILE = os.path.join(
    STAGE_6_RESULTS,
    "task5_optimized_translations.npy"
)

os.makedirs(
    STAGE_7_RESULTS,
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


# ============================================================
# STAGE 6 KEYFRAMES
# ============================================================

KEYFRAMES = np.array([
    0, 4, 7, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32,
    35, 45, 52, 62, 67,
    77, 87, 89, 93, 97,
    99, 109, 113, 118, 127,
    131, 134, 135
], dtype=int)


# ============================================================
# ASSOCIATION PARAMETERS
# ============================================================

# Maximum distance between a YOLO frame and its nearest
# SLAM keyframe.
MAX_KEYFRAME_DISTANCE = 10

# Minimum number of 3D points required for association.
MIN_3D_POINTS = 3

# Maximum number of points retained for one object observation.
MAX_3D_POINTS = 100

# Use central part of bounding box to reduce background points.
CENTRAL_BBOX_RATIO = 0.80


# ============================================================
# FILE CHECK
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT FILES")
print("=" * 70)

required_files = {
    "YOLO detections": DETECTION_FILE,
    "Optimized map": MAP_FILE,
    "Optimized rotations": ROTATION_FILE,
    "Optimized translations": TRANSLATION_FILE
}

for name, path in required_files.items():

    if os.path.exists(path):

        print(f"[OK] {name}")

    else:

        print(f"[MISSING] {name}")
        print(path)

        raise SystemExit


# ============================================================
# LOAD DETECTIONS
# ============================================================

print("\n" + "=" * 70)
print("LOADING YOLO DETECTIONS")
print("=" * 70)

with open(DETECTION_FILE, "r") as f:
    detections_data = json.load(f)

total_detections = sum(
    len(frame["detections"])
    for frame in detections_data
)

print("Detection frames:", len(detections_data))
print("Total detections:", total_detections)


# ============================================================
# LOAD MAP
# ============================================================

print("\n" + "=" * 70)
print("LOADING OPTIMIZED 3D MAP")
print("=" * 70)

map_points = np.load(MAP_FILE)

map_points = np.asarray(
    map_points,
    dtype=np.float64
)

print("Map points:", len(map_points))
print("Map shape:", map_points.shape)


# ============================================================
# LOAD CAMERA POSES
# ============================================================

print("\n" + "=" * 70)
print("LOADING CAMERA POSES")
print("=" * 70)

rotations = np.load(
    ROTATION_FILE
)

translations = np.load(
    TRANSLATION_FILE
)

rotations = np.asarray(
    rotations,
    dtype=np.float64
)

translations = np.asarray(
    translations,
    dtype=np.float64
)

print("Rotation shape:", rotations.shape)
print("Translation shape:", translations.shape)
print("Keyframes:", len(KEYFRAMES))


# ============================================================
# VALIDATION
# ============================================================

pose_count = min(
    len(KEYFRAMES),
    len(rotations),
    len(translations)
)

KEYFRAMES = KEYFRAMES[:pose_count]
rotations = rotations[:pose_count]
translations = translations[:pose_count]

print("\nUsable poses:", pose_count)


# ============================================================
# FIND NEAREST KEYFRAME
# ============================================================

def find_nearest_keyframe(frame_index):

    distances = np.abs(
        KEYFRAMES - frame_index
    )

    nearest_index = int(
        np.argmin(distances)
    )

    nearest_frame = int(
        KEYFRAMES[nearest_index]
    )

    frame_distance = int(
        distances[nearest_index]
    )

    return (
        nearest_index,
        nearest_frame,
        frame_distance
    )


# ============================================================
# PROJECT 3D POINTS
# ============================================================

def project_points(
    points,
    R,
    t
):

    rvec, _ = cv2.Rodrigues(R)

    tvec = t.reshape(
        3,
        1
    )

    projected, _ = cv2.projectPoints(
        points,
        rvec,
        tvec,
        K,
        None
    )

    projected = projected.reshape(
        -1,
        2
    )

    # Camera coordinates
    camera_points = (
        R @ points.T
    ).T + t.reshape(
        1,
        3
    )

    depths = camera_points[:, 2]

    return projected, depths


# ============================================================
# RESULT STORAGE
# ============================================================

semantic_objects = []

processed = 0
successful = 0
failed = 0
too_far = 0

nearest_keyframe_distances = []


# ============================================================
# MAIN ASSOCIATION
# ============================================================

print("\n" + "=" * 70)
print("STARTING NEAREST-KEYFRAME ASSOCIATION")
print("=" * 70)


for frame_data in detections_data:

    frame_index = int(
        frame_data["frame_index"]
    )

    detections = frame_data[
        "detections"
    ]

    if len(detections) == 0:
        continue


    # --------------------------------------------------------
    # FIND NEAREST KEYFRAME
    # --------------------------------------------------------

    pose_index, nearest_frame, frame_distance = \
        find_nearest_keyframe(
            frame_index
        )


    print(
        f"\nFrame {frame_index:03d}"
        f" -> nearest keyframe "
        f"{nearest_frame:03d}"
        f" | distance = {frame_distance}"
    )


    if frame_distance > MAX_KEYFRAME_DISTANCE:

        print(
            "Skipped: nearest keyframe too far."
        )

        too_far += len(detections)

        continue


    nearest_keyframe_distances.append(
        frame_distance
    )


    R = rotations[
        pose_index
    ]

    t = translations[
        pose_index
    ]


    # --------------------------------------------------------
    # PROJECT MAP
    # --------------------------------------------------------

    projected, depths = project_points(
        map_points,
        R,
        t
    )


    # --------------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------------

    for detection in detections:

        processed += 1

        class_name = detection[
            "class_name"
        ]

        confidence = float(
            detection["confidence"]
        )

        bbox = detection[
            "bbox"
        ]

        x1 = float(
            bbox["x1"]
        )

        y1 = float(
            bbox["y1"]
        )

        x2 = float(
            bbox["x2"]
        )

        y2 = float(
            bbox["y2"]
        )


        # ----------------------------------------------------
        # BBOX CENTER
        # ----------------------------------------------------

        center_x = (
            x1 + x2
        ) / 2.0

        center_y = (
            y1 + y2
        ) / 2.0


        width = x2 - x1
        height = y2 - y1


        # ----------------------------------------------------
        # CENTRAL REGION
        # ----------------------------------------------------

        cx1 = (
            center_x
            -
            width * CENTRAL_BBOX_RATIO / 2
        )

        cx2 = (
            center_x
            +
            width * CENTRAL_BBOX_RATIO / 2
        )

        cy1 = (
            center_y
            -
            height * CENTRAL_BBOX_RATIO / 2
        )

        cy2 = (
            center_y
            +
            height * CENTRAL_BBOX_RATIO / 2
        )


        # ----------------------------------------------------
        # FIND 3D POINTS INSIDE BBOX
        # ----------------------------------------------------

        inside = (

            (depths > 0.1)

            &

            (projected[:, 0] >= cx1)

            &

            (projected[:, 0] <= cx2)

            &

            (projected[:, 1] >= cy1)

            &

            (projected[:, 1] <= cy2)

        )


        candidate_indices = np.where(
            inside
        )[0]


        # ----------------------------------------------------
        # FAILED ASSOCIATION
        # ----------------------------------------------------

        if len(candidate_indices) < MIN_3D_POINTS:

            failed += 1

            print(
                f"  {class_name:15s}"
                f" | 3D points = "
                f"{len(candidate_indices):3d}"
                f" | FAILED"
            )

            continue


        # ----------------------------------------------------
        # LIMIT POINT COUNT
        # ----------------------------------------------------

        if len(candidate_indices) > MAX_3D_POINTS:

            selected_points = projected[
                candidate_indices
            ]

            distances = np.sqrt(

                (
                    selected_points[:, 0]
                    -
                    center_x
                ) ** 2

                +

                (
                    selected_points[:, 1]
                    -
                    center_y
                ) ** 2

            )

            order = np.argsort(
                distances
            )

            candidate_indices = (
                candidate_indices[
                    order[
                        :MAX_3D_POINTS
                    ]
                ]
            )


        # ----------------------------------------------------
        # ASSOCIATED 3D POINTS
        # ----------------------------------------------------

        points_3d = map_points[
            candidate_indices
        ]


        # ----------------------------------------------------
        # ROBUST CENTER
        # ----------------------------------------------------

        object_center = np.median(
            points_3d,
            axis=0
        )


        object_min = np.min(
            points_3d,
            axis=0
        )

        object_max = np.max(
            points_3d,
            axis=0
        )


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        result = {

            "detection_frame":
                frame_index,

            "nearest_keyframe":
                nearest_frame,

            "frame_to_keyframe_distance":
                frame_distance,

            "pose_index":
                pose_index,

            "class_name":
                class_name,

            "confidence":
                confidence,

            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },

            "image_center": {
                "x": center_x,
                "y": center_y
            },

            "associated_3d_point_count":
                int(len(candidate_indices)),

            "associated_map_indices":
                candidate_indices.tolist(),

            "3d_center": {
                "x": float(object_center[0]),
                "y": float(object_center[1]),
                "z": float(object_center[2])
            },

            "3d_min": {
                "x": float(object_min[0]),
                "y": float(object_min[1]),
                "z": float(object_min[2])
            },

            "3d_max": {
                "x": float(object_max[0]),
                "y": float(object_max[1]),
                "z": float(object_max[2])
            }

        }


        semantic_objects.append(
            result
        )

        successful += 1


        print(
            f"  {class_name:15s}"
            f" | 3D points = "
            f"{len(candidate_indices):3d}"
            f" | 3D center = "
            f"("
            f"{object_center[0]:7.2f}, "
            f"{object_center[1]:7.2f}, "
            f"{object_center[2]:7.2f}"
            f")"
        )


# ============================================================
# SAVE JSON
# ============================================================

json_output = os.path.join(
    STAGE_7_RESULTS,
    "task4_nearest_keyframe_semantic_3d.json"
)

with open(
    json_output,
    "w"
) as f:

    json.dump(
        semantic_objects,
        f,
        indent=2
    )


# ============================================================
# SAVE NUMPY
# ============================================================

npy_output = os.path.join(
    STAGE_7_RESULTS,
    "task4_nearest_keyframe_semantic_3d.npy"
)

np.save(
    npy_output,
    np.array(
        semantic_objects,
        dtype=object
    ),
    allow_pickle=True
)


# ============================================================
# CLASS STATISTICS
# ============================================================

class_results = {}

for obj in semantic_objects:

    name = obj[
        "class_name"
    ]

    if name not in class_results:

        class_results[name] = []

    class_results[name].append(
        obj
    )


# ============================================================
# STATISTICS FILE
# ============================================================

stats_output = os.path.join(
    STAGE_7_RESULTS,
    "task4_nearest_keyframe_statistics.txt"
)

with open(
    stats_output,
    "w"
) as f:

    f.write(
        "STAGE 7 - TASK 4\n"
    )

    f.write(
        "NEAREST-KEYFRAME 2D TO 3D "
        "SEMANTIC ASSOCIATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Total YOLO detections: "
        f"{total_detections}\n"
    )

    f.write(
        f"Processed detections: "
        f"{processed}\n"
    )

    f.write(
        f"Successful associations: "
        f"{successful}\n"
    )

    f.write(
        f"Failed associations: "
        f"{failed}\n"
    )

    f.write(
        f"Too-far detections: "
        f"{too_far}\n"
    )

    if processed > 0:

        rate = (
            successful
            /
            processed
            *
            100
        )

    else:

        rate = 0.0

    f.write(
        f"Association success rate: "
        f"{rate:.2f}%\n"
    )

    if len(nearest_keyframe_distances) > 0:

        f.write(
            "\nNearest-keyframe statistics\n"
        )

        f.write(
            f"Minimum frame distance: "
            f"{min(nearest_keyframe_distances)}\n"
        )

        f.write(
            f"Maximum frame distance: "
            f"{max(nearest_keyframe_distances)}\n"
        )

        f.write(
            f"Average frame distance: "
            f"{np.mean(nearest_keyframe_distances):.2f}\n"
        )

    f.write(
        "\nSEMANTIC OBJECT SUMMARY\n"
    )

    for name in sorted(
        class_results.keys()
    ):

        objects = class_results[
            name
        ]

        centers = np.array([

            [
                obj["3d_center"]["x"],
                obj["3d_center"]["y"],
                obj["3d_center"]["z"]
            ]

            for obj in objects

        ])

        mean_center = np.mean(
            centers,
            axis=0
        )

        f.write(
            f"\n{name}\n"
        )

        f.write(
            f"Observations: "
            f"{len(objects)}\n"
        )

        f.write(
            "Mean 3D position: "
            f"("
            f"{mean_center[0]:.4f}, "
            f"{mean_center[1]:.4f}, "
            f"{mean_center[2]:.4f}"
            f")\n"
        )


# ============================================================
# CLASS CENTERS
# ============================================================

class_centers = []

for name in sorted(
    class_results.keys()
):

    objects = class_results[
        name
    ]

    centers = np.array([

        [
            obj["3d_center"]["x"],
            obj["3d_center"]["y"],
            obj["3d_center"]["z"]
        ]

        for obj in objects

    ])

    mean_center = np.mean(
        centers,
        axis=0
    )

    class_centers.append({

        "class_name":
            name,

        "observation_count":
            len(objects),

        "mean_3d_position": {

            "x":
                float(mean_center[0]),

            "y":
                float(mean_center[1]),

            "z":
                float(mean_center[2])

        }

    })


class_center_output = os.path.join(
    STAGE_7_RESULTS,
    "task4_nearest_keyframe_object_centers.json"
)

with open(
    class_center_output,
    "w"
) as f:

    json.dump(
        class_centers,
        f,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TASK 4 FINAL SUMMARY")
print("=" * 70)

print(
    f"\nTotal YOLO detections       : "
    f"{total_detections}"
)

print(
    f"Processed detections       : "
    f"{processed}"
)

print(
    f"Successful associations    : "
    f"{successful}"
)

print(
    f"Failed associations        : "
    f"{failed}"
)

print(
    f"Too-far detections         : "
    f"{too_far}"
)

if processed > 0:

    print(
        f"Association success rate  : "
        f"{successful / processed * 100:.2f}%"
    )


print(
    "\nSemantic 3D objects:"
)


for name in sorted(
    class_results.keys()
):

    objects = class_results[
        name
    ]

    centers = np.array([

        [
            obj["3d_center"]["x"],
            obj["3d_center"]["y"],
            obj["3d_center"]["z"]
        ]

        for obj in objects

    ])

    mean_center = np.mean(
        centers,
        axis=0
    )

    print(
        f"{name:15s}"
        f" | Observations: "
        f"{len(objects):2d}"
        f" | Mean 3D: "
        f"("
        f"{mean_center[0]:7.2f}, "
        f"{mean_center[1]:7.2f}, "
        f"{mean_center[2]:7.2f}"
        f")"
    )


print(
    "\nResults saved in:"
)

print(
    STAGE_7_RESULTS
)

print(
    "\nGenerated files:"
)

print(
    "1. task4_nearest_keyframe_semantic_3d.json"
)

print(
    "2. task4_nearest_keyframe_semantic_3d.npy"
)

print(
    "3. task4_nearest_keyframe_statistics.txt"
)

print(
    "4. task4_nearest_keyframe_object_centers.json"
)

print(
    "\n" + "=" * 70
)

print(
    "TASK 4 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)