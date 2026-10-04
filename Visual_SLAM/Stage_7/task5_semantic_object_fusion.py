import os
import json
import numpy as np
from sklearn.cluster import DBSCAN


# ============================================================
# STAGE 7 - TASK 5
# SEMANTIC OBJECT FUSION / 3D CLUSTERING
# ============================================================

print("=" * 70)
print("STAGE 7 - TASK 5")
print("SEMANTIC OBJECT FUSION / 3D CLUSTERING")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

STAGE_7_DIR = os.path.dirname(
    SCRIPT_DIR
)

RESULTS_DIR = os.path.join(
    STAGE_7_DIR,
    "results"
)

INPUT_FILE = os.path.join(
    RESULTS_DIR,
    "task4_nearest_keyframe_semantic_3d.json"
)

OUTPUT_JSON = os.path.join(
    RESULTS_DIR,
    "task5_fused_semantic_objects.json"
)

OUTPUT_NPY = os.path.join(
    RESULTS_DIR,
    "task5_fused_semantic_objects.npy"
)

OUTPUT_STATS = os.path.join(
    RESULTS_DIR,
    "task5_semantic_object_statistics.txt"
)


# ============================================================
# PARAMETERS
# ============================================================

# Maximum distance between observations of the
# same semantic object.
DBSCAN_EPS = 2.0

# Minimum observations required to form a cluster.
DBSCAN_MIN_SAMPLES = 1


# ============================================================
# CHECK INPUT
# ============================================================

print("\n" + "=" * 70)
print("CHECKING INPUT")
print("=" * 70)

if not os.path.exists(INPUT_FILE):

    print("[ERROR] Input file not found:")
    print(INPUT_FILE)

    raise SystemExit

print("[OK] Semantic observations found")
print(INPUT_FILE)


# ============================================================
# LOAD OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("LOADING SEMANTIC OBSERVATIONS")
print("=" * 70)

with open(
    INPUT_FILE,
    "r"
) as f:

    observations = json.load(f)


print(
    "Total semantic observations:",
    len(observations)
)


# ============================================================
# GROUP BY CLASS
# ============================================================

objects_by_class = {}

for observation in observations:

    class_name = observation[
        "class_name"
    ]

    if class_name not in objects_by_class:

        objects_by_class[
            class_name
        ] = []

    objects_by_class[
        class_name
    ].append(
        observation
    )


print(
    "\nDetected semantic classes:"
)

for class_name in sorted(
    objects_by_class.keys()
):

    print(
        f" - {class_name}: "
        f"{len(objects_by_class[class_name])} "
        f"observations"
    )


# ============================================================
# CLUSTER EACH CLASS
# ============================================================

fused_objects = []

global_object_id = 0


for class_name in sorted(
    objects_by_class.keys()
):

    class_observations = objects_by_class[
        class_name
    ]


    print(
        "\n" + "-" * 70
    )

    print(
        f"Processing class: {class_name}"
    )


    # --------------------------------------------------------
    # EXTRACT 3D CENTERS
    # --------------------------------------------------------

    points = np.array([

        [

            obs["3d_center"]["x"],

            obs["3d_center"]["y"],

            obs["3d_center"]["z"]

        ]

        for obs in class_observations

    ], dtype=np.float64)


    print(
        "Observations:",
        len(points)
    )


    # --------------------------------------------------------
    # SINGLE OBSERVATION
    # --------------------------------------------------------

    if len(points) == 1:

        labels = np.array(
            [0]
        )

    else:

        # ----------------------------------------------------
        # DBSCAN
        # ----------------------------------------------------

        clustering = DBSCAN(

            eps=DBSCAN_EPS,

            min_samples=DBSCAN_MIN_SAMPLES

        ).fit(points)


        labels = clustering.labels_


    unique_labels = sorted(
        set(labels)
    )


    # --------------------------------------------------------
    # CREATE OBJECTS
    # --------------------------------------------------------

    for cluster_label in unique_labels:

        cluster_indices = np.where(
            labels == cluster_label
        )[0]


        cluster_points = points[
            cluster_indices
        ]


        cluster_observations = [

            class_observations[i]

            for i in cluster_indices

        ]


        # ----------------------------------------------------
        # ROBUST CENTER
        # ----------------------------------------------------

        median_center = np.median(
            cluster_points,
            axis=0
        )


        mean_center = np.mean(
            cluster_points,
            axis=0
        )


        # ----------------------------------------------------
        # SPATIAL EXTENT
        # ----------------------------------------------------

        min_xyz = np.min(
            cluster_points,
            axis=0
        )

        max_xyz = np.max(
            cluster_points,
            axis=0
        )

        extent = (
            max_xyz
            -
            min_xyz
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidences = [

            float(
                obs["confidence"]
            )

            for obs in cluster_observations

        ]


        mean_confidence = float(
            np.mean(confidences)
        )

        max_confidence = float(
            np.max(confidences)
        )


        # ----------------------------------------------------
        # FRAME INFORMATION
        # ----------------------------------------------------

        frames = [

            int(
                obs["detection_frame"]
            )

            for obs in cluster_observations

        ]


        keyframes = [

            int(
                obs["nearest_keyframe"]
            )

            for obs in cluster_observations

        ]


        # ----------------------------------------------------
        # OBJECT ID
        # ----------------------------------------------------

        object_id = (

            f"object_"
            f"{global_object_id:03d}"

        )


        global_object_id += 1


        # ----------------------------------------------------
        # CREATE FUSED OBJECT
        # ----------------------------------------------------

        fused_object = {

            "object_id":
                object_id,

            "class_name":
                class_name,

            "observation_count":
                int(
                    len(cluster_observations)
                ),

            "detection_frames":
                frames,

            "nearest_keyframes":
                keyframes,

            "mean_confidence":
                mean_confidence,

            "max_confidence":
                max_confidence,

            "3d_center_median": {

                "x":
                    float(
                        median_center[0]
                    ),

                "y":
                    float(
                        median_center[1]
                    ),

                "z":
                    float(
                        median_center[2]
                    )

            },

            "3d_center_mean": {

                "x":
                    float(
                        mean_center[0]
                    ),

                "y":
                    float(
                        mean_center[1]
                    ),

                "z":
                    float(
                        mean_center[2]
                    )

            },

            "3d_min": {

                "x":
                    float(
                        min_xyz[0]
                    ),

                "y":
                    float(
                        min_xyz[1]
                    ),

                "z":
                    float(
                        min_xyz[2]
                    )

            },

            "3d_max": {

                "x":
                    float(
                        max_xyz[0]
                    ),

                "y":
                    float(
                        max_xyz[1]
                    ),

                "z":
                    float(
                        max_xyz[2]
                    )

            },

            "3d_extent": {

                "x":
                    float(
                        extent[0]
                    ),

                "y":
                    float(
                        extent[1]
                    ),

                "z":
                    float(
                        extent[2]
                    )

            },

            "source_observations":
                cluster_observations

        }


        fused_objects.append(
            fused_object
        )


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(

            f"\n{object_id}"

        )

        print(

            f"Class          : "
            f"{class_name}"

        )

        print(

            f"Observations   : "
            f"{len(cluster_observations)}"

        )

        print(

            f"Mean confidence: "
            f"{mean_confidence:.3f}"

        )

        print(

            "3D center      : "
            f"("
            f"{median_center[0]:.3f}, "
            f"{median_center[1]:.3f}, "
            f"{median_center[2]:.3f}"
            f")"

        )

        print(

            "3D extent      : "
            f"("
            f"{extent[0]:.3f}, "
            f"{extent[1]:.3f}, "
            f"{extent[2]:.3f}"
            f")"

        )


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w"
) as f:

    json.dump(
        fused_objects,
        f,
        indent=2
    )


# ============================================================
# SAVE NUMPY
# ============================================================

np.save(
    OUTPUT_NPY,
    np.array(
        fused_objects,
        dtype=object
    ),
    allow_pickle=True
)


# ============================================================
# STATISTICS
# ============================================================

class_counts = {}

for obj in fused_objects:

    class_name = obj[
        "class_name"
    ]

    if class_name not in class_counts:

        class_counts[
            class_name
        ] = 0

    class_counts[
        class_name
    ] += 1


with open(
    OUTPUT_STATS,
    "w"
) as f:

    f.write(
        "STAGE 7 - TASK 5\n"
    )

    f.write(
        "SEMANTIC OBJECT FUSION / 3D CLUSTERING\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Input observations: "
        f"{len(observations)}\n"
    )

    f.write(
        f"Fused semantic objects: "
        f"{len(fused_objects)}\n"
    )

    f.write(
        f"DBSCAN epsilon: "
        f"{DBSCAN_EPS}\n"
    )

    f.write(
        f"DBSCAN minimum samples: "
        f"{DBSCAN_MIN_SAMPLES}\n"
    )

    f.write(
        "\nOBJECT SUMMARY\n"
    )

    for obj in fused_objects:

        center = obj[
            "3d_center_median"
        ]

        extent = obj[
            "3d_extent"
        ]

        f.write(
            "\n"
        )

        f.write(
            f"Object ID: "
            f"{obj['object_id']}\n"
        )

        f.write(
            f"Class: "
            f"{obj['class_name']}\n"
        )

        f.write(
            f"Observations: "
            f"{obj['observation_count']}\n"
        )

        f.write(
            f"Mean confidence: "
            f"{obj['mean_confidence']:.4f}\n"
        )

        f.write(
            "3D center: "
            f"("
            f"{center['x']:.4f}, "
            f"{center['y']:.4f}, "
            f"{center['z']:.4f}"
            f")\n"
        )

        f.write(
            "3D extent: "
            f"("
            f"{extent['x']:.4f}, "
            f"{extent['y']:.4f}, "
            f"{extent['z']:.4f}"
            f")\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TASK 5 SUMMARY")
print("=" * 70)

print(
    "\nInput semantic observations:",
    len(observations)
)

print(
    "Fused semantic objects:",
    len(fused_objects)
)

print(
    "\nSemantic object classes:"
)

for class_name in sorted(
    class_counts.keys()
):

    print(

        f"{class_name:15s}"
        f" : "
        f"{class_counts[class_name]} "
        f"3D object(s)"

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
    "1. task5_fused_semantic_objects.json"
)

print(
    "2. task5_fused_semantic_objects.npy"
)

print(
    "3. task5_semantic_object_statistics.txt"
)

print(
    "\n" + "=" * 70
)

print(
    "TASK 5 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)
