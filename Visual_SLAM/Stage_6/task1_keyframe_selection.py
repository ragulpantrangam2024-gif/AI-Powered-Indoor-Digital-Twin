import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


# ============================================================
# STAGE 6 - TASK 1
# KEYFRAME SELECTION
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


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("STAGE 6 - TASK 1")
print("KEYFRAME SELECTION")
print("=" * 70)

print("\nStage 6 directory:")
print(STAGE6_DIR)

print("\nImage directory:")
print(IMAGE_DIR)

print("\nResults directory:")
print(RESULT_DIR)


# ============================================================
# 3. CHECK IMAGE DIRECTORY
# ============================================================

if not os.path.exists(IMAGE_DIR):

    print("\nERROR: Image directory not found!")

    print(
        "\nExpected:"
    )

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


# ============================================================
# 5. CHECK IMAGE COUNT
# ============================================================

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
# 6. ORB DETECTOR
# ============================================================

orb = cv2.ORB_create(
    nfeatures=1000
)


# ============================================================
# 7. BF MATCHER
# ============================================================

bf = cv2.BFMatcher(
    cv2.NORM_HAMMING,
    crossCheck=False
)


# ============================================================
# 8. KEYFRAME PARAMETERS
# ============================================================

# Maximum number of frames allowed between
# two keyframes.

MAX_FRAME_INTERVAL = 10


# Minimum number of good matches.
#
# If the current frame has fewer matches with
# the current keyframe, the scene has changed
# significantly.

MIN_GOOD_MATCHES = 80


# Lowe ratio threshold

LOWE_RATIO = 0.75


# ============================================================
# 9. INITIAL KEYFRAME
# ============================================================

keyframe_indices = []

keyframe_match_counts = []

keyframe_intervals = []


# First frame is always a keyframe

current_keyframe_index = 0

keyframe_indices.append(
    0
)


# ============================================================
# 10. LOAD FIRST KEYFRAME
# ============================================================

current_keyframe = cv2.imread(
    image_files[0]
)

if current_keyframe is None:

    print(
        "\nERROR: Could not read first image."
    )

    exit()


current_gray = cv2.cvtColor(
    current_keyframe,
    cv2.COLOR_BGR2GRAY
)


current_kp, current_des = (
    orb.detectAndCompute(
        current_gray,
        None
    )
)


print("\n")
print("=" * 70)
print("KEYFRAME SELECTION STARTED")
print("=" * 70)

print(
    "\nInitial Keyframe: Frame 0"
)


# ============================================================
# 11. PROCESS FRAMES
# ============================================================

for i in range(
    1,
    len(image_files)
):

    # --------------------------------------------------------
    # Load current frame
    # --------------------------------------------------------

    frame = cv2.imread(
        image_files[i]
    )

    if frame is None:

        print(
            f"Frame {i}: Could not read image."
        )

        continue


    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Detect ORB features
    # --------------------------------------------------------

    kp, des = (
        orb.detectAndCompute(
            gray,
            None
        )
    )


    if (
        current_des is None
        or
        des is None
    ):

        good_match_count = 0

    else:

        # ----------------------------------------------------
        # KNN MATCHING
        # ----------------------------------------------------

        knn_matches = bf.knnMatch(
            current_des,
            des,
            k=2
        )


        # ----------------------------------------------------
        # LOWE RATIO TEST
        # ----------------------------------------------------

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


        good_match_count = len(
            good_matches
        )


    # --------------------------------------------------------
    # Frame interval
    # --------------------------------------------------------

    frame_interval = (
        i
        -
        current_keyframe_index
    )


    # --------------------------------------------------------
    # KEYFRAME DECISION
    # --------------------------------------------------------

    select_keyframe = False

    reason = ""


    # Condition 1:
    # Too few matches

    if (
        good_match_count
        <
        MIN_GOOD_MATCHES
    ):

        select_keyframe = True

        reason = (
            "Low feature overlap"
        )


    # Condition 2:
    # Maximum frame interval reached

    elif (
        frame_interval
        >=
        MAX_FRAME_INTERVAL
    ):

        select_keyframe = True

        reason = (
            "Maximum frame interval"
        )


    # --------------------------------------------------------
    # Select new keyframe
    # --------------------------------------------------------

    if select_keyframe:

        keyframe_indices.append(
            i
        )

        keyframe_match_counts.append(
            good_match_count
        )

        keyframe_intervals.append(
            frame_interval
        )


        print(
            f"Frame {i:03d} -> "
            f"KEYFRAME | "
            f"Matches: "
            f"{good_match_count:3d} | "
            f"Interval: "
            f"{frame_interval:2d} | "
            f"Reason: "
            f"{reason}"
        )


        # ----------------------------------------------------
        # Update current keyframe
        # ----------------------------------------------------

        current_keyframe_index = i

        current_gray = gray

        current_kp = kp

        current_des = des


# ============================================================
# 12. FINAL KEYFRAME INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("KEYFRAME SELECTION SUMMARY")
print("=" * 70)

print(
    f"\nTotal Frames       : "
    f"{len(image_files)}"
)

print(
    f"Selected Keyframes : "
    f"{len(keyframe_indices)}"
)


print("\nKeyframe Indices:")

print(
    keyframe_indices
)


# ============================================================
# 13. KEYFRAME INTERVAL STATISTICS
# ============================================================

if len(keyframe_indices) > 1:

    intervals = np.diff(
        keyframe_indices
    )

    print("\nKeyframe Interval Statistics")

    print(
        f"Minimum : "
        f"{np.min(intervals)}"
    )

    print(
        f"Maximum : "
        f"{np.max(intervals)}"
    )

    print(
        f"Average : "
        f"{np.mean(intervals):.2f}"
    )

else:

    intervals = np.array(
        []
    )


# ============================================================
# 14. SAVE KEYFRAME INDICES
# ============================================================

keyframe_array = np.array(
    keyframe_indices,
    dtype=np.int32
)

np.save(
    os.path.join(
        RESULT_DIR,
        "keyframe_indices.npy"
    ),
    keyframe_array
)


# ============================================================
# 15. SAVE KEYFRAME TEXT FILE
# ============================================================

keyframe_txt = os.path.join(
    RESULT_DIR,
    "keyframes.txt"
)

with open(
    keyframe_txt,
    "w"
) as file:

    file.write(
        "STAGE 6 - TASK 1\n"
    )

    file.write(
        "KEYFRAME SELECTION\n"
    )

    file.write(
        "=" * 70
        +
        "\n\n"
    )

    file.write(
        f"Total Frames: "
        f"{len(image_files)}\n"
    )

    file.write(
        f"Selected Keyframes: "
        f"{len(keyframe_indices)}\n\n"
    )

    file.write(
        "Keyframe Indices:\n"
    )

    for index in keyframe_indices:

        file.write(
            f"{index}\n"
        )


# ============================================================
# 16. SAVE STATISTICS
# ============================================================

statistics_file = os.path.join(
    RESULT_DIR,
    "keyframe_statistics.txt"
)

with open(
    statistics_file,
    "w"
) as file:

    file.write(
        "STAGE 6 - TASK 1\n"
    )

    file.write(
        "KEYFRAME SELECTION\n"
    )

    file.write(
        "=" * 70
        +
        "\n\n"
    )

    file.write(
        f"Total Frames       : "
        f"{len(image_files)}\n"
    )

    file.write(
        f"Selected Keyframes : "
        f"{len(keyframe_indices)}\n"
    )

    if len(intervals) > 0:

        file.write(
            f"Minimum Interval   : "
            f"{np.min(intervals)}\n"
        )

        file.write(
            f"Maximum Interval   : "
            f"{np.max(intervals)}\n"
        )

        file.write(
            f"Average Interval   : "
            f"{np.mean(intervals):.2f}\n"
        )

    file.write(
        "\nKeyframe Indices:\n"
    )

    file.write(
        str(keyframe_indices)
    )


# ============================================================
# 17. VISUALIZE KEYFRAME DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    range(len(image_files)),
    np.zeros(len(image_files)),
    marker=".",
    linestyle=""
)

plt.scatter(
    keyframe_indices,
    np.zeros(
        len(keyframe_indices)
    ),
    s=80,
    label="Keyframes"
)

plt.xlabel(
    "Frame Index"
)

plt.yticks([])

plt.title(
    "Stage 6 - Keyframe Selection"
)

plt.grid(
    True,
    axis="x"
)

plt.legend()

plt.tight_layout()


plot_path = os.path.join(
    RESULT_DIR,
    "keyframe_selection.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("TASK 1 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nResults saved in:")

print(
    RESULT_DIR
)

print("\nGenerated files:")

print(
    "1. keyframe_indices.npy"
)

print(
    "2. keyframes.txt"
)

print(
    "3. keyframe_statistics.txt"
)

print(
    "4. keyframe_selection.png"
)

print(
    "\nNo plot windows were opened."
)