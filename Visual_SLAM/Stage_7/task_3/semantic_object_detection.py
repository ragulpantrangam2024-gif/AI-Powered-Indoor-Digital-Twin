import os
import cv2
import json
import numpy as np

from ultralytics import YOLO


# ============================================================
# STAGE 7 - TASK 3
# ROOM-SPECIFIC SEMANTIC OBJECT DETECTION
# ============================================================

print("=" * 70)
print("STAGE 7 - TASK 3")
print("ROOM-SPECIFIC SEMANTIC OBJECT DETECTION")
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

PROJECT_DIR = os.path.dirname(
    STAGE_7_DIR
)

# Use the original 141 frames from Stage 6
IMAGE_DIR = os.path.join(
    PROJECT_DIR,
    "Stage_6",
    "images"
)

RESULTS_DIR = os.path.join(
    STAGE_7_DIR,
    "results"
)

DETECTION_IMAGE_DIR = os.path.join(
    RESULTS_DIR,
    "task3_filtered_detections"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    DETECTION_IMAGE_DIR,
    exist_ok=True
)


print("\nProject directory:")
print(PROJECT_DIR)

print("\nImage directory:")
print(IMAGE_DIR)

print("\nResults directory:")
print(RESULTS_DIR)


# ============================================================
# CHECK IMAGE DIRECTORY
# ============================================================

if not os.path.isdir(IMAGE_DIR):

    print(
        "\nERROR: Image directory not found:"
    )

    print(
        IMAGE_DIR
    )

    raise SystemExit


# ============================================================
# FIND IMAGES
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


image_files = [

    f for f in os.listdir(
        IMAGE_DIR
    )

    if f.lower().endswith(
        IMAGE_EXTENSIONS
    )

]


image_files.sort()


print(
    "\nImages found:",
    len(image_files)
)


if len(image_files) == 0:

    print(
        "ERROR: No images found."
    )

    raise SystemExit


print(
    "\nFirst five images:"
)

for filename in image_files[:5]:

    print(
        filename
    )


print(
    "\nLast image:"
)

print(
    image_files[-1]
)


# ============================================================
# LOAD YOLO
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "LOADING YOLO MODEL"
)

print(
    "=" * 70
)


MODEL_NAME = "yolo11n.pt"


print(
    "\nModel:",
    MODEL_NAME
)


model = YOLO(
    MODEL_NAME
)


print(
    "YOLO model loaded successfully."
)


# ============================================================
# DETECTION SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.50

IOU_THRESHOLD = 0.50


print(
    "\nConfidence threshold:",
    CONFIDENCE_THRESHOLD
)

print(
    "IoU threshold:",
    IOU_THRESHOLD
)


# ============================================================
# ROOM-SPECIFIC ALLOWED CLASSES
# ============================================================

ALLOWED_CLASSES = {

    "bed",

    "chair",

    "laptop",

    "dining table",

    "keyboard",

    "bottle",

    "cup",

    "backpack",

    "book"

}


print(
    "\nAllowed semantic classes:"
)

for class_name in sorted(
    ALLOWED_CLASSES
):

    print(
        " -",
        class_name
    )


# ============================================================
# DATA STORAGE
# ============================================================

all_detections = []

total_raw_detections = 0

total_filtered_detections = 0

frames_with_detections = 0

class_counts = {}


# ============================================================
# PROCESS FRAMES
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "STARTING ROOM-SPECIFIC OBJECT DETECTION"
)

print(
    "=" * 70
)


for frame_index, filename in enumerate(
    image_files
):

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )


    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            f"\nFrame {frame_index:03d}: "
            "Could not read image."
        )

        continue


    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

    results = model.predict(

        source=image,

        conf=CONFIDENCE_THRESHOLD,

        iou=IOU_THRESHOLD,

        verbose=False

    )


    result = results[0]


    frame_detections = []

    raw_count = 0


    # --------------------------------------------------------
    # EXTRACT DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes


        raw_count = len(boxes)

        total_raw_detections += raw_count


        for detection_index in range(
            len(boxes)
        ):

            box = boxes[
                detection_index
            ]


            # Bounding box
            xyxy = box.xyxy[
                0
            ].cpu().numpy()


            x1, y1, x2, y2 = (
                xyxy
            )


            # Confidence
            confidence = float(
                box.conf[
                    0
                ].cpu().numpy()
            )


            # Class ID
            class_id = int(
                box.cls[
                    0
                ].cpu().numpy()
            )


            # Class name
            class_name = model.names[
                class_id
            ]


            # ------------------------------------------------
            # FILTER CLASSES
            # ------------------------------------------------

            if class_name not in ALLOWED_CLASSES:

                continue


            # ------------------------------------------------
            # CENTER
            # ------------------------------------------------

            center_x = (
                x1 + x2
            ) / 2.0


            center_y = (
                y1 + y2
            ) / 2.0


            # ------------------------------------------------
            # WIDTH / HEIGHT
            # ------------------------------------------------

            width = (
                x2 - x1
            )


            height = (
                y2 - y1
            )


            # ------------------------------------------------
            # SAVE DETECTION
            # ------------------------------------------------

            detection = {

                "detection_id":
                    len(frame_detections),

                "class_id":
                    class_id,

                "class_name":
                    class_name,

                "confidence":
                    confidence,

                "bbox": {

                    "x1":
                        float(x1),

                    "y1":
                        float(y1),

                    "x2":
                        float(x2),

                    "y2":
                        float(y2)

                },

                "center": {

                    "x":
                        float(center_x),

                    "y":
                        float(center_y)

                },

                "width":
                    float(width),

                "height":
                    float(height)

            }


            frame_detections.append(
                detection
            )


            total_filtered_detections += 1


            if class_name not in class_counts:

                class_counts[
                    class_name
                ] = 0


            class_counts[
                class_name
            ] += 1


    # ========================================================
    # FRAME RESULT
    # ========================================================

    if len(frame_detections) > 0:

        frames_with_detections += 1


    frame_result = {

        "frame_index":
            frame_index,

        "filename":
            filename,

        "num_detections":
            len(frame_detections),

        "detections":
            frame_detections

    }


    all_detections.append(
        frame_result
    )


    # ========================================================
    # DRAW FILTERED DETECTIONS
    # ========================================================

    annotated_image = image.copy()


    for detection in frame_detections:

        x1 = int(
            detection["bbox"]["x1"]
        )

        y1 = int(
            detection["bbox"]["y1"]
        )

        x2 = int(
            detection["bbox"]["x2"]
        )

        y2 = int(
            detection["bbox"]["y2"]
        )


        class_name = detection[
            "class_name"
        ]


        confidence = detection[
            "confidence"
        ]


        label = (

            f"{class_name} "

            f"{confidence:.2f}"

        )


        # Bounding box
        cv2.rectangle(

            annotated_image,

            (x1, y1),

            (x2, y2),

            (0, 255, 0),

            2

        )


        # Label background
        text_size = cv2.getTextSize(

            label,

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            2

        )[0]


        text_width = text_size[0]

        text_height = text_size[1]


        label_y = max(
            y1 - 5,
            text_height + 5
        )


        cv2.rectangle(

            annotated_image,

            (
                x1,
                label_y - text_height - 5
            ),

            (
                x1 + text_width + 5,
                label_y + 2
            ),

            (0, 255, 0),

            -1

        )


        cv2.putText(

            annotated_image,

            label,

            (
                x1 + 2,
                label_y - 2
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (0, 0, 0),

            2

        )


    # ========================================================
    # FRAME INFORMATION
    # ========================================================

    info_text = (

        f"Frame: {frame_index:03d} | "

        f"Detections: "
        f"{len(frame_detections)}"

    )


    cv2.putText(

        annotated_image,

        info_text,

        (10, 25),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        (255, 255, 255),

        2

    )


    # ========================================================
    # SAVE EVERY FRAME
    # ========================================================

    output_path = os.path.join(

        DETECTION_IMAGE_DIR,

        f"detection_{frame_index:06d}.jpg"

    )


    cv2.imwrite(

        output_path,

        annotated_image

    )


    # ========================================================
    # PROGRESS
    # ========================================================

    print(

        f"Frame {frame_index:03d} | "

        f"Raw: {raw_count:2d} | "

        f"Accepted: "
        f"{len(frame_detections):2d}"

    )


# ============================================================
# SAVE JSON
# ============================================================

json_file = os.path.join(

    RESULTS_DIR,

    "task3_filtered_detections.json"

)


with open(

    json_file,

    "w"

) as f:

    json.dump(

        all_detections,

        f,

        indent=2

    )


# ============================================================
# SAVE NUMPY
# ============================================================

np_file = os.path.join(

    RESULTS_DIR,

    "task3_filtered_detections.npy"

)


np.save(

    np_file,

    np.array(

        all_detections,

        dtype=object

    ),

    allow_pickle=True

)


# ============================================================
# SORT CLASSES
# ============================================================

sorted_classes = sorted(

    class_counts.items(),

    key=lambda x: x[1],

    reverse=True

)


# ============================================================
# SAVE STATISTICS
# ============================================================

statistics_file = os.path.join(

    RESULTS_DIR,

    "task3_filtered_statistics.txt"

)


with open(

    statistics_file,

    "w"

) as f:

    f.write(

        "STAGE 7 - TASK 3\n"

    )

    f.write(

        "ROOM-SPECIFIC SEMANTIC OBJECT DETECTION\n"

    )

    f.write(

        "=" * 70 + "\n\n"

    )


    f.write(

        f"Total Images : "
        f"{len(image_files)}\n"

    )


    f.write(

        f"Confidence Threshold : "
        f"{CONFIDENCE_THRESHOLD}\n"

    )


    f.write(

        f"IoU Threshold : "
        f"{IOU_THRESHOLD}\n"

    )


    f.write(

        f"Raw Detections : "
        f"{total_raw_detections}\n"

    )


    f.write(

        f"Filtered Detections : "
        f"{total_filtered_detections}\n"

    )


    f.write(

        f"Frames With Detections : "
        f"{frames_with_detections}\n"

    )


    if len(image_files) > 0:

        average = (

            total_filtered_detections

            /

            len(image_files)

        )

    else:

        average = 0.0


    f.write(

        f"Average Detections Per Frame : "
        f"{average:.3f}\n"

    )


    f.write(

        "\nALLOWED CLASSES\n"

    )


    for class_name in sorted(
        ALLOWED_CLASSES
    ):

        f.write(

            f"{class_name}\n"

        )


    f.write(

        "\nDETECTED CLASS COUNTS\n"

    )


    for class_name, count in sorted_classes:

        f.write(

            f"{class_name} : "
            f"{count}\n"

        )


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "TASK 3 SUMMARY"
)

print(
    "=" * 70
)


print(
    "\nTotal Images:",
    len(image_files)
)


print(
    "Raw Detections:",
    total_raw_detections
)


print(
    "Filtered Detections:",
    total_filtered_detections
)


print(
    "Frames With Detections:",
    frames_with_detections
)


if len(image_files) > 0:

    print(

        "Average Filtered "
        "Detections Per Frame:",

        f"{total_filtered_detections / len(image_files):.3f}"

    )


print(
    "\nDetected Classes:"
)


if len(sorted_classes) == 0:

    print(
        "No allowed objects detected."
    )

else:

    for class_name, count in sorted_classes:

        print(

            f"{class_name:20s} : "
            f"{count}"

        )


print(
    "\nFiltered JSON:"
)

print(
    json_file
)


print(
    "\nFiltered NumPy:"
)

print(
    np_file
)


print(
    "\nStatistics:"
)

print(
    statistics_file
)


print(
    "\nAnnotated frames:"
)

print(
    DETECTION_IMAGE_DIR
)


print(
    "\n" + "=" * 70
)

print(
    "TASK 3 COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)