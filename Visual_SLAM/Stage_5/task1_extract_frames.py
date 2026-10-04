import cv2
import os

# =====================================================
# STAGE 5 - TASK 1
# VIDEO TO IMAGE SEQUENCE
# =====================================================

# -----------------------------------------------------
# Paths
# -----------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_DIR = os.path.join(BASE_DIR, "video")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------------------------------
# Input Video
# -----------------------------------------------------

video_path = os.path.join(
    VIDEO_DIR,
    "input_video.mp4"
)

# -----------------------------------------------------
# Open Video
# -----------------------------------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video.")
    print("\nExpected video location:")
    print(video_path)
    exit()

# -----------------------------------------------------
# Video Information
# -----------------------------------------------------

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = total_frames / fps if fps > 0 else 0

print("=" * 60)
print("STAGE 5 - TASK 1")
print("VIDEO TO IMAGE SEQUENCE")
print("=" * 60)

print("\nVideo Information")
print("-" * 30)

print(f"FPS            : {fps:.2f}")
print(f"Total Frames   : {total_frames}")
print(f"Resolution     : {width} x {height}")
print(f"Duration       : {duration:.2f} seconds")

# -----------------------------------------------------
# Frame Sampling
# -----------------------------------------------------

FRAME_INTERVAL = 5

print("\nFrame Sampling")
print("-" * 30)

print(f"Taking every {FRAME_INTERVAL}th frame")

# -----------------------------------------------------
# Extract Frames
# -----------------------------------------------------

frame_index = 0
saved_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Save every Nth frame
    if frame_index % FRAME_INTERVAL == 0:

        filename = os.path.join(
            IMAGE_DIR,
            f"frame_{saved_count:06d}.jpg"
        )

        cv2.imwrite(
            filename,
            frame
        )

        saved_count += 1

    frame_index += 1

# -----------------------------------------------------
# Release Video
# -----------------------------------------------------

cap.release()

# -----------------------------------------------------
# Results
# -----------------------------------------------------

print("\n" + "=" * 60)
print("EXTRACTION COMPLETED")
print("=" * 60)

print(f"\nOriginal Frames : {total_frames}")
print(f"Frames Saved    : {saved_count}")

print("\nImage Resolution")
print(f"{width} x {height}")

print("\nSaved Location:")
print(IMAGE_DIR)

print("\nTask 1 Completed Successfully!")