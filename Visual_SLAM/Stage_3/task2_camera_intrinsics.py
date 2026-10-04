import cv2
import numpy as np

# -----------------------------
# Camera Intrinsic Parameters
# -----------------------------

fx = 800
fy = 800

cx = 320
cy = 240

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

print("Camera Intrinsic Matrix:\n")
print(K)

# -----------------------------
# Image
# -----------------------------

image = np.ones((480, 640, 3), dtype=np.uint8) * 255

# -----------------------------
# 3D Points
# -----------------------------

points_3D = np.array([
    [0, 0, 5],
    [1, 1, 5],
    [-1, 1, 5],
    [2, -1, 8],
    [-2, -1, 8]
])

# -----------------------------
# Project Points
# -----------------------------

for point in points_3D:

    X, Y, Z = point

    u = int(fx * X / Z + cx)
    v = int(fy * Y / Z + cy)

    print(f"3D Point: {point} --> Pixel: ({u},{v})")

    cv2.circle(image, (u, v), 6, (0,0,255), -1)

    cv2.putText(
        image,
        f"({u},{v})",
        (u+5, v-5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255,0,0),
        1
    )

# -----------------------------
# Save Result
# -----------------------------

cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_3\results\projected_points.jpg",image)

cv2.imshow("Projected Points", image)

cv2.waitKey(0)

cv2.destroyAllWindows()