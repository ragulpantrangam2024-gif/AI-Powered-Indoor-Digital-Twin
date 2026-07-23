import cv2
import numpy as np

# -------------------------
# Intrinsic Matrix
# -------------------------

K = np.array([
    [800, 0, 320],
    [0, 800, 240],
    [0, 0, 1]
], dtype=float)

# -------------------------
# Rotation (Identity)
# -------------------------

R = np.eye(3)

# -------------------------
# Translation
# -------------------------

t = np.array([[0],
              [0],
              [0]], dtype=float)

# -------------------------
# World Points
# -------------------------

world_points = np.array([
    [0, 0, 5],
    [1, 1, 5],
    [-1, 1, 5],
    [2, -1, 8],
    [-2, -1, 8]
], dtype=float)

# Blank image
image = np.ones((480,640,3),dtype=np.uint8)*255

for P in world_points:

    P = P.reshape(3,1)

    # World → Camera
    P_camera = R @ P + t

    X = P_camera[0,0]
    Y = P_camera[1,0]
    Z = P_camera[2,0]

    # Camera → Image
    u = int(K[0,0]*X/Z + K[0,2])
    v = int(K[1,1]*Y/Z + K[1,2])

    print("World:",P.ravel()," Pixel:",(u,v))

    cv2.circle(image,(u,v),6,(0,0,255),-1)

cv2.imshow("Projection",image)

cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_3\results\projection.jpg",image)

cv2.waitKey(0)
cv2.destroyAllWindows()