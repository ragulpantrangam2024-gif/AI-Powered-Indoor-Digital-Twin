import numpy as np

# World point
P_world = np.array([
    [2],
    [1],
    [5]
])

# Identity rotation
R = np.eye(3)

# Camera translated 1 meter to the right
t = np.array([
    [1],
    [0],
    [0]
])

# Convert world point to camera coordinates
P_camera = R @ P_world + t

print("World Point:")
print(P_world)

print("\nRotation Matrix:")
print(R)

print("\nTranslation Vector:")
print(t)

print("\nCamera Coordinates:")
print(P_camera)