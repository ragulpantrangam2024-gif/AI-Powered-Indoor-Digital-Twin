import cv2

# ==========================================================
# Stage 1 - Task 2
# Image Basics: Pixels, ROI and Drawing Functions
# ==========================================================

# ----------------------------------------------------------
# 1. Read Image
# ----------------------------------------------------------

image = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_1\images\room.jpg")

if image is None:
    print("Error: Could not load image.")
    exit()

# ----------------------------------------------------------
# 2. Image Properties
# ----------------------------------------------------------

print("========== IMAGE INFORMATION ==========")
print(f"Shape      : {image.shape}")
print(f"Height     : {image.shape[0]}")
print(f"Width      : {image.shape[1]}")
print(f"Channels   : {image.shape[2]}")
print(f"Data Type  : {image.dtype}")

# ----------------------------------------------------------
# 3. Pixel Access
# ----------------------------------------------------------

pixel = image[767, 1023]

print("\n========== PIXEL VALUE ==========")
print("Pixel at (767,1023):", pixel)

# Individual color channels

blue = image[100,200,0]
green = image[100,200,1]
red = image[100,200,2]

print("\n========== COLOR CHANNELS ==========")
print("Blue  :", blue)
print("Green :", green)
print("Red   :", red)

# ----------------------------------------------------------
# 4. Modify Pixel Region
# ----------------------------------------------------------

# Create a black rectangle

image[0:200,200:300] = [0,0,0]

# ----------------------------------------------------------
# 5. Region of Interest (ROI)
# ----------------------------------------------------------

# Crop wardrobe

wardrobe = image[250:700,150:450]

# Uncomment to view ROI
cv2.imshow("Wardrobe", wardrobe)

# ----------------------------------------------------------
# 6. Drawing Functions
# ----------------------------------------------------------

# Blue line

cv2.line(
    image,
    (150,50),
    (450,50),
    (255,0,0),
    3
)

# Blue rectangle

cv2.rectangle(
    image,
    (250,60),
    (350,180),
    (255,0,0),
    3
)

# Green circle

cv2.circle(
    image,
    (180,250),
    50,
    (0,255,0),
    3
)

# Add text

cv2.putText(
    image,
    "Indoor Digital Twin",
    (100,100),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (241,120,40),
    2
)

# ----------------------------------------------------------
# 7. Save Result
# ----------------------------------------------------------

cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\basic_image.jpg", image)

# ----------------------------------------------------------
# 8. Display Result
# ----------------------------------------------------------

cv2.imshow("Task 2 Output", image)

cv2.waitKey(0)
cv2.destroyAllWindows()