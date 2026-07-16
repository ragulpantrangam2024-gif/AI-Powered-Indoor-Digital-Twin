import cv2

# ==========================================================
# Stage 1 - Task 3
# Image Preprocessing
# ==========================================================

# ----------------------------------------------------------
# 1. Read Image
# ----------------------------------------------------------

image = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_1\images\room.jpg")

if image is None:
    print("Error: Could not load image.")
    exit()

# ----------------------------------------------------------
# 2. Grayscale Conversion
# ----------------------------------------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ----------------------------------------------------------
# 3. Image Resizing
# ----------------------------------------------------------

resized = cv2.resize(image, (640, 480))

# ----------------------------------------------------------
# 4. Image Rotation
# ----------------------------------------------------------

rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

# ----------------------------------------------------------
# 5. Gaussian Blur
# ----------------------------------------------------------

gaussian = cv2.GaussianBlur(image, (5, 5), 0)

# ----------------------------------------------------------
# 6. Median Blur
# ----------------------------------------------------------

median = cv2.medianBlur(image, 5)

# ----------------------------------------------------------
# 7. Bilateral Filtering
# ----------------------------------------------------------

bilateral = cv2.bilateralFilter(image, 9, 75, 75)

# ----------------------------------------------------------
# 8. Display Results
# ----------------------------------------------------------

cv2.imshow("Original Image", image)
cv2.imshow("Grayscale Image", gray)
cv2.imshow("Resized Image", resized)
cv2.imshow("Rotated Image", rotated)
cv2.imshow("Gaussian Blur", gaussian)
cv2.imshow("Median Blur", median)
cv2.imshow("Bilateral Filter", bilateral)

# ----------------------------------------------------------
# 9. Save Results
# ----------------------------------------------------------

cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\grayscale.jpg", gray)
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\resized.jpg", resized)
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\rotated.jpg", rotated)
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\gaussian.jpg", gaussian)
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\median.jpg", median)
cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\bilateral.jpg", bilateral)

# ----------------------------------------------------------
# 10. Wait for Key Press and Close Windows
# ----------------------------------------------------------

cv2.waitKey(0)
cv2.destroyAllWindows()