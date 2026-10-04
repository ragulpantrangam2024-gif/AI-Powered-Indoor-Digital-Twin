import cv2

# ==========================================================
# Stage 1 - Task 4
# Edge Detection
# ==========================================================

# ----------------------------------------------------------
# 1. Read Image
# ----------------------------------------------------------

image = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_1\images\room.jpg")

if image is None:
    print("Error: Could not load image.")
    exit()
    

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_y = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

sobel = cv2.addWeighted(
    sobel_x,
    0.5,
    sobel_y,
    0.5,
    0
)

scharr_x = cv2.Scharr(
    gray,
    cv2.CV_64F,
    1,
    0
)

scharr_y = cv2.Scharr(
    gray,
    cv2.CV_64F,
    0,
    1
)

scharr_x = cv2.convertScaleAbs(scharr_x)
scharr_y = cv2.convertScaleAbs(scharr_y)

scharr = cv2.addWeighted(
    scharr_x,
    0.5,
    scharr_y,
    0.5,
    0
)

laplacian = cv2.Laplacian(
    gray,
    cv2.CV_64F
)

laplacian = cv2.convertScaleAbs(laplacian)

canny = cv2.Canny(
    gray,
    100,
    200
)

cv2.imshow("Original", image)

cv2.imshow("Gray", gray)

cv2.imshow("Sobel", sobel)

cv2.imshow("Scharr", scharr)

cv2.imshow("Laplacian", laplacian)

cv2.imshow("Canny", canny)


cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\sobel.jpg", sobel)

cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\scharr.jpg", scharr)

cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\laplacian.jpg", laplacian)

cv2.imwrite(r"D:\rwu\rwu jobs\TwinCube\Stage_1\results\canny.jpg", canny)


cv2.waitKey(0)

cv2.destroyAllWindows()