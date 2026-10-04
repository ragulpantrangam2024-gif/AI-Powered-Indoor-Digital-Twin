import cv2

# Read image
image = cv2.imread(r"D:\rwu\rwu jobs\TwinCube\Stage_3\images\chessboard1.jpeg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Your board has 8x8 squares -> 7x7 inner corners
pattern_size = (7, 7)

# Better detector (recommended)
ret, corners = cv2.findChessboardCornersSB(gray, pattern_size)

print("Chessboard Found:", ret)

if ret:
    cv2.drawChessboardCorners(image, pattern_size, corners, ret)

cv2.imshow("Detected Chessboard", image)
cv2.waitKey(0)
cv2.destroyAllWindows()