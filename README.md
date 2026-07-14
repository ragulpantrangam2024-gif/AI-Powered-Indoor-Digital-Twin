# Stage 1 – Task 1: Image Loading and Basic Image Properties

## Objective

The objective of this task is to understand how a computer reads and represents an image before performing any computer vision operations. This forms the foundation of the entire Visual SLAM and 3D Reconstruction pipeline.

## Theory

A digital image is a collection of pixels arranged in a two-dimensional grid. In OpenCV, images are stored as **NumPy arrays**, where each pixel contains intensity values.

* **Color Image:** Three channels representing **Blue, Green, and Red (BGR)**. OpenCV uses the BGR color format by default.

When an image is loaded using `cv2.imread()`, it is stored as a NumPy array of type `uint8`, meaning each pixel value is represented using an unsigned 8-bit integer.

## Implementation

In this task, the following operations were performed:

* Loaded an image using OpenCV (`cv2.imread()`).
* Verified that the image was successfully loaded.
* Displayed the image using `cv2.imshow()`.
* Retrieved the image dimensions (height, width, and number of channels).
* Printed the image data type.
* Saved a copy of the image using `cv2.imwrite()`.

## Output

The program displays:

* Original image
* Image height
* Image width
* Number of color channels
* Image data type (`uint8`)

A copy of the image is also saved in the `results` folder.

## Key Learning Outcomes

After completing this task, I learned:

* How OpenCV reads an image from disk.
* How images are represented internally as NumPy arrays.
* Why OpenCV uses the BGR color format.
* How to access basic image properties such as shape and data type.
* Why `uint8` is the standard data type for digital images.

## Relevance to the Final Project

This task is the first step toward building an **AI-powered Indoor Digital Twin**. Every computer vision pipeline—including feature detection, feature matching, camera pose estimation, Visual SLAM, and 3D reconstruction—begins by reading and understanding image data. A solid understanding of image representation is essential before implementing higher-level perception algorithms.

## Technologies Used

* Python 3
* OpenCV
* NumPy
* Visual Studio Code
* Git & GitHub
