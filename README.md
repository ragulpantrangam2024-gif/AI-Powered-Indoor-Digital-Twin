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



# Stage 1 – Task 2: Image Basics, Pixel Operations, ROI and Drawing Functions

## Objective

The objective of this task is to understand how a digital image is represented in memory and how individual pixels and groups of pixels can be accessed and modified using OpenCV and NumPy. This task also introduces basic drawing functions that are commonly used for visualization in computer vision applications.

---

## Theory

A digital image is represented as a three-dimensional NumPy array with the following structure:

```
(Height, Width, Channels)
```

For a color image, each pixel contains three color intensity values in **BGR (Blue, Green, Red)** format.

Example:

```
Pixel = [120, 180, 255]

Blue  = 120
Green = 180
Red   = 255
```

Pixel values are stored as **unsigned 8-bit integers (`uint8`)**, where each channel ranges from **0 to 255**.

Images can be manipulated by accessing individual pixels or by selecting a group of pixels using **NumPy slicing**.

---

## Implementation

The following operations were implemented in this task:

- Loaded a color image using OpenCV.
- Retrieved image properties including:
  - Image dimensions
  - Height
  - Width
  - Number of channels
  - Data type
- Accessed the value of a single pixel.
- Extracted individual Blue, Green and Red channel values.
- Modified a selected image region using NumPy slicing.
- Created a Region of Interest (ROI).
- Drew graphical objects on the image:
  - Line
  - Rectangle
  - Circle
- Added text to the image using OpenCV.
- Saved the modified image to the `results` folder.

---

## Output

The program displays:

- Image information
- Pixel values
- Individual BGR channel values
- Modified image containing:
  - Black region
  - Line
  - Rectangle
  - Circle
  - Text

The final output image is saved as:

```
results/task2_output.jpg
```

---

## Key Learning Outcomes

After completing this task, I learned:

- How images are stored as NumPy arrays.
- How to access and modify individual pixels.
- How OpenCV stores color images in BGR format.
- How to extract individual color channels.
- How NumPy slicing modifies an entire region of an image.
- What a Region of Interest (ROI) is and why it is useful.
- How to annotate images using OpenCV drawing functions.

---

## Relevance to the Final Project

Understanding pixel manipulation and image representation is essential before implementing advanced computer vision algorithms. These concepts will be used throughout the project for image preprocessing, feature detection, feature matching, camera pose estimation, Visual SLAM, and 3D reconstruction.

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Visual Studio Code
- Git & GitHub
* Git & GitHub



# Stage 1 – Task 3: Image Preprocessing

## Objective

The objective of this task is to understand the importance of image preprocessing before applying computer vision algorithms. Image preprocessing enhances image quality, reduces noise, and prepares the image for reliable feature detection and analysis.

---

## Theory

Image preprocessing is the first stage in most computer vision pipelines. Raw images captured by cameras may contain noise, unnecessary color information, or different image sizes that can negatively affect the performance of feature detection and object recognition algorithms.

Several preprocessing techniques are commonly applied before further image analysis:

- **Grayscale Conversion** reduces a color image from three channels (BGR) to a single intensity channel.
- **Image Resizing** changes the image dimensions while preserving its content, allowing algorithms to process images at a consistent resolution.
- **Image Rotation** changes the orientation of an image and is useful for image augmentation and geometric transformations.
- **Gaussian Blur** smooths the image by reducing high-frequency noise.
- **Median Blur** removes impulse (salt-and-pepper) noise while preserving object boundaries.
- **Bilateral Filtering** reduces noise while maintaining sharp edges, making it suitable for computer vision applications.

---

## Implementation

The following preprocessing operations were implemented using OpenCV:

- Loaded a color image.
- Converted the image to grayscale using `cv2.cvtColor()`.
- Resized the image using `cv2.resize()`.
- Rotated the image by 90° clockwise using `cv2.rotate()`.
- Applied Gaussian Blur using `cv2.GaussianBlur()`.
- Applied Median Blur using `cv2.medianBlur()`.
- Applied Bilateral Filtering using `cv2.bilateralFilter()`.
- Displayed all processed images for comparison.
- Saved the processed images in the `results` folder.

---

## Output

The following output images were generated:

- Original Image
- Grayscale Image
- Resized Image
- Rotated Image
- Gaussian Blur
- Median Blur
- Bilateral Filter

The processed images are saved in:

```
Stage_1/results/
```

---

## Key Learning Outcomes

After completing this task, I learned:

- Why image preprocessing is an essential step in computer vision.
- How grayscale conversion reduces computational complexity.
- How image resizing standardizes image dimensions.
- How image rotation performs geometric transformations.
- The purpose of Gaussian Blur for reducing image noise.
- The advantages of Median Blur for removing impulse noise.
- How Bilateral Filtering smooths an image while preserving edges.
- How different preprocessing techniques affect image quality and feature visibility.

---

## Applications

The preprocessing techniques implemented in this task are widely used in:

- Feature Detection
- Edge Detection
- Image Segmentation
- Object Detection
- Face Recognition
- Autonomous Driving
- Visual SLAM
- 3D Reconstruction
- Medical Image Processing

---

## Relevance to the Final Project

Image preprocessing improves the quality of input images before feature extraction. The techniques learned in this task provide the foundation for the next stages of the project, including edge detection, feature detection, feature matching, camera pose estimation, Visual SLAM, and AI-powered Indoor Digital Twin reconstruction.

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Visual Studio Code
- Git & GitHub
