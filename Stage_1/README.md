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

# Stage 1 – Task 4: Edge Detection

## Objective

The objective of this task is to understand how edges are detected in digital images using classical image processing techniques. Edge detection identifies object boundaries by measuring changes in image intensity and serves as the foundation for feature detection, object recognition, and Visual SLAM.

---

## Theory

An edge is a region in an image where the intensity changes rapidly. These intensity changes usually correspond to object boundaries, corners, or texture variations.

Edge detection converts an image into a representation that highlights structural information while suppressing homogeneous regions. Most edge detection algorithms operate on grayscale images because color information is not required for computing intensity gradients.

In this task, four commonly used edge detection algorithms were implemented:

- **Sobel Operator** – Computes horizontal and vertical intensity gradients.
- **Scharr Operator** – An improved version of Sobel that provides more accurate gradients for small kernels.
- **Laplacian Operator** – Computes the second derivative of the image to detect edges in all directions.
- **Canny Edge Detector** – A multi-stage algorithm that detects strong and continuous edges while reducing noise and false detections.

---

## Implementation

The following operations were implemented using OpenCV:

- Loaded the input image.
- Converted the image to grayscale.
- Applied the Sobel operator in both the x and y directions.
- Combined Sobel X and Sobel Y to obtain the final Sobel edge image.
- Applied the Scharr operator.
- Applied the Laplacian operator.
- Applied the Canny edge detector.
- Displayed the output of each edge detection algorithm.
- Saved the generated edge images in the `results` folder.

---

## Output

The following images were generated:

- Original Image
- Grayscale Image
- Sobel Edge Detection
- Scharr Edge Detection
- Laplacian Edge Detection
- Canny Edge Detection

The processed images are saved in:

```
Stage_1/results/
```

---

## Key Learning Outcomes

After completing this task, I learned:

- What an image edge represents.
- How image gradients are used to detect edges.
- The difference between first-order and second-order edge detection methods.
- How Sobel computes horizontal and vertical gradients.
- Why Scharr provides more accurate gradient estimation than Sobel for small kernels.
- How the Laplacian detects edges in all directions.
- Why the Canny Edge Detector produces cleaner and more reliable edge maps.
- The importance of edge detection before feature extraction.

---

## Applications

Edge detection is widely used in:

- Feature Detection
- Image Segmentation
- Object Detection
- Face Detection
- Autonomous Driving
- Robotics
- Medical Image Analysis
- Industrial Inspection
- Visual SLAM
- 3D Reconstruction

---

## Relevance to the Final Project

Edge detection is an important preprocessing step in computer vision pipelines. The detected edges reveal the structural information of a scene, making it easier for later algorithms to identify stable feature points.

In the next stage of the project, these concepts will be extended to feature detection algorithms such as Harris Corner Detector, FAST, and ORB, which form the basis of feature matching and Visual SLAM.

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Visual Studio Code
- Git & GitHub


# Stage 1 – Task 5: Feature Detection

## Objective

The objective of this task is to understand and implement classical feature detection algorithms used in computer vision. Feature detectors identify distinctive points in an image that can be reliably detected and matched across multiple images. These features form the foundation of image matching, camera pose estimation, object tracking, and Visual SLAM.

---

## Theory

A **feature** is a distinctive region in an image that can be recognized even when the viewpoint, scale, or illumination changes. Corners are generally preferred over edges because they contain intensity variations in multiple directions, making them easier to localize and track.

In this task, four popular feature detection algorithms were studied and implemented:

### 1. Harris Corner Detector
The Harris Corner Detector identifies corners by analyzing intensity changes in both horizontal and vertical directions using image gradients. It computes a corner response for every pixel and classifies regions as flat areas, edges, or corners.

### 2. Shi-Tomasi Corner Detector
Shi-Tomasi is an improved version of the Harris detector. Instead of using the Harris response equation, it evaluates the minimum eigenvalue of the gradient matrix to select stronger and more reliable corners. It is widely used in feature tracking applications.

### 3. FAST (Features from Accelerated Segment Test)
FAST is a high-speed corner detector designed for real-time computer vision applications. It compares the intensity of pixels surrounding a candidate pixel and classifies it as a corner based on a predefined threshold. FAST is widely used in robotics and embedded vision because of its computational efficiency.

### 4. ORB (Oriented FAST and Rotated BRIEF)
ORB combines the FAST detector with the BRIEF descriptor to create a fast, rotation-invariant feature extraction algorithm. It detects keypoints using FAST, assigns an orientation to each keypoint, and computes binary descriptors that can be efficiently matched between images. ORB is the feature extraction algorithm used in ORB-SLAM.

---

## Implementation

The following feature detection algorithms were implemented using OpenCV:

- Loaded the input image.
- Converted the image to grayscale.
- Detected Harris corners.
- Detected Shi-Tomasi corners.
- Detected FAST keypoints.
- Detected ORB keypoints and computed ORB descriptors.
- Visualized detected keypoints on the original image.
- Saved the output images for comparison.

---

## Output

The following outputs were generated:

- Harris Corner Detection
- Shi-Tomasi Corner Detection
- FAST Keypoint Detection
- ORB Keypoint Detection

The processed images are saved in:

```
Stage_1/results/
```

---

## Comparison of Feature Detectors

| Algorithm | Detects Corners | Computes Descriptor | Rotation Invariant | Speed |
|-----------|-----------------|---------------------|--------------------|-------|
| Harris | ✅ | ❌ | ❌ | Moderate |
| Shi-Tomasi | ✅ | ❌ | ❌ | Moderate |
| FAST | ✅ | ❌ | ❌ | Very Fast |
| ORB | ✅ | ✅ | ✅ | Very Fast |

---

## Key Learning Outcomes

After completing this task, I learned:

- What image features and keypoints are.
- Why corners are more reliable than edges for tracking.
- How Harris detects corners using image gradients.
- How Shi-Tomasi improves corner selection using eigenvalues.
- How FAST performs extremely fast corner detection for real-time applications.
- How ORB combines FAST and BRIEF to detect and describe image features.
- The difference between a **keypoint** and a **descriptor**.
- Why ORB descriptors are binary and matched using Hamming distance.
- Why ORB is widely used in Visual SLAM systems such as ORB-SLAM.

---

## Applications

Feature detection is widely used in:

- Feature Matching
- Object Recognition
- Image Stitching
- Panorama Generation
- Camera Pose Estimation
- Visual Odometry
- Augmented Reality
- Robotics
- Autonomous Driving
- Visual SLAM
- 3D Reconstruction

---

## Relevance to the Final Project

Feature detection is one of the most important stages in a Visual SLAM pipeline. Reliable keypoints enable the system to identify the same physical locations across multiple frames, allowing the estimation of camera motion and reconstruction of the surrounding environment.

The ORB feature detector implemented in this task serves as the core feature extraction method used in ORB-SLAM and will be used in the next stage for feature matching between images.

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Visual Studio Code
- Git & GitHub