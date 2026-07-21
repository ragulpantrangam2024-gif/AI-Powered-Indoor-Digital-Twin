# Stage 2 – Task 1: ORB Feature Matching using Brute Force Matcher

## Objective

The objective of this task is to detect ORB features in two images, compute their descriptors, and establish correspondences between the images using the Brute Force Matcher with Hamming Distance. Feature matching enables the identification of the same physical points across multiple images and serves as the foundation for camera motion estimation and Visual SLAM.

---

## Theory

Feature matching is the process of finding corresponding keypoints between two images of the same scene. After detecting keypoints and computing descriptors, each descriptor from the first image is compared with descriptors from the second image to determine the most similar feature.

In this task, ORB (Oriented FAST and Rotated BRIEF) is used to detect keypoints and generate binary descriptors. Since ORB descriptors are binary vectors, they are compared using the **Hamming Distance**, which measures the number of differing bits between two descriptors.

A **Brute Force Matcher (BFMatcher)** compares every descriptor in the first image with every descriptor in the second image and selects the closest match based on the minimum Hamming Distance.

To improve the reliability of the matches, **Cross Check** is enabled. A match is accepted only if two descriptors are each other's best match in both directions.

---

## Implementation

The following steps were implemented using OpenCV:

- Loaded two images of the same scene.
- Converted both images to grayscale.
- Detected ORB keypoints in each image.
- Computed ORB descriptors.
- Created a Brute Force Matcher using Hamming Distance.
- Matched descriptors between the two images.
- Sorted the matches based on descriptor distance.
- Displayed the top feature matches.
- Saved the matched image in the results folder.

---

## Output

The generated output includes:

- ORB keypoints detected in both images.
- Feature correspondences drawn between the two images.
- Total number of detected keypoints.
- Total number of matched features.

Example output:

```
Number of keypoints in Image 1: 500
Number of keypoints in Image 2: 500
Number of matches: 229
```

The matched image is saved in:

```
Stage_2/results/
```

---

## Key Concepts

### ORB (Oriented FAST and Rotated BRIEF)

ORB combines the FAST keypoint detector with the BRIEF descriptor to generate fast, rotation-invariant feature descriptors suitable for real-time computer vision applications.

### Descriptor

A descriptor is a binary representation of the appearance around a keypoint. It allows the same feature to be recognized across different images.

### Hamming Distance

Hamming Distance measures the number of differing bits between two binary descriptors. A smaller Hamming Distance indicates a better feature match.

### Brute Force Matcher

The Brute Force Matcher compares every descriptor in one image with every descriptor in the other image and selects the closest descriptor based on the chosen distance metric.

### Cross Check

Cross Check validates matches by ensuring that two descriptors are each other's best match. This helps remove many incorrect correspondences.

---

## Key Learning Outcomes

After completing this task, I learned:

- How feature matching establishes correspondences between two images.
- The difference between keypoints and descriptors.
- How ORB descriptors are matched using Hamming Distance.
- How the Brute Force Matcher performs descriptor comparison.
- The purpose of Cross Check in improving match reliability.
- How descriptor distance indicates the quality of a feature match.
- Why feature matching is the first step toward camera pose estimation and Visual SLAM.

---

## Applications

Feature matching is widely used in:

- Visual SLAM
- Visual Odometry
- Camera Pose Estimation
- Structure from Motion (SfM)
- Image Stitching
- Panorama Generation
- Augmented Reality
- Robotics
- Autonomous Driving
- 3D Reconstruction

---

## Relevance to the Final Project

Feature matching connects corresponding points between consecutive camera frames, enabling the estimation of camera motion and scene geometry. This is a fundamental step in Visual SLAM systems such as ORB-SLAM and forms the basis for constructing a consistent 3D map of an environment.

In the next task, feature matches will be refined by filtering weak correspondences, resulting in a more robust set of matches for geometric verification and camera pose estimation.

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Visual Studio Code
- Git & GitHub


# Stage 2 – Task 2: Good Match Filtering

## Objective

The objective of this task is to improve the quality of feature matching by removing weak and unreliable correspondences. Instead of using every feature match returned by the Brute Force Matcher, only matches with sufficiently small descriptor distances are retained. This improves the robustness of subsequent computer vision tasks such as camera pose estimation and Visual SLAM.

---

## Theory

Feature matching algorithms often produce incorrect or weak correspondences due to repetitive textures, image noise, illumination changes, or viewpoint variations. Although the Brute Force Matcher identifies the closest descriptor for each feature, not every match is reliable.

Each feature match is associated with a **Hamming Distance**, which measures the similarity between two binary ORB descriptors.

- Smaller Hamming Distance → Better match
- Larger Hamming Distance → Weaker match

By defining a distance threshold, weak matches can be discarded while retaining only the strongest correspondences.

In this task, matches with a Hamming Distance greater than the selected threshold were removed.

---

## Implementation

The following steps were implemented using OpenCV:

- Loaded two images of the same scene.
- Converted both images to grayscale.
- Detected ORB keypoints.
- Computed ORB descriptors.
- Matched descriptors using the Brute Force Matcher with Hamming Distance.
- Sorted matches based on descriptor distance.
- Filtered weak matches using a predefined distance threshold.
- Visualized only the good feature matches.
- Saved the filtered matching result.

---

## Output

Example output:

```
Number of keypoints in Image 1: 500
Number of keypoints in Image 2: 500

Total Matches: 229
Good Matches: 155
```

The generated output image displays only the filtered feature correspondences between the two images.

The processed image is saved in:

```
Stage_2/results/
```

---

## Key Concepts

### Hamming Distance

Hamming Distance measures the number of differing bits between two binary ORB descriptors.

- Smaller distance → Higher similarity
- Larger distance → Lower similarity

---

### Match Distance

Each feature match contains a distance value indicating how similar two descriptors are.

Example:

| Match | Distance |
|--------|---------:|
| A | 6 |
| B | 11 |
| C | 18 |
| D | 45 |
| E | 72 |

Matches with smaller distances are generally considered more reliable.

---

### Good Match Filtering

A distance threshold is used to remove weak feature correspondences.

Example:

```python
threshold = 40

if match.distance < threshold:
    good_matches.append(match)
```

Only matches whose descriptor distance is less than the threshold are retained.

---

## Key Learning Outcomes

After completing this task, I learned:

- How descriptor distance represents feature similarity.
- Why not all feature matches are reliable.
- How Hamming Distance is used to evaluate ORB feature matches.
- How to remove weak correspondences using a distance threshold.
- How filtering improves the quality of feature matching.
- Why strong feature correspondences are important before geometric verification.

---

## Applications

Good match filtering is widely used in:

- Visual SLAM
- Visual Odometry
- Camera Pose Estimation
- Image Registration
- Structure from Motion (SfM)
- Image Stitching
- Panorama Generation
- Robotics
- Autonomous Driving
- 3D Reconstruction

---

## Relevance to the Final Project

Reliable feature correspondences are essential for estimating camera motion and reconstructing a consistent map of the environment. By removing weak matches, the system becomes more robust against incorrect feature associations, improving the accuracy of later stages in the Visual SLAM pipeline.

In the next task, feature matching will be further improved using **K-Nearest Neighbors (KNN) Matching** and **Lowe's Ratio Test**, which eliminate ambiguous feature matches more effectively than simple distance thresholding.

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Visual Studio Code
- Git & GitHub


# Stage 2 – Task 3: KNN Matching and Lowe's Ratio Test

## Objective

The objective of this task is to improve the reliability of feature matching by identifying and removing ambiguous correspondences. Instead of selecting only the nearest descriptor, the K-Nearest Neighbors (KNN) Matcher retrieves the two closest descriptor matches, and Lowe's Ratio Test is applied to retain only distinctive and reliable feature matches.

---

## Theory

Feature matching based solely on the nearest descriptor can produce incorrect correspondences, especially when multiple features have similar appearances. To improve matching accuracy, KNN Matching retrieves the two nearest descriptor matches for every feature.

Lowe's Ratio Test compares the distance of the best match with the distance of the second-best match.

If the best match is significantly better than the second-best match, it is accepted. Otherwise, the match is rejected because it is considered ambiguous.

The acceptance criterion is given by:

\[
\frac{\text{Best Match Distance}}{\text{Second Best Match Distance}} < \text{Ratio Threshold}
\]

A ratio threshold of **0.75** was used in this implementation.

This approach significantly reduces false feature correspondences and is widely used in modern computer vision systems.

---

## Implementation

The following steps were implemented using OpenCV:

- Loaded two images of the same scene.
- Converted both images to grayscale.
- Detected ORB keypoints.
- Computed ORB descriptors.
- Performed K-Nearest Neighbor (KNN) feature matching.
- Retrieved the two nearest descriptor matches for every feature.
- Applied Lowe's Ratio Test to remove ambiguous matches.
- Visualized the filtered feature correspondences.
- Saved the final matching result.

---

## Output

Example output:

```
Keypoints Image 1: 500
Keypoints Image 2: 500

KNN Matches: 500
Good Matches: 152
```

The generated output image displays the reliable feature correspondences after applying Lowe's Ratio Test.

The processed image is saved in:

```
Stage_2/results/
```

---

## Key Concepts

### K-Nearest Neighbors (KNN) Matching

Instead of selecting only the nearest descriptor, KNN Matching retrieves the two closest descriptors for every feature.

This allows the algorithm to determine whether the best match is sufficiently better than the second-best match.

---

### Lowe's Ratio Test

Lowe's Ratio Test evaluates the uniqueness of a feature match.

For each feature:

- Best Match → Distance = m.distance
- Second Best Match → Distance = n.distance

The match is accepted only if:

```python
m.distance < 0.75 * n.distance
```

This removes ambiguous feature correspondences where multiple descriptors have similar distances.

---

### Ambiguous Match

Example:

| Best Match | Second Best Match |
|------------|------------------:|
| 15 | 16 |

Since both distances are nearly identical, the feature cannot be matched confidently and is rejected.

---

### Reliable Match

Example:

| Best Match | Second Best Match |
|------------|------------------:|
| 15 | 40 |

The best descriptor is significantly closer than the second-best descriptor, making the correspondence reliable.

---

## Key Learning Outcomes

After completing this task, I learned:

- The difference between Brute Force Matching and KNN Matching.
- Why selecting only the nearest descriptor may produce incorrect matches.
- How Lowe's Ratio Test removes ambiguous feature correspondences.
- Why descriptor uniqueness is important for reliable feature matching.
- How KNN Matching improves the robustness of feature matching.
- Why Lowe's Ratio Test is widely used before geometric verification.

---

## Applications

KNN Matching and Lowe's Ratio Test are widely used in:

- Visual SLAM
- Visual Odometry
- Structure from Motion (SfM)
- Camera Pose Estimation
- Image Stitching
- Panorama Generation
- Augmented Reality
- Robotics
- Autonomous Driving
- 3D Reconstruction

---

## Relevance to the Final Project

Reliable feature correspondences are essential for estimating camera motion and constructing accurate maps of the environment. Lowe's Ratio Test removes ambiguous matches before geometric verification, improving the robustness of Visual SLAM systems.

In the next task, the remaining feature correspondences will be verified using **RANSAC (Random Sample Consensus)**, which removes geometrically inconsistent matches and prepares the data for camera pose estimation.

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Visual Studio Code
- Git & GitHub


# Stage 2 – Task 4: RANSAC (Random Sample Consensus)

## Objective

The objective of this task is to improve the reliability of feature correspondences by removing geometrically inconsistent matches using the Random Sample Consensus (RANSAC) algorithm. RANSAC identifies the subset of feature matches that agree with a common geometric transformation while rejecting incorrect matches (outliers).

---

## Theory

Even after applying K-Nearest Neighbor (KNN) Matching and Lowe's Ratio Test, some incorrect feature correspondences may still remain. These incorrect matches are called **outliers** and can negatively affect camera motion estimation and 3D reconstruction.

RANSAC (Random Sample Consensus) is a robust estimation algorithm that identifies the largest group of feature matches that satisfy a common geometric model.

The algorithm works by repeatedly:

1. Randomly selecting a small subset of feature correspondences.
2. Estimating a geometric transformation (Homography).
3. Testing all remaining matches against the estimated model.
4. Classifying matches as:
   - **Inliers** – Matches consistent with the estimated transformation.
   - **Outliers** – Matches that do not satisfy the geometric model.
5. Repeating the process multiple times and selecting the model with the highest number of inliers.

This process significantly improves the quality of feature correspondences before camera pose estimation.

---

## Implementation

The following steps were implemented using OpenCV:

- Loaded two images of the same scene.
- Converted both images to grayscale.
- Detected ORB keypoints.
- Computed ORB descriptors.
- Performed KNN feature matching.
- Applied Lowe's Ratio Test to remove ambiguous matches.
- Extracted the coordinates of the matched keypoints.
- Estimated the Homography Matrix using RANSAC.
- Classified feature correspondences as inliers and outliers.
- Visualized only the inlier matches.
- Saved the final RANSAC result.

---

## Output

Example output:

```
Keypoints Image 1: 500
Keypoints Image 2: 500

Matches after Lowe Ratio Test: 152

RANSAC Inliers: 127
RANSAC Outliers: 25
```

The generated output image displays only the inlier feature correspondences after RANSAC.

The processed image is saved in:

```
Stage_2/results/
```

---

## Key Concepts

### Inlier

An inlier is a feature correspondence that agrees with the estimated geometric transformation between the two images.

These matches are considered reliable and are used for camera motion estimation.

---

### Outlier

An outlier is a feature correspondence that does not satisfy the estimated geometric model.

Outliers usually result from:

- Incorrect descriptor matching
- Repetitive textures
- Image noise
- Occlusions
- Perspective variations

These matches are rejected by RANSAC.

---

### Homography Matrix

A Homography Matrix describes the geometric relationship between two images of the same planar scene or two views related primarily by camera rotation.

RANSAC estimates this matrix while simultaneously identifying the feature correspondences that support it.

---

### RANSAC

Random Sample Consensus is a robust model estimation algorithm that iteratively searches for the largest set of feature correspondences that agree with a common geometric transformation.

Unlike descriptor-based matching, RANSAC verifies feature matches using geometric consistency.

---

## Key Learning Outcomes

After completing this task, I learned:

- Why descriptor similarity alone cannot guarantee correct feature matches.
- The difference between inliers and outliers.
- How RANSAC removes geometrically inconsistent correspondences.
- How a Homography Matrix is estimated from matched feature points.
- Why robust geometric verification is essential before camera pose estimation.
- How RANSAC improves the accuracy of feature matching in Visual SLAM.

---

## Applications

RANSAC is widely used in:

- Visual SLAM
- Visual Odometry
- Structure from Motion (SfM)
- Camera Pose Estimation
- Image Stitching
- Panorama Generation
- Augmented Reality
- Robotics
- Autonomous Driving
- 3D Reconstruction

---

## Relevance to the Final Project

Reliable feature correspondences are essential for accurately estimating camera motion and constructing a consistent digital representation of an environment.

RANSAC removes incorrect feature matches before geometric estimation, making it one of the core algorithms in feature-based Visual SLAM systems such as ORB-SLAM.

The inlier correspondences obtained in this task will be used in the next stage to estimate image transformations and understand camera movement.

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Visual Studio Code
- Git & GitHub

# Stage 2 – Task 5: Homography and Perspective Transformation

## Objective

The objective of this task is to estimate the Homography Matrix between two images using reliable feature correspondences and use it to perform a perspective transformation. This demonstrates how one image can be geometrically transformed to align with another image of the same scene.

---

## Theory

A **Homography** is a 3×3 transformation matrix that describes the relationship between two views of the same planar scene or two images captured primarily through camera rotation.

After detecting feature correspondences and removing incorrect matches using Lowe's Ratio Test and RANSAC, the remaining inlier correspondences are used to estimate the Homography Matrix.

The estimated Homography Matrix enables the transformation of points from one image to their corresponding locations in another image through perspective transformation.

This technique is widely used in panorama stitching, augmented reality, image registration, document rectification, and robotics.

---

## Implementation

The following steps were implemented using OpenCV:

- Loaded two images of the same scene.
- Converted both images to grayscale.
- Detected ORB keypoints.
- Computed ORB descriptors.
- Performed KNN feature matching.
- Applied Lowe's Ratio Test.
- Removed geometrically inconsistent matches using RANSAC.
- Estimated the Homography Matrix from the inlier correspondences.
- Applied perspective transformation using `cv2.warpPerspective()`.
- Displayed and saved the warped image.

---

## Output

The generated output includes:

- Estimated Homography Matrix.
- Warped version of Image 1.
- Original Image 1.
- Original Image 2.

Example console output:

```
Homography Matrix:

[[ 0.998  0.006 95.341]
 [-0.004  1.002  1.876]
 [ 0.000  0.000  1.000]]
```

The processed image is saved in:

```
Stage_2/results/
```

---

## Key Concepts

### Homography Matrix

A Homography Matrix is a 3×3 projective transformation matrix that maps points from one image to their corresponding locations in another image.

It is estimated from feature correspondences using RANSAC.

---

### Perspective Transformation

Perspective transformation maps every pixel in one image to its new position using the estimated Homography Matrix.

In OpenCV, this transformation is performed using:

```python
cv2.warpPerspective()
```

---

### Image Warping

Image warping transforms the appearance of one image so that it aligns with another image captured from a different viewpoint.

If the Homography Matrix is correctly estimated, the warped image closely matches the second image.

---

### Planar Scene

Homography accurately models the relationship between two images when the observed scene is approximately planar or when the camera undergoes pure rotational motion.

Examples include:

- Building facades
- Roads
- Floors
- Walls
- Posters
- Documents

---

## Key Learning Outcomes

After completing this task, I learned:

- What a Homography Matrix represents.
- How Homography is estimated from feature correspondences.
- The relationship between feature matching and geometric transformation.
- How perspective transformation is performed using `cv2.warpPerspective()`.
- Why Homography is suitable for planar scenes.
- The limitations of Homography for general 3D scenes.
- How Homography is used in image alignment and computer vision applications.

---

## Applications

Homography is widely used in:

- Panorama Stitching
- Image Registration
- Augmented Reality
- Document Scanning
- Perspective Correction
- Robotics
- Visual SLAM (planar scenes)
- Structure from Motion (SfM)
- Camera Motion Analysis
- 3D Reconstruction

---

## Relevance to the Final Project

Homography estimation demonstrates how reliable feature correspondences can be used to estimate the geometric relationship between two images.

Although modern Visual SLAM systems primarily estimate camera motion using the Essential Matrix and Perspective-n-Point (PnP) for general 3D scenes, Homography remains an important concept for planar environments, image alignment, and understanding geometric transformations.

This task completes the feature matching and geometric verification stage of the AI-Powered Indoor Digital Twin project.

---

## Technologies Used

- Python 3
- OpenCV
- NumPy
- Visual Studio Code
- Git & GitHub