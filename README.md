# HeartShapeAR

## AR Overlay Results
[Demo Video](demo.avi)

## How It Works

### Camera Calibration
The program first detects corners in the chessboard pattern from a video (e.g., `chessboard.mp4`). It uses these detected 2D image points alongside the known 3D object points (assuming the chessboard lies on a flat surface) to compute the camera matrix (intrinsic parameters) and distortion coefficients using OpenCV’s `cv.calibrateCamera()` function.

### Pose Estimation
With the calibration data available, the system estimates the camera’s pose relative to the chessboard plane in each frame using `cv.solvePnP()`. This step returns the rotation and translation vectors, which are essential for determining how the 3D AR object will be projected onto the 2D image.

### 3D Heart Model Generation
A heart shape is generated via a parametric function. To create a 3D effect, the heart is extruded by generating two sets of 3D points:
- **Front Face:** The original heart curve points (lying on the chessboard plane, i.e., z = 0).
- **Back Face:** A set of points translated along the negative Z-axis (e.g., z = -depth).

The two faces are then connected with lines to emphasize the extruded, volumetric appearance.

### AR Overlay and Video Saving
The generated 3D heart model is projected onto the image using the calculated pose and intrinsic parameters with `cv.projectPoints()`. The program then overlays the heart model onto the live video frame, drawing the front and back outlines and connecting edges to give a clear 3D impression. Finally, the AR results are simultaneously displayed on-screen and saved into an output video file (e.g., `demo.avi`) using OpenCV’s video writing tools.
