# Digital Image Processing App

## Description
This is a GUI application for digital image processing built with Python, Tkinter, and OpenCV. It allows users to load images and apply various processing techniques in real-time.

## Features
- Load images (JPG, JPEG, PNG, BMP)
- Adjust brightness and contrast with sliders
- Apply filters: Blur, Sharpen, Edge Detection, Histogram Equalization
- Frequency domain filtering with adjustable mask size
- Extract color channels (Blue, Green, Red)
- Denoising with customizable parameters
- Morphological operations (erosion)
- Thresholding
- Save processed images with compression
- Real-time histogram display of the processed image
- Reset to original image

## System Requirements
- Windows operating system
- No additional software installation required (standalone executable)

## Download and Installation
1. Locate the `image_helper.exe` file in the `dist/` folder of the project.
2. Copy or move the `image_helper.exe` file to a convenient location on your computer (e.g., Desktop or a dedicated folder).
3. No installation is needed; the executable is ready to run.

## Usage
1. Double-click the `image_helper.exe` file to launch the application.
2. Click the "Load Image" button to select and open an image file from your computer.
3. Use the sliders and buttons on the left panel to apply various image processing operations:
   - Adjust Brightness and Contrast sliders for basic adjustments.
   - Click buttons like "Blur", "Sharpen", "Edge Detection", etc., to apply filters.
   - Adjust parameters like Kernel Size, Threshold Value, etc., before applying operations.
4. View the original image on the left and the processed image on the right.
5. The histogram of the processed image is displayed on the right side.
6. Click "Save Compressed" to save the processed image as a JPEG with compression.
7. Click "RESET" to revert the processed image back to the original.

## Notes
- Some operations (like frequency filtering, denoising, morphological, and thresholding) run in the background using threading to keep the UI responsive.
- For best performance, ensure your system has sufficient RAM, especially when processing large images.
- The application supports common image formats and provides real-time updates for most adjustments.

## Troubleshooting
- If the application doesn't start, ensure you have the correct executable file and that it's not corrupted.
- For any issues with image loading or processing, verify that the image file is not corrupted and is in a supported format.