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
1. Locate the `image_helper.exe` file in the release tab on the right hand side of your Github monitor.
2. Copy, make a short cut or move the `image_helper.exe` file to a convenient location on your computer (e.g., Desktop or a dedicated folder).
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

## Building from Source
To build your own `.exe` file from the source code:

1. **Download the Repository**:
   - Go to the GitHub repository page.
   - Click the "Code" button and select "Download ZIP".
   - Extract the ZIP file to a local folder.

2. **Set Up Python Environment**:
   - Ensure Python 3.7+ is installed (download from python.org if needed).
   - Open a command prompt or terminal in the extracted folder.
   - Create a virtual environment (optional but recommended):
     ```
     python -m venv venv
     venv\Scripts\activate  # On Windows
     ```
   - Install required dependencies:
     ```
     pip install pyinstaller opencv-python pillow matplotlib numpy
     ```

3. **Build the Executable**:
   - Run PyInstaller using the provided spec file:
     ```
     pyinstaller image_helper.spec
     ```
   - This will generate the `.exe` file in the `dist/` folder (e.g., `dist/image_helper.exe`).

4. **Run the Application**:
   - Navigate to `dist/` and double-click `image_helper.exe` to launch the app.

## Automated Builds with GitHub Actions
This repository includes a GitHub Actions workflow that automatically builds the `.exe` file on Windows and attaches it to releases. To use this:
- Push changes to the repository.
- Create a new release on GitHub (under "Releases" in the repository).
- The workflow will trigger, build the executable, and upload it as a release asset for download.

## Troubleshooting
- If the application doesn't start, ensure you have the correct executable file and that it's not corrupted.
- For any issues with image loading or processing, verify that the image file is not corrupted and is in a supported format.
