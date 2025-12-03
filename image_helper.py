import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import hashlib

class DIPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Image Processing App")
        self.root.geometry("1200x800")

        # Variables
        self.original_image = None
        self.processed_image = None
        self.brightness = tk.DoubleVar(value=0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.threshold_value = tk.IntVar(value=127)
        self.kernel_size = tk.IntVar(value=5)
        self.morph_iterations = tk.IntVar(value=1)
        self.denoise_h = tk.IntVar(value=10)
        self.denoise_hColor = tk.IntVar(value=10)
        self.denoise_template = tk.IntVar(value=7)
        self.denoise_search = tk.IntVar(value=21)
        self.freq_mask_size = tk.IntVar(value=30)
        self.color_channel = tk.StringVar(value="Blue")
        self.hist_fig = None
        self.hist_ax = None
        self.hist_canvas_widget = None
        self.update_timer = None
        self.debounce_delay = 0.2  # 200ms
        self.cached_original_pil = None
        self.cached_processed_pil = None
        self.original_image_hash = None
        self.processed_image_hash = None

        # GUI Elements
        self.create_widgets()

    def create_widgets(self):
        # Frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        # Row 0: Load Image
        load_btn = tk.Button(control_frame, text="Load Image", command=self.load_image)
        load_btn.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")

        # Row 1: Brightness
        brightness_label = tk.Label(control_frame, text="Brightness")
        brightness_label.grid(row=1, column=0, sticky="w")
        brightness_slider = tk.Scale(control_frame, from_=-100, to=100, orient=tk.HORIZONTAL, variable=self.brightness, command=self.update_image)
        brightness_slider.grid(row=1, column=1, sticky="ew")

        # Row 2: Contrast
        contrast_label = tk.Label(control_frame, text="Contrast")
        contrast_label.grid(row=2, column=0, sticky="w")
        contrast_slider = tk.Scale(control_frame, from_=0.1, to=3.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.contrast, command=self.update_image)
        contrast_slider.grid(row=2, column=1, sticky="ew")

        # Row 3: Filters
        blur_btn = tk.Button(control_frame, text="Blur", command=self.apply_blur)
        blur_btn.grid(row=3, column=0, pady=2, sticky="ew")
        sharpen_btn = tk.Button(control_frame, text="Sharpen", command=self.apply_sharpen)
        sharpen_btn.grid(row=3, column=1, pady=2, sticky="ew")

        # Row 4: Edge and Hist
        edge_btn = tk.Button(control_frame, text="Edge Detection", command=self.apply_edge_detection)
        edge_btn.grid(row=4, column=0, pady=2, sticky="ew")
        hist_eq_btn = tk.Button(control_frame, text="Histogram Eq", command=self.apply_histogram_equalization)
        hist_eq_btn.grid(row=4, column=1, pady=2, sticky="ew")

        # Row 5: Frequency
        freq_label = tk.Label(control_frame, text="Freq Mask Size")
        freq_label.grid(row=5, column=0, sticky="w")
        freq_slider = tk.Scale(control_frame, from_=10, to=100, orient=tk.HORIZONTAL, variable=self.freq_mask_size)
        freq_slider.grid(row=5, column=1, sticky="ew")

        # Row 6: Frequency Button
        freq_btn = tk.Button(control_frame, text="Apply Frequency Filter", command=self.apply_frequency_filter)
        freq_btn.grid(row=6, column=0, columnspan=2, pady=5, sticky="ew")

        # Row 7: Color Channel
        comp_label = tk.Label(control_frame, text="Color Channel")
        comp_label.grid(row=7, column=0, sticky="w")
        comp_menu = tk.OptionMenu(control_frame, self.color_channel, "Blue", "Green", "Red", command=self.extract_color_channels)
        comp_menu.grid(row=7, column=1, sticky="ew")

        # Row 8: Denoise H
        denoise_h_label = tk.Label(control_frame, text="Denoise H")
        denoise_h_label.grid(row=8, column=0, sticky="w")
        denoise_h_slider = tk.Scale(control_frame, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.denoise_h)
        denoise_h_slider.grid(row=8, column=1, sticky="ew")

        # Row 9: Denoise HColor
        denoise_hColor_label = tk.Label(control_frame, text="Denoise HColor")
        denoise_hColor_label.grid(row=9, column=0, sticky="w")
        denoise_hColor_slider = tk.Scale(control_frame, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.denoise_hColor)
        denoise_hColor_slider.grid(row=9, column=1, sticky="ew")

        # Row 10: Denoise Template
        denoise_template_label = tk.Label(control_frame, text="Denoise Template")
        denoise_template_label.grid(row=10, column=0, sticky="w")
        denoise_template_slider = tk.Scale(control_frame, from_=1, to=15, orient=tk.HORIZONTAL, variable=self.denoise_template)
        denoise_template_slider.grid(row=10, column=1, sticky="ew")

        # Row 11: Denoise Search
        denoise_search_label = tk.Label(control_frame, text="Denoise Search")
        denoise_search_label.grid(row=11, column=0, sticky="w")
        denoise_search_slider = tk.Scale(control_frame, from_=1, to=50, orient=tk.HORIZONTAL, variable=self.denoise_search)
        denoise_search_slider.grid(row=11, column=1, sticky="ew")

        # Row 12: Denoise Button
        denoise_btn = tk.Button(control_frame, text="Apply Denoising", command=self.apply_denoising)
        denoise_btn.grid(row=12, column=0, columnspan=2, pady=5, sticky="ew")

        # Row 13: Kernel Size
        kernel_label = tk.Label(control_frame, text="Kernel Size")
        kernel_label.grid(row=13, column=0, sticky="w")
        kernel_slider = tk.Scale(control_frame, from_=3, to=15, orient=tk.HORIZONTAL, variable=self.kernel_size)
        kernel_slider.grid(row=13, column=1, sticky="ew")

        # Row 14: Morph Iterations
        morph_iter_label = tk.Label(control_frame, text="Morph Iterations")
        morph_iter_label.grid(row=14, column=0, sticky="w")
        morph_iter_slider = tk.Scale(control_frame, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.morph_iterations)
        morph_iter_slider.grid(row=14, column=1, sticky="ew")

        # Row 15: Morphological Button
        morph_btn = tk.Button(control_frame, text="Apply Morphological", command=self.apply_morphological)
        morph_btn.grid(row=15, column=0, columnspan=2, pady=5, sticky="ew")

        # Row 16: Threshold
        thresh_label = tk.Label(control_frame, text="Threshold Value")
        thresh_label.grid(row=16, column=0, sticky="w")
        thresh_slider = tk.Scale(control_frame, from_=0, to=255, orient=tk.HORIZONTAL, variable=self.threshold_value)
        thresh_slider.grid(row=16, column=1, sticky="ew")

        # Row 17: Threshold Button
        thresh_btn = tk.Button(control_frame, text="Apply Thresholding", command=self.apply_thresholding)
        thresh_btn.grid(row=17, column=0, columnspan=2, pady=5, sticky="ew")

        # Row 18: Compress
        compress_btn = tk.Button(control_frame, text="Save Compressed", command=self.save_compressed)
        compress_btn.grid(row=18, column=0, columnspan=2, pady=5, sticky="ew")

        # Row 19: Reset (larger)
        reset_btn = tk.Button(control_frame, text="RESET", command=self.reset_image, font=("Arial", 14, "bold"), height=2)
        reset_btn.grid(row=19, column=0, columnspan=2, pady=10, sticky="ew")

        # Configure grid weights
        control_frame.grid_columnconfigure(1, weight=1)

        # Frame for images and histogram
        right_frame = tk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        # Frame for images
        image_frame = tk.Frame(right_frame)
        image_frame.pack(side=tk.LEFT, padx=10, pady=10)

        # Original Image Label
        self.original_label = tk.Label(image_frame, text="Original Image")
        self.original_label.pack()

        # Processed Image Label
        self.processed_label = tk.Label(image_frame, text="Processed Image")
        self.processed_label.pack()

        # Frame for histogram
        hist_frame = tk.Frame(right_frame)
        hist_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        # Histogram Frame
        self.hist_canvas = tk.Frame(hist_frame)
        self.hist_canvas.pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.processed_image = self.original_image.copy()
            self.display_images()

    def update_image(self, *args):
        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(int(self.debounce_delay * 1000), self._perform_update)

    def _perform_update(self):
        if self.original_image is not None:
            self.processed_image = self.adjust_brightness_contrast(self.original_image, self.brightness.get(), self.contrast.get())
            self.display_images()
        self.update_timer = None

    def adjust_brightness_contrast(self, image, brightness, contrast):
        # Adjust brightness and contrast
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted

    def apply_blur(self):
        if self.processed_image is not None:
            kernel_size = self.kernel_size.get()
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            self.processed_image = cv2.GaussianBlur(self.processed_image, (kernel_size, kernel_size), 0)
            self.display_images()

    def apply_sharpen(self):
        if self.processed_image is not None:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            self.processed_image = cv2.filter2D(self.processed_image, -1, kernel)
            self.display_images()

    def apply_edge_detection(self):
        if self.processed_image is not None:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.display_images()

    def apply_histogram_equalization(self):
        if self.processed_image is not None:
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            self.processed_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
            self.display_images()

    def apply_frequency_filter(self, *args):
        if self.processed_image is not None:
            threading.Thread(target=self._apply_frequency_filter_thread).start()

    def _apply_frequency_filter_thread(self):
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        # Apply DFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        # Create low-pass filter mask
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        size = self.freq_mask_size.get()
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow-size:crow+size, ccol-size:ccol+size] = 1
        # Apply mask and inverse DFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        self.processed_image = cv2.cvtColor(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.root.after(0, self.display_images)

    def extract_color_channels(self, *args):
        if self.original_image is not None:
            channel = self.color_channel.get()
            b, g, r = cv2.split(self.original_image)
            if channel == "Blue":
                self.processed_image = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
            elif channel == "Green":
                self.processed_image = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
            elif channel == "Red":
                self.processed_image = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])
            self.display_images()

    def apply_denoising(self, *args):
        if self.processed_image is not None:
            threading.Thread(target=self._apply_denoising_thread).start()

    def _apply_denoising_thread(self):
        h = self.denoise_h.get()
        hColor = self.denoise_hColor.get()
        templateWindowSize = self.denoise_template.get()
        searchWindowSize = self.denoise_search.get()
        self.processed_image = cv2.fastNlMeansDenoisingColored(self.processed_image, None, h, hColor, templateWindowSize, searchWindowSize)
        self.root.after(0, self.display_images)

    def apply_morphological(self, *args):
        if self.processed_image is not None:
            threading.Thread(target=self._apply_morphological_thread).start()

    def _apply_morphological_thread(self):
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        kernel_size = self.kernel_size.get()
        iterations = self.morph_iterations.get()
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        eroded = cv2.erode(gray, kernel, iterations=iterations)
        self.processed_image = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        self.root.after(0, self.display_images)

    def apply_thresholding(self, *args):
        if self.processed_image is not None:
            threading.Thread(target=self._apply_thresholding_thread).start()

    def _apply_thresholding_thread(self):
        gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        thresh_value = self.threshold_value.get()
        _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        self.processed_image = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        self.root.after(0, self.display_images)

    def save_compressed(self):
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
            if file_path:
                cv2.imwrite(file_path, self.processed_image, [cv2.IMWRITE_JPEG_QUALITY, 50])
                messagebox.showinfo("Saved", "Image saved with compression.")

    def reset_image(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.brightness.set(0)
            self.contrast.set(1.0)
            self.display_images()

    def _compute_image_hash(self, image):
        return hashlib.md5(image.tobytes()).hexdigest()

    def display_images(self):
        if self.original_image is not None:
            # Compute current hashes
            current_original_hash = self._compute_image_hash(self.original_image)
            current_processed_hash = self._compute_image_hash(self.processed_image)

            # Check if original image has changed
            if self.original_image_hash != current_original_hash or self.cached_original_pil is None:
                original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.cached_original_pil = Image.fromarray(original_rgb).resize((400, 300))
                self.original_image_hash = current_original_hash

            # Check if processed image has changed
            if self.processed_image_hash != current_processed_hash or self.cached_processed_pil is None:
                processed_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
                self.cached_processed_pil = Image.fromarray(processed_rgb).resize((400, 300))
                self.processed_image_hash = current_processed_hash

            # Create Tk images from cached PIL images
            original_tk = ImageTk.PhotoImage(self.cached_original_pil)
            processed_tk = ImageTk.PhotoImage(self.cached_processed_pil)

            self.original_label.config(image=original_tk)
            self.original_label.image = original_tk

            self.processed_label.config(image=processed_tk)
            self.processed_label.image = processed_tk

            # Update histogram
            self.update_histogram()

    def update_histogram(self):
        if self.processed_image is not None:
            # Convert to grayscale for histogram
            gray = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)

            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

            # Reuse figure if exists, else create new
            if self.hist_fig is None:
                self.hist_fig, self.hist_ax = plt.subplots(figsize=(4, 3))
                self.hist_ax.set_title('Grayscale Histogram')
                self.hist_ax.set_xlabel('Pixel Value')
                self.hist_ax.set_ylabel('Frequency')
                self.hist_canvas_widget = FigureCanvasTkAgg(self.hist_fig, master=self.hist_canvas)
                self.hist_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                self.hist_ax.clear()
                self.hist_ax.set_title('Grayscale Histogram')
                self.hist_ax.set_xlabel('Pixel Value')
                self.hist_ax.set_ylabel('Frequency')

            self.hist_ax.plot(hist, color='black')
            self.hist_canvas_widget.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DIPApp(root)
    root.mainloop()
