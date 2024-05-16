"""
Threshold helper. Run without giving args.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import cv2
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)s %(levelname)s %(asctime)s:\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

def add_arguments(parser):
    return parser

class ImageAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Analysis GUI")

        # Initialize image attributes
        self.original_image = None
        self.tk_image = None

        # Create a frame for the controls
        self.control_frame = ttk.Frame(self.root, padding="10 10 10 10")
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # Create an "Open Image" button
        self.open_button = ttk.Button(self.control_frame, text="Open Image",
                                      command=self.open_image)
        self.open_button.pack(pady=5)

        # Create brightness slider
        self.white_thresh_label = ttk.Label(self.control_frame, text="white thresh")
        self.white_thresh_value = tk.StringVar(value="180")
        self.white_thresh_value_label = ttk.Label(self.control_frame,
                                                  textvariable=self.white_thresh_value)
        self.white_thresh_slider = ttk.Scale(self.control_frame, from_=0, to=255,
                                             orient=tk.HORIZONTAL,
                                             command=self.update_image)
        self.white_thresh_slider.set(180)  # Default value

        # Create contrast slider
        self.black_thresh_label = ttk.Label(self.control_frame, text="black thresh")
        self.black_thresh_value = tk.StringVar(value="70")
        self.black_thresh_value_label = ttk.Label(self.control_frame,
                                                  textvariable=self.black_thresh_value)
        self.black_thresh_slider = ttk.Scale(self.control_frame, from_=0, to=255,
                                             orient=tk.HORIZONTAL,
                                             command=self.update_image)
        self.black_thresh_slider.set(70)  # Default value

        # Create a frame for the image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.LEFT, padx=10, pady=10)

    def open_image(self):
        # Open a file dialog to select an image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.tif")])
        if file_path:
            # Load the selected image
            self.original_image = Image.open(file_path).convert("RGB")

            # Convert the image to PhotoImage
            self.tk_image = ImageTk.PhotoImage(self.original_image)

            # Create or update the image label
            if hasattr(self, 'image_label'):
                self.image_label.configure(image=self.tk_image)
            else:
                self.image_label = tk.Label(self.image_frame, image=self.tk_image)
                self.image_label.pack()

            # Display the sliders after loading an image
            self.white_thresh_label.pack(anchor=tk.W)
            self.white_thresh_slider.pack(fill=tk.X)
            self.white_thresh_value_label.pack(anchor=tk.E)
            self.black_thresh_label.pack(anchor=tk.W)
            self.black_thresh_slider.pack(fill=tk.X)
            self.black_thresh_value_label.pack(anchor=tk.E)

        self.update_image()

    def update_image(self, event=None):
        if self.original_image is None:
            return

        # Get the current slider values
        white_thresh = round(self.white_thresh_slider.get())
        black_thresh = round(self.black_thresh_slider.get())

        self.white_thresh_value.set(white_thresh)
        self.black_thresh_value.set(black_thresh)

        image_with_cells = np.array(self.original_image)

        contours = self.cell_detector_2(image_with_cells[:, :, 0],
                                        black_thresh=black_thresh,
                                        white_thresh=white_thresh
                                        )

        gray_contours, yellow_contours, green_contours = self.divide_contours(contours)

        image_with_cells = cv2.drawContours(
            image_with_cells, gray_contours, -1, (200, 200, 200), 1
        )
        image_with_cells = cv2.drawContours(
            image_with_cells, yellow_contours, -1, (175, 175, 75), 1
        )
        image_with_cells = cv2.drawContours(
            image_with_cells, green_contours, -1, (75, 200, 150), 1
        )

        # Convert the image to PhotoImage and update the label
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(image_with_cells))
        self.image_label.configure(image=self.tk_image)
        self.image_label.image = self.tk_image  # Keep a reference to avoid garbage collection

    def divide_contours(self, contours, convexity_min = 0.875, inertia_min = 0.6):

        gray_contours = []
        yellow_contours = []
        green_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 30 <= area < 500:
                convexity = self.cnt_convexity(cnt)
                inertia = self.cnt_inertia(cnt)
                if convexity >= convexity_min and inertia >= inertia_min:
                    green_contours.append(cnt)
                else:
                    yellow_contours.append(cnt)
            else:
                gray_contours.append(cnt)

        return gray_contours, yellow_contours, green_contours

    def cnt_convexity(self, cnt):
        hull = cv2.convexHull(cnt)

        cnt_area = cv2.contourArea(cnt)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            return None

        convexity = cnt_area / hull_area
        return convexity

    def cnt_inertia(self, cnt):
        # Cannot approximate as ellipse if less than 5 points
        if len(cnt) < 5:
            return None

        ellipse = cv2.fitEllipse(cnt)
        center, shape, angle = ellipse
        width, height = shape

        inertia = width / height
        return inertia

    def cnt_circularity(self, cnt):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if perimeter == 0:
            return None

        circularity = 4 * np.pi * area / (perimeter ** 2)
        return circularity

    def cell_detector_2(self, img, blur_kernel=(3, 3), black_thresh=70,
                        white_thresh=125, details: bool = False, qc_outdir=None,
                        auto_thresh=False):

        # Find black spots
        blurred = cv2.blur(img, blur_kernel)
        _, black_threshed = cv2.threshold(
            blurred, black_thresh, 255, cv2.THRESH_BINARY
        )
        black_thresh_inv = np.array(255) - black_threshed

        # find white spots
        _, white_threshed = cv2.threshold(
            img, white_thresh, 255, cv2.THRESH_BINARY
        )

        # combine white + black
        combined = cv2.bitwise_or(black_thresh_inv, white_threshed)  # type: ignore

        # Get contours of masked image
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        return contours


def main(args):

    root = tk.Tk()
    app = ImageAnalysisGUI(root)
    root.mainloop()
