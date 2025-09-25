# Ultrasound Image Visualization

This project demonstrates how to load, inspect, segment, and save **ultrasound images** using Python. It provides a simple but practical example of working with medical images in grayscale format, applying a predefined mask to highlight regions of interest, and saving the segmented results for further analysis.

## Overview

Ultrasound imaging is widely used in medical diagnostics to visualize internal organs, tissues, and abnormalities. Unlike CT or MRI, ultrasound produces real‑time images based on sound waves. This project focuses on basic visualization and segmentation of ultrasound images using **OpenCV** and **NumPy**. It shows how to:

- Load ultrasound images from disk
- Display them interactively
- Inspect pixel statistics such as intensity range, mean, and variance
- Apply a predefined mask to segment a region of interest (e.g., a tumor)
- Save the segmented image for later use

## Features

- **Image Loading and Display**  
  Read ultrasound images in grayscale and display them in a window.

- **Image Statistics**  
  Print useful information such as shape, size, data type, pixel range, mean, median, variance, and standard deviation.

- **Mask Loading and Display**  
  Load a binary mask image that defines the region of interest (e.g., a tumor boundary) and display it.

- **Segmentation**  
  Apply the mask to the ultrasound image using bitwise operations to isolate the region of interest.

- **Saving Results**  
  Save the segmented image into a dedicated output folder for later analysis or visualization.

## Libraries Used

- [OpenCV](https://opencv.org/) – for image loading, processing, and visualization  
- [NumPy](https://numpy.org/) – for numerical operations and pixel statistics  

## Example Workflow

1. Load an ultrasound image of a malignant case and display it.  
2. Print image statistics such as pixel intensity range and mean values.  
3. Load the corresponding tumor mask image and display it.  
4. Apply the mask to segment the tumor region from the ultrasound image.  
5. Save the segmented image into a `SegmentedImages` folder.

## Getting Started

To run the examples, install the required libraries:

```bash
pip install opencv-python numpy