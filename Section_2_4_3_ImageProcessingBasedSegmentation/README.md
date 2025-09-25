# Image Processing Based Segmentation

This project demonstrates how to perform **medical image segmentation** using classical image processing techniques in Python. It focuses on applying masks, thresholding, and interactive controls to segment regions of interest from medical images such as brain tumor scans and CT slices.

## Overview

Segmentation is a key step in medical image analysis, allowing specific structures or abnormalities to be isolated for closer study. This project shows two main approaches:

1. **Segmentation using available masks** – where a predefined mask is applied to highlight a region of interest (e.g., a tumor).  
2. **Segmentation using thresholding** – where pixel intensity ranges are used to generate masks dynamically, with interactive sliders to adjust thresholds and slice selection.

The examples are designed to be approachable for beginners while still demonstrating practical techniques used in medical imaging.

## Features

- **Mask-Based Segmentation**  
  Load an image and its corresponding mask, apply the mask to extract the region of interest, and overlay the mask in red on the original image for visualization.

- **Threshold-Based Segmentation**  
  Load a series of DICOM slices, normalize them, and interactively adjust upper and lower thresholds to generate segmentation masks.  
  Visualize the segmented regions as transparent red overlays on the original slices.

- **Interactive Visualization**  
  Use OpenCV trackbars (sliders) to dynamically change slice index and threshold values, making the segmentation process intuitive and exploratory.

## Libraries Used

- [OpenCV](https://opencv.org/) – for image loading, processing, and visualization  
- [NumPy](https://numpy.org/) – for numerical operations and array handling  
- [pydicom](https://pydicom.github.io/) – for reading DICOM medical images  

## Example Workflows

- Load a brain tumor image and its mask, apply the mask, and visualize the segmented tumor region.  
- Overlay the mask in red with transparency to highlight the tumor on the original grayscale image.  
- Load a CT scan series, normalize the slices, and interactively adjust thresholds to segment bone or soft tissue.  
- Scroll through slices with a slider to explore segmentation across the volume.

## Getting Started

To run the examples, install the required libraries:

```bash
pip install opencv-python numpy pydicom