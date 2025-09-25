# X-Ray Visualization

This project demonstrates how to load, display, and process **X-ray images** using Python. It provides simple but practical examples of working with medical images in common formats such as `.png` and `.jpg`. The focus is on visualization, basic analysis, and introductory image enhancement techniques.

## Overview

X-ray images are widely used in medical imaging to examine bones, lungs, and other internal structures. This project shows how to handle X-ray images with **OpenCV** and **NumPy**, covering tasks such as loading images, displaying them, zooming into regions of interest, applying enhancements, and saving processed results. The examples are designed to be approachable for beginners in image processing while still demonstrating useful techniques.

## Features

- **Load and Display X-Ray Images**  
  Open and visualize grayscale X-ray images using OpenCV’s display tools.

- **Basic Image Information**  
  Print essential details such as image shape, size, and data type.

- **Zoom into Regions of Interest (ROI)**  
  Select and display specific areas of an X-ray image for closer inspection.

- **Basic Image Enhancements**  
  Apply common processing techniques such as histogram equalization, Gaussian blur, and Canny edge detection to improve image quality and highlight structures.

- **Save Processed Images**  
  Export processed X-ray images to disk for later use or analysis.

## Libraries Used

- [OpenCV](https://opencv.org/) – for image loading, processing, and visualization  
- [NumPy](https://numpy.org/) – for numerical operations and array handling  

## Example Workflows

- Load a chest X-ray and display it in a window.  
- Print the resolution, pixel count, and data type of an X-ray image.  
- Define a rectangular region of interest (ROI) and zoom into that area for detailed viewing.  
- Enhance an X-ray image by equalizing its histogram, applying Gaussian blur, and detecting edges.  
- Save a thresholded dental X-ray image into a new folder for processed images.

## Getting Started

To run the examples, install the required libraries:

```bash
pip install opencv-python numpy