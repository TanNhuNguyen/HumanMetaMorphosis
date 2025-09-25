# DICOM Image Enhancements

This project demonstrates how to apply a variety of **image enhancement techniques** to medical images stored in the **DICOM (Digital Imaging and Communications in Medicine)** format. It provides practical examples of improving visualization and analysis of CT or MRI scans using Python and OpenCV.

## Overview

DICOM is the standard format for storing and transmitting medical imaging data. While raw DICOM images often contain valuable diagnostic information, they may require enhancement to improve clarity, highlight structures, or reduce noise. This project explores several enhancement techniques, including:

- Intensity rescaling
- Contrast and brightness adjustments
- Noise reduction
- Cropping and padding
- Histogram equalization
- Edge enhancement
- Frequency domain filtering

The examples are designed to be interactive, allowing users to adjust parameters dynamically with sliders and immediately see the effects on the images.

## Features

- **Intensity Rescaling**  
  Normalize pixel values to a specified range for better visualization.

- **Contrast and Brightness Adjustment**  
  Use interactive sliders to fine‑tune contrast and brightness in real time.

- **Noise Reduction**  
  Apply different filters such as Gaussian blur, median blur, bilateral filtering, and box filtering to reduce noise.

- **Cropping and Padding**  
  Demonstrate how to crop regions of interest and pad images with borders.

- **Histogram Equalization**  
  Enhance image contrast by redistributing pixel intensities.

- **Edge Enhancement**  
  Highlight structural boundaries using Canny edge detection.

- **Frequency Domain Filtering**  
  Apply low‑pass filtering in the Fourier domain to smooth images and reduce high‑frequency noise.

## Libraries Used

- [pydicom](https://pydicom.github.io/) – for reading DICOM medical images  
- [OpenCV](https://opencv.org/) – for image processing and visualization  
- [NumPy](https://numpy.org/) – for numerical operations  

## Example Workflows

- Load a lung CT DICOM series and rescale intensities for clearer visualization.  
- Adjust contrast and brightness interactively with trackbars to highlight soft tissue.  
- Apply Gaussian blur or median filtering to reduce noise in a noisy slice.  
- Crop a region of interest from a CT slice and pad it with borders for analysis.  
- Perform histogram equalization to enhance bone structures in a CT scan.  
- Detect edges in a DICOM slice to emphasize anatomical boundaries.  
- Use frequency domain filtering to smooth images while preserving overall structure.

## Getting Started

To run the examples, install the required libraries:

```bash
pip install pydicom opencv-python numpy