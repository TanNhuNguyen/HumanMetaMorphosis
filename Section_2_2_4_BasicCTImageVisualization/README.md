# CT Image Visualization

This project demonstrates how to load, explore, and visualize **CT (Computed Tomography) images** in Python. It focuses on working with DICOM series, extracting slices, and rendering 3D voxel data. The examples are designed to help learners understand how CT data is structured and how it can be processed for medical imaging and visualization tasks.

## Overview

CT scans are stored as a series of DICOM files, each representing a 2D slice of the body. By stacking these slices, a 3D volume can be reconstructed and visualized. This project shows how to:

- Load DICOM series from folders
- Inspect metadata such as patient information, image size, and slice thickness
- Display individual slices using OpenCV
- Navigate through slices dynamically with a slider
- Save slices as standard image formats (PNG or JPEG)
- Visualize the full 3D volume as voxel data using PyVista

## Features

- **DICOM Metadata Exploration**  
  Load a series of DICOM files and print key metadata such as patient ID, modality, pixel spacing, and number of slices.

- **Slice Visualization**  
  Display specific slices from a CT scan using OpenCV.  
  Navigate through slices interactively with a trackbar slider.

- **Image Export**  
  Save selected slices as `.png` or `.jpg` images for easier sharing or further processing.

- **3D Voxel Visualization**  
  Convert the stack of DICOM slices into a 3D volume and render it with PyVista.  
  Visualize anatomical structures in grayscale or with custom colormaps.

## Libraries Used

- [pydicom](https://pydicom.github.io/) – for reading DICOM medical images  
- [OpenCV](https://opencv.org/) – for displaying and saving images  
- [NumPy](https://numpy.org/) – for numerical operations  
- [PyVista](https://docs.pyvista.org/) – for 3D voxel visualization  

## Example Workflows

- Load a head CT scan and print metadata such as patient name, study date, and slice thickness.  
- Display the 10th slice of a liver CT scan in grayscale.  
- Use a slider to scroll through lung CT slices interactively.  
- Save a selected slice as both PNG and JPEG for external use.  
- Render a 3D voxel visualization of a spinopelvic CT dataset using PyVista.

## Getting Started

To run the examples, install the required libraries:

```bash
pip install pydicom opencv-python numpy pyvista