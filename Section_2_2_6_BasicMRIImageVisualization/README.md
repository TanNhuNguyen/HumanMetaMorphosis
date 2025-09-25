# MRI Image Visualization

This project demonstrates how to load, explore, and visualize **MRI (Magnetic Resonance Imaging) scans** in Python. It focuses on handling DICOM series, displaying individual slices, navigating through them interactively, and rendering full 3D voxel volumes. The examples are designed to help learners understand how MRI data is structured and how it can be processed for medical imaging and visualization tasks.

## Overview

MRI scans are typically stored as a series of DICOM files, each representing a 2D slice of the body. By stacking these slices, a 3D volume can be reconstructed and visualized. This project shows how to:

- Load DICOM series from folders
- Inspect metadata such as patient information, orientation, and slice thickness
- Display individual slices using OpenCV
- Navigate through slices dynamically with a slider
- Save slices as standard image formats
- Visualize the full 3D volume as voxel data using PyVista

## Features

- **DICOM Metadata Exploration**  
  Load a series of MRI DICOM files and print key metadata such as patient ID, study date, modality, image orientation, pixel spacing, and slice thickness.

- **Slice Visualization**  
  Display specific slices from an MRI scan in grayscale.  
  Normalize pixel values for better contrast and visualization.

- **Interactive Slice Navigation**  
  Use a trackbar slider to scroll through MRI slices dynamically in an OpenCV window.

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

- Load a head MRI sequence and print metadata such as patient name, study date, and slice thickness.  
- Display the first slice of an MRI scan in grayscale with normalized contrast.  
- Use a slider to scroll through brain MRI slices interactively.  
- Save a selected slice as a PNG or JPEG for external use.  
- Render a 3D voxel visualization of the MRI dataset using PyVista.

## Getting Started

To run the examples, install the required libraries:

```bash
pip install pydicom opencv-python numpy pyvista