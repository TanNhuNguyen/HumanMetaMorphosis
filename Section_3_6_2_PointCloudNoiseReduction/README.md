# Point Cloud Noise Reduction

This project demonstrates how to clean and denoise **3D point clouds** using Python. Point clouds captured from LiDAR, depth cameras, or 3D scanners often contain noise and outliers due to sensor limitations, reflections, or environmental conditions. Noise reduction is a crucial preprocessing step to improve the quality of point clouds for visualization, reconstruction, and analysis.

## Overview

Raw point clouds can be messy: they may include scattered outliers, uneven density, or measurement errors. This project explores different filtering and denoising techniques to make point clouds cleaner and more reliable. By applying these methods, the resulting data becomes easier to visualize and more suitable for downstream tasks such as surface reconstruction, registration, or machine learning.

The project shows how to:

- Load noisy point clouds from files  
- Apply statistical and radius‑based outlier removal  
- Downsample point clouds to reduce redundancy  
- Visualize original vs. denoised point clouds side by side  

## Features

- **Statistical Outlier Removal**  
  Removes points that deviate significantly from their neighbors, reducing random noise.  

- **Radius Outlier Removal**  
  Eliminates points that have too few neighbors within a given radius, useful for removing isolated outliers.  

- **Voxel Downsampling**  
  Reduces the number of points while preserving overall structure, making visualization and processing faster.  

- **Visualization**  
  Compare noisy and cleaned point clouds interactively to see the effects of noise reduction.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for point cloud processing and visualization  
- [NumPy](https://numpy.org/) – for numerical operations  

## Example Workflows

- Load a noisy `.ply` point cloud and apply **statistical outlier removal** to clean it.  
- Use **radius outlier removal** to eliminate isolated points from a LiDAR scan.  
- Downsample a dense point cloud with a voxel grid filter and visualize the result.  
- Compare the original noisy point cloud with the denoised version side by side.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d numpy