# Point Cloud and Surface Mesh Resampling

This project demonstrates how to **resample 3D point clouds and surface meshes** using Python. Resampling is an important step in geometry processing, as it allows you to adjust the density, uniformity, and distribution of points or mesh vertices for better visualization, analysis, or downstream tasks such as reconstruction and machine learning.

## Overview

Point clouds and meshes often come from scanners or datasets with uneven density, noise, or excessive detail. Resampling techniques help standardize these datasets by reducing redundancy, filling gaps, or redistributing points more evenly.  

This project shows how to:

- Load point clouds and meshes from common formats  
- Apply voxel downsampling and uniform resampling to point clouds  
- Sample points from surface meshes to create point clouds  
- Reconstruct meshes from resampled points  
- Visualize and compare original vs. resampled data  

## Features

- **Point Cloud Resampling**  
  Perform voxel grid downsampling to reduce point density.  
  Apply uniform random sampling to create lighter datasets.  
  Use Poisson disk sampling for more evenly distributed points.  

- **Mesh Resampling**  
  Sample points directly from surface meshes to generate point clouds.  
  Control the number of sampled points for different levels of detail.  
  Reconstruct meshes from sampled points for comparison.  

- **Visualization**  
  Display original and resampled data side by side.  
  Inspect how resampling affects density, smoothness, and structure.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for point cloud and mesh processing  
- [Trimesh](https://trimsh.org/) – for mesh handling and sampling  
- [PyVista](https://docs.pyvista.org/) – for visualization  
- [NumPy](https://numpy.org/) – for numerical operations  

## Example Workflows

- Load a dense LiDAR point cloud and apply voxel downsampling to reduce its size while preserving structure.  
- Sample 10,000 points from a surface mesh to create a point cloud for analysis.  
- Compare uniform random sampling with Poisson disk sampling to see differences in point distribution.  
- Reconstruct a mesh from resampled points and visualize it alongside the original.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d trimesh pyvista numpy