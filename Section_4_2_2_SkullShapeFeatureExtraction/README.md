# Skull Shape Feature Extraction

This project demonstrates how to extract **geometric and morphological features** from 3D skull models using Python. It focuses on analyzing skull meshes and point clouds to compute shape descriptors, distances, and other quantitative measures that can be used for medical research, anthropology, or computational geometry applications.

## Overview

Understanding the shape of the skull is important in fields such as medical imaging, forensic science, and evolutionary biology. By extracting features from 3D skull data, researchers can compare anatomical structures, detect abnormalities, or classify specimens.  

This project shows how to:

- Load skull meshes or point clouds from files  
- Compute basic geometric properties such as volume, surface area, and bounding box  
- Extract shape descriptors (e.g., curvature, centroid, principal axes)  
- Measure distances between anatomical landmarks  
- Visualize extracted features alongside the original skull model  

## Features

- **Mesh and Point Cloud Loading**  
  Import skull data from `.ply`, `.stl`, or `.obj` formats.  

- **Geometric Feature Extraction**  
  Compute surface area, volume, centroid, and bounding box dimensions.  

- **Shape Descriptors**  
  Estimate curvature, principal component axes, and orientation.  

- **Landmark Analysis**  
  Identify and measure distances between anatomical landmarks on the skull.  

- **Visualization**  
  Render skull models with overlays of extracted features for intuitive interpretation.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for mesh and point cloud processing  
- [Trimesh](https://trimsh.org/) – for geometry analysis and feature computation  
- [NumPy](https://numpy.org/) – for numerical operations  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D visualization  

## Example Workflows

- Load a 3D skull mesh and compute its surface area and volume.  
- Extract the centroid and principal axes to understand skull orientation.  
- Calculate curvature maps to highlight regions of high geometric variation.  
- Measure distances between predefined anatomical landmarks.  
- Visualize the skull with overlays showing extracted features.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d trimesh pyvista numpy