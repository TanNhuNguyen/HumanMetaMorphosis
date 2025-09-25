# Investigating Shape–Movement Relationship

This project explores the **relationship between 3D shape and movement** using computational geometry and statistical analysis. It focuses on how anatomical or object shapes influence motion patterns, and how motion data can, in turn, be used to infer or analyze shape characteristics. The project is designed as a resource for biomechanics, medical imaging, robotics, and computational anatomy.

## Overview

Understanding how shape relates to movement is crucial in many fields:
- In **biomechanics**, bone and joint shapes determine ranges of motion.  
- In **robotics**, object geometry affects how it can be manipulated.  
- In **medical imaging**, analyzing shape–movement correlations can help detect abnormalities.  

This project shows how to:
- Load 3D meshes or point clouds representing anatomical or mechanical structures  
- Extract geometric features (surface area, curvature, axes, landmarks)  
- Collect or simulate movement data (e.g., joint rotations, trajectories)  
- Apply statistical methods to correlate shape descriptors with movement patterns  
- Visualize both shape and motion data interactively  

## Features

- **Data Loading and Preprocessing**  
  Import 3D meshes or point clouds from `.ply`, `.stl`, or `.obj` formats.  
  Normalize and align shapes for consistent analysis.  

- **Feature Extraction**  
  Compute geometric descriptors such as centroid, bounding box, curvature, and principal axes.  
  Identify anatomical or structural landmarks.  

- **Movement Analysis**  
  Represent motion as trajectories, joint angles, or displacement fields.  
  Link motion data to shape features for correlation studies.  

- **Statistical Correlation**  
  Apply regression, PCA, or correlation analysis to quantify shape–movement relationships.  
  Identify which shape features most strongly influence movement.  

- **Visualization**  
  Render 3D shapes with overlays of movement vectors or trajectories.  
  Plot statistical results such as scatter plots and correlation matrices.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for mesh and point cloud processing  
- [Trimesh](https://trimsh.org/) – for geometry handling and analysis  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D visualization  
- [NumPy](https://numpy.org/) – for numerical operations  
- [SciPy](https://scipy.org/) – for statistical computations  
- [scikit-learn](https://scikit-learn.org/) – for PCA, regression, and correlation analysis  
- [Matplotlib](https://matplotlib.org/) – for plotting results  

## Example Workflows

- Load a pelvis mesh and analyze how its geometry correlates with hip joint movement.  
- Extract curvature features from a skull model and compare them with jaw movement data.  
- Simulate robotic arm trajectories and study how object shape affects grasping motion.  
- Perform PCA on shape descriptors and correlate principal components with movement ranges.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d trimesh pyvista numpy scipy scikit-learn matplotlib