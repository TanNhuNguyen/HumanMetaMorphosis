# Statistical Skull Shape Mesh Modeling

This project demonstrates how to build **statistical models of skull shapes** using 3D mesh data. It focuses on analyzing collections of skull meshes, extracting features, and applying statistical techniques such as Principal Component Analysis (PCA) to capture shape variability. The project is designed as a resource for medical imaging, anthropology, forensics, and computational geometry.

## Overview

Human skulls vary in shape due to genetics, age, sex, and population differences. By analyzing multiple skull meshes together, statistical modeling can capture these variations and represent them in a compact, interpretable way.  

This project shows how to:

- Load and preprocess skull meshes  
- Align meshes into a common coordinate system  
- Extract geometric features and landmarks  
- Build a statistical shape model using PCA or similar methods  
- Visualize mean skull shapes and modes of variation  

## Features

- **Mesh Preprocessing**  
  Import skull meshes from `.ply`, `.stl`, or `.obj` formats.  
  Normalize, align, and clean meshes for consistency.  

- **Feature Extraction**  
  Compute geometric descriptors such as surface area, volume, centroid, and curvature.  
  Identify anatomical landmarks for correspondence across samples.  

- **Statistical Shape Modeling**  
  Apply PCA to capture dominant modes of skull shape variation.  
  Generate mean skull meshes and visualize shape changes along principal components.  

- **Visualization**  
  Render skull meshes interactively with overlays of statistical features.  
  Compare original skulls with reconstructed or mean shapes.  

- **Export**  
  Save statistical models and reconstructed meshes for further analysis.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for mesh processing and visualization  
- [Trimesh](https://trimsh.org/) – for geometry handling and analysis  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D visualization  
- [NumPy](https://numpy.org/) – for numerical operations  
- [SciPy](https://scipy.org/) – for statistical computations  
- [scikit-learn](https://scikit-learn.org/) – for PCA and machine learning methods  

## Example Workflows

- Load a dataset of skull meshes and align them into a common orientation.  
- Compute the mean skull shape from the dataset.  
- Apply PCA to extract the top modes of variation in skull geometry.  
- Visualize how skull shape changes along the first few principal components.  
- Save reconstructed meshes for further study or visualization.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d trimesh pyvista numpy scipy scikit-learn