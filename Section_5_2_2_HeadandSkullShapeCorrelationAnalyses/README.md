# Head–Skull Shape Correlation Analysis

This project demonstrates how to analyze the **correlation between head shape and skull shape** using 3D mesh and point cloud data. It focuses on extracting geometric features, aligning datasets, and applying statistical methods to study how external head morphology relates to underlying skull structure. The project is designed as a resource for medical imaging, anthropology, forensics, and computational geometry.

## Overview

Understanding the relationship between head and skull shapes is important in fields such as craniofacial surgery, forensic reconstruction, and evolutionary biology. By comparing 3D models of the head and skull, researchers can quantify correlations, identify patterns, and build predictive models.  

This project shows how to:

- Load and preprocess head and skull meshes  
- Align datasets into a common coordinate system  
- Extract geometric features and landmarks  
- Compute correlations between head and skull shape descriptors  
- Visualize relationships and statistical results  

## Features

- **Data Loading and Preprocessing**  
  Import head and skull meshes from `.ply`, `.stl`, or `.obj` formats.  
  Normalize, align, and clean meshes for consistency.  

- **Feature Extraction**  
  Compute geometric descriptors such as surface area, volume, centroid, curvature, and bounding box dimensions.  
  Identify anatomical landmarks for correspondence across head and skull models.  

- **Correlation Analysis**  
  Apply statistical methods (e.g., Pearson correlation, regression, PCA) to study relationships between head and skull features.  
  Quantify how external head shape reflects underlying skull morphology.  

- **Visualization**  
  Render head and skull meshes interactively.  
  Overlay landmarks and highlight correlated regions.  
  Plot statistical results such as scatter plots and correlation matrices.  

- **Export**  
  Save correlation results, processed meshes, and visualizations for further analysis.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for mesh and point cloud processing  
- [Trimesh](https://trimsh.org/) – for geometry handling and analysis  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D visualization  
- [NumPy](https://numpy.org/) – for numerical operations  
- [SciPy](https://scipy.org/) – for statistical computations  