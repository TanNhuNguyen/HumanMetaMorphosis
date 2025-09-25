# Skull Shape Personalization

This project demonstrates how to **personalize 3D skull shapes** using Python. It focuses on adapting generic skull models to match individual anatomical variations by applying transformations, morphing techniques, and feature‑based adjustments. The project is designed as a resource for exploring how skull geometry can be customized for research, medical, or educational applications.

## Overview

Every human skull has unique characteristics. In medical imaging, anthropology, or forensic science, it is often necessary to adapt a base skull model to reflect individual differences. This project shows how to:

- Load generic skull meshes or point clouds  
- Identify and extract key anatomical features  
- Apply scaling, rotation, and deformation to personalize skull geometry  
- Morph skulls based on feature correspondences or statistical models  
- Visualize original vs. personalized skulls interactively  

## Features

- **Base Skull Loading**  
  Import skull models from `.ply`, `.stl`, or `.obj` formats.  

- **Feature Extraction**  
  Identify landmarks and compute geometric descriptors (e.g., centroid, axes, curvature).  

- **Personalization Techniques**  
  - Apply non‑uniform scaling to simulate cranial width/height variations  
  - Morph skulls using feature correspondences or interpolation  
  - Deform meshes with radial basis functions (RBF) or cage‑based methods  

- **Visualization**  
  Render original and personalized skulls side by side for comparison.  
  Overlay landmarks and feature points for intuitive interpretation.  

- **Export**  
  Save personalized skulls for reuse in analysis, visualization, or 3D printing.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for mesh and point cloud processing  
- [Trimesh](https://trimsh.org/) – for geometry handling and mesh operations  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D visualization  
- [NumPy](https://numpy.org/) – for numerical operations  
- [SciPy](https://scipy.org/) – for interpolation and deformation methods  

## Example Workflows

- Load a generic skull mesh and apply scaling to match an individual’s cranial proportions.  
- Extract anatomical landmarks and use them to guide morphing of the skull shape.  
- Apply RBF deformation to adjust localized regions of the skull.  
- Visualize the original and personalized skulls side by side with highlighted landmarks.  
- Save the personalized skull model for further study or 3D printing.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d trimesh pyvista numpy scipy