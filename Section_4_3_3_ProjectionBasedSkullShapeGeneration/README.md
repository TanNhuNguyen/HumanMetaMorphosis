# Project-Based Skull Shape Generation

This project demonstrates how to **generate and manipulate 3D skull shapes** using Python. It focuses on creating synthetic skull models, applying transformations, and visualizing them for research, education, or computational geometry applications. The project is designed as a hands‑on resource for exploring how skull shapes can be modeled and studied programmatically.

## Overview

3D skull models are widely used in medical imaging, anthropology, forensics, and computer graphics. However, real skull datasets can be difficult to obtain due to privacy and ethical concerns. This project provides a framework for generating synthetic skull shapes and experimenting with them in a controlled environment.  

It shows how to:

- Generate base skull‑like geometries  
- Apply geometric transformations (translation, rotation, scaling)  
- Modify and refine shapes to simulate variations  
- Visualize skulls interactively in 3D  
- Save generated models for reuse in other applications  

## Features

- **Synthetic Skull Generation**  
  Create simplified skull‑like meshes or point clouds as starting models.  

- **Shape Transformation**  
  Apply scaling, rotation, and translation to simulate anatomical variations.  

- **Refinement and Modification**  
  Smooth, subdivide, or deform generated skulls to increase realism.  

- **Visualization**  
  Render skulls in 3D with interactive controls for inspection.  

- **Export**  
  Save generated skull models in formats such as `.ply` or `.stl` for further use.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for mesh and point cloud processing  
- [Trimesh](https://trimsh.org/) – for geometry handling and mesh generation  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D visualization  
- [NumPy](https://numpy.org/) – for numerical operations  

## Example Workflows

- Generate a base skull mesh and visualize it in 3D.  
- Apply scaling along the x‑axis to simulate cranial width variation.  
- Rotate and translate a skull model to align it with a reference orientation.  
- Refine a coarse skull mesh with subdivision for smoother surfaces.  
- Save the generated skull model and load it into external 3D software.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d trimesh pyvista numpy