# Voxelization of 3D Shapes

This project demonstrates how to convert **3D geometric shapes** into **voxel representations** using Python. Voxelization is the process of dividing a 3D object into a grid of small cubes (voxels), similar to how pixels represent 2D images. This technique is widely used in computer graphics, 3D modeling, scientific visualization, and machine learning applications.

## Overview

Voxelization provides a way to approximate continuous 3D surfaces with discrete volumetric data. It is useful for tasks such as collision detection, physics simulations, shape analysis, and preparing data for deep learning models.  

This project shows how to:

- Generate simple 3D shapes such as spheres, cubes, and cylinders  
- Convert these shapes into voxel grids  
- Visualize voxelized shapes interactively  
- Save voxel data for later use in analysis or modeling  

## Features

- **Shape Generation**  
  Create basic 3D primitives (sphere, cube, cylinder) as meshes or point clouds.  

- **Voxelization**  
  Convert meshes into voxel grids with adjustable resolution.  
  Explore how voxel size affects the accuracy and detail of the representation.  

- **Visualization**  
  Render voxelized shapes in 3D using libraries like Open3D or PyVista.  
  Compare original meshes with their voxelized versions.  

- **Data Export**  
  Save voxelized shapes for reuse in other applications or experiments.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for voxelization, mesh handling, and visualization  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D plotting  
- [Trimesh](https://trimsh.org/) – for geometry processing  
- [NumPy](https://numpy.org/) – for numerical operations  

## Example Workflows

- Generate a sphere mesh and voxelize it at different resolutions to see how detail changes.  
- Voxelize a cube and visualize both the original mesh and the voxel grid side by side.  
- Create a cylinder, voxelize it, and export the voxel data for further analysis.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d pyvista trimesh numpy