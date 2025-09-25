# Point Cloud Visualization

This project focuses on visualizing **3D point clouds** using Python. It demonstrates how to load, display, and manipulate point cloud data with several popular libraries, including **Open3D**, **PyVista**, and **Trimesh**. The goal is to provide a practical introduction to working with point clouds, from basic loading and saving to applying colors, computing normals, and rendering interactive visualizations.

## Overview

Point clouds are collections of 3D points that represent the shape of objects or environments. They are often captured by 3D scanners, LiDAR sensors, or generated from meshes. This project shows how to:

- Load point clouds from files such as `.ply`
- Visualize them in interactive windows
- Add and modify colors
- Compute and display normals
- Save generated or modified point clouds back to disk

By comparing different libraries, the project highlights the strengths of each approach and gives learners flexibility in choosing the right tool for their workflow.

## Features

- **Visualization with Open3D**  
  Load point clouds, inspect their properties, and render them interactively.  
  Supports adding colors, computing normals, and applying transformations.

- **Visualization with PyVista**  
  Use PyVista’s plotting tools to render point clouds with customizable colors, backgrounds, and point styles.  
  Demonstrates how to add random or uniform colors and visualize normals as arrows.

- **Visualization with Trimesh**  
  Load and display point clouds using Trimesh’s lightweight viewer.  
  Add vertex colors and compute normals from meshes for enhanced visualization.

- **Point Cloud Generation and Saving**  
  Generate simple point clouds (e.g., spheres) and save them in `.ply` format using different libraries.  
  Compare how Open3D, PyVista, and Trimesh handle saving and reloading.

## Libraries Used

- [Open3D](http://www.open3d.org/) – for point cloud processing and visualization  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D plotting and mesh handling  
- [Trimesh](https://trimsh.org/) – for geometry processing and lightweight visualization  
- [NumPy](https://numpy.org/) – for numerical operations  
- [VTK](https://vtk.org/) – backend support for PyVista visualizations  

## Example Workflows

- Load a `.ply` file of a brain scan and visualize it with Open3D, printing its bounding box and checking for colors or normals.  
- Use PyVista to render a lung CT point cloud in red, with points displayed as spheres against a white background.  
- Load a pelvis point cloud with Trimesh, add random colors, and visualize it interactively.  
- Generate a synthetic sphere point cloud, compute normals, and visualize them as arrows to understand surface orientation.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d pyvista trimesh numpy vtk