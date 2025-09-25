# Mesh Visualization

This project demonstrates how to visualize **3D triangle surface meshes** in Python using several popular libraries. It is designed as a learning resource for anyone interested in computer graphics, geometry processing, or scientific visualization. The examples cover different visualization modes, normal computation, color mapping, and texture mapping.

## Overview

Meshes are a fundamental way of representing 3D objects, consisting of vertices, edges, and faces that define a surface. This project shows how to load meshes from files, inspect their properties, and render them in different styles. It also demonstrates how to compute and visualize normals, apply color maps, and add textures to meshes.

## Features

- **Mesh Visualization with Open3D**  
  Load meshes from files and display them in surface, wireframe, and vertex‑only modes.  
  Compute vertex normals if they are missing and visualize them alongside the mesh.

- **Normal Computation**  
  Generate simple meshes (such as spheres) and compute vertex normals.  
  Visualize normals as line sets to better understand surface orientation.

- **Color Mapping**  
  Apply color maps to meshes based on vertex attributes (e.g., z‑coordinate).  
  Use colormaps such as *viridis* to create smooth gradients across the surface.

- **Texture Mapping with PyVista**  
  Generate UV coordinates for a mesh and apply an image texture.  
  Demonstrates how to wrap a 2D image (e.g., a chessboard) onto a 3D surface.

## Libraries Used

- [Open3D](http://www.open3d.org/) – for mesh processing and visualization  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D plotting and texture mapping  
- [Trimesh](https://trimsh.org/) – for geometry handling  
- [NumPy](https://numpy.org/) – for numerical operations  
- [Matplotlib](https://matplotlib.org/) – for colormaps  
- [Pillow](https://python-pillow.org/) – for image handling  

## Example Workflows

- Load a head mesh from a `.ply` file, compute missing normals, and visualize it in surface and wireframe modes.  
- Generate a low‑resolution sphere mesh, compute normals, and display them as red arrows extending from the surface.  
- Apply a rainbow colormap to a sphere mesh based on vertex height (z‑coordinate).  
- Map a chessboard texture onto a sphere using PyVista’s texture mapping tools.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d pyvista trimesh numpy matplotlib pillow