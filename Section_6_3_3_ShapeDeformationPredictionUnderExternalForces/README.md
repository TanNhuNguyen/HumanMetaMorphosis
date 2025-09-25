# Shape Deformation Estimation Under External Force

This project demonstrates how to estimate and visualize **shape deformation** of 3D objects when subjected to **external forces**. It combines geometry processing with basic physical modeling to simulate how meshes or point clouds deform under applied stresses. The project is designed as a resource for biomechanics, engineering, computer graphics, and computational geometry.

## Overview

When external forces act on an object, its shape changes depending on material properties, geometry, and boundary conditions. Estimating these deformations is important in fields such as **biomechanics (bone/tissue modeling), structural engineering, and animation**.  

This project shows how to:

- Load 3D meshes or point clouds representing objects or anatomical structures  
- Define external forces and boundary conditions  
- Estimate deformation using simplified physical or mathematical models  
- Visualize original vs. deformed shapes interactively  
- Save results for further analysis or simulation  

## Features

- **Mesh and Point Cloud Loading**  
  Import 3D objects from `.ply`, `.stl`, or `.obj` formats.  

- **Force Application**  
  Define external forces (e.g., compression, tension, bending) applied to the object.  

- **Deformation Estimation**  
  Use simplified models (linear elasticity, displacement fields, interpolation) to approximate deformation.  
  Apply radial basis functions (RBF) or cage‑based deformation for flexible shape changes.  

- **Visualization**  
  Render original and deformed shapes side by side.  
  Overlay displacement vectors to illustrate how points move under force.  

- **Export**  
  Save deformed meshes for reuse in analysis, visualization, or simulation.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for mesh and point cloud processing  
- [Trimesh](https://trimsh.org/) – for geometry handling  
- [PyVista](https://docs.pyvista.org/) – for interactive 3D visualization  
- [NumPy](https://numpy.org/) – for numerical operations  
- [SciPy](https://scipy.org/) – for interpolation and deformation methods  

## Example Workflows

- Load a pelvis mesh and apply a compressive force to simulate deformation.  
- Use RBF interpolation to estimate displacement of vertices under external stress.  
- Visualize displacement vectors as arrows overlaid on the mesh.  
- Compare original and deformed skull meshes side by side.  
- Save the deformed mesh for further biomechanical analysis.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d trimesh pyvista numpy scipy