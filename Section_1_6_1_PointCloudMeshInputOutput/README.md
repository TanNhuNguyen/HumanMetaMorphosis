# Point Cloud and Mesh Input/Output

This project demonstrates how to work with **3D point clouds** and **surface meshes** using Python. It is designed as an educational resource to show how different file formats, libraries, and geometric operations come together in 3D graphics and computational geometry.

## Overview

The code provides examples of reading, generating, and writing 3D data in a variety of formats. It covers both **point clouds** (collections of 3D points, often from scanners or sensors) and **meshes** (surfaces made of connected polygons). By exploring these examples, you can learn how to handle real-world 3D data and how to create synthetic shapes for testing or visualization.

## Features

- **Point Cloud I/O**  
  Read point clouds from common formats such as **PLY**, **PCD**, and **PTS**.  
  Generate synthetic point clouds for shapes like spheres and cubes.  
  Save point clouds back to disk with colors and normals.

- **Mesh I/O**  
  Read meshes from formats including **PLY**, **STL**, **OBJ**, and **OFF**.  
  Generate simple meshes such as spheres, cubes, and cylinders.  
  Convert between mesh representations using different libraries.

- **Geometry Processing**  
  Estimate and orient normals for point clouds.  
  Assign uniform colors to generated shapes.  
  Compute and inspect mesh properties such as vertices, faces, and normals.

## Libraries Used

The project makes use of several powerful Python libraries for 3D geometry:

- **Open3D** – for point cloud and mesh processing  
- **PyVista** – for mesh creation and visualization  
- **Trimesh** – for mesh generation and file export  
- **NumPy** – for numerical operations

## Example Workflows

- Load a scanned head model from a `.ply` or `.pcd` file and inspect its points, colors, and normals.  
- Generate a synthetic sphere point cloud, paint it red, and save it as a `.ply` file.  
- Create a cube surface point cloud, estimate normals, and export it as a `.pcd` file.  
- Build a cylinder mesh with Trimesh and export it as an `.obj`.  
- Read meshes from different formats and compute missing normals for visualization.

## Project Structure

The code is organized into functions that demonstrate specific tasks, such as reading a point cloud from a file, generating synthetic geometry, or writing a mesh to disk. A simple `main()` function is included to run one of the examples.

## Getting Started

To run the examples, you will need:

- Python 3.8 or later  
- The required libraries installed:  
  ```bash
  pip install open3d pyvista trimesh numpy