# 3D Surface Mesh Reconstruction from Point Clouds

This project demonstrates how to reconstruct **3D surface meshes** from raw **point cloud data** using Python. It provides practical examples of turning unstructured points into continuous surfaces, making the data easier to visualize, analyze, and use in downstream applications such as 3D modeling, simulation, or medical imaging.

## Overview

Point clouds are often captured by 3D scanners, LiDAR sensors, or generated from imaging data. While they represent the geometry of an object or scene, they lack explicit surface connectivity. Mesh reconstruction bridges this gap by creating a polygonal surface (usually triangles) that approximates the underlying shape.  

This project explores several reconstruction techniques, including:

- **Alpha Shapes** – for generating watertight surfaces from point sets.  
- **Ball Pivoting** – for rolling a virtual ball over points to form a mesh.  
- **Poisson Surface Reconstruction** – for producing smooth, watertight surfaces from noisy data.  

## Features

- Load point clouds from `.ply` or `.pcd` files.  
- Estimate normals required for surface reconstruction.  
- Apply different reconstruction algorithms (Alpha Shapes, Ball Pivoting, Poisson).  
- Visualize reconstructed meshes interactively.  
- Save reconstructed meshes to disk in formats such as `.ply` or `.stl`.  

## Libraries Used

- [Open3D](http://www.open3d.org/) – for point cloud processing, normal estimation, and mesh reconstruction  
- [Trimesh](https://trimsh.org/) – for mesh handling and export  
- [NumPy](https://numpy.org/) – for numerical operations  

## Example Workflows

- Load a scanned object point cloud and estimate normals.  
- Reconstruct a watertight mesh using **Poisson Surface Reconstruction**.  
- Use **Ball Pivoting** to generate a mesh that preserves fine details.  
- Experiment with **Alpha Shapes** to control surface tightness with different radii.  
- Save the reconstructed mesh and visualize it in external 3D software.  

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d trimesh numpy