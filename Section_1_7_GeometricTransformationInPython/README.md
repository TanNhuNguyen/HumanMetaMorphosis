# Geometrical Transformation

This project demonstrates how to perform and visualize a variety of **geometric transformations** on 3D meshes using Python. It covers fundamental operations such as translation, rotation, scaling, and affine transformations, as well as more advanced techniques like morphing and registration. The goal is to provide a practical introduction to how 3D objects can be manipulated, aligned, and deformed in computational geometry and computer graphics.

## Overview

Geometric transformations are essential tools in 3D modeling, computer vision, and graphics. They allow us to move, rotate, resize, and align objects in space, as well as to deform them for tasks like shape matching or animation. This project explores both **rigid transformations** (which preserve shape and size) and **non‑rigid transformations** (which allow deformation).

## Features

- **Translation, Rotation, and Scaling**  
  Demonstrates how to move, rotate, and resize meshes using transformation matrices.  
  Visualizes original and transformed meshes side by side with coordinate frames.

- **Affine Transformations**  
  Combines translation, rotation, and scaling into a single transformation matrix.  
  Shows how affine transformations can be applied to meshes for more complex manipulations.

- **Mesh Morphing**  
  Uses cage‑based deformation and radial basis functions (RBF) to smoothly morph a mesh into a new shape.  
  Demonstrates how bounding boxes and cages can be used to control deformation.

- **Rigid Registration**  
  Aligns two meshes by estimating the best rigid transformation between corresponding feature points.  
  Visualizes the alignment process and compares estimated transformations with ground truth.

- **Non‑Rigid Registration**  
  Applies RBF interpolation to deform a source mesh so that its features align with those of a target mesh.  
  Useful for shape matching, medical imaging, and other applications where objects differ in form.

## Libraries Used

- [Open3D](http://www.open3d.org/) – for mesh processing and visualization  
- [Trimesh](https://trimsh.org/) – for geometry handling, registration, and morphing  
- [NumPy](https://numpy.org/) – for numerical operations  
- [SciPy](https://scipy.org/) – for radial basis function interpolation  
- [XML](https://docs.python.org/3/library/xml.etree.elementtree.html) – for reading picked feature points  

## Example Workflows

- Translate a pelvis mesh 100 units along the x‑axis and visualize the original and translated meshes together.  
- Rotate a head mesh 45° around the y‑axis and compare it with the original.  
- Scale a pelvis mesh by a factor of 2 along the x‑axis while keeping y and z unchanged.  
- Apply an affine transformation that combines translation, rotation, and non‑uniform scaling.  
- Morph a pelvis mesh by moving the top of its bounding cage upward and applying RBF deformation.  
- Perform rigid registration between two head meshes using manually picked feature points.  
- Carry out non‑rigid registration between skull meshes to align anatomical features.

## Getting Started

To run the examples, install the required libraries:

```bash
pip install open3d trimesh numpy scipy pillow matplotlib