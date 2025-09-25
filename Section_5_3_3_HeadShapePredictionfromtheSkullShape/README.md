# Head ↔ Skull Shape Prediction

This repository contains two related projects that explore the relationship between **head shape** and **skull shape** using 3D mesh and point cloud data. Both projects focus on predictive modeling, feature extraction, and visualization, but approach the problem from opposite directions:

1. **Head Shape Prediction from Skull Shape**  
2. **Skull Shape Prediction from Head Shape**

Together, they provide a framework for studying correlations between external morphology and internal anatomy, with applications in **medical imaging, anthropology, forensics, and computational geometry**.

---

## 1. Head Shape Prediction from Skull Shape

### Overview
This project demonstrates how to predict **external head morphology** from the underlying **skull geometry**. Since the skull provides the structural foundation for the head, statistical and machine learning methods can be used to estimate head shape from skull features.

### Features
- Load and preprocess paired skull and head meshes  
- Align datasets into a common coordinate system  
- Extract geometric features and anatomical landmarks  
- Train predictive models (e.g., regression, PCA, neural networks)  
- Visualize predicted head meshes alongside ground truth  

### Example Workflow
- Import skull meshes and extract descriptors (surface area, curvature, axes)  
- Train a regression model to map skull features to head features  
- Predict head meshes and compare them with actual head models  
- Visualize prediction errors and correlation metrics  

---

## 2. Skull Shape Prediction from Head Shape

### Overview
This project demonstrates how to predict the **underlying skull geometry** from the **external head shape**. Since head morphology reflects skull structure but is influenced by soft tissue, predictive modeling can help estimate skull shape from external features.

### Features
- Load and preprocess paired head and skull meshes  
- Extract geometric descriptors and landmarks from head models  
- Train predictive models to map head features to skull features  
- Generate predicted skull meshes and compare them with ground truth  
- Visualize overlays of predicted vs. actual skulls  

### Example Workflow
- Import head meshes and compute descriptors (volume, centroid, curvature)  
- Train a statistical or ML model to predict skull shape  
- Visualize predicted skull meshes alongside actual skulls  
- Evaluate accuracy using error metrics (e.g., surface distance, Hausdorff distance)  

---

## Shared Libraries Used

- [Open3D](http://www.open3d.org/) – mesh and point cloud processing  
- [Trimesh](https://trimsh.org/) – geometry handling and analysis  
- [PyVista](https://docs.pyvista.org/) – interactive 3D visualization  
- [NumPy](https://numpy.org/) – numerical operations  
- [SciPy](https://scipy.org/) – statistical computations  
- [scikit-learn](https://scikit-learn.org/) – regression, PCA, ML methods  
- [Matplotlib](https://matplotlib.org/) – plotting and error visualization  

---

## Getting Started

Install the required libraries:

```bash
pip install open3d trimesh pyvista numpy scipy scikit-learn matplotlib