# Section 2.2.4 – Basic CT Data Visualization

This module provides a hands-on introduction to loading, organizing, and visualizing computed tomography (CT) datasets in Python. It is part of the book  
"Human Anatomical Shape Metamorphosis: Statistical Shape Modelling, AI-Driven Prediction, and Applications"  
by Tan-Nhu Nguyen and Tien-Tuan Dao (ISTE Group, London, 2025).

Readers will learn to reconstruct volumetric anatomy from DICOM-formatted CT series, extract spatial metadata, and visualize anatomical slices in different planes—preparing them for more advanced processing and segmentation pipelines in future sections.

## Short Contents

- Reading and sorting CT DICOM image series using `pydicom`  
- Extracting voxel dimensions and pixel spacing from CT metadata  
- Constructing 3D NumPy volumes from 2D CT slices  
- Visualizing slices along axial, sagittal, and coronal axes  
- Saving reconstructed data volumes for downstream use