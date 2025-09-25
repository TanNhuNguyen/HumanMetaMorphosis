#******************************************************************************************************************#
#**********************************************PROJECTION INFORMATION**********************************************#
#******************************************************************************************************************#

#******************************************************************************************************************#
#**********************************************SUPPORTING LIBRARIES************************************************#
#******************************************************************************************************************#
import os;
import numpy as np;
import copy;
import xml.etree.ElementTree as ET;

import open3d as o3d;
import trimesh;
from trimesh.registration import procrustes;
from scipy.interpolate import RBFInterpolator;

#******************************************************************************************************************#
#**********************************************SUPPORTING BUFFERS**************************************************#
#******************************************************************************************************************#
dataFolder = "../../../Data/Section_1_7_GeometricTransformation";

#******************************************************************************************************************#
#**********************************************SUPPORTING FUNCTIONS**************************************************#
#******************************************************************************************************************#
def loadPickedPoints(filePath):
    """
    Load x, y, z coordinates from a .pp (PickedPoints) XML file and return as a NumPy array of shape (N, 3).
    
    Parameters:
        filePath (str): Path to the .pp file.
    
    Returns:
        np.ndarray: Array of shape (N, 3) with x, y, z coordinates.
    """
    # Parse the XML file
    tree = ET.parse(filePath);
    # Get the root element
    root = tree.getroot();
    # Extract points from the XML structure
    pickedPoints = [];
    # Loop through each 'point' element and extract coordinates
    for point in root.findall('point'):
        xCoord = float(point.get('x')); # Extract x coordinate
        yCoord = float(point.get('y')); # Extract y coordinate
        zCoord = float(point.get('z')); # Extract z coordinate
        pickedPoints.append([xCoord, yCoord, zCoord]); # Append the coordinates as a list
    return np.array(pickedPoints); # Return the array of picked points
def createSpheresForMeshFeatures(meshfeatures, radius=0.5, subdivisions=2, color=[0.8, 0.1, 0.1, 1.0]):
    """
    Create spheres for mesh features.
    
    Parameters:
        meshfeatures (np.ndarray): Array of shape (N, 3) with x, y, z coordinates of mesh features.
        radius (float): Radius of the spheres.
        subdivisions (int): Number of subdivisions for the sphere.
        color (list): Color of the spheres in RGBA format.
    
    Returns:
        list: List of trimesh objects representing the spheres.
    """
    spheres = [];
    for feature in meshfeatures:
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions);
        sphere.apply_translation(feature); # Move the sphere to the feature position
        sphere.visual.vertex_colors = color; # Set the color of the sphere
        spheres.append(sphere); # Append the sphere to the list
    return spheres; # Return the list of spheres
def visualizeMeshesWithFeatures(sourceMesh, sourceMeshFeatures, targetMesh, targetMeshFeatures):
    """
    Visualize the source and target meshes along with their features.

    Parameters:
        sourceMesh (trimesh.Trimesh): The source mesh object.
        sourceMeshFeatures (list): List of feature points for the source mesh.
        targetMesh (trimesh.Trimesh): The target mesh object.
        targetMeshFeatures (list): List of feature points for the target mesh.
    """
    # Create blue sphere for source mesh features
    sourceMeshFeaturesSpheres = createSpheresForMeshFeatures(sourceMeshFeatures,
                                                             0.005,
                                                             2,
                                                             [0.1, 0.1, 0.8, 1.0]);

    # Create red sphere for target mesh features
    targetMeshFeaturesSpheres = createSpheresForMeshFeatures(targetMeshFeatures,
                                                             0.005,
                                                             2,
                                                             [0.8, 0.1, 0.1, 1.0]);

    # Create original coordinate frame
    originalCoordinateFrame = trimesh.creation.axis(origin_size=0.010, axis_length=0.1);

    # Visualize the meshes and their features
    scene = trimesh.Scene([sourceMesh] + sourceMeshFeaturesSpheres +
                            [targetMesh] + targetMeshFeaturesSpheres +
                            [originalCoordinateFrame]);
    
    # Show the scene with a specific resolution
    scene.show(resolution=(800, 600));
def estimateRigidTransform(sourceMeshFeatures, targetMeshFeatures):
    """
    Estimate the rigid transformation from source mesh features to target mesh features.
    
    Parameters:
        sourceMeshFeatures (np.ndarray): Array of shape (N, 3) with x, y, z coordinates of source mesh features.
        targetMeshFeatures (np.ndarray): Array of shape (N, 3) with x, y, z coordinates of target mesh features.
    
    Returns:
        np.ndarray: Estimated transformation matrix.
    """
    # Use trimesh's registration module to estimate the transformation
    estimatedTransform, _, _ = procrustes(sourceMeshFeatures, 
                                          targetMeshFeatures, 
                                          reflection=False,
                                          scale=False);
    return estimatedTransform; # Return the estimated transformation matrix
def applyRigidTransformTo3DPoints(points, transformMatrix):
    """
    Apply a rigid transformation to a set of 3D points.
    
    Parameters:
        points (np.ndarray): Array of shape (N, 3) with x, y, z coordinates of the points.
        transformMatrix (np.ndarray): Transformation matrix of shape (4, 4).
    
    Returns:
        np.ndarray: Transformed points of shape (N, 3).
    """
    # Convert points to homogeneous coordinates
    homogeneousPoints = np.hstack((points, np.ones((points.shape[0], 1))));

    # Apply the transformation
    transformedPoints = homogeneousPoints @ transformMatrix.T;

    # Return only the x, y, z coordinates
    return transformedPoints[:, :3];
def deformMeshWithRadialBasisFunctions(sourceMesh, sourceFeatures, targetFeatures):
    """
    Deform the source mesh using radial basis functions (RBF) based on the source and target features.
    
    Parameters:
        sourceMesh (trimesh.Trimesh): The source mesh object.
        sourceFeatures (np.ndarray): Array of shape (N, 3) with x, y, z coordinates of source mesh features.
        targetFeatures (np.ndarray): Array of shape (N, 3) with x, y, z coordinates of target mesh features.
    
    Returns:
        deformedMesh (trimesh.Trimesh): The deformed mesh object.
        deformedMeshFeatures (np.ndarray): Array of shape (N, 3) with deformed mesh features.
    """
    # Calculate the displacements for each feature
    displacements = targetFeatures - sourceFeatures;

    # Create the radial basis function interpolator
    rbfInterpolator = RBFInterpolator(sourceFeatures, displacements, kernel='thin_plate_spline');

    # Apply the RBF to deform the mesh vertices
    deformedVertices = sourceMesh.vertices + rbfInterpolator(sourceMesh.vertices);

    # Create a new mesh with the deformed vertices
    deformedMesh = trimesh.Trimesh(vertices=deformedVertices, faces=sourceMesh.faces, process=False);

    # Deform features as well
    deformedMeshFeatures = sourceFeatures + rbfInterpolator(sourceFeatures);
    
    # Return the deformed mesh and its features
    return deformedMesh, deformedMeshFeatures;
def loadInputDataForNonRigidRegistration(sourceMeshFilePath, sourceMeshFeatureFilePath, 
                                         targetMeshFilePath, targetMeshFeatureFilePath):
    # Load the source mesh
    if not os.path.exists(sourceMeshFilePath):
        print(f"Source mesh file not found at {sourceMeshFilePath}");
        return;
    sourceMesh = trimesh.load(sourceMeshFilePath);

    # Load source mesh features
    if not os.path.exists(sourceMeshFeatureFilePath):
        print(f"Source mesh feature file not found at {sourceMeshFeatureFilePath}");
        return;
    sourceMeshFeatures = loadPickedPoints(sourceMeshFeatureFilePath);

    # Colorize the source mesh as bone color
    sourceMesh.visual.vertex_colors = [0.8, 0.8, 0.6, 0.7];

    # Load target mesh
    if not os.path.exists(targetMeshFilePath):
        print(f"\t Target mesh file not found at {targetMeshFilePath}");
        return;
    targetMesh = trimesh.load(targetMeshFilePath);

    # Load target mesh features
    if not os.path.exists(targetMeshFeatureFilePath):
        print(f"\t Target mesh feature file not found at {targetMeshFeatureFilePath}");
        return;
    targetMeshFeatures = loadPickedPoints(targetMeshFeatureFilePath);

    # Colorize the target mesh as dark bone color
    targetMesh.visual.vertex_colors = [0.6, 0.6, 0.4, 1.0];

    # Return the loaded meshes and features
    return sourceMesh, sourceMeshFeatures, targetMesh, targetMeshFeatures;
def showMeshAndFeaturesInformationForNonRigidRegistration(sourceMesh, sourceMeshFeatures,
                                                            targetMesh, targetMeshFeatures):
    # Show mesh and feature information
    print(f"\t The number of source mesh vertices: {sourceMesh.vertices.shape[0]}");
    print(f"\t The number of source mesh features: {sourceMeshFeatures.shape[0]}");
    print(f"\t The number of target mesh vertices: {targetMesh.vertices.shape[0]}");
    print(f"\t The number of target mesh features: {targetMeshFeatures.shape[0]}");

#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def meshTranslation():
    # Information: This function instructs the readers to conduct mesh translation. This is one of the
    # fundamental operations in geometric transformations, allowing the movement of mesh objects in space.
    # Initialize
    print("Initializing ...");

    # Load an example mesh with open3d
    print("Loading an example mesh ...");
    ## Forming file path
    meshFilePath = os.path.join(dataFolder, "MalePelvis.ply");
    ## Checking and reading mesh
    if not os.path.exists(meshFilePath):
        print(f"Mesh file not found at {meshFilePath}");
        return;
    mesh = o3d.io.read_triangle_mesh(meshFilePath);
    ## Compute mesh normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals();
    ## Adding the blue color to the mesh vertices
    mesh.paint_uniform_color([0.1, 0.1, 0.8]);  # Set the mesh color to blue

    # Generate translation matrix using open3d
    print("Generating translation matrix ...");
    translationVector = np.array([100, 0, 0]);  # Translate by 100 units along the x-axis
    translationMatrix = np.eye(4);  # Create a 4x4 identity matrix
    translationMatrix[:3, 3] = translationVector  # Set the translation vector in the last column

    # Apply translation to the mesh
    print("Applying translation to the mesh ...");
    ## Clone the original mesh to avoid modifying it directly
    translatedMesh = copy.deepcopy(mesh);  # Clone the original mesh
    ## Transform the cloned mesh using the translation matrix
    translatedMesh.transform(translationMatrix);
    ## Set the red color to the translated mesh vertices
    translatedMesh.paint_uniform_color([0.8, 0.1, 0.1]);  # Set the translated mesh color to red
    
    # Visualize the original mesh and the translated mesh using open3d with coordinate system
    print("Visualizing the original and translated meshes ...");
    ## Create a coordinate frame for original mesh
    originalMeshCenter = mesh.get_center();
    originalMeshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=originalMeshCenter);
    ## Create a coordinate frame for translated mesh
    translatedMeshCenter = translatedMesh.get_center();
    translatedMeshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=translatedMeshCenter);
    ## Visualize both meshes with their coordinate frames
    o3d.visualization.draw_geometries([mesh, originalMeshFrame, translatedMesh, translatedMeshFrame],
                                       window_name="Mesh Translation", width=800, height=600);

    # Finished processing
    print("Finished processing.");
def meshRotation():
    # Information: This function instructs the readers to conduct mesh rotation. This is one of the
    # fundamental operations in geometric transformations, allowing the rotation of mesh objects in space.
    # Initialize
    print("Initializing ...");

    # Load an example mesh with open3d
    print("Loading an example mesh ...");
    ## Forming file path
    meshFilePath = os.path.join(dataFolder, "FemaleHeadMesh_TriMesh.ply");
    ## Checking and reading mesh
    if not os.path.exists(meshFilePath):
        print(f"Mesh file not found at {meshFilePath}");
        return;
    mesh = o3d.io.read_triangle_mesh(meshFilePath);
    ## Compute mesh normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals();
    ## Adding the blue color to the mesh vertices
    mesh.paint_uniform_color([0.1, 0.1, 0.8]);  # Set the mesh color to blue

    # Generate rotation matrix using open3d
    print("Generating rotation matrix ...");
    ## Define the rotation angle and axis
    rotationAngle = np.pi / 4;  # Rotate by 45 degrees
    ## Create a rotation matrix using the axis-angle representation
    rotationAxis = np.array([0, 1, 0]);  # Rotate around the y-axis
    ## Get the rotation matrix from the axis-angle representation
    rotationMatrix3x3 = o3d.geometry.get_rotation_matrix_from_axis_angle(rotationAxis * rotationAngle);
    ## Create a 4x4 transformation matrix
    rotationMatrix = np.eye(4);  # Create a 4x4 identity matrix
    rotationMatrix[:3, :3] = rotationMatrix3x3;  # Set the rotation matrix in the top-left 3x3 submatrix
    ## Set the translation part to zero (no translation)
    rotationMatrix[3, :3] = 0;  # No translation in the last row

    # Apply rotation to the mesh
    print("Applying rotation to the mesh ...");
    ## Clone the original mesh to avoid modifying it directly
    rotatedMesh = copy.deepcopy(mesh);  # Clone the original mesh
    ## Transform the cloned mesh using the rotation matrix
    rotatedMesh.transform(rotationMatrix);
    ## Set the red color to the rotated mesh vertices
    rotatedMesh.paint_uniform_color([0.8, 0.1, 0.1]);  # Set the rotated mesh color to red

    # Visualize the original mesh and the rotated mesh using open3d with coordinate system
    print("Visualizing the original and rotated meshes ...");
    ## Create original coordinate frame
    originalCoordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=[0, 0, 0]);
    ## Create a coordinate frame for original mesh
    originalMeshCenter = mesh.get_center();
    originalMeshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=originalMeshCenter);
    ## Create a coordinate frame for rotated mesh
    rotatedMeshCenter = rotatedMesh.get_center();
    rotatedMeshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20, origin=rotatedMeshCenter);
    ## Visualize both meshes with their coordinate frames
    o3d.visualization.draw_geometries([mesh, originalMeshFrame, 
                                       rotatedMesh, rotatedMeshFrame, 
                                       originalCoordinateFrame],
                                       window_name="Mesh Rotation", width=800, height=600);

    # Finished processing
    print("Finished processing.");
def meshScaling():
    # Information: This function instructs the readers to conduct mesh scaling. This is one of the
    # fundamental operations in geometric transformations, allowing the scaling of mesh objects in space.
    # Initialize
    print("Initializing ...");

    # Load an example mesh with open3d
    print("Loading an example mesh ...");
    ## Forming file path
    meshFilePath = os.path.join(dataFolder, "MalePelvis.ply");
    ## Checking and reading mesh
    if not os.path.exists(meshFilePath):
        print(f"Mesh file not found at {meshFilePath}");
        return;
    mesh = o3d.io.read_triangle_mesh(meshFilePath);
    ## Compute mesh normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals();
    ## Adding the blue color to the mesh vertices
    mesh.paint_uniform_color([0.1, 0.1, 0.8]);  # Set the mesh color to blue

    # Generate scaling matrix using open3d
    print("Generating scaling matrix ...");
    ## Define the scaling factors for each axis
    scalingFactors = np.array([2, 1, 1]);  # Scale by 2 along the x-axis, no scaling along y and z
    ## Create a scaling matrix
    scalingMatrix = np.eye(4);  # Create a 4x4 identity matrix
    scalingMatrix[0, 0] = scalingFactors[0];  # Set the scaling factor for x-axis
    scalingMatrix[1, 1] = scalingFactors[1];  # Set the scaling factor for y-axis
    scalingMatrix[2, 2] = scalingFactors[2];  # Set the scaling factor for z-axis
    
    # Apply scaling to the mesh
    print("Applying scaling to the mesh ...");
    ## Clone the original mesh to avoid modifying it directly
    scaledMesh = copy.deepcopy(mesh);  # Clone the original mesh
    ## Transform the cloned mesh using the scaling matrix
    scaledMesh.transform(scalingMatrix);
    ## Set the red color to the scaled mesh vertices
    scaledMesh.paint_uniform_color([0.8, 0.1, 0.1]);  # Set the scaled mesh color to red

    # Visualize the original mesh and the scaled mesh using open3d with coordinate system
    print("Visualizing the original and scaled meshes ...");
    ## Create a coordinate frame for original mesh
    originalMeshCenter = mesh.get_center();
    originalMeshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=originalMeshCenter);
    ## Create a coordinate frame for scaled mesh
    scaledMeshCenter = scaledMesh.get_center();
    scaledMeshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=scaledMeshCenter);
    ## Visualize both meshes with their coordinate frames
    o3d.visualization.draw_geometries([mesh, originalMeshFrame,
                                       scaledMesh, scaledMeshFrame],
                                       window_name="Mesh Scaling", width=800, height=600);

    # Finished processing
    print("Finished processing.");
def meshTransformation():
    # Information: This function instructs the readers to conduct mesh transformation. This is one of the
    # fundamental operations in geometric transformations, allowing the transformation of mesh objects in space.
    # Transformation can include a combination of translation, rotation, and scaling.
    # Initialize
    print("Initializing ...");

    # Load an example mesh with open3d
    print("Loading an example mesh ...");
    ## Forming file path
    meshFilePath = os.path.join(dataFolder, "MalePelvis.ply");
    ## Checking and reading mesh
    if not os.path.exists(meshFilePath):
        print(f"Mesh file not found at {meshFilePath}");
        return;
    mesh = o3d.io.read_triangle_mesh(meshFilePath);
    ## Compute mesh normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals();
    ## Adding the blue color to the mesh vertices
    mesh.paint_uniform_color([0.1, 0.1, 0.8]);  # Set the mesh color to blue

    # Generate transformation matrix using open3d
    print("Generating transformation matrix ...");
    ## Define the translation vector
    translationVector = np.array([100, 0, 0]);  # Translate by 100 units along the x-axis
    ## Define the rotation angle and axis
    rotationAngle = np.pi / 4;  # Rotate by 45 degrees
    rotationAxis = np.array([0, 1, 0]);  # Rotate around the y-axis
    ## Create a 4x4 transformation matrix
    transformationMatrix = np.eye(4);  # Create a 4x4 identity matrix
    ## Set the translation part in the last column
    transformationMatrix[:3, 3] = translationVector;  # Set the translation vector in the last column
    ## Create the rotation matrix using the axis-angle representation
    rotationMatrix3x3 = o3d.geometry.get_rotation_matrix_from_axis_angle(rotationAxis * rotationAngle);
    ## Set the rotation matrix in the top-left 3x3 submatrix
    transformationMatrix[:3, :3] = rotationMatrix3x3;
    
    # Apply transformation to the mesh
    print("Applying transformation to the mesh ...");
    ## Clone the original mesh to avoid modifying it directly
    transformedMesh = copy.deepcopy(mesh);  # Clone the original mesh
    ## Transform the cloned mesh using the transformation matrix
    transformedMesh.transform(transformationMatrix);
    ## Set the red color to the transformed mesh vertices
    transformedMesh.paint_uniform_color([0.8, 0.1, 0.1]);  # Set the transformed mesh color to red
    
    # Visualize the original mesh and the transformed mesh using open3d with coordinate system
    print("Visualizing the original and transformed meshes ...");
    ## Create a coordinate frame for original mesh
    originalCoordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0]);    
    ## Create a coordinate frame for original mesh  
    originalMeshCenter = mesh.get_center();
    originalMeshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100,
                                                                          origin=originalMeshCenter);
    ## Create a coordinate frame for transformed mesh
    transformedMeshCenter = transformedMesh.get_center();
    transformedMeshFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100,
                                                                          origin=transformedMeshCenter);
    ## Visualize both meshes with their coordinate frames
    o3d.visualization.draw_geometries([mesh, transformedMesh, 
                                       originalMeshFrame, transformedMeshFrame,
                                       originalCoordinateFrame],
                                       window_name="Original and Transformed Meshes",
                                       width=640, height=480);

    # Finished processing
    print("Finished processing.");
def meshAffineTransformation():
    # Information: This function instructs the readers to conduct mesh affine transformation. This is one of the
    # fundamental operations in geometric transformations, allowing the combination of linear transformations
    # (like rotation, scaling, and translation) with the ability to change the origin of the coordinate system.
    # This transformation is useful for more complex manipulations of mesh objects.
    # Initialize
    print("Initializing ...");

    # Load an example mesh with open3d
    print("Loading an example mesh ...");
    ## Forming file path
    meshFilePath = os.path.join(dataFolder, "MalePelvis.ply");
    ## Checking and reading mesh
    if not os.path.exists(meshFilePath):
        print(f"Mesh file not found at {meshFilePath}");
        return;
    mesh = o3d.io.read_triangle_mesh(meshFilePath);
    ## Compute mesh normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals();
    ## Adding the blue color to the mesh vertices
    mesh.paint_uniform_color([0.1, 0.1, 0.8]);  # Set the mesh color to blue

    # Generate affine transformation matrix using open3d
    print("Generating affine transformation matrix ...");
    ## Define the translation vector
    translationVector = np.array([100, 0, 0]);  # Translate by
    ## Define the rotation angle and axis
    rotationAngle = np.pi / 2;  # Rotate by 45 degrees
    rotationAxis = np.array([0, 1, 0]);  # Rotate around the y-axis
    ## Create a 4x4 affine transformation matrix
    affineTransformationMatrix = np.eye(4);  # Create a 4x4 identity matrix
    ## Set the translation part in the last column
    affineTransformationMatrix[:3, 3] = translationVector;  # Set the translation vector in the last column
    ## Create the rotation matrix using the axis-angle representation
    rotationMatrix3x3 = o3d.geometry.get_rotation_matrix_from_axis_angle(rotationAxis * rotationAngle);
    ## Set the rotation matrix in the top-left 3x3 submatrix
    affineTransformationMatrix[:3, :3] = rotationMatrix3x3;
    ## Set the scaling factors for each axis
    scalingFactors = np.array([1, 1.5, 2]);  # No scaling in this case
    affineTransformationMatrix[0, 0] *= scalingFactors[0];  # Set the scaling factor for x-axis
    affineTransformationMatrix[1, 1] *= scalingFactors[1];  # Set the scaling factor for y-axis
    affineTransformationMatrix[2, 2] *= scalingFactors[2];  # Set the scaling factor for z-axis

    # Apply affine transformation to the mesh
    print("Applying affine transformation to the mesh ...");
    ## Clone the original mesh to avoid modifying it directly
    transformedMesh = copy.deepcopy(mesh);  # Clone the original mesh
    ## Transform the cloned mesh using the affine transformation matrix
    transformedMesh.transform(affineTransformationMatrix);
    ## Set the red color to the transformed mesh vertices
    transformedMesh.paint_uniform_color([0.8, 0.1, 0.1]);  # Set the transformed mesh color to red

    # Visualize the original mesh and the transformed mesh using open3d with coordinate system
    print("Visualizing the original and transformed meshes ...");
    ## Create an original coordinate frame
    originalCoordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0]);
    ## Add mesh and transformed mesh to the visualization
    o3d.visualization.draw_geometries([mesh, transformedMesh,
                                       originalCoordinateFrame],
                                       window_name="Original and Transformed Meshes",
                                       width=640, height=480);

    # Finished processing
    print("Finished processing.");
def meshMorphing():
    # This function will instruct the cage based deformation of the mesh. First, the mesh will be loaded,
    # Then the cage of the mesh will be created, and finally, the mesh will be morphed using the cage.
    
    # Initialize
    print("Initializing ...");

    # Load the mesh using trimesh
    print("Loading an example mesh ...");
    ## Forming file path
    meshFilePath = os.path.join(dataFolder, "MalePelvis.ply");
    ## Checking and reading mesh
    if not os.path.exists(meshFilePath):
        print(f"Mesh file not found at {meshFilePath}");
        return;
    ## Load the mesh using trimesh
    mesh = trimesh.load(meshFilePath);
    ## Adding the blue color to the mesh vertices
    mesh.visual.vertex_colors = [0.1, 0.1, 0.8, 1.0];

    # Create a cage for the mesh using trimesh
    print("Creating a cage for the mesh ...");
    ## Create a bounding box for the mesh
    boundingBox = mesh.bounding_box;
    ## Create cage by the bounding box
    cage = boundingBox.to_mesh();
    ## Set the color of the cage to green with opacity 0.5
    cage.visual.vertex_colors = [0.1, 0.8, 0.1, 0.5];
    
    # Visualize the mesh with the cage using trimesh
    print("Visualizing the mesh with the cage ...");    
    ## Add the coordinate frame to the scene
    coordinateFrame = trimesh.creation.axis(origin_size=10, axis_length=100);
    ## Create a scene with the mesh and the cage
    scene = trimesh.Scene([mesh, cage, coordinateFrame]);
    ## Show the scene with a specific resolution
    scene.show(resolution=(800, 600));
    
    # Copy the mesh preparing for morphing
    print("Preparing the mesh for morphing ...");
    ## Generate the morphed mesh by cloning the original mesh
    morphedMesh = trimesh.Trimesh(vertices=mesh.vertices.copy(), 
                                  faces=mesh.faces.copy(), 
                                  process=False);
    ## Generate the morphed cage by cloning the original mesh
    morphedCage = trimesh.Trimesh(vertices=boundingBox.vertices.copy(), 
                                  faces=boundingBox.faces.copy(),
                                  process=False);
    ## Set the color of the morphed cage to green with opacity 0.5
    morphedCage.visual.vertex_colors = [0.1, 0.8, 0.1, 0.5];
    
    # Conduct mesh morphing using radial basis functions (RBF) from trimesh
    print("Conducting mesh morphing using radial basis functions (RBF) ...");
    ## Move the cage vertices up by 200 units
    upperCageVertexIndices = [0, 1, 2, 3];
    morphedCage.vertices[upperCageVertexIndices] += np.array([0, 200, 0]);
    ## Apply the RBF morphing to the morphed mesh using the morphed cage
    cageVertexDisplacements = morphedCage.vertices - boundingBox.vertices;
    ## Create the radial basis function (RBF) for morphing
    rbfMorphing = RBFInterpolator(boundingBox.vertices, 
                                  cageVertexDisplacements, 
                                  kernel='thin_plate_spline', 
                                  smoothing=1e-3);
    ## Apply the RBF morphing to the morphed mesh vertices
    morphedMeshVertexDisplacements = rbfMorphing(mesh.vertices);
    ## Update the morphed mesh vertices with the displacements
    morphedMesh.vertices += morphedMeshVertexDisplacements;

    # Visualize the morphed mesh with the morphed cage using trimesh
    print("Visualizing the morphed mesh with the morphed cage ...");
    ## Add the coordinate frame to the scene
    coordinateFrame = trimesh.creation.axis(origin_size=10, axis_length=100);
    ## Create a scene with the morphed mesh and the morphed cage
    scene = trimesh.Scene([morphedMesh, morphedCage, coordinateFrame]);
    ## Show the scene with a specific resolution
    scene.show(resolution=(800, 600));
    
    # Finished processing
    print("Finished processing.");
def meshRigidRegistration():
    # Information: This function instructs the readers to conduct mesh registration. This is one of the
    # fundamental operations in geometric transformations, allowing the alignment of multiple mesh objects
    # into a common coordinate system. Mesh registration can involve techniques such as feature matching,
    # iterative closest point (ICP) algorithms, and other optimization methods to minimize the differences
    # between the meshes. Mesh registration is essential for applications such as 3D reconstruction,
    # object recognition, and scene understanding.
    # Initialize
    print("Initializing ...");

    # Prepare the data for mesh registration
    print("Preparing the data for mesh registration ...");
    ## Load the source mesh
    sourceMeshFilePath = os.path.join(dataFolder, 
                                      "FemaleHeadMesh_TriMesh.ply");
    if not os.path.exists(sourceMeshFilePath):
        print(f"Source mesh file not found at {sourceMeshFilePath}");
        return;
    sourceMesh = trimesh.load(sourceMeshFilePath);
    ## Source mesh features
    sourceMeshFeatureFilePath = os.path.join(dataFolder, 
                                             "FemaleHeadMesh_TriMesh_picked_points.pp");
    sourceMeshFeatures = loadPickedPoints(sourceMeshFeatureFilePath);
    ## Colorize the source mesh as bone color
    sourceMesh.visual.vertex_colors = [0.8, 0.8, 0.6, 1.0];
    ## Show mesh and feature information
    print(f"\t The number of source mesh vertices: {sourceMesh.vertices.shape[0]}");
    print(f"\t The number of source mesh features: {sourceMeshFeatures.shape[0]}");

    # Generate the target mesh and features
    print("Generating the target mesh and features ...");
    ## Generate the target mesh by cloning the source mesh
    targetMesh = copy.deepcopy(sourceMesh);
    ## Generate the target mesh features by cloning the source mesh features
    targetMeshFeatures = copy.deepcopy(sourceMeshFeatures);
    ## Generate a transformation matrix to apply to the target mesh
    transformMatrix = np.array([[0, -1, 0, 20],
                                [1, 0, 0, 20],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]);
    ## Transform target mesh using trimesh
    targetMesh.apply_transform(transformMatrix);
    ## Transform target mesh features using the same transformation matrix
    targetMeshFeatures = np.dot(targetMeshFeatures, 
                                transformMatrix[:3, :3].T) + \
                                    transformMatrix[:3, 3];
    ## Colorize the target mesh as red bone color
    targetMesh.visual.vertex_colors = [0.8, 0.1, 0.1, 1.0];
    ## Show mesh and feature information
    print(f"\t The number of target mesh vertices: {targetMesh.vertices.shape[0]}");
    print(f"\t The number of target mesh features: {targetMeshFeatures.shape[0]}");

    # Visualize the source and target meshes with their features
    print("Visualizing the source and target meshes with their features ...");
    ## Create source mesh features as spheres
    sourceMeshFeatureSpheres = [trimesh.creation.icosphere(radius=0.5,
                                                           subdivisions=2)
                                for feature in sourceMeshFeatures];
    ## Move the source mesh features to their respective positions
    for i, feature in enumerate(sourceMeshFeatureSpheres):
        ## Apply translation to each feature sphere
        feature.apply_translation(sourceMeshFeatures[i]);
        ## Set the color of the feature sphere to red
        feature.visual.vertex_colors = [0.8, 0.1, 0.1, 1.0];
    ## Create target mesh features as spheres
    targetMeshFeatureSpheres = [trimesh.creation.icosphere(radius=0.5, subdivisions=2)
                                for feature in targetMeshFeatures];
    ## Move the target mesh features to their respective positions
    for i, feature in enumerate(targetMeshFeatureSpheres):
        ## Apply translation to each feature sphere
        feature.apply_translation(targetMeshFeatures[i]);
        ## Set the color of the feature sphere to blue
        feature.visual.vertex_colors = [0.1, 0.1, 0.8, 1.0];
    ## Create scene with source mesh, target mesh, and their features
    scene = trimesh.Scene([sourceMesh] + sourceMeshFeatureSpheres +
                          [targetMesh] + targetMeshFeatureSpheres +
                          [trimesh.creation.axis(origin_size=2, axis_length=10)]);
    ## Show the scene with a specific resolution
    scene.show(resolution=(800, 600));

    # Estimate rigid transformation from source features to target features
    print("Estimating rigid transformation from source features to target features ...");
    ## Use trimesh's registration module to estimate the transformation
    estimatedTransform, _, _ = procrustes(sourceMeshFeatures, 
                                          targetMeshFeatures, 
                                          reflection=False,
                                          scale=False);
    ## Compare estimated transform with the ground truth transform
    print(f"\t Estimated transformation matrix:\n{estimatedTransform}");
    print(f"\t Ground truth transformation matrix:\n{transformMatrix}");
    ## Apply the estimated transformation to the source mesh
    transformedMesh = copy.deepcopy(sourceMesh);
    transformedMesh.apply_transform(estimatedTransform);
    ## Colorize the transformed mesh as skin color
    transformedMesh.visual.vertex_colors = [0.8, 0.6, 0.4, 0.7];
    ## Change color of the target mesh to have the alpha channel of 0.7
    targetMesh.visual.vertex_colors = [0.8, 0.1, 0.1, 0.7];

    # Visualize the transformed mesh with target mesh and their features
    print("Visualizing the transformed mesh with target mesh and their features ...");
    ## Create transformed mesh features as spheres
    transformedMeshFeatureSpheres = [trimesh.creation.icosphere(radius=0.5, 
                                                                subdivisions=2)
                                     for feature in targetMeshFeatures];
    ## Move the transformed mesh features to their respective positions
    for i, feature in enumerate(transformedMeshFeatureSpheres):
        ## Apply translation to each feature sphere
        feature.apply_translation(targetMeshFeatures[i]);
        ## Set the color of the feature sphere to red
        feature.visual.vertex_colors = [0.8, 0.1, 0.1, 1.0];
    ## Create scene with transformed mesh, target mesh, and their features
    scene = trimesh.Scene([transformedMesh] + transformedMeshFeatureSpheres + \
                          [targetMesh] + targetMeshFeatureSpheres);
    ## Show the scene with a specific resolution
    scene.show(resolution=(800, 600));

    # Finished processing
    print("Finished processing.");
def meshNonRigidRegistration():
    # Initialize
    print("Initializing ...");

    # Load input data
    print("Loading input data ...");
    sourceMeshFilePath = os.path.join(dataFolder, "TempSkullShapeWithParts.ply");
    sourceMeshFeatureFilePath = os.path.join(dataFolder, "TempSkullShapeWithParts_picked_points.pp");
    targetMeshFilePath = os.path.join(dataFolder, "119219-SkullShape.ply");
    targetMeshFeatureFilePath = os.path.join(dataFolder, "119219-SkullShape_picked_points.pp");
    sourceMesh, sourceMeshFeatures, targetMesh, targetMeshFeatures = \
        loadInputDataForNonRigidRegistration(sourceMeshFilePath,
                                              sourceMeshFeatureFilePath,
                                              targetMeshFilePath,
                                              targetMeshFeatureFilePath);
    
    # Show mesh and feature information
    print("Showing mesh and feature information ...");
    showMeshAndFeaturesInformationForNonRigidRegistration(sourceMesh, sourceMeshFeatures,
                                                           targetMesh, targetMeshFeatures);

    # Visualize meshes and features
    print("Visualize meshes and features ...");
    visualizeMeshesWithFeatures(sourceMesh, sourceMeshFeatures, 
                                targetMesh, targetMeshFeatures);
    
    # Rigid transform the source mesh to the target mesh
    print("Rigid transforming the source mesh to the target mesh ...");
    ## Create deformed mesh and features by cloning the source mesh and features
    deformedMesh = copy.deepcopy(sourceMesh);
    deformedMeshFeatures = copy.deepcopy(sourceMeshFeatures);
    ## Estimate the rigid transformation from source mesh features to target mesh features
    rigidTransform = estimateRigidTransform(sourceMeshFeatures, targetMeshFeatures);
    ## Apply the estimated rigid transformation to the source mesh
    deformedMesh.apply_transform(rigidTransform);
    ## Apply the rigid transformation to the source mesh features
    deformedMeshFeatures = applyRigidTransformTo3DPoints(deformedMeshFeatures, rigidTransform);
    ## Visualize deformed mesh and features
    visualizeMeshesWithFeatures(deformedMesh, deformedMeshFeatures, 
                                targetMesh, targetMeshFeatures);

    # Non-rigid registration using radial basis functions (RBF)
    print("Non-rigid registration using radial basis functions (RBF) ...");
    ## Using radial basis functions to deform the mesh
    deformedMesh, deformedMeshFeatures = deformMeshWithRadialBasisFunctions(deformedMesh, 
                                                                            deformedMeshFeatures, 
                                                                            targetMeshFeatures);
    ## Visualize the deformed mesh with target mesh and their features
    visualizeMeshesWithFeatures(deformedMesh, deformedMeshFeatures, 
                                targetMesh, targetMeshFeatures);

    # Finished processing
    print("Finished processing.");

#******************************************************************************************************************#
#**********************************************MAIN FUNCTION*******************************************************#
#******************************************************************************************************************#
def main():
    os.system("cls");
    meshNonRigidRegistration();
if __name__ == "__main__":
    main()