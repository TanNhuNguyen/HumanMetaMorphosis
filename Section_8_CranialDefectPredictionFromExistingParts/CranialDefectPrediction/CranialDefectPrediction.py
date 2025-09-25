#***********************************************************************************************************************************************#
#************************************************************SUPPORTING LIBRARIES***************************************************************#
#***********************************************************************************************************************************************#
import os
import sys;
import numpy as np;
import re;
import copy;
from pycpd import AffineRegistration;
import open3d as o3d;
import trimesh;
import pymeshlab;
from scipy.interpolate import RBFInterpolator;
from scipy.spatial import KDTree;
import xml.etree.ElementTree as ET;
from sklearn.preprocessing import StandardScaler;
from sklearn.decomposition import PCA;
from sklearn.multioutput import MultiOutputRegressor;
from sklearn.linear_model import Ridge, RidgeCV;
from sklearn.linear_model import LinearRegression;
import matplotlib.pyplot as plt;

#***********************************************************************************************************************************************#
#*************************************************************VISUALIZER CLASS******************************************************************#
#***********************************************************************************************************************************************#

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING BUFFERS*****************************************************************#
#***********************************************************************************************************************************************#
mainFolder = "../../../Data/Section_8_CranialDefectPredictionFromExistingParts";
templateSkullGeometryFolder = mainFolder + "/TemplateSkullGeometries";
skullMeshAndFeatureFolder = mainFolder + "/SkullMeshAndFeatures";
postProcessedMeshAndFeatureFolder = mainFolder + "/PostSkullMeshAndFeatures";
normalizedSkullMeshAndFeatureFolder = mainFolder + "/NormSkullMeshAndFeatures";
syntheticSkullDefectFolder = mainFolder + "/SyntheticSkullDefects";
realSkullDefectFolder = mainFolder + "/RealSkullDefects";
crossValidationFolder = mainFolder + "/CrossValidation";

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
#*************************** GENERAL SUPPORTING FUNCTIONS ***************************#
def pause():
    programPause = input("Press the <ENTER> key to continue...");
    exit();
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)));
    filledLength = int(length * iteration // total);
    bar = fill * filledLength + '-' * (length - filledLength);
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd);
    if iteration == total:
        print();
def listAllFilesInFolder(folderPath, fileExtension):
    fileList = [];
    for file in os.listdir(folderPath):
        if file.endswith(fileExtension):
            fileList.append(file);
    return fileList;
def estimateRigidSVDTransform(inSourcePoints, inTargetPoints):
    # Checking input points
    if len(inSourcePoints) != len(inTargetPoints) or len(inSourcePoints) < 3:
        print("estimateRigidSVDTransform:: ERROR: Number of source and target points must be the same and at least 3!")
        return None

    # Convert to numpy arrays
    sourcePoints = np.asarray(inSourcePoints, dtype=np.float64)
    targetPoints = np.asarray(inTargetPoints, dtype=np.float64)

    # Compute centroids
    centroidSrc = np.mean(sourcePoints, axis=0)
    centroidTgt = np.mean(targetPoints, axis=0)

    # Center the points
    srcCentered = sourcePoints - centroidSrc
    tgtCentered = targetPoints - centroidTgt

    # Compute covariance matrix
    H = srcCentered.T @ tgtCentered

    # SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = centroidTgt - R @ centroidSrc

    # Build transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    # Return the transformation matrix
    return T
def transform3DPoints(points, transformMatrix):
    """
    Transform 3D points using a 4x4 transformation matrix.
    Args:
        points: Nx3 numpy array
        transformMatrix: 4x4 numpy array
    Returns:
        transformedPoints: Nx3 numpy array
    """
    points = np.asarray(points, dtype=np.float64);
    N = points.shape[0];
    # Convert to homogeneous coordinates
    points_h = np.hstack([points, np.ones((N, 1))]);
    # Apply transformation
    transformed_h = (transformMatrix @ points_h.T).T;
    # Convert back to 3D
    transformedPoints = transformed_h[:, :3];
    return transformedPoints;
def estimateAffineCPDTransform(sourcePoints, targetPoints):
    """
    Estimate affine transform using CPD algorithm.
    Args:
        sourcePoints: Nx3 numpy array or list
        targetPoints: Nx3 numpy array or list
    Returns:
        affineMatrix: 4x4 numpy array (affine transform matrix)
    """
    sourcePoints = np.asarray(sourcePoints, dtype=np.float64)
    targetPoints = np.asarray(targetPoints, dtype=np.float64)

    reg = AffineRegistration(X=targetPoints, Y=sourcePoints)
    TY, (B, t) = reg.register()  # TY: transformed source, B: 3x3, t: 3x1

    # Build affine matrix (4x4)
    affineMatrix = np.eye(4)
    affineMatrix[:3, :3] = B
    affineMatrix[:3, 3] = t.reshape(-1)

    return affineMatrix
def cloneO3DMesh(inMesh):
    # Clone the mesh using copy
    return copy.deepcopy(inMesh);
def computeBarycentricCoordinatesForTrimesh(inTriMesh, inPoints):
    """
    Compute the barycentric coordinates of points with respect to the triangles in a 3D mesh.

    Parameters:
    mesh (trimesh.Trimesh): The input 3D mesh.
    points (np.ndarray): An array of shape (n, 3) representing the 3D points.

    Returns:
    list of np.ndarray: A list of barycentric coordinates for each point.
    """
    barycentricCoords = [];

    # Find the nearest points on the surface and the corresponding face indices
    nearestPoints, nearestDistances, nearestFaceIndices = inTriMesh.nearest.on_surface(inPoints);

    for point, face_index in zip(inPoints, nearestFaceIndices):
        face_vertices = inTriMesh.vertices[inTriMesh.faces[face_index]];

        # Compute barycentric coordinates for the point with respect to the closest face
        A, B, C = face_vertices;
        AB = B - A;
        AC = C - A;
        AP = point - A;

        areaABC = np.linalg.norm(np.cross(AB, AC));
        areaPBC = np.linalg.norm(np.cross(B - point, C - point));
        areaPCA = np.linalg.norm(np.cross(C - point, A - point));

        u = areaPBC / areaABC;
        v = areaPCA / areaABC;
        w = 1 - u - v;

        barycentricCoords.append(np.array([u, v, w]));

    return np.array(barycentricCoords), np.array(nearestFaceIndices);
def computeBarycentricCoordinatesForO3DMesh(inO3DMesh, inPoints):
    """
    Compute the barycentric coordinates of points with respect to the triangles in an Open3D mesh.

    Parameters:
    inO3DMesh (o3d.geometry.TriangleMesh): The input Open3D mesh.
    inPoints (np.ndarray): An array of shape (n, 3) representing the 3D points.

    Returns:
    list of np.ndarray: A list of barycentric coordinates for each point.
    """
    # Convert Open3D mesh to trimesh
    triMesh = trimesh.Trimesh(vertices=np.asarray(inO3DMesh.vertices), faces=np.asarray(inO3DMesh.triangles));

    # Use the trimesh function to compute barycentric coordinates
    return computeBarycentricCoordinatesForTrimesh(triMesh, inPoints);
def radialBasisFunctionBasedBlendShapeDeformation(inO3DSourceMesh, inSourceFeatures, inTargetFeatures, inRBFType='thin_plate_spline', inSmooth=1e-3):
    # Checking inputs
    if inO3DSourceMesh is None:
        print("radialBasisFunctionBasedBlendShapeDeformation:: ERROR: Source mesh is None!");
        return None;
    if len(inSourceFeatures) != len(inTargetFeatures) or len(inSourceFeatures) < 3:
        print("radialBasisFunctionBasedBlendShapeDeformation:: ERROR: Number of source and target features must be the same and at least 3!");
        return None;
    if inRBFType not in ['inverse_multiquadric', 'cubic', 'quintic', 'inverse_quadratic', 'gaussian', 'multiquadric', 'thin_plate_spline', 'linear']:
        print("radialBasisFunctionBasedBlendShapeDeformation:: ERROR: Unsupported RBF type!");
        return None;
    if inSmooth <= 0:
        print("radialBasisFunctionBasedBlendShapeDeformation:: ERROR: RBF smoothing must be positive!");
        return None;

    # Use the radial basis function
    featureDisplacements = inTargetFeatures - inSourceFeatures;
    rbf = RBFInterpolator(inSourceFeatures, featureDisplacements, kernel=inRBFType, smoothing=1e-3);
    vertexDisplacements = rbf(inO3DSourceMesh.vertices);
    deformedMesh = copy.deepcopy(inO3DSourceMesh);
    deformedMesh.vertices = o3d.utility.Vector3dVector(np.asarray(inO3DSourceMesh.vertices) + vertexDisplacements);

    # Compute the vertex normals
    deformedMesh.compute_vertex_normals();

    # Return the deformed open3d mesh
    return deformedMesh;
def reconstruct3DPointsFromBarycentricCoordinatesForTriMesh(inTriMesh, inBaryCoords, inFaceIndices):
    """
    Reconstruct 3D points from barycentric coordinates and face indices.

    Parameters:
    mesh (trimesh.Trimesh): The input 3D mesh.
    baryCoords (np.ndarray): An array of shape (n, 3) representing the barycentric coordinates.
    faceIndices (np.ndarray): An array of shape (n,) representing the face indices.

    Returns:
    np.ndarray: An array of shape (n, 3) representing the reconstructed 3D points.
    """
    reconstructedPoints = [];

    for baryCoord, face_index in zip(inBaryCoords, inFaceIndices):
        face_vertices = inTriMesh.vertices[inTriMesh.faces[face_index]];
        A, B, C = face_vertices;

        # Reconstruct the point using barycentric coordinates
        point = baryCoord[0] * A + baryCoord[1] * B + baryCoord[2] * C;
        reconstructedPoints.append(point);

    return np.array(reconstructedPoints);
def reconstruct3DPointsFromBarycentricCoordinatesForO3DMesh(inO3DMesh, inBaryCoords, inFaceIndices):
    """
    Reconstruct 3D points from barycentric coordinates and face indices for an Open3D mesh.

    Parameters:
    inO3DMesh (o3d.geometry.TriangleMesh): The input Open3D mesh.
    inBaryCoords (np.ndarray): An array of shape (n, 3) representing the barycentric coordinates.
    inFaceIndices (np.ndarray): An array of shape (n,) representing the face indices.

    Returns:
    np.ndarray: An array of shape (n, 3) representing the reconstructed 3D points.
    """
    # Convert Open3D mesh to trimesh
    triMesh = trimesh.Trimesh(vertices=np.asarray(inO3DMesh.vertices), faces=np.asarray(inO3DMesh.triangles));

    # Use the trimesh function to reconstruct 3D points
    return reconstruct3DPointsFromBarycentricCoordinatesForTriMesh(triMesh, inBaryCoords, inFaceIndices);
def formO3DMesh(inVertices, inTriangles):
    # Create an Open3D TriangleMesh object
    mesh = o3d.geometry.TriangleMesh();

    # Set vertices and triangles
    mesh.vertices = o3d.utility.Vector3dVector(inVertices);
    mesh.triangles = o3d.utility.Vector3iVector(inTriangles);

    # Compute vertex normals for better visualization
    mesh.compute_vertex_normals();

    # Return the formed mesh
    return mesh;
def isotropicRemeshO3DMeshWithResampling(inO3DMesh, inNumOfTargetVertices):
    # Poisson disk sampling to get target number of vertices
    sampledPointCloud = inO3DMesh.sample_points_poisson_disk(number_of_points=inNumOfTargetVertices);

    # Surface reconstruction using point pivot algorithm
    distances = sampledPointCloud.compute_nearest_neighbor_distance();
    avgDist = np.mean(distances);
    radius = 3 * avgDist;
    bpaMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(sampledPointCloud, o3d.utility.DoubleVector([radius, radius * 2]));

    # Fixing the mesh
    bpaMesh.remove_duplicated_vertices();
    bpaMesh.remove_degenerate_triangles();
    bpaMesh.remove_duplicated_triangles();
    bpaMesh.remove_non_manifold_edges();
    bpaMesh.compute_vertex_normals();

    # Convert to pymeshlab to close holes more effectively
    pymeshlabMesh = pymeshlab.MeshSet();
    tempMesh = pymeshlab.Mesh(np.asarray(bpaMesh.vertices), np.asarray(bpaMesh.triangles));
    pymeshlabMesh.add_mesh(tempMesh, "tempMesh");

    # Fill holes
    pymeshlabMesh.meshing_close_holes(maxholesize=1000);
    tempMesh = pymeshlabMesh.current_mesh();

    # Convert back to Open3D mesh
    bpaMesh = formO3DMesh(tempMesh.vertex_matrix(), tempMesh.face_matrix());

    # Return the remeshed mesh
    return bpaMesh;
def fromO3DMeshToTriMesh(inO3DMesh):
    # Convert Open3D mesh to trimesh
    return trimesh.Trimesh(vertices=np.asarray(inO3DMesh.vertices), faces=np.asarray(inO3DMesh.triangles), process=False);
def fromTriMeshToO3DMesh(inTriMesh):
    # Convert back to open3d mesh
    o3DMesh = o3d.geometry.TriangleMesh();
    o3DMesh.vertices = o3d.utility.Vector3dVector(inTriMesh.vertices);
    o3DMesh.triangles = o3d.utility.Vector3iVector(inTriMesh.faces);
    o3DMesh.compute_vertex_normals();
    return o3DMesh;
def estimateNearestIndicesKDTreeBased(inSourcePoints, inTargetPoints, inThreshold=1e-6):
    # Prepare buffers
    sourcePoints = np.array(inSourcePoints);
    targetPoints = np.array(inTargetPoints);

    # Create a KD-tree for the body vertices
    targetPointTree = KDTree(targetPoints);
    
    # Find the distances from each head vertex to the nearest body vertex
    distances, indices = targetPointTree.query(sourcePoints);
    
    # Return buffer
    return indices;
def formTrimesh(inVertices, inFaces):
    # Create a trimesh object
    return trimesh.Trimesh(vertices=inVertices, faces=inFaces, process=False);
def nonRigidTransformICPAmberg(inO3DSourceMesh, inSourceFeatures, inO3DTargetMesh, inTargetFeatures):
    # Checking inputs
    if inO3DSourceMesh is None or inO3DTargetMesh is None:
        print("nonRigidTransformICPAmberg:: ERROR: Source or target mesh is None!");
        return None;
    if len(inSourceFeatures) != len(inTargetFeatures) or len(inSourceFeatures) < 3:
        print("nonRigidTransformICPAmberg:: ERROR: Number of source and target features must be the same and at least 3!");
        return None;

    # Original source mesh
    originalSourceMesh = fromO3DMeshToTriMesh(inO3DSourceMesh);

    # Convert open3d meshes to trimesh
    sourceMesh = trimesh.Trimesh(vertices=np.asarray(inO3DSourceMesh.vertices), faces=np.asarray(inO3DSourceMesh.triangles));
    targetMesh = trimesh.Trimesh(vertices=np.asarray(inO3DTargetMesh.vertices), faces=np.asarray(inO3DTargetMesh.triangles));
    sourceFeatures = np.asarray(inSourceFeatures, dtype=np.float64);
    targetFeatures = np.asarray(inTargetFeatures, dtype=np.float64);

    # Compute original source indices
    originalSourceIndices = estimateNearestIndicesKDTreeBased(sourceMesh.vertices, originalSourceMesh.vertices);
    modifiedSourceMeshVertices = originalSourceMesh.vertices[originalSourceIndices];
    modifiedSourceMeshFaces = sourceMesh.faces;
    modifiedSourceMesh = formTrimesh(modifiedSourceMeshVertices, modifiedSourceMeshFaces);    

    # Compute barycentric coordinates
    sourceBaryCoords, sourceFaceIndices = computeBarycentricCoordinatesForTrimesh(sourceMesh, sourceFeatures);

    # Deform using the non-rigid ICP amberg algorithm using the built-in function from trimesh
    deformedVertices = trimesh.registration.nricp_amberg(
        source_mesh=sourceMesh,
        target_geometry=targetMesh,
        source_landmarks=(sourceFaceIndices, sourceBaryCoords),
        target_positions=targetFeatures
    );
    deformedMesh = trimesh.Trimesh(vertices=deformedVertices, faces=sourceMesh.faces, process=False);

    # Convert back to open3d mesh
    deformedO3DMesh = fromTriMeshToO3DMesh(deformedMesh);
    modifiedO3DSourceMesh = fromTriMeshToO3DMesh(modifiedSourceMesh);

    # Return the deformed open3d mesh
    return deformedO3DMesh, modifiedO3DSourceMesh;
def estimateNearestPointsKDTreeBased(inSourcePoints, inTargetPoints, inThreshold=1e-6):
    # Prepare buffers
    sourcePoints = np.array(inSourcePoints);
    targetPoints = np.array(inTargetPoints);

    # Create a KD-tree for the body vertices
    targetPointTree = KDTree(targetPoints);
    
    # Find the distances from each head vertex to the nearest body vertex
    distances, indices = targetPointTree.query(sourcePoints);
    
    # Prepare output buffer
    nearestPoints = targetPoints[indices];
    
    # Return buffer
    return nearestPoints;
def poissonDiskSamplingOnO3DMesh(inO3DMesh, inNumberOfPoints):
    # Checking inputs
    if inO3DMesh is None:
        print("poissonDiskSamplingOnO3DMesh:: ERROR: Input mesh is None!");
        return None;
    if inNumberOfPoints <= 0:
        print("poissonDiskSamplingOnO3DMesh:: ERROR: Number of points must be positive!");
        return None;

    # Sample points using Poisson disk sampling
    sampledPointCloud = inO3DMesh.sample_points_poisson_disk(number_of_points=inNumberOfPoints);

    # Return the sampled points as a numpy array
    return np.asarray(sampledPointCloud.points, dtype=np.float64);
def readXYZPointsFromPP(inFilePath):
    tree = ET.parse(inFilePath)
    root = tree.getroot()
    points = []
    for point in root.findall('.//point'):
        x = float(point.get('x'))
        y = float(point.get('y'))
        z = float(point.get('z'))
        points.append([x, y, z])
    return np.array(points)
def estimate3DConvexHull(inPointCloud):
    cloudMesh = trimesh.Trimesh(vertices=inPointCloud, faces=[]);
    return cloudMesh.convex_hull;
def trimeshToMeshSet(meshInTrimesh):
    # Getting the vertices and faces
    vertices = np.asarray(meshInTrimesh.vertices);
    faces = np.asarray(meshInTrimesh.faces);

    # Create a pymeshlab Mesh
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces);

    # Create a MeshSet and add the mesh
    ms = pymeshlab.MeshSet();
    ms.add_mesh(mesh);
    
    return ms;
def meshSetToTrimesh(meshInPyMeshLab):
   mesh = meshInPyMeshLab.current_mesh();
   vertices = np.asarray(mesh.vertex_matrix());
   faces = np.asarray(mesh.face_matrix());    
   outTrimesh = trimesh.Trimesh(vertices=vertices, faces=faces);    
   return outTrimesh;
def isotropicRemeshToTargetEdgeLength(mesh, targetLen = 0.01):
    bbox = mesh.bounding_box;
    bbox_diagonal = np.linalg.norm(bbox.extents);
    targetLengthPercentage = (targetLen / bbox_diagonal) * 100;
    ms = trimeshToMeshSet(mesh);
    ms.meshing_isotropic_explicit_remeshing(targetlen = pymeshlab.Percentage(targetLengthPercentage));
    outMesh = meshSetToTrimesh(ms);
    return outMesh;
def formTriMesh(inVertices, inFacets, inProcess = False):
    # Checking the vertices and faces
    if (len(inVertices.flatten()) == 0):
        print("formTriMesh:: The inVertices are empty."); return None;
    if (len(inFacets.flatten()) == 0):
        print("formTriMesh:: The inFaces are empty."); return None;
    
    # Forming the mesh
    mesh = trimesh.Trimesh(inVertices, inFacets, process=inProcess);
    return mesh;
def poissonDiskSampleMesh(input_mesh, sample_num=100000):
    # Create a MeshSet
    ms = pymeshlab.MeshSet()

    # Load the mesh from the trimesh object
    ms.add_mesh(pymeshlab.Mesh(input_mesh.vertices, input_mesh.faces))

    # Perform Poisson disk sampling
    ms.generate_sampling_poisson_disk(samplenum=sample_num)

    # Retrieve the sampled mesh
    sampled_mesh = ms.current_mesh()

    # Convert the sampled mesh vertices to a Nx3 numpy array
    sampled_vertices = sampled_mesh.vertex_matrix()

    return sampled_vertices
def surfaceReconstructionBallPivoting(points):
    # Create a MeshSet
    ms = pymeshlab.MeshSet()

    # Convert points to a PyMeshLab mesh
    ms.add_mesh(pymeshlab.Mesh(points))

    # Perform surface reconstruction using Ball Pivoting algorithm
    ms.generate_surface_reconstruction_ball_pivoting()

    # Retrieve the reconstructed mesh
    reconstructedMesh = ms.current_mesh()

    # Convert the reconstructed mesh back to a trimesh object
    reconstructedVertices = reconstructedMesh.vertex_matrix()
    reconstructedFaces = reconstructedMesh.face_matrix()
    reconstructedTrimesh = trimesh.Trimesh(vertices=reconstructedVertices, faces=reconstructedFaces)

    return reconstructedTrimesh
def computeCenterToVertexNormals(mesh):
    """
    Computes direction vectors from mesh center to each vertex.
    
    Parameters:
    - mesh (trimesh.Trimesh): The input mesh.

    Returns:
    - normals (np.ndarray): Array of shape (n_vertices, 3) with unit vectors.
    """
    center = mesh.vertices.mean(axis=0)  # mesh center
    directions = mesh.vertices - center  # direction vectors to each vertex
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normals = directions / norms  # normalize to unit vectors
    return normals
def closeHolesWithMeshLab(inMesh, inMaxHoleSize = 10):
    meshSet = trimeshToMeshSet(inMesh);
    meshSet.meshing_close_holes(maxholesize = inMaxHoleSize, newfaceselected = False, selfintersection = False)
    outMesh = meshSetToTrimesh(meshSet);
    return outMesh;
def saveTriMeshToPLY(inTriMesh, inFilePath):
    inTriMesh.export(inFilePath, file_type='ply');
    return;
def save3DPointsToPLY(inPoints, inFilePath):
    # Convert points to Open3D point cloud
    pointCloud = o3d.geometry.PointCloud();
    pointCloud.points = o3d.utility.Vector3dVector(inPoints);
    
    # Save to PLY file
    o3d.io.write_point_cloud(inFilePath, pointCloud);
    return;
def readIndicesFromTXT(inFilePath):
    with open(inFilePath, 'r') as file:
        indices = [int(line.strip()) for line in file if line.strip().isdigit()]
    return indices;
def readFacesFromTXT(inFilePath):
    faces = [];
    with open(inFilePath, 'r') as file:
        for line in file:
            parts = line.strip().split();
            if len(parts) >= 3:
                try:
                    face = [int(parts[0]), int(parts[1]), int(parts[2])];
                    faces.append(face);
                except ValueError:
                    continue;
    return np.array(faces);
def estimateRigidTransformSVD(inSourcePoints, inTargetPoints):
    # Checking input points
    if len(inSourcePoints) != len(inTargetPoints) or len(inSourcePoints) < 3:
        print("estimateRigidTransformSVD:: ERROR: Number of source and target points must be the same and at least 3!");
        return None;

    # Convert to numpy arrays
    sourcePoints = np.asarray(inSourcePoints, dtype=np.float64);
    targetPoints = np.asarray(inTargetPoints, dtype=np.float64);

    # Compute centroids
    centroidSrc = np.mean(sourcePoints, axis=0);
    centroidTgt = np.mean(targetPoints, axis=0);

    # Center the points
    srcCentered = sourcePoints - centroidSrc;
    tgtCentered = targetPoints - centroidTgt;

    # Compute covariance matrix
    H = srcCentered.T @ tgtCentered;

    # SVD
    U, S, Vt = np.linalg.svd(H);
    R = Vt.T @ U.T;

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1;
        R = Vt.T @ U.T;

    # Compute translation
    t = centroidTgt - R @ centroidSrc;

    # Build transformation matrix
    T = np.eye(4);
    T[:3, :3] = R;
    T[:3, 3] = t;

    # Return the transformation matrix
    return T;
def transform3DPoints(inPoints, inTransformMatrix):
    """
    Transform 3D points using a 4x4 transformation matrix.
    Args:
        points: Nx3 numpy array
        transformMatrix: 4x4 numpy array
    Returns:
        transformedPoints: Nx3 numpy array
    """
    points = np.asarray(inPoints, dtype=np.float64);
    N = points.shape[0];
    # Convert to homogeneous coordinates
    points_h = np.hstack([points, np.ones((N, 1))]);
    # Apply transformation
    transformed_h = (inTransformMatrix @ points_h.T).T;
    # Convert back to 3D
    transformedPoints = transformed_h[:, :3];
    return transformedPoints;
def fixO3DMesh(inO3DMesh):
    # Remove duplicated vertices
    inO3DMesh.remove_duplicated_vertices();
    # Remove degenerate triangles
    inO3DMesh.remove_degenerate_triangles();
    # Remove duplicated triangles
    inO3DMesh.remove_duplicated_triangles();
    # Remove non-manifold edges
    inO3DMesh.remove_non_manifold_edges();
    # Compute vertex normals
    inO3DMesh.compute_vertex_normals();
    return inO3DMesh;
def saveNumPyArrayToCSVFile(inArray, inFilePath):
    np.savetxt(inFilePath, inArray, delimiter=',');
    return;
def readNumPyArrayFromCSVFile(inFilePath):
    return np.loadtxt(inFilePath, delimiter=',');

#*************************** SKULL PERSONALIZATION FUNCTIONS ***************************#
def estimateTriMeshShapeFromTriMesh(inTriMesh, inNumRemeshes = 4, inTargetEdgeLength = 0.001):
    # Estimate the convex hull of the pelvis
    print("estimateTriMeshShapeFromTriMesh::INFO:: Estimating the shape from the input mesh...");
    meshConvexHull = estimate3DConvexHull(inTriMesh.vertices);

    # Resample the convex hull
    print("estimateTriMeshShapeFromTriMesh::INFO:: Resampling the convex hull...");
    remeshedConvexHull = isotropicRemeshToTargetEdgeLength(meshConvexHull, inTargetEdgeLength);

    # Project remeshed pelvis convex hull onto the pelvis mesh
    print("estimateTriMeshShapeFromTriMesh::INFO:: Projecting the remeshed convex hull onto the input mesh...");
    projectedConvexHullVertices = estimateNearestPointsKDTreeBased(remeshedConvexHull.vertices, inTriMesh.vertices);
    projectedConvexHull = formTriMesh(projectedConvexHullVertices, remeshedConvexHull.faces);

    # Repeat of remeshing and projecting
    print("estimateTriMeshShapeFromTriMesh::INFO:: Repeating remeshing and projecting...");
    for i in range(inNumRemeshes):
        # Isotropic remesh and project
        remeshedProjectedConvexHull = isotropicRemeshToTargetEdgeLength(projectedConvexHull, inTargetEdgeLength);
        projectedConvexHullVertices = estimateNearestPointsKDTreeBased(remeshedConvexHull.vertices, remeshedProjectedConvexHull.vertices);
        projectedConvexHull = formTriMesh(projectedConvexHullVertices, remeshedConvexHull.faces);

    # Isotropic remesh the final time
    print("estimateTriMeshShapeFromTriMesh::INFO:: Final isotropic remeshing...");
    shape = isotropicRemeshToTargetEdgeLength(projectedConvexHull, inTargetEdgeLength);

    # Sample the shape
    print("estimateTriMeshShapeFromTriMesh::INFO:: Sampling the shape...");
    pelvisShapeSamples = poissonDiskSampleMesh(shape, 100000);

    # Surface reconstruct the shape
    print("estimateTriMeshShapeFromTriMesh::INFO:: Surface reconstructing the shape...");
    reconPelvisShape = surfaceReconstructionBallPivoting(pelvisShapeSamples);

    # Fix the normals of the shape
    print("estimateTriMeshShapeFromTriMesh::INFO:: Fixing the normals of the shape...");
    newNormals = computeCenterToVertexNormals(reconPelvisShape);
    reconPelvisShape.vertex_normals = newNormals;

    # Close holes the mesh
    print("estimateTriMeshShapeFromTriMesh::INFO:: Closing holes in the shape...");
    reconPelvisShape = closeHolesWithMeshLab(reconPelvisShape, 30000);

    # Return the shape
    print("estimateTriMeshShapeFromTriMesh::INFO:: Finished estimating the shape from the input mesh.");
    return reconPelvisShape;
def personalizeSkullShape(targetSkullShape, targetFeatures, templateSkullShape, templateFeatures):
    """
    Personalize a template skull mesh to match a target skull mesh using feature-based registration and deformation.
    Args:
        targetSkullShape: o3d.geometry.TriangleMesh
        targetFeatures: Nx3 numpy array or list
        templateSkullShape: o3d.geometry.TriangleMesh
        templateFeatures: Nx3 numpy array or list
    Returns:
        personalizedSkullShape: o3d.geometry.TriangleMesh
    """
    # Step 1: Rigid registration (SVD)
    transformation = estimateRigidSVDTransform(templateFeatures, targetFeatures)
    if transformation is None:
        print("ERROR: Could not estimate rigid transformation!")
        return None
    deformedSkullShape = cloneO3DMesh(templateSkullShape)
    deformedSkullShape.compute_vertex_normals()
    deformedSkullShape.transform(transformation)
    deformedFeatures = transform3DPoints(templateFeatures, transformation)

    # Step 2: Affine registration (CPD)
    affineTransform = estimateAffineCPDTransform(deformedFeatures, targetFeatures)
    deformedSkullShape.transform(affineTransform)
    deformedFeatures = transform3DPoints(deformedFeatures, affineTransform)

    # Step 3: RBF blend shape deformation
    deformedFeaturesBaryCoords, deformedFeaturesFaceIndices = computeBarycentricCoordinatesForO3DMesh(deformedSkullShape, deformedFeatures)
    deformedSkullShape = radialBasisFunctionBasedBlendShapeDeformation(
        deformedSkullShape, deformedFeatures, targetFeatures, inRBFType='thin_plate_spline', inSmooth=1e-3
    )
    deformedFeatures = reconstruct3DPointsFromBarycentricCoordinatesForO3DMesh(
        deformedSkullShape, deformedFeaturesBaryCoords, deformedFeaturesFaceIndices
    )

    # Step 4: Non-rigid ICP (Amberg) using downsampled mesh
    coarseDeformedSkullShape = isotropicRemeshO3DMeshWithResampling(deformedSkullShape, 2000)
    defCoarseDeformedSkullShape, coarseDeformedSkullShape = nonRigidTransformICPAmberg(
        coarseDeformedSkullShape, deformedFeatures, targetSkullShape, targetFeatures
    )
    coarseDeformedSkullVertices = np.asarray(coarseDeformedSkullShape.vertices, dtype=np.float64)
    defCoarseDeformedSkullVertices = np.asarray(defCoarseDeformedSkullShape.vertices, dtype=np.float64)
    deformedSkullShape = radialBasisFunctionBasedBlendShapeDeformation(
        deformedSkullShape, coarseDeformedSkullVertices, defCoarseDeformedSkullVertices
    )

    # Step 5: Projection refinement
    deformedSkullShapeVertices = np.asarray(deformedSkullShape.vertices, dtype=np.float64)
    targetSkullShapeVertices = np.asarray(targetSkullShape.vertices, dtype=np.float64)
    projectedSkullShapeVertices = estimateNearestPointsKDTreeBased(deformedSkullShapeVertices, targetSkullShapeVertices)
    personalizedSkullShape = cloneO3DMesh(deformedSkullShape)
    personalizedSkullShape.vertices = o3d.utility.Vector3dVector(projectedSkullShapeVertices)
    personalizedSkullShape.compute_vertex_normals()

    # Return the personalized skull shape
    return personalizedSkullShape;
def personalizeSkullMesh(inTargetSkullMesh, inTargetSkullShape, inTargetSkullFeatures, inTemplateSkullMesh, inTemplateSkullShape, inTemplateSkullFeatures):
    # Reading template skull mesh, skull shape, and features
    print("personalizeSkullMesh::INFO:: Personalizing the skull mesh...");
    ## Reading template mesh and shape
    targetSkullMesh = inTargetSkullMesh
    targetSkullShape = inTargetSkullShape
    ## Compute mesh normals
    targetSkullMesh.compute_vertex_normals();
    targetSkullShape.compute_vertex_normals();
    targetSkullMesh.paint_uniform_color([0.9, 0.1, 0.1]);
    targetSkullShape.paint_uniform_color([0.9, 0.1, 0.1]);
    ## Reading template features
    targetSkullFeatures = inTargetSkullFeatures;
    if targetSkullMesh is None or targetSkullFeatures is None:
        print("personalizeSkullMesh::ERROR:: Could not read the template skull mesh or features!");
        return;
    
    # Reading template skull shape and mesh and features
    print("personalizeSkullMesh::INFO:: Reading the template skull mesh, shape, and features...");
    ## Reading template skull mesh and shape
    templateSkullMesh = inTemplateSkullMesh;
    templateSkullShape = inTemplateSkullShape;
    ## Compute mesh normals
    templateSkullMesh.compute_vertex_normals();
    templateSkullShape.compute_vertex_normals();
    templateSkullMesh.paint_uniform_color([0.7, 0.7, 0.7]);
    templateSkullShape.paint_uniform_color([0.7, 0.7, 0.7]);
    ## Reading template features
    templateFeatures = inTemplateSkullFeatures;
    if templateSkullMesh is None or templateFeatures is None:
        print("personalizeSkullMesh::ERROR:: Could not read the target skull mesh or features!");
        return;
    
    # Use rigid and affine transform to align the template skull shape to the target skull mesh
    print("personalizeSkullMesh::INFO:: Aligning the template skull shape to the target skull mesh...");
    ## Define buffers
    defSkullShape = copy.deepcopy(templateSkullShape);
    defSkullFeatures = copy.deepcopy(templateFeatures);
    defSkullMesh = copy.deepcopy(templateSkullMesh);
    defSkullShape.compute_vertex_normals();
    defSkullMesh.compute_vertex_normals();
    defSkullMesh.paint_uniform_color([0.7, 0.7, 0.7]);
    ## Estimate the rigid transformation using SVD based on features points
    transformation = estimateRigidSVDTransform(defSkullFeatures, templateFeatures);
    if transformation is None:
        print("personalizeSkullMesh::ERROR:: Could not estimate the rigid transformation!");
        return;
    ## Transform the template skull shape to the target using the estimated transformation
    defSkullShape.transform(transformation);
    defSkullMesh.transform(transformation);
    defSkullFeatures = transform3DPoints(defSkullFeatures, transformation);
    ## Estimate the affine transformation using CPD based on features points
    affineTransform = estimateAffineCPDTransform(defSkullFeatures, targetSkullFeatures);
    ## Transform the deformed skull shape to the target using the estimated affine transformation
    defSkullShape.transform(affineTransform);
    defSkullMesh.transform(affineTransform);
    defSkullFeatures = transform3DPoints(defSkullFeatures, affineTransform);
    
    # Deform the deformed skull mesh to the target skull mesh based on skull shape deformation
    print("personalizeSkullMesh::INFO:: Deforming the deformed skull mesh to the target skull mesh based on skull shape deformation...");
    ## Personalize the deformed skull shape to the target skull shape using the features
    personalizedSkullShape = personalizeSkullShape(targetSkullShape, targetSkullFeatures, defSkullShape, defSkullFeatures);
    if personalizedSkullShape is None:
        print("personalizeSkullMesh::ERROR:: Could not personalize the skull shape!");
        return;
    personalizedSkullShape.compute_vertex_normals();
    personalizedSkullShape.paint_uniform_color([0.7, 0.7, 0.7]);
    ## Get the deformed skull shape vertices and personalized skull shape vertices
    deformedSkullShapeVertices = np.asarray(defSkullShape.vertices, dtype=np.float32);
    personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float32);
    ## Generate selected indices of 1000 indices randomly from 0 to number of vertices
    numOfSampledPoints = 5000;
    if len(deformedSkullShapeVertices) > numOfSampledPoints:
        selectedIndices = np.random.choice(len(deformedSkullShapeVertices) - 1, size=numOfSampledPoints, replace=False);
    else:
        selectedIndices = np.arange(len(deformedSkullShapeVertices));
    ## Deform the deformed skull mesh to the target skull mesh using RBF based on the personalized skull shape
    personalizedSkullMesh = radialBasisFunctionBasedBlendShapeDeformation(defSkullMesh, 
                                                                          deformedSkullShapeVertices[selectedIndices], 
                                                                          personalizedSkullShapeVertices[selectedIndices]);
    personalizedSkullMesh.compute_vertex_normals();
    personalizedSkullMesh.paint_uniform_color([0.7, 0.7, 0.7]);
    ## Visualize the personalized skull shape with target skull shape with the same space
    o3d.visualization.draw_geometries([personalizedSkullMesh] + [targetSkullMesh]);

    # Deform using the skull mesh using non rigid ICP
    print("personalizeSkullMesh::INFO:: Deforming the personalized skull mesh to the target skull mesh using non-rigid ICP...");
    ## Downsample the personalized skull mesh to get sampled features
    personalizedSkullMeshFeatures = poissonDiskSamplingOnO3DMesh(personalizedSkullMesh, 2000);
    targetSkullMeshVertices = np.asarray(targetSkullMesh.vertices, dtype=np.float64);
    targetSkullMeshFeatures = estimateNearestPointsKDTreeBased(personalizedSkullMeshFeatures, targetSkullMeshVertices);
    ## Deform the coarse personalized skull mesh to the target using the non rigid icp amberg from trimesh
    defPersonalizedSkullMesh, modifiedPersonalizedSkullMesh = nonRigidTransformICPAmberg(personalizedSkullMesh,
                                                                                         personalizedSkullMeshFeatures,
                                                                                         targetSkullMesh,
                                                                                         targetSkullMeshFeatures);
   
    # Return the personalized skull mesh
    print("personalizeSkullMesh::INFO:: Returning the deformed personalized skull mesh.");
    return defPersonalizedSkullMesh;

#***********************************************************************************************************************************************#
#************************************************************PROCESSING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
#*************************** DATA PROCESSING FUNCTIONS ***************************#
def subjectIDEstimation():
    # Initialize
    print("Subject ID Estimation");

    # List all skull mesh names inside the skullMeshAndFeatureFolder
    print("Listing all skull mesh names inside the skullMeshAndFeatureFolder");
    ## Getting the skull mesh file names
    print("\t Looking for skull mesh files inside the folder: " + skullMeshAndFeatureFolder);
    skullFileNames = listAllFilesInFolder(skullMeshAndFeatureFolder, ".ply");
    print("\t\t The number of skull mesh files found: " + str(len(skullFileNames)));
    ## Extracting the subject IDs from the skull mesh file names
    print("\t Extracting the subject IDs from the skull mesh file names");
    ids = [re.match(r"(.+?)-SkullMesh\.ply", name).group(1) for name in skullFileNames];
    print("\t\t The number of subject IDs found: " + str(len(ids)));

    # Save the subject IDs to a text file
    print("Saving the subject IDs to a text file");
    np.savetxt(mainFolder + "/SubjectIDs.txt", ids, fmt="%s");

    # Finished processing.
    print("Finished processing.");
def generateSkullShapesFromSkullMeshes():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("Usage: python CranialDefectPrediction.py <StartSubjectIndex> <EndSubjectIndex>");
        return;
    startIndex = int(sys.argv[1]);
    endIndex = int(sys.argv[2]);
    print("\t Generating skull shapes from skull meshes from subject index " + str(startIndex) + " to " + str(endIndex));

    # Reading full subject IDs
    print("Reading full subject IDs...");
    subjectIDs = np.loadtxt(mainFolder + "/SubjectIDs.txt", dtype=str);

    # Repeat for each subject
    for i in range(startIndex, endIndex + 1):
        print("/********************************* Processing subject index: " + str(i));
        subjectID = subjectIDs[i];

        # Reading target skull mesh
        print("\t Reading target skull mesh...");
        targetSkullMeshPath = skullMeshAndFeatureFolder + "/" + subjectID + "-SkullMesh.ply";
        targetSkullMesh = o3d.io.read_triangle_mesh(targetSkullMeshPath);
        if targetSkullMesh is None:
            print("generateSkullShapesFromSkullMeshes::ERROR:: Could not read the target skull mesh from: " + targetSkullMeshPath);
            continue;
        targetSkullMesh.compute_vertex_normals();
        targetSkullMesh.paint_uniform_color([0.9, 0.1, 0.1]);

        # Estimate the skull shape from the skull mesh
        print("\t Estimating the skull shape from the skull mesh...");
        targetSkullShape = estimateTriMeshShapeFromTriMesh(fromO3DMeshToTriMesh(targetSkullMesh), inNumRemeshes=4, inTargetEdgeLength=0.001);
        if targetSkullShape is None:
            print("generateSkullShapesFromSkullMeshes::ERROR:: Could not estimate the skull shape from the skull mesh for subject ID: " + subjectID);
            continue;
        targetSkullShape = fromTriMeshToO3DMesh(targetSkullShape);
        targetSkullShape.compute_vertex_normals();
        targetSkullShape.paint_uniform_color([0.9, 0.1, 0.1]);

        # Save the estimated skull shape
        print("\t Saving the estimated skull shape...");
        outputSkullShapePath = skullMeshAndFeatureFolder + "/" + subjectID + "-SkullShape.ply";
        o3d.io.write_triangle_mesh(outputSkullShapePath, targetSkullShape);

    # Finished processing
    print("Finished processing.");
def personalizeSkullShapes():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("Usage: python CranialDefectPrediction.py <StartSubjectIndex> <EndSubjectIndex>");
        return;
    startIndex = int(sys.argv[1]);
    endIndex = int(sys.argv[2]);
    print("\t Personalizing skull meshes from subject index " + str(startIndex) + " to " + str(endIndex));

    # Reading full subject IDs
    print("Reading full subject IDs...");
    subjectIDs = np.loadtxt(mainFolder + "/SubjectIDs.txt", dtype=str);

    # Reading template skull mesh, shape, and features
    print("Reading template skull mesh, shape, and features...");
    ## Reading template skull mesh
    templateSkullMeshPath = templateSkullGeometryFolder + "/TempSkullWithParts.ply";
    templateSkullMesh = o3d.io.read_triangle_mesh(templateSkullMeshPath);
    if templateSkullMesh is None:
        print("personalizeSkullMeshes::ERROR:: Could not read the template skull mesh from: " + templateSkullMeshPath);
        return;
    ## Reading template skull shape
    templateSkullShapePath = templateSkullGeometryFolder + "/TempSkullShapeWithParts.ply";
    templateSkullShape = o3d.io.read_triangle_mesh(templateSkullShapePath);
    if templateSkullShape is None:
        print("personalizeSkullMeshes::ERROR:: Could not read the template skull shape from: " + templateSkullShapePath);
        return;
    ## Reading template features
    templateFeaturesPath = templateSkullGeometryFolder + "/TempSkullShapeWithParts_picked_points.pp";
    templateFeatures = readXYZPointsFromPP(templateFeaturesPath);
    if templateFeatures is None:
        print("personalizeSkullMeshes::ERROR:: Could not read the template skull features from: " + templateFeaturesPath);
        return;

    # Repeat for each subject
    for i in range(startIndex, endIndex + 1):
        print("Processing subject index: " + str(i));
        subjectID = subjectIDs[i];

        # Reading target skull mesh, shape, and features
        print("\t Reading target skull mesh, shape, and features...");
        ## Reading target skull shape
        targetSkullShapePath = skullMeshAndFeatureFolder + "/" + subjectID + "-SkullShape.ply";
        targetSkullShape = o3d.io.read_triangle_mesh(targetSkullShapePath);
        if targetSkullShape is None:
            print("personalizeSkullMeshes::ERROR:: Could not read the target skull shape from: " + targetSkullShapePath);
            continue;
        ## Reading target features
        targetFeaturesPath = skullMeshAndFeatureFolder + "/" + subjectID + "-SkullMesh_picked_points.pp";
        targetFeatures = readXYZPointsFromPP(targetFeaturesPath);
        if targetFeatures is None:
            print("personalizeSkullMeshes::ERROR:: Could not read the target skull features from: " + targetFeaturesPath);
            continue;

        # Fix the target skull shape
        print("\t Fixing the target skull shape...");
        targetSkullShape = fixO3DMesh(targetSkullShape);

        # Personalize the skull shape
        print("\t Personalizing the skull shape...");
        personalizedSkullShape = personalizeSkullShape(targetSkullShape, targetFeatures, templateSkullShape, templateFeatures);
        if personalizedSkullShape is None:
            print("personalizeSkullMeshes::ERROR:: Could not personalize the skull shape for subject ID: " + subjectID);
            continue;

        # Save the personalized skull shape
        print("\t Saving the personalized skull shape...");
        outputPersonalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
        o3d.io.write_triangle_mesh(outputPersonalizedSkullShapePath, personalizedSkullShape);

    # Finished processing.
    print("Finished processing.");
def estimateExistingMissingPartIndicesAndFaces():
    # Initialize
    print("Initializing ...");

    ## Iterate for each area index and case index
    print("Iterating for each area index and case index...");
    for areaIndex in range(0, 3):
        for caseIndex in range(0, 3):
            print("/********************************* Processing area index: " + str(areaIndex) + ", case index: " + str(caseIndex));
            if areaIndex == 0:
                syntheticSkullShapeFilePath = mainFolder + f"/SyntheticSkullDefects/SmallAreas/SmallAreas_Case_{caseIndex}.ply";
                caseFolder = mainFolder + f"/SyntheticSkullDefects/SmallAreas";
            elif areaIndex == 1:
                syntheticSkullShapeFilePath = mainFolder + f"/SyntheticSkullDefects/MediumAreas/MediumAreas_Case_{caseIndex}.ply";
                caseFolder = mainFolder + f"/SyntheticSkullDefects/MediumAreas";
            elif areaIndex == 2:
                syntheticSkullShapeFilePath = mainFolder + f"/SyntheticSkullDefects/LargeAreas/LargeAreas_Case_{caseIndex}.ply";
                caseFolder = mainFolder + f"/SyntheticSkullDefects/LargeAreas";
            else:
                print("trainValidTestForMissingPartPrediction::ERROR:: Invalid area index: " + str(areaIndex));
                return;

            ## Reading the template skull shape and synthetic skull mesh
            print("\t Reading the template skull shape and synthetic skull mesh...");
            templateSkullShape = o3d.io.read_triangle_mesh(templateSkullGeometryFolder + "/TempSkullShapeWithParts.ply");
            if templateSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the template skull shape from: " + 
                    templateSkullGeometryFolder + "/TempSkullShapeWithParts.ply");
                return;
            syntheticSkullShape = o3d.io.read_triangle_mesh(syntheticSkullShapeFilePath);
            if syntheticSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the synthetic skull mesh from: " + syntheticSkullShapeFilePath);
                return;
            ## Estimate the existing part indices and faces
            print("\t Estimating the existing part indices and faces...");
            syntheticSkullShapeVertices = np.asarray(syntheticSkullShape.vertices, dtype=np.float64);
            templateSkullShapeVertices = np.asarray(templateSkullShape.vertices, dtype=np.float64);
            existingPartIndices = estimateNearestIndicesKDTreeBased(syntheticSkullShapeVertices, templateSkullShapeVertices);
            existingPartFaces = syntheticSkullShape.triangles;
            ## Estimate the missing part indices and faces
            print("\t Estimating the missing part indices and faces...");
            allIndices = np.arange(len(templateSkullShape.vertices));
            missingPartIndices = np.setdiff1d(allIndices, existingPartIndices);
            fullFaces = np.asarray(templateSkullShape.triangles, dtype=np.int32);
            missingPartFaces = [];
            for face in fullFaces:
                if np.any(np.isin(face, missingPartIndices)):
                    missingPartFaces.append(face);
            missingPartFaces = np.array(missingPartFaces);
            tempTrimesh = trimesh.Trimesh(vertices=templateSkullShapeVertices, faces=missingPartFaces, process=True);
            tempTrimesh.remove_unreferenced_vertices();
            missingPartFaces = np.asarray(tempTrimesh.faces, dtype=np.int32);
            ## Save the existing and missing part indices and faces
            np.savetxt(caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt", existingPartIndices, fmt="%d");
            np.savetxt(caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt", missingPartIndices, fmt="%d");
            np.savetxt(caseFolder + f"/ExistingPartFaces_Case_{caseIndex}.txt", existingPartFaces, fmt="%d");
            np.savetxt(caseFolder + f"/MissingPartFaces_Case_{caseIndex}.txt", missingPartFaces, fmt="%d");

    # Finished processing.
    print("Finished processing.");
def normalizeRigidDifferencesAmongSkullShapes():
    # Initialize
    print("Initializing ...");

    # Read the subject ids
    print("Reading the subject ids...");
    subjectIDs = np.loadtxt(mainFolder + "/SubjectIDs.txt", dtype=str);

    # Get the first personalized skull shape as the reference
    print("Getting the first personalized skull shape as the reference...");
    referenceSubjectID = subjectIDs[0];
    referencePersonalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + referenceSubjectID + "-PersonalizedSkullShape.ply";
    referencePersonalizedSkullShape = o3d.io.read_triangle_mesh(referencePersonalizedSkullShapePath);
    if referencePersonalizedSkullShape is None:
        print("normalizeRigidDifferencesAmongSkullShapes::ERROR:: Could not read the reference personalized skull shape from: " + referencePersonalizedSkullShapePath);
        return;

    # Register the reference skull shape to the origin using svd
    print("Registering the reference skull shape to the origin using SVD...");
    referenceVertices = np.asarray(referencePersonalizedSkullShape.vertices, dtype=np.float64);
    referenceCenter = np.mean(referenceVertices, axis=0);
    referenceVertices -= referenceCenter;

    # Repeat for each subject
    print("Repeating for each subject...");
    for i in range(len(subjectIDs)):
        # Print progress bar
        printProgressBar(i + 1, len(subjectIDs), prefix = 'Progress:', suffix = 'Complete', length = 50);
    
        # Reading the personalized skull shape
        subjectID = subjectIDs[i];
        skullShapeFilePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
        skullShape = o3d.io.read_triangle_mesh(skullShapeFilePath);
        if skullShape is None:
            print("normalizeRigidDifferencesAmongSkullShapes::ERROR:: Could not read the personalized skull shape from: " + skullShapeFilePath);
            return;

        # Estimate the rigid transform using svd based on the vertices
        skullShapeVertices = np.asarray(skullShape.vertices, dtype=np.float64);
        transformation = estimateRigidTransformSVD(skullShapeVertices, referenceVertices);
    
        # Transform the skull shape using the estimated transformation
        if transformation is not None:
            skullShapeVertices = np.asarray(skullShape.vertices, dtype=np.float64);
            skullShapeVertices = transform3DPoints(skullShapeVertices, transformation);
            skullShape.vertices = o3d.utility.Vector3dVector(skullShapeVertices);
    
        # Save the normalized skull shape
        normalizedSkullShapeFilePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape_Normalized.ply";
        o3d.io.write_triangle_mesh(normalizedSkullShapeFilePath, skullShape);

    # Finished processing.
    print("Finished processing.");

#*************************** CROSS-VALIDATION FUNCTIONS **************************#
def trainTestSplit():
    # Initialize
    print("Initializing ...");

    # Reading full subject IDs
    print("Reading full subject IDs...");
    subjectIDs = np.loadtxt(mainFolder + "/SubjectIDs.txt", dtype=str);

    # Shuffle the subject IDs
    print("Shuffling the subject IDs...");
    np.random.seed(42);  # For reproducibility    

    # Iterate 10 times for cross-validation
    print("Iterating 10 times for cross-validation...");
    trainRate = 0.7; validRate = 0.2; testRate = 0.1;
    numSubjects = len(subjectIDs);
    numOfTrain = int(numSubjects * trainRate);
    numOfValid = int(numSubjects * validRate);
    numOfTest = numSubjects - numOfTrain - numOfValid;
    trainTestSplitFolder = mainFolder + "/CrossValidationSplits";
    if not os.path.exists(trainTestSplitFolder): os.makedirs(trainTestSplitFolder);
    for fold in range(10):
        # Debugging
        print("\t Processing fold: " + str(fold + 1));

        # Shuffle the subject IDs
        np.random.shuffle(subjectIDs);
    
        # Get the train, valid, and test IDs for the current fold
        trainIDs = subjectIDs[:numOfTrain];
        validIDs = subjectIDs[numOfTrain:numOfTrain + numOfValid];
        testIDs = subjectIDs[numOfTrain + numOfValid:];
    
        # Save the train, valid, and test IDs to text files to the cross-validation folder
        np.savetxt(trainTestSplitFolder + "/TrainIDs_Fold" + str(fold + 1) + ".txt", trainIDs, fmt="%s");
        np.savetxt(trainTestSplitFolder + "/ValidIDs_Fold" + str(fold + 1) + ".txt", validIDs, fmt="%s");
        np.savetxt(trainTestSplitFolder + "/TestIDs_Fold" + str(fold + 1) + ".txt", testIDs, fmt="%s");

    # Finished processing.
    print("Finished processing.");
def trainAndValidateMissingPartPrediction_LinearRegression():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Usage: python CranialDefectPrediction.py <AreaIndex> <CaseIndex> <StartValidIndex> <EndValidIndex> <StartNumComps> <EndNumComps>");
        return;
    areaIndex = int(sys.argv[1]); caseIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);

    # Create model cross validation folder
    print("Creating model cross validation folder...");
    modelCrossValidationFolder = crossValidationFolder + "/LinearRegressionModels";
    if not os.path.exists(modelCrossValidationFolder): os.makedirs(modelCrossValidationFolder);

    # Forming validation folder, and case folder
    print("Forming validation folder, and case folder...");
    caseFolder = ""; validationFolder = "";
    missingIndexFilePath = ""; missingFacetFilePath = ""; existingIndexFilePath = ""; existingFacetFilePath = "";
    if areaIndex == 0:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/SmallAreas";
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/SmallAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    elif areaIndex == 1:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/MediumAreas";
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/MediumAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    elif areaIndex == 2:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/LargeAreas";
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/LargeAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    else:
        print("trainValidTestForMissingPartPrediction::ERROR:: Invalid area index: " + str(areaIndex));
        return;
    if not os.path.exists(validationFolder): os.makedirs(validationFolder);

    # Reading template data
    print("Reading template data ...");
    missingPartVertexIndices = readIndicesFromTXT(missingIndexFilePath);
    existingPartVertexIndices = readIndicesFromTXT(existingIndexFilePath);

    # Iterate from startValidIndex to endValidIndex
    print("Iterate for cross-validation from valid index " + str(startValidIndex) + " to " + str(endValidIndex));
    for validIndex in range(startValidIndex, endValidIndex + 1):
        # Debugging
        print("/********************************* Processing valid index: " + str(validIndex));
    
        # Reading train, valid, and test IDs
        print("Reading train, valid, and test IDs...");
        trainIDs = np.loadtxt(crossValidationFolder + "/CrossValidationSplits/TrainIDs_Fold" + str(validIndex) + ".txt", dtype=str);
        validIDs = np.loadtxt(crossValidationFolder + "/CrossValidationSplits/ValidIDs_Fold" + str(validIndex) + ".txt", dtype=str);
        testIDs = np.loadtxt(crossValidationFolder + "/CrossValidationSplits/TestIDs_Fold" + str(validIndex) + ".txt", dtype=str);
    
        # Forming data
        print("Forming data...");
        ## Forming training data
        print("\t Forming training data...");
        trainExistingPartData = []; trainMissingPartData = [];
        for i, subjectID in enumerate(trainIDs):
            ## Print progress bar
            printProgressBar(i, len(trainIDs), prefix = 'Progress', suffix = 'Complete', length = 50);

            ## Reading the personalized skull shape
            personalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
            personalizedSkullShape = o3d.io.read_triangle_mesh(personalizedSkullShapePath);
            if personalizedSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the personalized skull shape from: " + personalizedSkullShapePath);
                return;
            personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float64);

            ## Extracting the existing part vertices and missing part vertices
            existingPartVertices = personalizedSkullShapeVertices[existingPartVertexIndices];
            missingPartVertices = personalizedSkullShapeVertices[missingPartVertexIndices];

            ## Append to the training data lists
            trainExistingPartData.append(existingPartVertices.flatten());
            trainMissingPartData.append(missingPartVertices.flatten());
        trainExistingPartData = np.array(trainExistingPartData);
        trainMissingPartData = np.array(trainMissingPartData);
        ## Forming validation data
        print("\t Forming validation data...");
        validExistingPartData = []; validMissingPartData = [];
        for i, subjectID in enumerate(validIDs):
            ## Print progress bar
            printProgressBar(i, len(validIDs), prefix = 'Progress', suffix = 'Complete', length = 50);

            ## Reading the personalized skull shape
            personalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
            personalizedSkullShape = o3d.io.read_triangle_mesh(personalizedSkullShapePath);
            if personalizedSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the personalized skull shape from: " + personalizedSkullShapePath);
                return;
            personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float64);

            ## Extracting the existing part vertices and missing part vertices
            existingPartVertices = personalizedSkullShapeVertices[existingPartVertexIndices];
            missingPartVertices = personalizedSkullShapeVertices[missingPartVertexIndices];

            ## Append to the validation data lists
            validExistingPartData.append(existingPartVertices.flatten());
            validMissingPartData.append(missingPartVertices.flatten());
        validExistingPartData = np.array(validExistingPartData);
        validMissingPartData = np.array(validMissingPartData);
    
        # Iterate for the number of components from 1 to 200
        print("\t Iterate for the number of components from " + str(startNumComps) + " to " + str(endNumComps));
        for numComps in range(startNumComps, endNumComps + 1):
            # Debugging
            print("/************* Processing number of components: " + str(numComps) + " Validation Index: " + str(validIndex), end="", flush=True);
            
            # Determine the target number of components
            targetNumComps = min(numComps, min(trainMissingPartData.shape[0], trainMissingPartData.shape[1]));

            # Parameterize the missing part data
            missingPartScaler = StandardScaler();
            trainMissingPartScaledData = missingPartScaler.fit_transform(trainMissingPartData);
            validMissingPartScaledData = missingPartScaler.transform(validMissingPartData);
            missingPartPCAModel = PCA(n_components=targetNumComps, svd_solver='full');
            missingPartPCAModel.fit(trainMissingPartScaledData);
    
            # Parameterize the existing part data
            existingPartScaler = StandardScaler();
            trainExistingPartScaledData = existingPartScaler.fit_transform(trainExistingPartData);
            validExistingPartScaledData = existingPartScaler.transform(validExistingPartData);
            existingPartPCAModel = PCA(n_components=targetNumComps, svd_solver='full');
            existingPartPCAModel.fit(trainExistingPartScaledData);
    
            # Compute the trainingX and trainingY and validationX and validationY
            trainX = existingPartPCAModel.transform(trainExistingPartScaledData);
            trainY = missingPartPCAModel.transform(trainMissingPartScaledData);
            validX = existingPartPCAModel.transform(validExistingPartScaledData);

            # Train regression using multivariate linear regression
            regressionModel = LinearRegression();
            regressionModel.fit(trainX, trainY);
    
            # Predict the validation data
            predY = regressionModel.predict(validX);
            predMissingPartData = missingPartScaler.inverse_transform(missingPartPCAModel.inverse_transform(predY));

            # Compute the validation error
            validationErrors = [];
            for i in range(len(predMissingPartData)):
                predMissingPartVertices = predMissingPartData[i].reshape(-1, 3);
                gtMissingPartVertices = validMissingPartData[i].reshape(-1, 3);
                error = np.linalg.norm(predMissingPartVertices - gtMissingPartVertices, axis=1);
                meanError = np.mean(error);
                validationErrors.append(meanError);
            validationErrors = np.array(validationErrors);
            meanValidationError = np.mean(validationErrors);
    
            # Save the validation errors to text files to the validation folder
            saveNumPyArrayToCSVFile(validationErrors, validationFolder + f"/ValidationErrors_Fold{validIndex}_Comps{numComps}.csv");
    
            # Debugging
            print(f", Mean Validation Error: {meanValidationError:.4f}");

    # Finished processing.
    print("Finished processing.");
def drawValidationErrors_LinearRegression():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("Usage: python CranialDefectPrediction.py <AreaIndex> <CaseIndex>");
        return;
    areaIndex = int(sys.argv[1]); caseIndex = int(sys.argv[2]);

    # Forming validation folder, and case folder
    print("Forming validation folder, and case folder...");
    modelCrossValidationFolder = crossValidationFolder + "/LinearRegressionModels";
    validationFolder = "";
    if areaIndex == 0:
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/SmallAreas_Case_{caseIndex}";
    elif areaIndex == 1:
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/MediumAreas_Case_{caseIndex}";
    elif areaIndex == 2:
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/LargeAreas_Case_{caseIndex}";
    else:
        print("drawValidationErrors::ERROR:: Invalid area index: " + str(areaIndex));
        return;

    # Reading the validation errors from the validation folder
    print("Reading the validation errors from the validation folder...");
    allValidationErrors = [];
    numCompsList = [];
    for numComps in range(1, 201):
        foldValidationErrors = [];
        for validIndex in range(1, 11):
            validationErrorFilePath = validationFolder + f"/ValidationErrors_Fold{validIndex}_Comps{numComps}.csv";
            if os.path.exists(validationErrorFilePath):
                foldErrors = readNumPyArrayFromCSVFile(validationErrorFilePath);
                foldValidationErrors.extend(foldErrors.tolist());
        if len(foldValidationErrors) > 0:
            meanError = np.mean(foldValidationErrors);
            allValidationErrors.append(meanError);
            numCompsList.append(numComps);
            print(f"Number of Components: {numComps}, Mean Validation Error: {meanError:.4f}");
    
    # Plotting the validation errors
    print("Plotting the validation errors...");
    ## Setting up the plot
    plt.figure(figsize=(10, 6));
    ## Draw the plot
    plt.plot(numCompsList, allValidationErrors, marker='o', linestyle='-', color='b');
    ## Draw the minimum error point
    minErrorIndex = np.argmin(allValidationErrors);
    plt.plot(numCompsList[minErrorIndex], allValidationErrors[minErrorIndex], marker='o', color='r', markersize=10);
    plt.annotate(f'Min Error: {allValidationErrors[minErrorIndex]:.4f} at Comps: {numCompsList[minErrorIndex]}', 
                 xy=(numCompsList[minErrorIndex], allValidationErrors[minErrorIndex]), textcoords="offset points", 
                 arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=10);
    ## Setting up the name of the plot
    plt.title("Validation Errors vs Number of Components");
    plt.xlabel("Number of Components");
    plt.ylabel("Mean Validation Error");
    plt.grid();
    plt.show();
def testMissingPartPrediction_LinearRegression():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Usage: python CranialDefectPrediction.py <AreaIndex> <CaseIndex> <StartValidIndex> <EndValidIndex>");
        return;
    areaIndex = int(sys.argv[1]); caseIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    optimalNumComps = 100; # You can change this based on the validation results

    # Create model cross-validation folder
    print("Creating model cross-validation folder...");
    modelCrossValidationFolder = crossValidationFolder + "/LinearRegressionModels";
    if not os.path.exists(modelCrossValidationFolder): os.makedirs(modelCrossValidationFolder);

    # Forming validation folder, and case folder
    print("Forming validation folder, and case folder...");
    caseFolder = ""; testingFolder = "";
    missingIndexFilePath = ""; existingIndexFilePath = "";
    if areaIndex == 0:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/SmallAreas";
        testingFolder = modelCrossValidationFolder + f"/TestingErrors/SmallAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    elif areaIndex == 1:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/MediumAreas";
        testingFolder = modelCrossValidationFolder + f"/TestingErrors/MediumAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    elif areaIndex == 2:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/LargeAreas";
        testingFolder = modelCrossValidationFolder + f"/TestingErrors/LargeAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    else:
        print("trainValidTestForMissingPartPrediction::ERROR:: Invalid area index: " + str(areaIndex));
        return;
    if not os.path.exists(testingFolder): os.makedirs(testingFolder);

    # Reading template data
    print("Reading template data ...");
    missingPartVertexIndices = readIndicesFromTXT(missingIndexFilePath);
    existingPartVertexIndices = readIndicesFromTXT(existingIndexFilePath);

    # Iterate from startValidIndex to endValidIndex
    print("Iterate for cross-validation from valid index " + str(startValidIndex) + " to " + str(endValidIndex));
    for validIndex in range(startValidIndex, endValidIndex + 1):
        # Debugging
        print("/********************************* Processing valid index: " + str(validIndex));
    
        # Reading train and test IDs
        print("Reading train and test IDs...");
        trainIDs = np.loadtxt(crossValidationFolder + "/CrossValidationSplits/TrainIDs_Fold" + str(validIndex) + ".txt", dtype=str);
        testIDs = np.loadtxt(crossValidationFolder + "/CrossValidationSplits/TestIDs_Fold" + str(validIndex) + ".txt", dtype=str);
    
        # Forming training data
        print("Forming training data...");
        trainExistingPartData = []; trainMissingPartData = [];
        for i, subjectID in enumerate(trainIDs):
            # Print progress bar
            printProgressBar(i, len(trainIDs), prefix = 'Progress', suffix = 'Complete', length = 50);

            # Reading the personalized skull shape
            personalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
            personalizedSkullShape = o3d.io.read_triangle_mesh(personalizedSkullShapePath);
            if personalizedSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the personalized skull shape from: " + personalizedSkullShapePath);
                return;
            personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float64);

            # Extracting the existing part vertices and missing part vertices
            existingPartVertices = personalizedSkullShapeVertices[existingPartVertexIndices];
            missingPartVertices = personalizedSkullShapeVertices[missingPartVertexIndices];

            # Append to the training data lists
            trainExistingPartData.append(existingPartVertices.flatten());
            trainMissingPartData.append(missingPartVertices.flatten());
        trainExistingPartData = np.array(trainExistingPartData);
        trainMissingPartData = np.array(trainMissingPartData);

        # Forming testing data
        print("Forming testing data...");
        testExistingPartData = []; testMissingPartData = [];
        for i, subjectID in enumerate(testIDs):
            # Print progress bar
            printProgressBar(i, len(testIDs), prefix = 'Progress', suffix = 'Complete', length = 50);

            # Reading the personalized skull shape
            personalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
            personalizedSkullShape = o3d.io.read_triangle_mesh(personalizedSkullShapePath);
            if personalizedSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the personalized skull shape from: " + personalizedSkullShapePath);
                return;
            personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float64);

            # Extracting the existing part vertices and missing part vertices
            existingPartVertices = personalizedSkullShapeVertices[existingPartVertexIndices];
            missingPartVertices = personalizedSkullShapeVertices[missingPartVertexIndices];

            # Append to the testing data lists
            testExistingPartData.append(existingPartVertices.flatten());
            testMissingPartData.append(missingPartVertices.flatten());
        testExistingPartData = np.array(testExistingPartData);
        testMissingPartData = np.array(testMissingPartData);

        # Training data again to train the final model
        print("Training data again to train the final model...");
        ## Determine the target number of components
        targetNumComps = min(optimalNumComps, min(trainMissingPartData.shape[0], trainMissingPartData.shape[1]));
        ## Parameterize the missing part data
        missingPartScaler = StandardScaler();
        trainMissingPartScaledData = missingPartScaler.fit_transform(trainMissingPartData);
        missingPartPCAModel = PCA(n_components=targetNumComps, svd_solver='full');
        missingPartPCAModel.fit(trainMissingPartScaledData);
        ## Parameterize the existing part data
        existingPartScaler = StandardScaler();
        trainExistingPartScaledData = existingPartScaler.fit_transform(trainExistingPartData);
        existingPartPCAModel = PCA(n_components=targetNumComps, svd_solver='full');
        existingPartPCAModel.fit(trainExistingPartScaledData);
        ## Compute the trainingX and trainingY
        trainX = existingPartPCAModel.transform(trainExistingPartScaledData);
        trainY = missingPartPCAModel.transform(trainMissingPartScaledData);
        ## Train regression using multivariate linear regression
        regressionModel = LinearRegression();
        regressionModel.fit(trainX, trainY);
        ## Compute the testingX
        testExistingPartScaledData = existingPartScaler.transform(testExistingPartData);
        testX = existingPartPCAModel.transform(testExistingPartScaledData);
        ## Predict the testing data
        predY = regressionModel.predict(testX);
        predMissingPartData = missingPartScaler.inverse_transform(missingPartPCAModel.inverse_transform(predY));

        # Compute the testing error
        print("Computing the testing error...", end="", flush=True);
        testingErrors = [];
        for i in range(len(predMissingPartData)):
            predMissingPartVertices = predMissingPartData[i].reshape(-1, 3);
            gtMissingPartVertices = testMissingPartData[i].reshape(-1, 3);
            error = np.linalg.norm(predMissingPartVertices - gtMissingPartVertices, axis=1);
            meanError = np.mean(error);
            testingErrors.append(meanError);
        testingErrors = np.array(testingErrors);
        meanTestingError = np.mean(testingErrors);
        print(" -> Mean testing error: ", meanTestingError);

        # Save the testing errors to text files to the testing folder
        saveNumPyArrayToCSVFile(testingErrors, testingFolder + f"/TestingErrors_Fold{validIndex}_Comps{optimalNumComps}.csv");

    # Finished processing.
    print("Finished processing.");
def trainAndValidateMissingPartPrediction_MultiOutputRidgeRegression():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Usage: python CranialDefectPrediction.py <AreaIndex> <CaseIndex> <StartValidIndex> <EndValidIndex> <StartNumComps> <EndNumComps>");
        return;
    areaIndex = int(sys.argv[1]); caseIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    startNumComps = int(sys.argv[5]); endNumComps = int(sys.argv[6]);

    # Create model cross-validation folder
    print("Creating model cross-validation folder...");
    modelCrossValidationFolder = crossValidationFolder + "/MultiOutputRidgeRegressionModels";
    if not os.path.exists(modelCrossValidationFolder): os.makedirs(modelCrossValidationFolder);

    # Forming validation folder, and case folder
    print("Forming validation folder, and case folder...");
    caseFolder = ""; validationFolder = "";
    missingIndexFilePath = ""; existingIndexFilePath = ""; existingFacetFilePath = "";
    if areaIndex == 0:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/SmallAreas";
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/SmallAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    elif areaIndex == 1:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/MediumAreas";
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/MediumAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    elif areaIndex == 2:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/LargeAreas";
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/LargeAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    else:
        print("trainValidTestForMissingPartPrediction::ERROR:: Invalid area index: " + str(areaIndex));
        return;
    if not os.path.exists(validationFolder): os.makedirs(validationFolder);

    # Reading template data
    print("Reading template data ...");
    missingPartVertexIndices = readIndicesFromTXT(missingIndexFilePath);
    existingPartVertexIndices = readIndicesFromTXT(existingIndexFilePath);

    # Iterate from startValidIndex to endValidIndex
    print("Iterate for cross-validation from valid index " + str(startValidIndex) + " to " + str(endValidIndex));
    for validIndex in range(startValidIndex, endValidIndex + 1):
        # Debugging
        print("/********************************* Processing valid index: " + str(validIndex));
    
        # Reading train, valid, and test IDs
        print("Reading train, valid, and test IDs...");
        trainIDs = np.loadtxt(crossValidationFolder + "/CrossValidationSplits/TrainIDs_Fold" + str(validIndex) + ".txt", dtype=str);
        validIDs = np.loadtxt(crossValidationFolder + "/CrossValidationSplits/ValidIDs_Fold" + str(validIndex) + ".txt", dtype=str);
    
        # Forming data
        print("Forming data...");
        ## Forming training data
        print("\t Forming training data...");
        trainExistingPartData = []; trainMissingPartData = [];
        for i, subjectID in enumerate(trainIDs):
            ## Print progress bar
            printProgressBar(i, len(trainIDs), prefix = 'Progress', suffix = 'Complete', length = 50);

            ## Reading the personalized skull shape
            personalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
            personalizedSkullShape = o3d.io.read_triangle_mesh(personalizedSkullShapePath);
            if personalizedSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the personalized skull shape from: " + personalizedSkullShapePath);
                return;
            personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float64);

            ## Extracting the existing part vertices and missing part vertices
            existingPartVertices = personalizedSkullShapeVertices[existingPartVertexIndices];
            missingPartVertices = personalizedSkullShapeVertices[missingPartVertexIndices];

            ## Append to the training data lists
            trainExistingPartData.append(existingPartVertices.flatten());
            trainMissingPartData.append(missingPartVertices.flatten());
        trainExistingPartData = np.array(trainExistingPartData);
        trainMissingPartData = np.array(trainMissingPartData);
        ## Forming validation data
        print("\t Forming validation data...");
        validExistingPartData = []; validMissingPartData = [];
        for i, subjectID in enumerate(validIDs):
            ## Print progress bar
            printProgressBar(i, len(validIDs), prefix = 'Progress', suffix = 'Complete', length = 50);

            ## Reading the personalized skull shape
            personalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
            personalizedSkullShape = o3d.io.read_triangle_mesh(personalizedSkullShapePath);
            if personalizedSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the personalized skull shape from: " + personalizedSkullShapePath);
                return;
            personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float64);

            ## Extracting the existing part vertices and missing part vertices
            existingPartVertices = personalizedSkullShapeVertices[existingPartVertexIndices];
            missingPartVertices = personalizedSkullShapeVertices[missingPartVertexIndices];

            ## Append to the validation data lists
            validExistingPartData.append(existingPartVertices.flatten());
            validMissingPartData.append(missingPartVertices.flatten());
        validExistingPartData = np.array(validExistingPartData);
        validMissingPartData = np.array(validMissingPartData);
    
        # Iterate for the number of components from 1 to 200
        print("\t Iterate for the number of components from " + str(startNumComps) + " to " + str(endNumComps));
        for numComps in range(startNumComps, endNumComps + 1):
            # Debugging
            print("/************* Processing number of components: " + str(numComps) + " Validation Index: " + str(validIndex), end="", flush=True);
            
            # Determine the target number of components
            targetNumComps = min(numComps, min(trainMissingPartData.shape[0], trainMissingPartData.shape[1]));

            # Parameterize the missing part data
            missingPartScaler = StandardScaler();
            trainMissingPartScaledData = missingPartScaler.fit_transform(trainMissingPartData);
            missingPartPCAModel = PCA(n_components=targetNumComps, svd_solver='full');
            missingPartPCAModel.fit(trainMissingPartScaledData);
    
            # Parameterize the existing part data
            existingPartScaler = StandardScaler();
            trainExistingPartScaledData = existingPartScaler.fit_transform(trainExistingPartData);
            validExistingPartScaledData = existingPartScaler.transform(validExistingPartData);
            existingPartPCAModel = PCA(n_components=targetNumComps, svd_solver='full');
            existingPartPCAModel.fit(trainExistingPartScaledData);
    
            # Compute the trainingX and trainingY and validationX and validationY
            trainX = existingPartPCAModel.transform(trainExistingPartScaledData);
            trainY = missingPartPCAModel.transform(trainMissingPartScaledData);
            validX = existingPartPCAModel.transform(validExistingPartScaledData);

            # Train regression using multi-output ridge regression
            regressionModel = MultiOutputRegressor(Ridge(alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4, solver='auto'));
            regressionModel.fit(trainX, trainY);
    
            # Predict the validation data
            predY = regressionModel.predict(validX);
            predMissingPartData = missingPartScaler.inverse_transform(missingPartPCAModel.inverse_transform(predY));

            # Compute the validation error
            validationErrors = [];
            for i in range(len(predMissingPartData)):
                predMissingPartVertices = predMissingPartData[i].reshape(-1, 3);
                gtMissingPartVertices = validMissingPartData[i].reshape(-1, 3);
                error = np.linalg.norm(predMissingPartVertices - gtMissingPartVertices, axis=1);
                meanError = np.mean(error);
                validationErrors.append(meanError);
            validationErrors = np.array(validationErrors);
            meanValidationError = np.mean(validationErrors);
    
            # Save the validation errors to text files to the validation folder
            saveNumPyArrayToCSVFile(validationErrors, validationFolder + f"/ValidationErrors_Fold{validIndex}_Comps{numComps}.csv");
    
            # Debugging
            print(f", Mean Validation Error: {meanValidationError:.4f}");

    # Finished processing.
    print("Finished processing.");
def drawValidationErrors_MultiOutputRidgeRegression():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("Usage: python CranialDefectPrediction.py <AreaIndex> <CaseIndex>");
        return;
    areaIndex = int(sys.argv[1]); caseIndex = int(sys.argv[2]);

    # Forming validation folder, and case folder
    print("Forming validation folder, and case folder...");
    modelCrossValidationFolder = crossValidationFolder + "/MultiOutputRidgeRegressionModels";
    validationFolder = "";
    if areaIndex == 0:
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/SmallAreas_Case_{caseIndex}";
    elif areaIndex == 1:
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/MediumAreas_Case_{caseIndex}";
    elif areaIndex == 2:
        validationFolder = modelCrossValidationFolder + f"/ValidationErrors/LargeAreas_Case_{caseIndex}";
    else:
        print("drawValidationErrors::ERROR:: Invalid area index: " + str(areaIndex));
        return;

    # Reading the validation errors from the validation folder
    print("Reading the validation errors from the validation folder...");
    allValidationErrors = [];
    numCompsList = [];
    for numComps in range(1, 201):
        foldValidationErrors = [];
        for validIndex in range(1, 11):
            validationErrorFilePath = validationFolder + f"/ValidationErrors_Fold{validIndex}_Comps{numComps}.csv";
            if os.path.exists(validationErrorFilePath):
                foldErrors = readNumPyArrayFromCSVFile(validationErrorFilePath);
                foldValidationErrors.extend(foldErrors.tolist());
        if len(foldValidationErrors) > 0:
            meanError = np.mean(foldValidationErrors);
            allValidationErrors.append(meanError);
            numCompsList.append(numComps);
            print(f"Number of Components: {numComps}, Mean Validation Error: {meanError:.4f}");
    
    # Plotting the validation errors
    print("Plotting the validation errors...");
    ## Setting up the plot
    plt.figure(figsize=(10, 6));
    ## Draw the plot
    plt.plot(numCompsList, allValidationErrors, marker='o', linestyle='-', color='b');
    ## Draw the minimum error point
    minErrorIndex = np.argmin(allValidationErrors);
    plt.plot(numCompsList[minErrorIndex], allValidationErrors[minErrorIndex], marker='o', color='r', markersize=10);
    plt.annotate(f'Min Error: {allValidationErrors[minErrorIndex]:.4f} at Comps: {numCompsList[minErrorIndex]}', 
                 xy=(numCompsList[minErrorIndex], allValidationErrors[minErrorIndex]), textcoords="offset points", 
                 arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=10);
    ## Setting up the name of the plot 
    plt.title("Validation Errors vs Number of Components");
    plt.xlabel("Number of Components");
    plt.ylabel("Mean Validation Error");
    plt.grid();
    plt.show();
def testMissingPartPrediction_MultiOutputRidgeRegression():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 5):
        print("Usage: python CranialDefectPrediction.py <AreaIndex> <CaseIndex> <StartValidIndex> <EndValidIndex>");
        return;
    areaIndex = int(sys.argv[1]); caseIndex = int(sys.argv[2]);
    startValidIndex = int(sys.argv[3]); endValidIndex = int(sys.argv[4]);
    optimalNumComps = 100; # You can change this based on the validation results

    # Create model cross-validation folder
    print("Creating model cross-validation folder...");
    modelCrossValidationFolder = crossValidationFolder + "/MultiOutputRidgeRegressionModels";
    if not os.path.exists(modelCrossValidationFolder): os.makedirs(modelCrossValidationFolder);

    # Forming validation folder, and case folder
    print("Forming validation folder, and case folder...");
    caseFolder = ""; testingFolder = "";
    missingIndexFilePath = ""; existingIndexFilePath = "";
    if areaIndex == 0:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/SmallAreas";
        testingFolder = modelCrossValidationFolder + f"/TestingErrors/SmallAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    elif areaIndex == 1:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/MediumAreas";
        testingFolder = modelCrossValidationFolder + f"/TestingErrors/MediumAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    elif areaIndex == 2:
        caseFolder = mainFolder + f"/SyntheticSkullDefects/LargeAreas";
        testingFolder = modelCrossValidationFolder + f"/TestingErrors/LargeAreas_Case_{caseIndex}";
        missingIndexFilePath = caseFolder + f"/MissingPartIndices_Case_{caseIndex}.txt";
        existingIndexFilePath = caseFolder + f"/ExistingPartIndices_Case_{caseIndex}.txt";
    else:
        print("trainValidTestForMissingPartPrediction::ERROR:: Invalid area index: " + str(areaIndex));
        return;
    if not os.path.exists(testingFolder): os.makedirs(testingFolder);

    # Reading template data
    print("Reading template data ...");
    missingPartVertexIndices = readIndicesFromTXT(missingIndexFilePath);
    existingPartVertexIndices = readIndicesFromTXT(existingIndexFilePath);

    # Iterate from startValidIndex to endValidIndex
    print("Iterate for cross-validation from valid index " + str(startValidIndex) + " to " + str(endValidIndex));
    for validIndex in range(startValidIndex, endValidIndex + 1):
        # Debugging
        print("/********************************* Processing valid index: " + str(validIndex));
    
        # Reading train and test IDs
        print("Reading train and test IDs...");
        trainIDs = np.loadtxt(crossValidationFolder + "/CrossValidationSplits/TrainIDs_Fold" + str(validIndex) + ".txt", dtype=str);
        testIDs = np.loadtxt(crossValidationFolder + "/CrossValidationSplits/TestIDs_Fold" + str(validIndex) + ".txt", dtype=str);
    
        # Forming training data
        print("Forming training data...");
        trainExistingPartData = []; trainMissingPartData = [];
        for i, subjectID in enumerate(trainIDs):
            # Print progress bar
            printProgressBar(i, len(trainIDs), prefix = 'Progress', suffix = 'Complete', length = 50);

            # Reading the personalized skull shape
            personalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
            personalizedSkullShape = o3d.io.read_triangle_mesh(personalizedSkullShapePath);
            if personalizedSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the personalized skull shape from: " + personalizedSkullShapePath);
                return;
            personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float64);

            # Extracting the existing part vertices and missing part vertices
            existingPartVertices = personalizedSkullShapeVertices[existingPartVertexIndices];
            missingPartVertices = personalizedSkullShapeVertices[missingPartVertexIndices];

            # Append to the training data lists
            trainExistingPartData.append(existingPartVertices.flatten());
            trainMissingPartData.append(missingPartVertices.flatten());
        trainExistingPartData = np.array(trainExistingPartData);
        trainMissingPartData = np.array(trainMissingPartData);

        # Forming testing data
        print("Forming testing data...");
        testExistingPartData = []; testMissingPartData = [];
        for i, subjectID in enumerate(testIDs):
            # Print progress bar
            printProgressBar(i, len(testIDs), prefix = 'Progress', suffix = 'Complete', length = 50);

            # Reading the personalized skull shape
            personalizedSkullShapePath = postProcessedMeshAndFeatureFolder + "/" + subjectID + "-PersonalizedSkullShape.ply";
            personalizedSkullShape = o3d.io.read_triangle_mesh(personalizedSkullShapePath);
            if personalizedSkullShape is None:
                print("trainValidTestForMissingPartPrediction::ERROR:: Could not read the personalized skull shape from: " + personalizedSkullShapePath);
                return;
            personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float64);
            # Extracting the existing part vertices and missing part vertices
            existingPartVertices = personalizedSkullShapeVertices[existingPartVertexIndices];   
            missingPartVertices = personalizedSkullShapeVertices[missingPartVertexIndices];
            # Append to the testing data lists
            testExistingPartData.append(existingPartVertices.flatten());
            testMissingPartData.append(missingPartVertices.flatten());
        testExistingPartData = np.array(testExistingPartData);
        testMissingPartData = np.array(testMissingPartData);

        # Training data again to train the final model
        print("Training data again to train the final model...");
        ## Determine the target number of components
        targetNumComps = min(optimalNumComps, min(trainMissingPartData.shape[0], trainMissingPartData.shape[1]));
        ## Parameterize the missing part data
        missingPartScaler = StandardScaler();
        trainMissingPartScaledData = missingPartScaler.fit_transform(trainMissingPartData);
        missingPartPCAModel = PCA(n_components=targetNumComps, svd_solver='full');
        missingPartPCAModel.fit(trainMissingPartScaledData);
        ## Parameterize the existing part data
        existingPartScaler = StandardScaler();
        trainExistingPartScaledData = existingPartScaler.fit_transform(trainExistingPartData);
        existingPartPCAModel = PCA(n_components=targetNumComps, svd_solver='full');
        existingPartPCAModel.fit(trainExistingPartScaledData);
        ## Compute the trainingX and trainingY
        trainX = existingPartPCAModel.transform(trainExistingPartScaledData);
        trainY = missingPartPCAModel.transform(trainMissingPartScaledData);
        ## Train regression using multi-output ridge regression
        regressionModel = MultiOutputRegressor(Ridge(alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4, solver='auto'));
        regressionModel.fit(trainX, trainY);
        ## Compute the testingX
        testExistingPartScaledData = existingPartScaler.transform(testExistingPartData);
        testX = existingPartPCAModel.transform(testExistingPartScaledData);
        ## Predict the testing data
        predY = regressionModel.predict(testX);
        predMissingPartData = missingPartScaler.inverse_transform(missingPartPCAModel.inverse_transform(predY));
        
        # Compute the testing error
        print("Computing the testing error...", end="", flush=True);
        testingErrors = [];
        for i in range(len(predMissingPartData)):
            predMissingPartVertices = predMissingPartData[i].reshape(-1, 3);
            gtMissingPartVertices = testMissingPartData[i].reshape(-1, 3);
            error = np.linalg.norm(predMissingPartVertices - gtMissingPartVertices, axis=1);
            meanError = np.mean(error);
            testingErrors.append(meanError);
        testingErrors = np.array(testingErrors);
        meanTestingError = np.mean(testingErrors);
        print(" -> Mean testing error: ", meanTestingError);

        # Save the testing errors to text files to the testing folder
        saveNumPyArrayToCSVFile(testingErrors, testingFolder + f"/TestingErrors_Fold{validIndex}_Comps{optimalNumComps}.csv");

    # Finished processing.
    print("Finished processing.");
def drawBoxplotForTestingErrorsOfLinearRegressionAndMultiOutputRidgeRegression():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("Usage: python CranialDefectPrediction.py <AreaIndex> <CaseIndex>");
        return;
    areaIndex = int(sys.argv[1]); caseIndex = int(sys.argv[2]);
    linearRegressionOptimalNumComps = 100; # You can change this based on the validation results
    multiOutputRidgeRegressionOptimalNumComps = 100; # You can change this based on the validation results

    # Forming testing folder, and case folder
    print("Forming testing folder, and case folder...");
    modelCrossValidationFolder = crossValidationFolder + "/MultiOutputRidgeRegressionModels";
    linearRegressionTestingFolder = ""; multiOutputRidgeRegressionTestingFolder = "";
    if areaIndex == 0:
        linearRegressionTestingFolder = modelCrossValidationFolder + f"/TestingErrors/SmallAreas_Case_{caseIndex}";
        multiOutputRidgeRegressionTestingFolder = modelCrossValidationFolder + f"/TestingErrors/SmallAreas_Case_{caseIndex}";
    elif areaIndex == 1:
        linearRegressionTestingFolder = modelCrossValidationFolder + f"/TestingErrors/MediumAreas_Case_{caseIndex}";
        multiOutputRidgeRegressionTestingFolder = modelCrossValidationFolder + f"/TestingErrors/MediumAreas_Case_{caseIndex}";
    elif areaIndex == 2:
        linearRegressionTestingFolder = modelCrossValidationFolder + f"/TestingErrors/LargeAreas_Case_{caseIndex}";
        multiOutputRidgeRegressionTestingFolder = modelCrossValidationFolder + f"/TestingErrors/LargeAreas_Case_{caseIndex}";
    else:
        print("drawBoxplotForTestingErrorsOfLinearRegressionAndMultiOutputRidgeRegression::ERROR:: Invalid area index: " + str(areaIndex));
        return;

    # Reading the testing errors from the testing folder
    print("Reading the testing errors from the testing folder...");
    linearRegressionTestingErrors = [];
    multiOutputRidgeRegressionTestingErrors = [];
    for validIndex in range(1, 11):
        linearRegressionErrorFilePath = linearRegressionTestingFolder + f"/TestingErrors_Fold{validIndex}_Comps{linearRegressionOptimalNumComps}.csv";
        if os.path.exists(linearRegressionErrorFilePath):
            foldErrors = readNumPyArrayFromCSVFile(linearRegressionErrorFilePath);
            linearRegressionTestingErrors.extend(foldErrors.tolist());
        multiOutputRidgeRegressionErrorFilePath = multiOutputRidgeRegressionTestingFolder + f"/TestingErrors_Fold{validIndex}_Comps{multiOutputRidgeRegressionOptimalNumComps}.csv";
        if os.path.exists(multiOutputRidgeRegressionErrorFilePath):
            foldErrors = readNumPyArrayFromCSVFile(multiOutputRidgeRegressionErrorFilePath);
            multiOutputRidgeRegressionTestingErrors.extend(foldErrors.tolist());
    linearRegressionTestingErrors = np.array(linearRegressionTestingErrors);
    multiOutputRidgeRegressionTestingErrors = np.array(multiOutputRidgeRegressionTestingErrors);
    print(f"Linear Regression - Total Testing Samples: {len(linearRegressionTestingErrors)}, Mean Testing Error: {np.mean(linearRegressionTestingErrors):.4f}");
    print(f"Multi-Output Ridge Regression - Total Testing Samples: {len(multiOutputRidgeRegressionTestingErrors)}, Mean Testing Error: {np.mean(multiOutputRidgeRegressionTestingErrors):.4f}");    

    # Plotting the boxplot for testing errors
    print("Plotting the boxplot for testing errors...");
    ## Setting up the plot
    plt.figure(figsize=(8, 6));
    ## Draw the boxplot
    plt.boxplot([linearRegressionTestingErrors, multiOutputRidgeRegressionTestingErrors], labels=['Linear Regression', 'Multi-Output Ridge Regression']);
    ## Setting up the name of the plot
    plt.title("Testing Errors Comparison");
    plt.ylabel("Mean Testing Error");
    plt.grid(axis='y');
    plt.show();

    # Finished processing.
    print("Finished processing.");

#*************************** TESTING FUNCTIONS ***********************************#

#***********************************************************************************************************************************************#
#************************************************************MAIN FUNCTIONS*********************************************************************#
#***********************************************************************************************************************************************#
def main():
    os.system("cls");
    personalizeSkullShapes();
if __name__ == "__main__":
    main()