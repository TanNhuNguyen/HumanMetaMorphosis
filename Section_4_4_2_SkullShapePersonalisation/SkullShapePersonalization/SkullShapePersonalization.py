#***********************************************************************************************************************************************#
#************************************************************SUPPORTING LIBRARIES***************************************************************#
#***********************************************************************************************************************************************#
import os;
import numpy as np;
import open3d as o3d;
import xml.etree.ElementTree as ET;
import copy;
from pycpd import AffineRegistration;
import trimesh;
from scipy.interpolate import RBFInterpolator;
import pymeshlab;
from scipy.spatial import KDTree;
from scipy.spatial.distance import directed_hausdorff

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING BUFFERS*****************************************************************#
#***********************************************************************************************************************************************#
mainFolder = "../../../Data/Section_4_4_2_SkullShapePersonalisation";

#***********************************************************************************************************************************************#
#************************************************************PROCESSING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
def pause():
    os.system("pause");
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
def formTrimesh(inVertices, inFaces):
    # Create a trimesh object
    return trimesh.Trimesh(vertices=inVertices, faces=inFaces, process=False);
def readO3DMesh(filePath):
    # Checking file path
    if not os.path.exists(filePath):
        print("readO3DMesh:: ERROR: File path does not exist!");
        return None;

    # Reading the mesh
    mesh = o3d.io.read_triangle_mesh(filePath);
    if mesh.is_empty():
        return None;

    # Return the mesh
    return mesh;
def read3DPointsFromPPFile(filePath):
    # Checking file path
    if not os.path.exists(filePath):
        print("read3DPointsFromPPFile:: ERROR: File path does not exist!")
        return None

    # Reading the points using XML parser
    points = []
    tree = ET.parse(filePath)
    root = tree.getroot()
    for point_elem in root.findall('.//point'):
        try:
            x = float(point_elem.attrib.get('x'))
            y = float(point_elem.attrib.get('y'))
            z = float(point_elem.attrib.get('z'))
            points.append([x, y, z])
        except (TypeError, ValueError):
            print(f"read3DPointsFromPPFile:: WARNING: Could not parse point attributes: {point_elem.attrib}")

    # Return the points
    return points
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
def createSphereAtPoint(point, radius=0.010, resolution=20):
    # Create a sphere mesh at the given point
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution);

    # Translate and color the sphere
    sphere.translate(point);

    # Set the sphere material
    sphere.paint_uniform_color([1.0, 0.0, 0.0]);

    # Compute normals
    sphere.compute_vertex_normals();

    # Return the sphere
    return sphere;
def transform3DPoints(points, transformMatrix):
    """
    Transform 3D points using a 4x4 transformation matrix.
    Args:
        points: Nx3 numpy array
        transformMatrix: 4x4 numpy array
    Returns:
        transformedPoints: Nx3 numpy array
    """
    points = np.asarray(points, dtype=np.float64)
    N = points.shape[0]
    # Convert to homogeneous coordinates
    points_h = np.hstack([points, np.ones((N, 1))])
    # Apply transformation
    transformed_h = (transformMatrix @ points_h.T).T
    # Convert back to 3D
    transformedPoints = transformed_h[:, :3]
    return transformedPoints;
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
def computeHausdorffDistanceBetweenO3DMeshes(inO3DMeshA, inO3DMeshB):
    # Checking inputs
    if inO3DMeshA is None or inO3DMeshB is None:
        print("computeHausdorffDistanceBetweenO3DMeshes:: ERROR: Input meshes are None!");
        return None;
    if inO3DMeshA.is_empty() or inO3DMeshB.is_empty():
        print("computeHausdorffDistanceBetweenO3DMeshes:: ERROR: Input meshes are empty!");
        return None;

    # Get vertices as numpy arrays
    verticesA = np.asarray(inO3DMeshA.vertices)
    verticesB = np.asarray(inO3DMeshB.vertices)

    # Compute directed Hausdorff distances
    hausdorff_AB = directed_hausdorff(verticesA, verticesB)[0]
    hausdorff_BA = directed_hausdorff(verticesB, verticesA)[0]

    # Return the maximum (true Hausdorff distance)
    return max(hausdorff_AB, hausdorff_BA)
def personalizeSkullShapeFromMeshes(targetSkullShape, targetFeatures, templateSkullShape, templateFeatures):
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
    print("personalizeSkullShapeFromMeshes:: Step 1: Rigid registration (SVD)");
    transformation = estimateRigidSVDTransform(templateFeatures, targetFeatures)
    if transformation is None:
        print("ERROR: Could not estimate rigid transformation!")
        return None
    deformedSkullShape = cloneO3DMesh(templateSkullShape)
    deformedSkullShape.compute_vertex_normals()
    deformedSkullShape.transform(transformation)
    deformedFeatures = transform3DPoints(templateFeatures, transformation)

    # Step 2: Affine registration (CPD)
    print("personalizeSkullShapeFromMeshes:: Step 2: Affine registration (CPD)")
    affineTransform = estimateAffineCPDTransform(deformedFeatures, targetFeatures)
    deformedSkullShape.transform(affineTransform)
    deformedFeatures = transform3DPoints(deformedFeatures, affineTransform)

    # Step 3: RBF blend shape deformation
    print("personalizeSkullShapeFromMeshes:: Step 3: RBF blend shape deformation")
    deformedFeaturesBaryCoords, deformedFeaturesFaceIndices = computeBarycentricCoordinatesForO3DMesh(deformedSkullShape, deformedFeatures)
    deformedSkullShape = radialBasisFunctionBasedBlendShapeDeformation(
        deformedSkullShape, deformedFeatures, targetFeatures, inRBFType='thin_plate_spline', inSmooth=1e-3
    )
    deformedFeatures = reconstruct3DPointsFromBarycentricCoordinatesForO3DMesh(
        deformedSkullShape, deformedFeaturesBaryCoords, deformedFeaturesFaceIndices
    )

    # Step 4: Non-rigid ICP (Amberg) using downsampled mesh
    print("personalizeSkullShapeFromMeshes:: Step 4: Non-rigid ICP (Amberg) using downsampled mesh")
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
    print("personalizeSkullShapeFromMeshes:: Step 5: Projection refinement")
    deformedSkullShapeVertices = np.asarray(deformedSkullShape.vertices, dtype=np.float64)
    targetSkullShapeVertices = np.asarray(targetSkullShape.vertices, dtype=np.float64)
    projectedSkullShapeVertices = estimateNearestPointsKDTreeBased(deformedSkullShapeVertices, targetSkullShapeVertices)
    personalizedSkullShape = cloneO3DMesh(deformedSkullShape)
    personalizedSkullShape.vertices = o3d.utility.Vector3dVector(projectedSkullShapeVertices)
    personalizedSkullShape.compute_vertex_normals()

    # Return the personalized skull shape
    print("personalizeSkullShapeFromMeshes:: Personalization complete.")
    return personalizedSkullShape;

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
def personalizeSkullShape():
    # Initialize
    print("Initializing ...");

    # Reading the template skull shape with features
    print("Reading the template skull shape with features ...");
    ## Reading template mesh
    print("\t Reading template mesh ...");
    templateSkullShape = readO3DMesh(os.path.join(mainFolder, "TempSkullShapeWithParts.ply"));
    ## Compute mesh normals
    print("\t Computing mesh normals ...");
    templateSkullShape.compute_vertex_normals();
    ## Reading template features
    print("\t Reading template features ...");
    templateFeatures = read3DPointsFromPPFile(os.path.join(mainFolder, "TempSkullShapeWithParts_picked_points.pp"));
    if templateSkullShape is None or templateFeatures is None:
        print("ERROR: Could not read the template skull shape or features!");
        return;
    print("\t Number of template features:", len(templateFeatures));
    # Visualize the template skull shape and features
    print("\t Visualizing the template skull shape and features ...");
    templateSkullShape.paint_uniform_color([0.7, 0.7, 0.7]);
    templateFeatureSpheres = [];
    for feature in templateFeatures:
        sphere = createSphereAtPoint(feature, radius=0.005);
        templateFeatureSpheres.append(sphere);
    o3d.visualization.draw_geometries([templateSkullShape] + templateFeatureSpheres);

    # Reading the target skull shape with features
    print("Reading the target skull shape with features ...");
    ## Reading target mesh
    print("\t Reading target mesh ...");
    targetSkullShape = readO3DMesh(os.path.join(mainFolder, "119219-SkullShape.ply"));
    ## Compute mesh normals
    print("\t Computing mesh normals ...");
    targetSkullShape.compute_vertex_normals();
    ## Reading target features
    print("\t Reading target features ...");
    targetFeatures = read3DPointsFromPPFile(os.path.join(mainFolder, "119219-SkullShape_picked_points.pp"));
    if targetSkullShape is None or targetFeatures is None:
        print("ERROR: Could not read the target skull shape or features!");
        return;
    print("\t Number of target features:", len(targetFeatures));
    ## Visualize the target skull shape and features
    print("\t Visualizing the target skull shape and features ...");
    targetSkullShape.paint_uniform_color([0.7, 0.7, 0.7]);
    targetFeatureSpheres = [];
    for feature in targetFeatures:
        sphere = createSphereAtPoint(feature, radius=0.005);
        targetFeatureSpheres.append(sphere);
    o3d.visualization.draw_geometries([targetSkullShape] + targetFeatureSpheres);

    # Initialize register the template skull shape to the target skull shape
    print("Initializing register the template skull shape to the target skull shape ...");
    ## Estimate the rigid transformation using SVD based on features points
    print("\t Estimating the rigid transformation using SVD based on features points ...");
    transformation = estimateRigidSVDTransform(templateFeatures, targetFeatures);
    if transformation is None:
        print("ERROR: Could not estimate the rigid transformation!");
        return;
    print("\t Transformation matrix:\n", transformation);
    ## Transform the template skull shape to the target using the estimated transformation
    print("\t Transforming the template skull shape to the target using the estimated transformation ...");
    deformedSkullShape = cloneO3DMesh(templateSkullShape);
    deformedSkullShape.compute_vertex_normals();
    deformedSkullShape.transform(transformation);
    ## Transform also the template features to the target using the estimated transformation
    deformedFeatures = copy.deepcopy(templateFeatures);
    deformedFeatures = transform3DPoints(deformedFeatures, transformation);
    ## Visualize the deformed skull shape with the target skull shape
    print("\t Visualizing the deformed skull shape with the target skull shape ...");
    deformedSkullShape.paint_uniform_color([0.7, 0.7, 0.7]);
    targetSkullShape.paint_uniform_color([0.9, 0.1, 0.1]);
    deformedSkullShapeMaterial = o3d.visualization.rendering.MaterialRecord();
    deformedSkullShapeMaterial.base_color = [0.7, 0.7, 0.7, 0.5];
    deformedSkullShapeMaterial.shader = "defaultLitTransparency";
    targetSkullShapeMaterial = o3d.visualization.rendering.MaterialRecord();
    targetSkullShapeMaterial.base_color = [0.9, 0.1, 0.1, 0.5];
    targetSkullShapeMaterial.shader = "defaultLitTransparency";
    o3d.visualization.draw([{ "name": "Deformed", "geometry": deformedSkullShape, "material": deformedSkullShapeMaterial}, 
                            { "name": "Target", "geometry": targetSkullShape, "material": targetSkullShapeMaterial}],
                            bg_color=(1, 1, 1, 1), 
                            show_skybox=False);
    
    # Deform the template skull shape to the target skull shape using affine transform
    print("Deforming the template skull shape to the target skull shape using affine transform ...");
    ## Estimate the affine transformation using CPD based on features points
    print("\t Estimating the affine transformation using CPD based on features points ...");
    affineTransform = estimateAffineCPDTransform(deformedFeatures, targetFeatures);
    print("\t Affine transformation matrix:\n", affineTransform);
    ## Transform the deformed skull shape to the target using the estimated affine transformation
    print("\t Transforming the deformed skull shape to the target using the estimated affine transformation ...");
    deformedSkullShape.transform(affineTransform);
    ## Transform also the deformed features to the target using the estimated affine transformation
    deformedFeatures = transform3DPoints(deformedFeatures, affineTransform);
    ## Visualize the deformed skull shape with the target skull shape
    print("\t Visualizing the deformed skull shape with the target skull shape ...");
    deformedSkullShape.paint_uniform_color([0.7, 0.7, 0.7]);
    targetSkullShape.paint_uniform_color([0.9, 0.1, 0.1]);
    deformedSkullShapeMaterial = o3d.visualization.rendering.MaterialRecord();
    deformedSkullShapeMaterial.base_color = [0.7, 0.7, 0.7, 0.5];
    deformedSkullShapeMaterial.shader = "defaultLitTransparency";
    targetSkullShapeMaterial = o3d.visualization.rendering.MaterialRecord();
    targetSkullShapeMaterial.base_color = [0.9, 0.1, 0.1, 0.5];
    targetSkullShapeMaterial.shader = "defaultLitTransparency";
    o3d.visualization.draw([{ "name": "Deformed", "geometry": deformedSkullShape, "material": deformedSkullShapeMaterial}, 
                            { "name": "Target", "geometry": targetSkullShape, "material": targetSkullShapeMaterial}],
                            bg_color=(1, 1, 1, 1), 
                            show_skybox=False);
    
    # Deform using the radial basis function based blend shape deformation
    print("Deform using the radial basis function based blend shape deformation ...");
    ## Compute barycentric coordinates for the deformed features
    print("\t Computing barycentric coordinates for the deformed features ...");
    deformedFeaturesBaryCoords, deformedFeaturesFaceIndices = computeBarycentricCoordinatesForO3DMesh(deformedSkullShape, deformedFeatures);
    ## Deform using the radial basis function based blend shape deformation
    print("\t Deforming using the radial basis function based blend shape deformation ...");
    deformedSkullShape = radialBasisFunctionBasedBlendShapeDeformation(deformedSkullShape, deformedFeatures, targetFeatures, inRBFType='thin_plate_spline', inSmooth=1e-3);
    ## Update the deformed features
    deformedFeatures = reconstruct3DPointsFromBarycentricCoordinatesForO3DMesh(deformedSkullShape, deformedFeaturesBaryCoords, deformedFeaturesFaceIndices);
    ## Visualize the deformed skull shape with the target skull shape
    print("\t Visualizing the deformed skull shape with the target skull shape ...");
    deformedSkullShape.paint_uniform_color([0.7, 0.7, 0.7]);
    targetSkullShape.paint_uniform_color([0.9, 0.1, 0.1]);
    deformedSkullShapeMaterial = o3d.visualization.rendering.MaterialRecord();
    deformedSkullShapeMaterial.base_color = [0.7, 0.7, 0.7, 0.75];
    deformedSkullShapeMaterial.shader = "defaultLitTransparency";
    targetSkullShapeMaterial = o3d.visualization.rendering.MaterialRecord();
    targetSkullShapeMaterial.base_color = [0.9, 0.1, 0.1, 0.75];
    targetSkullShapeMaterial.shader = "defaultLitTransparency";
    o3d.visualization.draw([{ "name": "Deformed", "geometry": deformedSkullShape, "material": deformedSkullShapeMaterial}, { "name": "Target", "geometry": targetSkullShape, "material": targetSkullShapeMaterial}], bg_color=(1, 1, 1, 1), show_skybox=False);
    
    # Deform skull shape using non rigid ICP using sampled features on the surface
    print("Deform skull shape using non rigid ICP using sampled features on the surface ...");
    ## Downsample the deformed skull shape to get sampled features
    print("\t Downsampling the deformed skull shape to get sampled features ...");
    coarseDeformedSkullShape = isotropicRemeshO3DMeshWithResampling(deformedSkullShape, 2000);
    ## Deform the coarse deformed skull shape to the target using the non rigid icp amberg from trimesh
    print("\t Deforming the coarse deformed skull shape to the target using the non rigid icp amberg from trimesh ...");
    defCoaseDeformedSkullShape, coarseDeformedSkullShape = nonRigidTransformICPAmberg(coarseDeformedSkullShape, deformedFeatures, targetSkullShape, targetFeatures);
    ## Deform the fine skull shape using the RBF based on the coarse deformed skull shape
    print("\t Deforming the fine skull shape using the RBF based on the coarse deformed skull shape ...");
    coarseDeformedSkullVertices = np.asarray(coarseDeformedSkullShape.vertices, dtype=np.float64);
    defCoarseDeformedSkullVertices = np.asarray(defCoaseDeformedSkullShape.vertices, dtype=np.float64);
    deformedSkullShape = radialBasisFunctionBasedBlendShapeDeformation(deformedSkullShape, 
                                                                       coarseDeformedSkullVertices, 
                                                                       defCoarseDeformedSkullVertices);
    ## Visualize the coarse deformed skull shape with the target skull shape
    print("\t Visualizing the coarse deformed skull shape with the target skull shape ...");    
    deformedSkullShape.compute_vertex_normals();
    deformedSkullShape.paint_uniform_color([1.0, 0.0, 0.0]);
    o3d.visualization.draw_geometries([deformedSkullShape, targetSkullShape]);

    # Refine deformation using projection to the nearest skull shape vertices
    print("Refining deformation using projection to the nearest skull shape vertices ...");
    ## Project the deformed skull shape to the nearest target skull shape vertices
    print("\t Projecting the deformed skull shape to the nearest target skull shape vertices ...");
    deformedSkullShapeVertices = np.asarray(deformedSkullShape.vertices, dtype=np.float64);
    projectedSkullShapeVertices = np.asarray(deformedSkullShape.vertices, dtype=np.float64);
    targetSkullShapeVertices = np.asarray(targetSkullShape.vertices, dtype=np.float64);
    projectedSkullShapeVertices = estimateNearestPointsKDTreeBased(projectedSkullShapeVertices, targetSkullShapeVertices);
    projectedSkullShape = copy.deepcopy(deformedSkullShape);
    projectedSkullShape.vertices = o3d.utility.Vector3dVector(projectedSkullShapeVertices);
    projectedSkullShape.compute_vertex_normals();
    ## Estimate the personalized skull shape using the deformed skull shape and the projected skull shape
    personalizedSkullShape = copy.deepcopy(projectedSkullShape);
    ## Visualize the coarse deformed skull shape with the target skull shape
    print("\t Visualizing the coarse deformed skull shape with the target skull shape ...");
    personalizedSkullShape.compute_vertex_normals();
    personalizedSkullShape.paint_uniform_color([1.0, 0.0, 0.0]);
    o3d.visualization.draw_geometries([personalizedSkullShape, targetSkullShape]);

    # Finished processing.
    print("Finished processing.");
def personalizeSkullMesh():
    # Initialize
    print("Initializing ...");

    # Reading template skull mesh, skull shape, and features
    print("Reading template skull mesh and features ...");
    ## Reading template mesh and shape
    print("\t Reading template mesh and shape ...");
    targetSkullMesh = readO3DMesh(os.path.join(mainFolder, "HN-CHUM-005-SkullMesh.ply"));
    targetSkullShape = readO3DMesh(os.path.join(mainFolder, "HN-CHUM-005-SkullShape.ply"));
    ## Compute mesh normals
    print("\t Computing mesh normals ...");
    targetSkullMesh.compute_vertex_normals();
    targetSkullShape.compute_vertex_normals();
    targetSkullMesh.paint_uniform_color([0.9, 0.1, 0.1]);
    targetSkullShape.paint_uniform_color([0.9, 0.1, 0.1]);
    ## Reading template features
    print("\t Reading template features ...");
    targetSkullFeatures = read3DPointsFromPPFile(os.path.join(mainFolder, "HN-CHUM-005-SkullMesh_picked_points.pp"));
    if targetSkullMesh is None or targetSkullFeatures is None:
        print("ERROR: Could not read the template skull mesh or features!");
        return;
    print("\t Number of template features:", len(targetSkullFeatures));
    # Visualize the template skull mesh and features
    print("\t Visualizing the template skull mesh and features ...");
    targetSkullMesh.paint_uniform_color([0.7, 0.7, 0.7]);
    targetFeatureSpheres = [];
    for feature in targetSkullFeatures:
        sphere = createSphereAtPoint(feature, radius=0.005);
        targetFeatureSpheres.append(sphere);
    o3d.visualization.draw_geometries([targetSkullMesh] + targetFeatureSpheres);

    # Reading template skull shape and mesh and features
    print("Reading target skull shape and features ...");
    ## Reading template skull mesh and shape
    print("\t Reading target skull mesh and shape ...");
    templateSkullMesh = readO3DMesh(os.path.join(mainFolder, "TempSkullWithParts.ply"));
    templateSkullShape = readO3DMesh(os.path.join(mainFolder, "TempSkullShapeWithParts.ply"));
    ## Compute mesh normals
    print("\t Computing mesh normals ...");
    templateSkullMesh.compute_vertex_normals();
    templateSkullShape.compute_vertex_normals();
    templateSkullMesh.paint_uniform_color([0.7, 0.7, 0.7]);
    templateSkullShape.paint_uniform_color([0.7, 0.7, 0.7]);
    ## Reading template features
    print("\t Reading target features ...");
    templateFeatures = read3DPointsFromPPFile(os.path.join(mainFolder, "TempSkullShapeWithParts_picked_points.pp"));
    if templateSkullMesh is None or templateFeatures is None:
        print("ERROR: Could not read the target skull mesh or features!");
        return;
    print("\t Number of target features:", len(templateFeatures));
    # Visualize the target skull mesh and features
    print("\t Visualizing the target skull mesh and features ...");
    templateFeatureSpheres = [];
    for feature in templateFeatures:
        sphere = createSphereAtPoint(feature, radius=0.005);
        templateFeatureSpheres.append(sphere);
    o3d.visualization.draw_geometries([templateSkullShape] + templateFeatureSpheres);

    # Use rigid and affine transform to align the template skull shape to the target skull mesh
    print("Use rigid and affine transform to align the template skull shape to the target skull mesh ...");
    ## Define buffers
    print("\t Defining buffers ...");
    defSkullShape = copy.deepcopy(templateSkullShape);
    defSkullFeatures = copy.deepcopy(templateFeatures);
    defSkullMesh = copy.deepcopy(templateSkullMesh);
    defSkullShape.compute_vertex_normals();
    defSkullMesh.compute_vertex_normals();
    defSkullMesh.paint_uniform_color([0.7, 0.7, 0.7]);
    ## Estimate the rigid transformation using SVD based on features points
    print("\t Estimating the rigid transformation using SVD based on features points ...");
    transformation = estimateRigidSVDTransform(defSkullFeatures, templateFeatures);
    if transformation is None:
        print("ERROR: Could not estimate the rigid transformation!");
        return;
    print("\t Transformation matrix:\n", transformation);
    ## Transform the template skull shape to the target using the estimated transformation
    print("\t Transforming the template skull shape to the target using the estimated transformation ...");    
    defSkullShape.transform(transformation);
    defSkullMesh.transform(transformation);
    defSkullFeatures = transform3DPoints(defSkullFeatures, transformation);
    ## Estimate the affine transformation using CPD based on features points
    print("\t Estimating the affine transformation using CPD based on features points ...");
    affineTransform = estimateAffineCPDTransform(defSkullFeatures, targetSkullFeatures);
    print("\t Affine transformation matrix:\n", affineTransform);
    ## Transform the deformed skull shape to the target using the estimated affine transformation
    print("\t Transforming the deformed skull shape to the target using the estimated affine transformation ...");
    defSkullShape.transform(affineTransform);
    defSkullMesh.transform(affineTransform);
    defSkullFeatures = transform3DPoints(defSkullFeatures, affineTransform);
    ## Visualize the deformed skull mesh and target skull mesh in the same space
    print("\t Visualizing the deformed skull mesh and target skull mesh in the same space ...");
    o3d.visualization.draw_geometries([defSkullMesh] + [targetSkullMesh]);

    # Deform the deformed skull mesh to the target skull mesh based on skull shape deformation
    print("Deform the deformed skull mesh to the target skull mesh based on skull shape deformation ...");
    ## Personalize the deformed skull shape to the target skull shape using the features
    print("\t Personalizing the deformed skull shape to the target skull shape using the features ...");
    personalizedSkullShape = personalizeSkullShapeFromMeshes(targetSkullShape, targetSkullFeatures, defSkullShape, defSkullFeatures);
    if personalizedSkullShape is None:
        print("ERROR: Could not personalize the skull shape!");
        return;
    personalizedSkullShape.compute_vertex_normals();
    personalizedSkullShape.paint_uniform_color([0.7, 0.7, 0.7]);
    ## Get the deformed skull shape vertices and personalized skull shape vertices
    print("\t Get the deformed skull shape vertices and personalized skull shape vertices ...");
    deformedSkullShapeVertices = np.asarray(defSkullShape.vertices, dtype=np.float32);
    personalizedSkullShapeVertices = np.asarray(personalizedSkullShape.vertices, dtype=np.float32);
    ## Generate selected indices of 1000 indices randomly from 0 to number of vertices
    print("\t Generating selected indices of 1000 indices randomly from 0 to number of vertices ...");
    numOfSampledPoints = 5000;
    if len(deformedSkullShapeVertices) > numOfSampledPoints:
        selectedIndices = np.random.choice(len(deformedSkullShapeVertices) - 1, size=numOfSampledPoints, replace=False);
    else:
        selectedIndices = np.arange(len(deformedSkullShapeVertices));
    ## Deform the deformed skull mesh to the target skull mesh using RBF based on the personalized skull shape
    print("\t Deforming the deformed skull mesh to the target skull mesh using RBF ...");
    personalizedSkullMesh = radialBasisFunctionBasedBlendShapeDeformation(defSkullMesh, 
                                                                          deformedSkullShapeVertices[selectedIndices], 
                                                                          personalizedSkullShapeVertices[selectedIndices]);
    personalizedSkullMesh.compute_vertex_normals();
    personalizedSkullMesh.paint_uniform_color([0.7, 0.7, 0.7]);
    ## Visualize the personalized skull shape with target skull shape with the same space
    print("\t Visualizing the personalized skull shape with target skull shape with the same space ...");
    o3d.visualization.draw_geometries([personalizedSkullMesh] + [targetSkullMesh]);

    # Deform using the skull mesh using non rigid ICP
    print("Deform using the skull mesh using non rigid ICP ...");
    ## Downsample the personalized skull mesh to get sampled features
    print("\t Downsampling the personalized skull mesh to get sampled features ...");
    personalizedSkullMeshFeatures = poissonDiskSamplingOnO3DMesh(personalizedSkullMesh, 2000);
    targetSkullMeshVertices = np.asarray(targetSkullMesh.vertices, dtype=np.float64);
    targetSkullMeshFeatures = estimateNearestPointsKDTreeBased(personalizedSkullMeshFeatures, targetSkullMeshVertices);
    ## Deform the coarse personalized skull mesh to the target using the non rigid icp amberg from trimesh
    print("\t Deforming the coarse personalized skull mesh to the target using the non rigid icp amberg from trimesh ...");
    defPersonalizedSkullMesh, modifiedPersonalizedSkullMesh = nonRigidTransformICPAmberg(personalizedSkullMesh,
                                                                                         personalizedSkullMeshFeatures,
                                                                                         targetSkullMesh,
                                                                                         targetSkullMeshFeatures);
    ## Visualize the deformed personalized skull mesh with target skull mesh with the same space
    print("\t Visualizing the deformed personalized skull mesh with target skull mesh with the same space ...");
    defPersonalizedSkullMesh.compute_vertex_normals();
    defPersonalizedSkullMesh.paint_uniform_color([0.7, 0.7, 0.7]);
    o3d.visualization.draw_geometries([defPersonalizedSkullMesh] + [targetSkullMesh]);

    # Finished processing.
    print("Finished processing.");

#***********************************************************************************************************************************************#
#************************************************************MAIN FUNCTIONS*********************************************************************#
#***********************************************************************************************************************************************#
def main():
    os.system("cls");
    personalizeSkullMesh();
if __name__ == "__main__":
    main()