#******************************************************************************************************************#
#**********************************************PROJECTION INFORMATION**********************************************#
#******************************************************************************************************************#
# This code is part of the book "Human MetaMorphosis" by Dr. Tan-Nhu NGUYEN.
# It is designed to demonstrate the use of point clouds and meshes in 3D graphics.
# The code includes functions to read and write point clouds in various formats,
# generate 3D shapes like spheres, cubes, and cylinders, and convert between different mesh formats.
# The code uses libraries such as Open3D, PyVista, and Trimesh for 3D geometry processing.
# The main functions include reading point clouds from PLY, PCD, and PTS files,
# generating point clouds for spheres and cubes, and saving meshes in formats like PLY, STL, OBJ, and OFF.

#******************************************************************************************************************#
#**********************************************SUPPORTING LIBRARIES************************************************#
#******************************************************************************************************************#
import os;
import numpy as np;
import open3d as o3d;
import pyvista as pv;
import trimesh as tri;

#******************************************************************************************************************#
#**********************************************SUPPORTING BUFFERS**************************************************#
#******************************************************************************************************************#
dataFolder = "../../../Data/Section_1_6_1_PointCloudMeshInputOutput";

#******************************************************************************************************************#
#**********************************************SUPPORTING FUNCTIONS************************************************#
#******************************************************************************************************************#
def generateSpherePoints(inCenter, inRadius, inNumPoints=1000):
    """
    Generates a 3D point cloud on the surface of a sphere using spherical coordinates.

    Parameters:
    - inCenter (list or array): [x, y, z] coordinates of the sphere's center.
    - inRadius (float): Radius of the sphere.
    - inNumPoints (int): Number of points to generate on the sphere surface (default is 1000).

    Returns:
    - points (ndarray): Nx3 array of 3D coordinates on the sphere surface.
    - normals (ndarray): Nx3 array of unit normal vectors pointing outward from the center.

    Method:
    - This function uses inverse transform sampling to generate points uniformly distributed
      over the surface of a sphere. Uniform sampling in spherical coordinates requires:
        * φ (azimuthal angle) to be uniformly sampled from [0, 2π)
        * cos(θ) (cosine of the polar angle) to be uniformly sampled from [-1, 1]
      This ensures that the resulting points are not clustered near the poles, which would
      happen if θ were sampled uniformly instead.

    - The spherical coordinates (θ, φ) are then converted to Cartesian coordinates (x, y, z)
      using the standard transformation:
        x = sin(θ) * cos(φ)
        y = sin(θ) * sin(φ)
        z = cos(θ)

    - These unit vectors represent points on a unit sphere centered at the origin. To scale
      the sphere to the desired radius and move it to the specified center, each point is
      scaled by the radius and translated by the center vector.

    - The same unit vectors are returned as surface normals, since on a perfect sphere,
      the normal at each point is the vector from the center to that point, normalized.
    """

    # Generate uniform spherical coordinates
    phi = np.random.uniform(0, 2 * np.pi, inNumPoints)        # azimuthal angle
    cosTheta = np.random.uniform(-1, 1, inNumPoints)         # cos of polar angle
    theta = np.arccos(cosTheta)                             # polar angle

    # Convert spherical to Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Stack into Nx3 array of unit vectors
    normals = np.stack((x, y, z), axis=1)

    # Scale by radius and shift by center
    points = inRadius * normals + np.array(inCenter)

    return points, normals
def generateCubeSurfacePoints(inCubeCenter, inCubeSideLength, inNumPoints=1000):
    """
    Generates 3D points uniformly distributed on the surface of a cube.

    Parameters:
    - inCubeCenter (list or array): [x, y, z] coordinates of the cube center.
    - inCubeSideLength (float): Length of each side of the cube.
    - inNumPoints (int): Total number of points to generate on the cube surface.

    Returns:
    - points (ndarray): Nx3 array of 3D points on the cube surface.
    """

    # Compute half the side length to define face boundaries relative to the center
    half = inCubeSideLength / 2.0
    inCubeCenter = np.array(inCubeCenter)

    # Divide the total number of points evenly across the 6 cube faces
    points_per_face = inNumPoints // 6
    remainder = inNumPoints % 6  # Handle leftover points by distributing them one per face

    # Helper function to generate points on a single square face of the cube
    def sampleFace(inAxis, inValue, inSize, n):
        """
        Generates `n` points on a square face of the cube.

        Parameters:
        - inAxis (int): The axis normal to the face (0 for x, 1 for y, 2 for z).
        - inValue (float): The fixed coordinate value along the face's normal axis.
        - inSize (float): The length of the face (equal to side_length).
        - n (int): Number of points to generate on this face.

        Returns:
        - points (ndarray): n x 3 array of points on the specified face.
        """

        # Generate 2D coordinates uniformly within the square face
        coords = np.random.uniform(-inSize / 2, inSize / 2, size=(n, 2))
        points = np.zeros((n, 3))

        # Assign the fixed coordinate along the face's normal axis
        # and fill the other two coordinates with the sampled values
        if inAxis == 0:  # x = constant (left or right face)
            points[:, 0] = inValue
            points[:, 1:] = coords
        elif inAxis == 1:  # y = constant (front or back face)
            points[:, 1] = inValue
            points[:, [0, 2]] = coords
        else:  # z = constant (top or bottom face)
            points[:, 2] = inValue
            points[:, :2] = coords

        return points

    # Generate points for each of the 6 cube faces
    faces = []
    for axis in range(3):  # Loop over x, y, z axes
        for sign in [-1, 1]:  # For each axis, generate both negative and positive face
            # Distribute any leftover points (from integer division) across the first few faces
            n = points_per_face + (1 if remainder > 0 else 0)
            remainder -= 1
            # Sample points on the current face and add to the list
            face = sampleFace(axis, sign * half, inCubeSideLength, n)
            faces.append(face)

    # Combine all face points into a single array and shift them to the cube's center
    points = np.vstack(faces) + inCubeCenter
    return points
def read3DPointsFromCSVFile(inFilePath, delimiter=","):
    """
    Reads 3D point data from a CSV file without a header.

    Parameters:
    - filename (str): Path to the CSV file.
    - delimiter (str): Delimiter used in the file (default is comma).

    Returns:
    - points (ndarray): Nx3 NumPy array of 3D points.
    """
    points = np.loadtxt(inFilePath, delimiter=delimiter)
    return points
def generateSphereMesh(inCenter=(0.0, 0.0, 0.0), inRadius=1.0, inResolution=20):
    """
    Generate a 3D sphere mesh using Open3D with specified center, radius, and resolution.

    Parameters:
        inCenter (tuple of float): (x, y, z) coordinates of the sphere center.
        inRadius (float): Radius of the sphere.
        inResolution (int): Mesh resolution (higher = smoother sphere).

    Returns:
        o3d.geometry.TriangleMesh: The generated sphere mesh.
    """
    # Create the sphere at the origin
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=inRadius, resolution=inResolution)
    
    # Compute normals for shading and rendering
    sphere.compute_vertex_normals()
    
    # Translate the sphere to the desired center
    sphere.translate(inCenter)

    return sphere
def generateCubeMesh(inCenter=(0, 0, 0), inSideLength=1.0):
    """
    Create a cube mesh in PyVista given the center and side length.

    Parameters:
        center (tuple): The (x, y, z) coordinates of the cube center.
        side_length (float): The length of each side of the cube.

    Returns:
        pv.PolyData: The generated cube surface mesh.
    """
    half = inSideLength / 2.0
    bounds = (
        inCenter[0] - half, inCenter[0] + half,
        inCenter[1] - half, inCenter[1] + half,
        inCenter[2] - half, inCenter[2] + half
    )
    return pv.Cube(bounds=bounds)
def generateCylinderMesh(inRadius=1.0, inHeight=2.0, inSections=32):
    """
    Create a cylinder mesh using trimesh.

    Parameters:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder along the Z-axis.
        sections (int): Number of segments around the circumference.

    Returns:
        trimesh.Trimesh: The generated cylinder mesh.
    """
    cylinder = tri.creation.cylinder(radius=inRadius, height=inHeight, sections=inSections)
    return cylinder
def open3dToTriMesh(inO3DMesh):
    """
    Convert an Open3D TriangleMesh to a Trimesh mesh.

    Parameters:
        inO3DMesh (o3d.geometry.TriangleMesh): The Open3D mesh to convert.

    Returns:
        trimesh.Trimesh: The converted Trimesh mesh.
    """
    vertices = np.asarray(inO3DMesh.vertices)
    faces = np.asarray(inO3DMesh.triangles)

    # Optional: include vertex normals or colors if available
    vertex_normals = np.asarray(inO3DMesh.vertex_normals) if inO3DMesh.has_vertex_normals() else None
    vertex_colors = np.asarray(inO3DMesh.vertex_colors) if inO3DMesh.has_vertex_colors() else None

    # Create the Trimesh object
    mesh = tri.Trimesh(vertices=vertices, faces=faces,
                           vertex_normals=vertex_normals,
                           vertex_colors=vertex_colors,
                           process=False)  # Avoid auto-cleanup
    return mesh

#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def read3DPointCloud_fromPLYFile():
    """
    Reads a 3D point cloud from a PLY file using Open3D and prints its structure.

    What is a PLY file?
    -------------------
    A PLY (Polygon File Format) file is a common format for storing 3D data such as point clouds and meshes.
    It consists of two main parts:

    1. Header:
       - Starts with the word 'ply' and specifies the format (e.g., 'ascii 1.0' or 'binary_little_endian 1.0').
       - Declares the elements (like 'vertex' or 'face') and their properties.
       - For a point cloud, the header might look like this:

         ply
         format ascii 1.0
         element vertex 10000
         property float x
         property float y
         property float z
         property uchar red
         property uchar green
         property uchar blue
         property float nx
         property float ny
         property float nz
         end_header

       This means the file contains 10,000 vertices, each with:
       - 3D coordinates (x, y, z)
       - RGB color values (red, green, blue)
       - Surface normals (nx, ny, nz)

    2. Data:
       - Follows the header.
       - Each line (in ASCII format) or binary block (in binary format) contains the values for one point.
       - Example line in ASCII:
         0.1 0.2 0.3 255 200 180 0.0 0.0 1.0

    What this function does:
    -------------------------
    1. Constructs the file path to the PLY file.
    2. Uses Open3D to read the point cloud, which automatically parses the header and loads the data.
    3. Converts the point, color, and normal data to NumPy arrays.
    4. Prints the shape and the first few entries of each attribute for inspection.
    """

    # Initializing
    print("Initializing ...");
    pointCloudFilePath = dataFolder + "/FemaleHeadPointCloud.ply";

    # Reading the point cloud
    print("Reading the point cloud ...");
    pointCloud = o3d.io.read_point_cloud(pointCloudFilePath);

    # Print point cloud information
    print("Print point cloud information ...");
    points = np.asarray(pointCloud.points);
    colors = np.asarray(pointCloud.colors);
    normals = np.asarray(pointCloud.normals);
    print("\t Point shape: ", points.shape);
    print("\t Color shape: ", colors.shape);
    print("\t Normals shape: ", normals.shape);
    print("\t First three points: \n", points[:3, :]);
    print("\t First three colors: \n", colors[:3, :]);
    print("\t First three normals: \n", normals[:3, :]);

    # Finished processing
    print("Finished processing.");
def read3DPointCloud_fromPCDFile():
    """
    Reads a 3D point cloud from a PCD file using Open3D and prints its structure.

    What is a PCD file?
    -------------------
    A PCD (Point Cloud Data) file is a format developed by the Point Cloud Library (PCL)
    for storing 3D point cloud data. It is designed to be simple, efficient, and flexible,
    supporting both ASCII and binary encodings.

    Structure of a PCD file:
    ------------------------
    1. Header (always in ASCII):
       - Describes the structure and content of the point cloud.
       - Example:
         VERSION .7
         FIELDS x y z rgb
         SIZE 4 4 4 4
         TYPE F F F F
         COUNT 1 1 1 1
         WIDTH 10000
         HEIGHT 1
         VIEWPOINT 0 0 0 1 0 0 0
         POINTS 10000
         DATA ascii

       Explanation:
       - FIELDS: names of each attribute per point (e.g., x, y, z, rgb)
       - SIZE: number of bytes per field
       - TYPE: data type (F = float, U = unsigned int, I = signed int)
       - COUNT: number of values per field (usually 1)
       - WIDTH × HEIGHT = total number of points
       - VIEWPOINT: sensor origin and orientation
       - DATA: format of the data section (ascii, binary, or binary_compressed)

    2. Data Section:
       - Follows the header.
       - Contains the actual point data in the format specified.
       - Example (ASCII):
         0.1 0.2 0.3 4.2108e+06
         0.4 0.5 0.6 4.2108e+06
         ...

       - If 'rgb' is included as a single float, it encodes packed RGB values.

    What this function does:
    -------------------------
    1. Constructs the file path to the PCD file.
    2. Uses Open3D to read the point cloud, which parses the header and loads the data.
    3. Converts the point, color, and normal data to NumPy arrays.
    4. Prints the shape and the first few entries of each attribute for inspection.
    """
    # Initializing
    print("Initializing ...");
    pointCloudFilePath = dataFolder + "/FemaleHeadPointCloud.pcd";

    # Reading the point cloud
    print("Reading the point cloud ...");
    pointCloud = o3d.io.read_point_cloud(pointCloudFilePath);

    # Print point cloud information
    print("Print point cloud information ...");
    points = np.asarray(pointCloud.points);
    colors = np.asarray(pointCloud.colors);
    normals = np.asarray(pointCloud.normals);
    print("\t Point shape: ", points.shape);
    print("\t Color shape: ", colors.shape);
    print("\t Normals shape: ", normals.shape);
    print("\t First three points: \n", points[:3, :]);
    print("\t First three colors: \n", colors[:3, :]);
    print("\t First three normals: \n", normals[:3, :]);

    # Finished processing
    print("Finished processing.");
def read3DPointCloud_fromPTSFile():
    """
    Reads a 3D point cloud from a PTS file using Open3D and prints its structure.

    What is a PTS file?
    -------------------
    A PTS (Point Text Scan) file is a plain text format commonly used to store point cloud data
    collected from 3D scanners such as LiDAR. It is simple and human-readable.

    Structure of a PTS file:
    ------------------------
    1. Header:
       - The first line often contains the number of points (optional in some cases).

    2. Data:
       - Each subsequent line represents one point.
       - Common formats include:
         x y z
         x y z r g b
         x y z intensity r g b
         x y z r g b intensity
       - Some tools may include normals (nx ny nz), but this is not standard and not reliably supported.

    Important Notes:
    ----------------
    - PTS files do not have a formal specification, so the structure can vary.
    - Open3D expects at least x, y, z coordinates.
    - If RGB values are present, Open3D will attempt to parse them.
    - Normals are not typically included or parsed from PTS files by Open3D.

    What this function does:
    -------------------------
    1. Constructs the file path to the PTS file.
    2. Uses Open3D to read the point cloud.
    3. Converts the point and color data to NumPy arrays.
    4. Attempts to access normals (will be empty if not present).
    5. Prints the shape and the first few entries of each attribute.
    """

    # Initializing
    print("Initializing ...");
    pointCloudFilePath = dataFolder + "/FemaleHeadPointCloud.pts";

    # Reading the point cloud
    print("Reading the point cloud ...");
    pointCloud = o3d.io.read_point_cloud(pointCloudFilePath);

    # Print point cloud information
    print("Print point cloud information ...");
    points = np.asarray(pointCloud.points);
    colors = np.asarray(pointCloud.colors);
    normals = np.asarray(pointCloud.normals);
    print("\t Point shape: ", points.shape);
    print("\t Color shape: ", colors.shape);
    print("\t Normals shape: ", normals.shape);
    print("\t First three points: \n", points[:3, :]);
    print("\t First three colors: \n", colors[:3, :]);
    print("\t First three normals: \n", normals[:3, :]);

    # Finished processing
    print("Finished processing.");

def write3DSpherePointCloud_toPLYFile():
    """
    Generates a 3D point cloud of a sphere, assigns normals and red color to each point,
    and saves the result to a PLY file using Open3D.

    Overview:
    ---------
    This function performs the following steps:
    1. Generates 3D points and normals on the surface of a sphere using geometric math.
    2. Converts the points and normals into an Open3D PointCloud object.
    3. Assigns a uniform red color to all points.
    4. Prints diagnostic information about the point cloud.
    5. Saves the point cloud to a PLY file in ASCII format.

    Details:
    --------
    - The sphere is centered at the origin [0, 0, 0] with a radius of 0.5.
    - 1000 points are generated uniformly on the sphere surface using spherical coordinates.
    - Normals are unit vectors pointing outward from the center (same as the direction of each point).
    - The point cloud is colored uniformly red ([1.0, 0.0, 0.0]).
    - The output file is saved to 'SpherePointCloud.ply' in the specified data folder.
    - The PLY file includes position, normal, and color attributes for each point.
    """

    # Initializing
    print("Initializing ...");

    # Create sphere mesh using open3d
    print("Creating sphere mesh using open3d ...");
    sphereCenter = np.array([0.0, 0.0, 0.0]);
    sphereRadius = 0.5;
    numOfPoints = 1000;
    spherePoints, sphereNormals = generateSpherePoints(sphereCenter, sphereRadius, numOfPoints);

    # Form the open3d point cloud
    print("Forming the open3d point cloud ...");
    spherePointCloud = o3d.geometry.PointCloud();
    spherePointCloud.points = o3d.utility.Vector3dVector(spherePoints);
    spherePointCloud.normals = o3d.utility.Vector3dVector(sphereNormals);

    # Paint the red color to the point cloud
    print("Painting the red color to the point cloud ...");
    sphereColor = np.array([1.0, 0.0, 0.0]);
    spherePointCloud.paint_uniform_color(sphereColor);

    # Checking the point cloud elements
    print("Checking the point cloud elements ...");
    print("\t Dimension of Points: ", np.asarray(spherePointCloud.points).shape);
    print("\t Dimension of Normals: ", np.asarray(spherePointCloud.normals).shape);
    print("\t Dimension of Colors: ", np.asarray(spherePointCloud.colors).shape);
    print("\t First three rows of Points: \n", np.asarray(spherePointCloud.points)[:3, :]);
    print("\t First three rows of Normals: \n", np.asarray(spherePointCloud.normals)[:3, :]);
    print("\t First three rows of Colors: \n", np.asarray(spherePointCloud.colors)[:3, :]);

    # Save the point cloud to ply file
    print("Saving the point cloud to ply file ...");
    o3d.io.write_point_cloud(dataFolder + "/SpherePointCloud.ply", spherePointCloud, write_ascii=True);

    # Finish processing
    print("Finished processing.");
def write3DCubePointCloud_toPCDFile():
    """
    Generates a 3D point cloud representing the surface of a cube, estimates and orients normals,
    assigns a uniform green color, and saves the result to a PCD file using Open3D.

    Overview:
    ---------
    This function performs the following steps:
    1. Generates 3D points on the surface of a cube using geometric math.
    2. Converts the points into an Open3D PointCloud object.
    3. Estimates normals based on local neighborhoods.
    4. Orients normals to face away from the origin and then inverts them.
    5. Assigns a uniform green color to all points.
    6. Prints diagnostic information about the point cloud.
    7. Saves the point cloud to a PCD file in ASCII format.

    Details:
    --------
    - The cube is centered at the origin [0, 0, 0] with a side length of 1.0 (radius = 0.5).
    - 1000 points are generated uniformly across the 6 faces of the cube.
    - Normals are estimated using a KD-tree search with a radius of 0.1 and up to 30 neighbors.
    - Normals are initially oriented to face outward from the origin, then flipped inward.
    - The point cloud is painted uniformly green ([0.0, 1.0, 0.0]).
    - The output file is saved as 'CubePointCloud.pcd' in the specified data folder.
    """

    # Initializing
    print("Initializing ...");

    # Create sphere mesh using open3d
    print("Creating sphere mesh using open3d ...");
    cubeCenter = np.array([0.0, 0.0, 0.0]);
    cubeRadius = 0.5;
    numOfPoints = 1000;
    cubePoints = generateCubeSurfacePoints(cubeCenter, cubeRadius, numOfPoints);

    # Form the open3d point cloud
    print("Forming the open3d point cloud ...");
    cubePointCloud = o3d.geometry.PointCloud();
    cubePointCloud.points = o3d.utility.Vector3dVector(cubePoints);
    cubePointCloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    cubePointCloud.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
    cubeNormals = np.asarray(cubePointCloud.normals);
    cubePointCloud.normals = o3d.utility.Vector3dVector(-cubeNormals)

    # Paint the red color to the point cloud
    print("Painting the red color to the point cloud ...");
    cubeColor = np.array([0.0, 1.0, 0.0]);
    cubePointCloud.paint_uniform_color(cubeColor);

    # Checking the point cloud elements
    print("Checking the point cloud elements ...");
    print("\t Dimension of Points: ", np.asarray(cubePointCloud.points).shape);
    print("\t Dimension of Normals: ", np.asarray(cubePointCloud.normals).shape);
    print("\t Dimension of Colors: ", np.asarray(cubePointCloud.colors).shape);
    print("\t First three rows of Points: \n", np.asarray(cubePointCloud.points)[:3, :]);
    print("\t First three rows of Normals: \n", np.asarray(cubePointCloud.normals)[:3, :]);
    print("\t First three rows of Colors: \n", np.asarray(cubePointCloud.colors)[:3, :]);

    # Save the point cloud to ply file
    print("Saving the point cloud to ply file ...");
    o3d.io.write_point_cloud(dataFolder + "/CubePointCloud.pcd", cubePointCloud, write_ascii=True);

    # Finish processing
    print("Finished processing.");
def write3DCylinderPointCloud_toPTSFile():
    """
    Reads 3D cylinder point data from a CSV file, constructs an Open3D point cloud,
    estimates and orients normals, assigns a red color to all points, and saves the
    result to a PTS file.

    Overview:
    ---------
    This function performs the following steps:
    1. Loads 3D point data from a CSV file (assumed to contain x, y, z coordinates).
    2. Converts the points into an Open3D PointCloud object.
    3. Estimates surface normals using local neighborhood analysis.
    4. Orients normals to face outward from the origin, then inverts them.
    5. Assigns a uniform red color to all points.
    6. Prints diagnostic information about the point cloud.
    7. Saves the point cloud to a PTS file in ASCII format.

    Notes:
    ------
    - The input CSV file should contain only 3 columns (x, y, z) without a header.
    - The PTS format supports positions and colors, but not normals.
    - Normals are still computed for visualization or further processing, but will not be saved in the PTS file.
    """

    # Initializing
    print("Initializing ...");

    # Read cylinder points from csv file
    print("Reading cylinder points from csv file ...");
    cylinderPoints = read3DPointsFromCSVFile(dataFolder + "/Cylinder3DPoints.csv");
    
    # Form the open3d point cloud
    print("Forming the open3d point cloud ...");
    cylinderPointCloud = o3d.geometry.PointCloud();
    cylinderPointCloud.points = o3d.utility.Vector3dVector(cylinderPoints);
    cylinderPointCloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    cylinderPointCloud.orient_normals_towards_camera_location(camera_location=[0, 0, 0])
    cubeNormals = np.asarray(cylinderPointCloud.normals);
    cylinderPointCloud.normals = o3d.utility.Vector3dVector(-cubeNormals)

    # Paint the red color to the point cloud
    print("Painting the red color to the point cloud ...");
    sphereColor = np.array([1.0, 0.0, 0.0]);
    cylinderPointCloud.paint_uniform_color(sphereColor);

    # Checking the point cloud elements
    print("Checking the point cloud elements ...");
    print("\t Dimension of Points: ", np.asarray(cylinderPointCloud.points).shape);
    print("\t Dimension of Normals: ", np.asarray(cylinderPointCloud.normals).shape);
    print("\t Dimension of Colors: ", np.asarray(cylinderPointCloud.colors).shape);
    print("\t First three rows of Points: \n", np.asarray(cylinderPointCloud.points)[:3, :]);
    print("\t First three rows of Normals: \n", np.asarray(cylinderPointCloud.normals)[:3, :]);
    print("\t First three rows of Colors: \n", np.asarray(cylinderPointCloud.colors)[:3, :]);

    # Save the point cloud to ply file
    print("Saving the point cloud to ply file ...");
    o3d.io.write_point_cloud(dataFolder + "/CylinderPointCloud.ply", cylinderPointCloud, write_ascii=True);

    # Finish processing
    print("Finished processing.");

def read3DTriangleSurfaceMesh_fromPLYFile():
    """
    Reads a 3D triangle surface mesh from a PLY file using Open3D.

    This function loads a triangle mesh from a predefined file path,
    prints the shapes of its vertex and face arrays, checks for the
    presence of vertex and triangle normals, computes them if missing,
    and displays updated mesh information.

    Note:
        - The variable `dataFolder` must be defined globally or before calling this function.
        - The mesh file is expected to be named 'FemaleHeadMesh_TriMesh.ply' and located in `dataFolder`.

    Returns:
        None
    """

    # Initialize
    print("Initializing ...");
    triangleMeshFilePath = dataFolder + "/FemaleHeadMesh_TriMesh.ply";

    # Read the mesh
    print("Reading the mesh ...");
    triangleMesh = o3d.io.read_triangle_mesh(triangleMeshFilePath);

    # Show information from the mesh
    print("Show information from the mesh ...");
    print("\t The memory shape of vertices: ", np.asarray(triangleMesh.vertices).shape);
    print("\t The memory shape of faces: ", np.asarray(triangleMesh.triangles).shape);
    print("\t The memory shape vertex normals: ", np.asarray(triangleMesh.vertex_normals).shape);
    print("\t The memory shape of triangle normals: ", np.asarray(triangleMesh.triangle_normals).shape);

    # Compute mesh normals
    print("Computing vertex normals and triangle normals ...");
    if (not triangleMesh.has_triangle_normals()): triangleMesh.compute_triangle_normals();
    if (not triangleMesh.has_vertex_normals()): triangleMesh.compute_vertex_normals();
    print("\t Mesh information after computing vertex normals and triangle normals: ");
    print("\t The memory shape of vertices: ", np.asarray(triangleMesh.vertices).shape);
    print("\t The memory shape of faces: ", np.asarray(triangleMesh.triangles).shape);
    print("\t The memory shape vertex normals: ", np.asarray(triangleMesh.vertex_normals).shape);
    print("\t The memory shape of triangle normals: ", np.asarray(triangleMesh.triangle_normals).shape);

    # Finished processing.
    print("Finished processing.");
def read3DSurfaceMesh_fromSTLFile():
    """
    Reads a 3D surface triangle mesh from an STL file using Open3D.

    This function loads a triangle mesh from a predefined STL file path,
    prints the shapes of its vertex and face arrays, checks for the presence
    of vertex and triangle normals, computes them if missing, and displays
    updated mesh information.

    Note:
        - The variable `dataFolder` must be defined before calling this function.
        - The STL file is expected to be named 'MaleVsFemalePelvis_TriMesh.stl'
          and located inside the `dataFolder` directory.

    Returns:
        None
    """

    # Initialize
    print("Initializing ...");
    triangleMeshFilePath = dataFolder + "/MaleVsFemalePelvis_TriMesh.stl";

    # Read the mesh
    print("Reading the mesh ...");
    triangleMesh = o3d.io.read_triangle_mesh(triangleMeshFilePath);

    # Show information from the mesh
    print("Show information from the mesh ...");
    print("\t The memory shape of vertices: ", np.asarray(triangleMesh.vertices).shape);
    print("\t The memory shape of faces: ", np.asarray(triangleMesh.triangles).shape);
    print("\t The memory shape vertex normals: ", np.asarray(triangleMesh.vertex_normals).shape);
    print("\t The memory shape of triangle normals: ", np.asarray(triangleMesh.triangle_normals).shape);

    # Compute mesh normals
    print("Computing vertex normals and triangle normals ...");
    if (not triangleMesh.has_triangle_normals()): triangleMesh.compute_triangle_normals();
    if (not triangleMesh.has_vertex_normals()): triangleMesh.compute_vertex_normals();
    print("\t Mesh information after computing vertex normals and triangle normals: ");
    print("\t The memory shape of vertices: ", np.asarray(triangleMesh.vertices).shape);
    print("\t The memory shape of faces: ", np.asarray(triangleMesh.triangles).shape);
    print("\t The memory shape vertex normals: ", np.asarray(triangleMesh.vertex_normals).shape);
    print("\t The memory shape of triangle normals: ", np.asarray(triangleMesh.triangle_normals).shape);

    # Finished processing.
    print("Finished processing.");
def read3DSurfaceMesh_fromOBJFile():
    """
    Reads a 3D surface mesh with quadrilateral elements from an OBJ file using PyVista.

    This function loads a quad-based surface mesh from a predefined OBJ file path,
    prints information about the mesh including the number of vertices, number of cells,
    and number of vertices in the first element (cell), and confirms successful processing.

    Note:
        - The variable `dataFolder` must be defined before calling this function.
        - The OBJ file is expected to be named 'BabyBodyMesh_QuadMesh.obj'
          and located inside the `dataFolder` directory.

    Returns:
        None
    """

    # Initialize
    print("Initializing ...");
    quadMeshFilePath = dataFolder + "/BabyBodyMesh_QuadMesh.obj";

    # Read the quad mesh from the file path
    print("Reading the quad mesh from file ...");
    quadMesh = pv.read(quadMeshFilePath);
    
    # Print quad mesh information
    print("Printing quad mesh information ...");
    print("\t The memory size of vertices: ", quadMesh.points.shape);
    print("\t The number of cells: ", quadMesh.n_cells)
    print("\t The number of element vertices: ", quadMesh.extract_cells([0]).n_points);

    # Finished processing.
    print("Finished processing.");
def read3DSurfaceMesh_fromOFFFile():
    """
    Reads a 3D triangle surface mesh from an OFF file using Trimesh.

    This function loads a triangle mesh from a predefined OFF file path,
    disables automatic processing to preserve the original mesh structure,
    and prints detailed information about the mesh including vertex count,
    face count, and available normals.

    Note:
        - The variable `dataFolder` must be defined before calling this function.
        - The OFF file is expected to be named 'HandMesh_TriMesh.off'
          and located inside the `dataFolder` directory.

    Returns:
        None
    """

    # Initialize
    print("Initializing ...");
    meshFilePath = dataFolder + "/HandMesh_TriMesh.off";

    # Reading mesh
    print("Reading mesh ...");
    mesh = tri.load_mesh(meshFilePath, process=False);

    # Print mesh information
    print("Print mesh information ...");
    print("\t The memory size of mesh vertices: ", mesh.vertices.shape);
    print("\t The memory size of mesh faces: ", mesh.faces.shape);
    print("\t The memory size of vertex normals: ", mesh.vertex_normals.shape);
    print("\t The memory size of facet normals: ", mesh.facets_normal.shape);

    # Finished processing.
    print("Finished processing.");

def write3DTriangleSurfaceMesh_toPLYFile():
    """
    Generates a 3D sphere mesh using Open3D, saves it to a PLY file, and prints mesh statistics.
    This function performs the following steps:
    1. Initializes the process and sets parameters for the sphere mesh.
    2. Generates a sphere mesh with specified center, radius, and resolution.
    3. Prints the number of vertices and faces in the generated mesh.
    4. Saves the mesh to a PLY file using Open3D's write_triangle_mesh function.
    5. Prints a message indicating that processing is complete.
    """
    # Initialize
    print("Initializing ...");

    # Generate sphere mesh using open3d library
    print("Generating sphere mesh using open3d library ...");
    sphereCenter = np.array([0.0, 0.0, 0.0]); sphereRadius = 0.5; sphereResolution = 10;
    sphereMesh = generateSphereMesh(sphereCenter, sphereRadius, sphereResolution);
    print("\t The number of vertices: ", np.asarray(sphereMesh.vertices).shape[0]);
    print("\t The number of faces: ", np.asarray(sphereMesh.triangles).shape[0]);

    # Save sphere mesh to file using open3d
    print("Saving sphere mesh to file using open3d ...");
    o3d.io.write_triangle_mesh(dataFolder + "/GeneratedSphereMesh.ply", sphereMesh);

    # Finished processing
    print("Finished processing.");
def write3DTriangleSurfaceMesh_toSTLFile():
    """
    Generates a 3D cube mesh using the PyVista library, saves it to an STL file, and prints mesh statistics.
    This function performs the following steps:
    1. Initializes the process and sets parameters for the cube mesh.
    2. Generates a cube mesh with specified center and side length using PyVista.
    3. Prints the number of vertices and faces in the generated mesh.
    4. Saves the mesh to an STL file using PyVista's save method.
    5. Prints a message indicating that processing is complete.
    """
    # Initialize
    print("Initializing ...");

    # Generate cube mesh
    print("Generate cube mesh using the pyvista library ...");
    cubeCenter = np.array([0.0, 0.0, 0.0]); cubeSideLength = 0.5;
    cubeMesh = generateCubeMesh(cubeCenter, cubeSideLength);
    print("\t The number of vertices: ", cubeMesh.n_points);
    print("\t The number of faces: ", cubeMesh.n_cells);

    # Save the mesh to stl file using pyvista library
    print("Saving the mesh to stl file using the pyvistar library ...");
    cubeMesh.save(dataFolder + "/GeneratedCubeMesh.stl", cubeMesh);

    # Finished processing
    print("Finished processing.");
def write3DTriangleSurfaceMesh_toOBJFile():
    """
    Generates a 3D cylinder mesh using the trimesh library, saves it to an OBJ file, and prints mesh statistics.
    This function performs the following steps:
    1. Initializes the process and sets parameters for the cylinder mesh.
    2. Generates a cylinder mesh with specified radius, height, and sections using trimesh.
    3. Prints the number of vertices and faces in the generated mesh.
    4. Saves the mesh to an OBJ file using trimesh's export method.
    5. Prints a message indicating that processing is complete.
    """
    # Initialize
    print("Initializing ...");

    # Generate the cylinder mesh using the trimesh
    print("Generate cylinder mesh using the trimesh ...");
    cylinderRadius = 0.5; cylinderHeight = 1.0; cylinderSections = 10;
    cylinderMesh = generateCylinderMesh(cylinderRadius, cylinderHeight, cylinderSections);
    print("\t The number of vertices: ", cylinderMesh.vertices.shape[0]);
    print("\t The number of faces: ", cylinderMesh.faces.shape[0]);

    # Save the cylinder mesh to obj file
    print("Save the cylinder mesh to obj file using trimesh ...");
    cylinderMesh.export(dataFolder + "/GeneratedCylinderMesh.obj");

    # Finished processing
    print("Finished processing.");
def write3DTriangleSurfaceMesh_toOFFFile():
    """
    Generates a 3D sphere mesh using Open3D, converts it to a Trimesh object, and saves it to an OFF file.
    This function performs the following steps:
    1. Initializes the process and sets parameters for the sphere mesh.
    2. Generates a sphere mesh with specified center, radius, and resolution using Open3D.
    3. Converts the Open3D mesh to a Trimesh object for compatibility with
    4. Prints the number of vertices and faces in the Trimesh object.
    5. Saves the Trimesh object to an OFF file using the Trimesh library.
    6. Prints a message indicating that processing is complete.
    """
    # Initialize
    print("Initializing ...");

    # Generate sphere mesh using open3d
    print("Generating the sphere mesh using the open3d ...");
    sphereCenter = np.array([0.0, 0.0, 0.0]); sphereRadius = 0.5; sphereResolution = 10;
    sphereMesh = generateSphereMesh(sphereCenter, sphereRadius, sphereResolution);

    # Convert open3d mesh to trimesh mesh
    print("Converting open3d mesh to trimesh ...");
    trimeshMesh = open3dToTriMesh(sphereMesh);
    print("\t The number of vertices: ", trimeshMesh.vertices.shape[0]);
    print("\t The number of faces: ", trimeshMesh.faces.shape[0]);

    # Save the trimesh mesh to off file using the trimesh library
    print("Saving the trimesh to off file using the trimesh library ...");
    trimeshMesh.export(dataFolder + "/GeneratedSphereMesh.off");

    # Finished processing
    print("Finished processing.");

#******************************************************************************************************************#
#**********************************************MAIN FUNCTION*******************************************************#
#******************************************************************************************************************#
def main():
    os.system("cls");
    write3DTriangleSurfaceMesh_toOFFFile();
if __name__ == "__main__":
    main()