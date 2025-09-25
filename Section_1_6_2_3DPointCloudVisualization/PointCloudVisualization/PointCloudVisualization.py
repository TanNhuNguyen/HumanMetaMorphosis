#******************************************************************************************************************#
#**********************************************PROJECTION INFORMATION**********************************************#
#******************************************************************************************************************#
# This project focus on visualizing 3-D point clouds using various libraries in Python. Various types of point cloud
# can be visualized, including black and white point clouds, colored point clouds, and point clouds with normals.
# The project will demonstrate how to load, visualize, and manipulate point clouds using libraries such as Open3D, 
# PyVista, and Trimesh. The goal is to provide a comprehensive understanding of how to work with 3-D point clouds in 
# Python, including loading data from files, visualizing point clouds, and applying transformations or filters to 
# the point clouds.

#******************************************************************************************************************#
#**********************************************SUPPORTING LIBRARIES************************************************#
#******************************************************************************************************************#
import os;
import numpy as np;
import open3d as o3d;
import pyvista as pv;
import trimesh as tri;
import vtk;

#******************************************************************************************************************#
#**********************************************SUPPORTING BUFFERS**************************************************#
#******************************************************************************************************************#
dataFolder = "../../../Data/Section_1_6_2_3DPointCloudVisualization";

#******************************************************************************************************************#
#**********************************************SUPPORTING FUNCTIONS************************************************#
#******************************************************************************************************************#

#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def loadAndVisualizeAPointCloudFromFile_usingOpen3D():
    """
    Load and visualize a 3D point cloud from a file using Open3D.
    This function demonstrates how to load a point cloud from a file, visualize it, and print its properties.
    It uses Open3D library to handle the point cloud data and visualization. 
    The point cloud is expected to be in PLY format, and the function will check if the file exists before attempting to load it.
    The function will also print various properties of the point cloud, such as the number of points, bounding boxes, and colors.
    If the point cloud is loaded successfully, it will be visualized in a window using Open3D's visualization tools.
    If the file does not exist or the point cloud is empty, appropriate error messages will be printed to the console.
    """
    # Initialize
    print("Initializing ...");

    # Load point cloud from file
    print("Loading point cloud from file ...");
    ## Define the file path
    pointCloudFilePath = os.path.join(dataFolder, "BrainPointCloud.ply");
    ## Check if the file exists
    if not os.path.exists(pointCloudFilePath):
        print(f"Error: The file {pointCloudFilePath} does not exist. Please check the file path.");
        return;
    ## Load the point cloud
    pointCloud = o3d.io.read_point_cloud(pointCloudFilePath);
    ## Check if the point cloud is loaded successfully
    if pointCloud.is_empty():
        print("Error: Point cloud is empty. Please check the file path and format.");
        return;
    print("Point cloud loaded successfully.");

    # Print the point cloud information
    print("Point cloud information:");
    print(f"Number of points: {len(pointCloud.points)}");
    print(f"Number of colors: {len(pointCloud.colors)}");
    print(f"Number of normals: {len(pointCloud.normals)}");
    print(f"Point cloud bounding box: {pointCloud.get_axis_aligned_bounding_box()}");
    print(f"Point cloud bounding box (oriented): {pointCloud.get_oriented_bounding_box()}");
    print(f"Point cloud has colors: {'Yes' if pointCloud.has_colors() else 'No'}");
    print(f"Point cloud has normals: {'Yes' if pointCloud.has_normals() else 'No'}");

    # Visualize the point cloud
    print("Visualizing point cloud ...");
    o3d.visualization.draw_geometries([pointCloud], window_name="Point Cloud Visualization");
    print("\t Point cloud visualization completed.");

    # Finished processing
    print("Finished processing.");
def loadAndVisualizeAPointCloudFromFile_usingPyVista():
    """
    Load and visualize a 3D point cloud from a file using PyVista.
    This function demonstrates how to load a point cloud from a file, visualize it, and print its properties.
    It uses PyVista library to handle the point cloud data and visualization.
    The point cloud is expected to be in PLY format, and the function will check if the file exists before attempting to load it.
    The function will also print various properties of the point cloud, such as the number of points, bounding boxes, and colors.
    If the point cloud is loaded successfully, it will be visualized in a window using PyVista's visualization tools.
    If the file does not exist or the point cloud is empty, appropriate error messages will be printed to the console.
    """
    # Initialize
    print("Initializing ...");

    # Load point cloud from file
    print("Loading point cloud from file ...");
    ## Define the file path
    pointCloudFilePath = os.path.join(dataFolder, "LungCTPointCloud.ply");
    ## Check if the file exists
    if not os.path.exists(pointCloudFilePath):
        print(f"Error: The file {pointCloudFilePath} does not exist. Please check the file path.");
        return;
    ## Load the point cloud
    pointCloud = pv.read(pointCloudFilePath);
    ## Check if the point cloud is loaded successfully
    if pointCloud.n_points == 0:
        print("Error: Point cloud is empty. Please check the file path and format.");
        return;
    print("\t Point cloud loaded successfully.");

    # Print the point cloud information
    print("Point cloud information:");
    print(f"\t Number of points: {pointCloud.n_points}");
    print(f"\t Point cloud bounding box: {pointCloud.bounds}");

    # Visualize the point cloud
    print("Visualizing point cloud ...");
    plotter = pv.Plotter();
    plotter.add_mesh(pointCloud, color='red', point_size=5, render_points_as_spheres=True);
    plotter.set_background('white');
    plotter.show();
    print("\t Point cloud visualization completed.");

    # Finished processing
    print("Finished processing.");
def loadAndVisualizeAPointCloudFromFile_usingTrimesh():
    """
    Load and visualize a 3D point cloud from a file using Trimesh.
    This function demonstrates how to load a point cloud from a file, visualize it, and print its properties.
    It uses Trimesh library to handle the point cloud data and visualization. 
    The point cloud is expected to be in PLY format, and the function will check if the file exists before attempting to load it.
    The function will also print various properties of the point cloud, such as the number of points, bounding boxes, and colors.
    If the point cloud is loaded successfully, it will be visualized in a window using Trimesh's visualization tools.
    If the file does not exist or the point cloud is empty, appropriate error messages will be printed to the console.
    """
    # Initialize
    print("Initializing ...");

    # Load point cloud from file
    print("Loading point cloud from file ...");
    ## Define the file path
    pointCloudFilePath = os.path.join(dataFolder, "PelvisPointCloud.ply");
    ## Check if the file exists
    if not os.path.exists(pointCloudFilePath):
        print(f"Error: The file {pointCloudFilePath} does not exist. Please check the file path.");
        return;
    ## Load the point cloud
    pointCloud = tri.load(pointCloudFilePath);
    ## Check if the point cloud is loaded successfully
    if pointCloud.is_empty:
        print("Error: Point cloud is empty. Please check the file path and format.");
        return;
    print("\t Point cloud loaded successfully.");

    # Print the point cloud information
    print("Point cloud information:");
    print(f"\t Number of points: {len(pointCloud.vertices)}");
    print(f"\t Point cloud bounding box: {pointCloud.bounding_box.bounds}");

    # Visualize the point cloud
    print("Visualizing point cloud ...");
    pointCloud.show();
    print("\t Point cloud visualization completed.");

    # Finished processing
    print("Finished processing.");

def saveAPointCloudToFile_usingOpen3D():
    """
    Save a 3D point cloud to a file using Open3D. This function demonstrates how to generate a simple sphere point cloud,
    and save it to a PLY file using Open3D's I/O capabilities. The point cloud is generated using Open3D's built-in functions,
    and the function will print the file path where the point cloud is saved.
    """
    # Initialize
    print("Initializing ...");

    # Generate sphere point cloud using open3d
    print("Generating sphere point cloud using Open3D ...");
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20);
    spherePointCloud = sphere.sample_points_uniformly(number_of_points=1000);

    # Visualize the sphere point cloud before saving
    print("Visualizing sphere point cloud ...");
    o3d.visualization.draw_geometries([spherePointCloud], window_name="Sphere Point Cloud Visualization");
    print("\t Sphere point cloud visualization completed.");

    # Save the point cloud to a file
    print("Saving point cloud to file ...");
    pointCloudFilePath = os.path.join(dataFolder, "SpherePointCloud.ply");
    o3d.io.write_point_cloud(pointCloudFilePath, spherePointCloud);
    print(f"\t Point cloud saved to {pointCloudFilePath}.");

    # Finished processing
    print("Finished processing.");
def saveAPointCloudToFile_PyVista():
    """
    Save a 3D point cloud to a file using PyVista. This function demonstrates how to generate a simple sphere point cloud,
    and save it to a PLY file using PyVista's I/O capabilities. The point cloud is generated using PyVista's built-in functions,
    and the function will print the file path where the point cloud is saved.
    """
    # Initialize
    print("Initializing ...");

    # Generate a sphere point cloud using PyVista
    print("Generating sphere point cloud using PyVista ...");
    sphere = pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20);
    spherePointCloud = pv.PolyData(sphere.points);

    # Visualize the sphere point cloud before saving
    print("Visualizing sphere point cloud ...");
    plotter = pv.Plotter();
    plotter.add_mesh(spherePointCloud, color='red', point_size=5, render_points_as_spheres=True);
    plotter.set_background('white');
    plotter.show();

    # Save the point cloud to a file
    print("Saving point cloud to file ...");
    pointCloudFilePath = os.path.join(dataFolder, "SpherePointCloud.ply");
    spherePointCloud.save(pointCloudFilePath);
    print(f"\t Point cloud saved to {pointCloudFilePath}.");

    # Finished processing
    print("Finished processing.");
def saveAPointCloudToFile_usingTrimesh():
    """
    Save a 3D point cloud to a file using Trimesh. This function demonstrates how to generate a simple sphere point cloud,
    and save it to a PLY file using Trimesh's I/O capabilities. The point cloud is generated using Trimesh's built-in functions,
    and the function will print the file path where the point cloud is saved.
    """
    # Initialize
    print("Initializing ...");

    # Generate sphere point cloud using trimesh
    print("Generating sphere point cloud using Trimesh ...");
    sphere = tri.creation.icosphere(subdivisions=2, radius=1.0);
    spherePointCloud = tri.points.PointCloud(sphere.vertices);

    # Visualize the sphere point cloud before saving
    print("Visualizing sphere point cloud ...");
    spherePointCloud.show();

    # Save the point cloud to a file
    print("Saving point cloud to file ...");
    pointCloudFilePath = os.path.join(dataFolder, "SpherePointCloud.ply");
    spherePointCloud.export(pointCloudFilePath);
    print(f"\t Point cloud saved to {pointCloudFilePath}.");

    # Finished processing
    print("Finished processing.");

def addColorsToAPointCloud_usingOpen3D():
    """
    Add colors to a 3D point cloud using Open3D. This function demonstrates how to load a point cloud,
    add random colors to it, and visualize the colored point cloud. The point cloud is expected to be in PLY format,
    and the function will check if the file exists before attempting to load it. It will also print various properties of the point cloud,
    such as the number of points, bounding boxes, and whether it has colors or normals.
    If the point cloud is loaded successfully, it will be visualized in a window using Open3D's visualization tools.
    If the file does not exist or the point cloud is empty, appropriate error messages will be printed to the console.
    The function will also demonstrate how to change the colors of the point cloud to a specific color (red in this case).
    """
    # Initialize
    print("Initializing ...");

    # Load no color point cloud from file
    print("Loading point cloud from file ...");
    ## Define the file path
    pointCloudFilePath = os.path.join(dataFolder, "BruceLeePointCloud.ply");
    ## Check if the file exists
    if not os.path.exists(pointCloudFilePath):
        print(f"Error: The file {pointCloudFilePath} does not exist. Please check the file path.");
        return;
    ## Load the point cloud
    pointCloud = o3d.io.read_point_cloud(pointCloudFilePath);
    ## Check if the point cloud is loaded successfully
    if pointCloud.is_empty():
        print("Error: Point cloud is empty. Please check the file path and format.");
        return;
    print("\t Point cloud loaded successfully.");

    # Print the point cloud information
    print("Point cloud information:");
    print(f"\t Number of points: {len(pointCloud.points)}");
    print(f"\t Point cloud bounding box: {pointCloud.get_axis_aligned_bounding_box()}");
    print(f"\t Point cloud bounding box (oriented): {pointCloud.get_oriented_bounding_box()}");
    print(f"\t Point cloud has colors: {'Yes' if pointCloud.has_colors() else 'No'}");
    print(f"\t Point cloud has normals: {'Yes' if pointCloud.has_normals() else 'No'}");

    # Visualize the no color point cloud
    print("Visualizing point cloud ...");
    o3d.visualization.draw_geometries([pointCloud], window_name="Point Cloud Visualization - No Color");
    print("\t Point cloud visualization completed.");

    # Add colors to the point cloud
    print("Adding colors to the point cloud ...");
    ## Generate random colors for each point
    colors = np.random.rand(len(pointCloud.points), 3);
    pointCloud.colors = o3d.utility.Vector3dVector(colors);

    # Visualize the colored point cloud
    print("Visualizing colored point cloud ...");
    o3d.visualization.draw_geometries([pointCloud], window_name="Point Cloud Visualization - Colored");
    print("\t Point cloud visualization completed.");

    # Change the colors of the point cloud
    print("Changing colors of the point cloud ...");
    ## Change the colors to red
    redColor = np.array([1, 0, 0]);
    pointCloud.colors = o3d.utility.Vector3dVector(np.tile(redColor, (len(pointCloud.points), 1)));

    # Visualize the point cloud with changed colors
    print("Visualizing point cloud with changed colors ...");
    o3d.visualization.draw_geometries([pointCloud], window_name="Point Cloud Visualization - Changed Colors");
    print("\t Point cloud visualization completed.");

    # Finished processing
    print("Finished processing.");
def addColorsToAPointCloud_usingPyVista():
    """
    Add colors to a 3D point cloud using PyVista.
    This function demonstrates how to load a point cloud, add random colors to it, and visualize the colored point cloud.
    The point cloud is expected to be in PLY format, and the function will check if the file exists before attempting to load it.
    It will also print various properties of the point cloud, such as the number of points, bounding boxes, and whether it has colors or normals.
    If the point cloud is loaded successfully, it will be visualized in a window using PyVista's visualization tools.
    If the file does not exist or the point cloud is empty, appropriate error messages will be printed to the console.
    The function will also demonstrate how to change the colors of the point cloud to a specific color (red in this case).  
    """
    # Initialize
    print("Initializing ...");

    # Load no color point cloud from file
    print("Loading point cloud from file ...");
    ## Define the file path
    pointCloudFilePath = os.path.join(dataFolder, "PelvisPointCloud.ply");
    ## Check if the file exists
    if not os.path.exists(pointCloudFilePath):
        print(f"Error: The file {pointCloudFilePath} does not exist. Please check the file path.");
        return;
    ## Load the point cloud
    pointCloud = pv.read(pointCloudFilePath);
    ## Check if the point cloud is loaded successfully
    if pointCloud.n_points == 0:
        print("Error: Point cloud is empty. Please check the file path and format.");
        return;
    print("\t Point cloud loaded successfully.");

    # Print the point cloud information
    print("Point cloud information:");
    print(f"\t Number of points: {pointCloud.n_points}");
    print(f"\t Point cloud bounding box: {pointCloud.bounds}");

    # Visualize the no color point cloud
    print("Visualizing point cloud ...");
    plotter = pv.Plotter();
    plotter.add_mesh(pointCloud, color='blue', point_size=5, render_points_as_spheres=True);
    plotter.set_background('white');
    plotter.show();
    print("\t Point cloud visualization completed.");

    # Add colors to the point cloud
    print("Adding colors to the point cloud ...");
    ## Generate random colors for each point
    colors = np.random.rand(pointCloud.n_points, 3);
    pointCloud.point_data['colors'] = colors;
    ## Set the colors to the point cloud
    pointCloud.set_active_scalars('colors');
    ## Visualize the colored point cloud
    print("Visualizing colored point cloud ...");
    plotter = pv.Plotter();
    plotter.add_mesh(pointCloud, scalars='colors', rgb=True, point_size=5, render_points_as_spheres=True);
    plotter.set_background('white');
    plotter.show();
    print("\t Point cloud visualization completed.");

    # Change the colors of the point cloud
    print("Changing colors of the point cloud ...");
    ## Change the colors to red
    redColor = np.array([[1.0, 0.0, 0.0]]);
    pointCloud.point_data['colors'] = np.tile(redColor, (pointCloud.n_points, 1));
    ## Set the colors to the point cloud
    pointCloud.set_active_scalars('colors');
    ## Visualize the point cloud with changed colors
    print("Visualizing point cloud with changed colors ...");
    plotter = pv.Plotter();
    plotter.add_mesh(pointCloud, scalars='colors', rgb=True, point_size=5, render_points_as_spheres=True);
    plotter.set_background('white');
    plotter.show();
    print("\t Point cloud visualization completed.");

    # Finished processing
    print("Finished processing.");
def addColorsToAPointCloud_usingTrimesh():
    """
    Adds colors to a point cloud using the Trimesh library.
    This function demonstrates how to load a point cloud from a file, add random colors to it, and visualize the colored point cloud.
    The point cloud is expected to be in PLY format, and the function will check if the file exists before attempting to load it.
    It will also print various properties of the point cloud, such as the number of points, bounding boxes, and whether it has colors or normals.
    If the point cloud is loaded successfully, it will be visualized in a window using Trimesh's visualization tools.
    If the file does not exist or the point cloud is empty, appropriate error messages will be printed to the console.
    The function will also demonstrate how to change the colors of the point cloud to a specific color (red in this case).
    """
    # Initialize
    print("Initializing ...");

    # Load no color point cloud from file
    print("Loading point cloud from file ...");
    ## Define the file path
    pointCloudFilePath = os.path.join(dataFolder, "PelvisPointCloud.ply");
    ## Check if the file exists
    if not os.path.exists(pointCloudFilePath):
        print(f"Error: The file {pointCloudFilePath} does not exist. Please check the file path.");
        return;
    ## Load the point cloud
    pointCloud = tri.load(pointCloudFilePath);
    ## Check if the point cloud is loaded successfully
    if pointCloud.is_empty:
        print("Error: Point cloud is empty. Please check the file path and format.");
        return;
    print("\t Point cloud loaded successfully.");

    # Print the point cloud information
    print("Point cloud information:");
    print(f"\t Number of points: {len(pointCloud.vertices)}");
    print(f"\t Point cloud bounding box: {pointCloud.bounding_box.bounds}");

    # Visualize the no color point cloud
    print("Visualizing point cloud ...");
    pointCloud.show();
    print("\t Point cloud visualization completed.");

    # Add colors to the point cloud
    print("Adding colors to the point cloud ...");
    ## Generate random colors for each point
    colors = np.random.rand(len(pointCloud.vertices), 3);
    pointCloud.visual.vertex_colors = colors;
    
    # Visualize the colored point cloud
    print("Visualizing colored point cloud ...");
    pointCloud.show();
    print("\t Point cloud visualization completed.");

    # Change the colors of the point cloud
    print("Changing colors of the point cloud ...");
    ## Change the colors to red
    redColor = np.array([1.0, 0.0, 0.0]);
    pointCloud.visual.vertex_colors = np.tile(redColor, (len(pointCloud.vertices), 1));

    # Visualize the point cloud with changed colors
    print("Visualizing point cloud with changed colors ...");
    pointCloud.show();
    print("\t Point cloud visualization completed.");

    # Finished processing
    print("Finished processing.");

def computeNormalsForAPointCloud_usingOpen3D():
    '''
    Computes normals for a point cloud using Open3D.
    This function demonstrates how to generate a simple sphere point cloud, compute normals for it,
    and visualize the point cloud with and without normals. The point cloud is generated using Open3D's built-in functions,
    and the function will print various properties of the point cloud before and after computing normals. 
    The function will also visualize the point cloud in a window using Open3D's visualization tools.
    The point cloud is expected to be a uniform sampling of a sphere, and the function will print the number of points,
    whether it has normals, and the bounding box of the point cloud.
    The function will also demonstrate how to visualize the point cloud with normals using Open3D's visualization tools.
    The function will print messages to indicate the progress of the operations, including initialization, point cloud
    generation, visualization, and normal computation.
    It will also print the point cloud information before and after computing normals, including the number of
    points, whether it has normals, and the bounding box of the point cloud.
    The function will also visualize the point cloud with normals using Open3D's visualization tools.
    '''
    # Initialize
    print("Initializing ...");

    # Generate a simple sphere without normals
    print("Generating sphere point cloud without normals ...");
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0);
    spherePointCloud = sphere.sample_points_uniformly(number_of_points=1000);
    print("\t Sphere point cloud generated successfully.");

    # Show point cloud information
    print("Point cloud information:");
    print(f"\t Number of points: {len(spherePointCloud.points)}");
    print(f"\t Point cloud has normals: {'Yes' if spherePointCloud.has_normals() else 'No'}");
    print(f"\t Point cloud bounding box: {spherePointCloud.get_axis_aligned_bounding_box()}");

    # Visualize the point cloud without normals
    print("Visualizing point cloud without normals ...");
    o3d.visualization.draw_geometries([spherePointCloud], window_name="Point Cloud Visualization - No Normals");
    print("\t Point cloud visualization completed.");

    # Compute normals for the point cloud
    print("Computing normals for the point cloud ...");
    spherePointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30));
    print("\t Normals computed successfully.");

    # Show point cloud information after computing normals
    print("Point cloud information after computing normals:");
    print(f"\t Number of points: {len(spherePointCloud.points)}");
    print(f"\t Point cloud has normals: {'Yes' if spherePointCloud.has_normals() else 'No'}");
    print(f"\t Point cloud bounding box: {spherePointCloud.get_axis_aligned_bounding_box()}");

    # Visualize the point cloud with normals
    print("Visualizing point cloud with normals ...");
    o3d.visualization.draw_geometries([spherePointCloud], window_name="Point Cloud Visualization - With Normals",
                                       point_show_normal=True);
    print("\t Point cloud visualization completed.");

    # Finished processing
    print("Finished processing.");
def computeNormalsForAPointCloud_usingPyVista():
    """
    Computes normals for a point cloud using PyVista.
    This function demonstrates how to generate a simple sphere point cloud, compute normals for it,
    and visualize the point cloud with and without normals. The point cloud is generated using PyVista's built-in functions,
    and the function will print various properties of the point cloud before and after computing normals.
    The function will also visualize the point cloud in a window using PyVista's visualization tools.
    The point cloud is expected to be a uniform sampling of a sphere, and the function will print the number of points,
    whether it has normals, and the bounding box of the point cloud.
    The function will also demonstrate how to visualize the point cloud with normals using PyVista's visualization tools.
    The function will print messages to indicate the progress of the operations, including initialization, point cloud
    generation, visualization, and normal computation.
    It will also print the point cloud information before and after computing normals, including the number of points,
    whether it has normals, and the bounding box of the point cloud. 
    The function will also visualize the point cloud with normals using PyVista's visualization tools.
    """
    # Initialize
    print("Initializing ...");

    # Generate a simple sphere and extract points only
    print("Generating sphere point cloud without normals ...");
    sphere = pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20);
    spherePointCloud = pv.PolyData(sphere.points);
    print("\t Sphere point cloud generated successfully.");

    # Show point cloud information
    print("Point cloud information:");
    print(f"\t Number of points: {spherePointCloud.n_points}");
    print(f"\t Number of cells: {spherePointCloud.n_cells}");
    has_normals = 'Normals' in spherePointCloud.point_data;
    print(f"\t Point cloud has normals: {'Yes' if has_normals else 'No'}");

    # Visualize the point cloud without normals
    print("Visualizing point cloud without normals ...");
    ## Initialize the plotter
    plotter = pv.Plotter();
    ## Add the point cloud
    plotter.add_mesh(spherePointCloud, color="red", point_size=5, render_points_as_spheres=True);
    ## Set the background color
    plotter.set_background("white");
    ## Show the plotter
    plotter.show();
    print("\t Point cloud visualization completed.");

    # For point clouds without faces, we can estimate normals using the original sphere
    print("Estimating normals for the point cloud ...");
    ## Use the original sphere to compute normals, then transfer to point cloud
    sphere_with_normals = sphere.copy();
    ## Compute normals for the sphere
    sphere_with_normals.compute_normals(inplace=True);
    ## Transfer normals to the point cloud
    spherePointCloud.point_data['Normals'] = sphere_with_normals.point_data['Normals'];
    print("\t Normals estimated successfully.");

    # Show point cloud information after computing normals
    print("Point cloud information after computing normals:");
    print(f"\t Number of points: {spherePointCloud.n_points}");
    has_normals = 'Normals' in spherePointCloud.point_data;
    print(f"\t Point cloud has normals: {'Yes' if has_normals else 'No'}");

    # Visualize the point cloud with normals
    print("Visualizing point cloud with normals ...");
    ## Initialize the plotter
    plotter = pv.Plotter();
    ## Add the point cloud with normals
    plotter.add_mesh(spherePointCloud, scalars="Normals", rgb=True, point_size=5, render_points_as_spheres=True);
    ## Add the normals as arrows
    plotter.add_arrows(spherePointCloud.points, spherePointCloud.point_data['Normals'], mag=0.1, color='blue', opacity=0.5);
    ## Set the background color
    plotter.set_background("white");
    ## Show the plotter
    plotter.show();
    print("\t Point cloud visualization completed.");

    # Finished processing
    print("Finished processing.");
def computeNormalsForAPointCloud_usingTrimesh():
    '''
    Compute normals for a point cloud using the Trimesh library. This function demonstrates how to compute and visualize normals for a point cloud
    generated from a sphere mesh using the Trimesh library.  The function generates a simple sphere point cloud without normals, computes the normals 
    using the original sphere mesh, and visualizes the point cloud with normal vectors as lines. The point cloud is expected to be a uniform sampling 
    of a sphere, and the function will print the number of points, whether it has normals, and the bounding box of the point cloud. 
    The function will also demonstrate how to visualize the point cloud with normals using Trimesh's visualization tools. The function will print messages 
    to indicate the progress of the operations, including initialization, point cloud generation, visualization, and normal computation. It will also print 
    the point cloud information before and after computing normals, including the number of points, whether it has normals, and the bounding box of the point 
    cloud. The function will also visualize the point cloud with normals using Trimesh's visualization tools. The function will also create normal vector lines 
    using Trimesh's Path3D and visualize them in a scene. The function will print messages to indicate the progress of the operations, including initialization, 
    point cloud generation, visualization, and normal computation. The function will also print the point cloud information before and after computing normals, 
    including the number of points, whether it has normals, and the bounding box of the point cloud. The function will also visualize the point cloud with normals
    using Trimesh's visualization tools. The function will also create normal vector lines using Trimesh's Path3D and visualize them in a scene.
    '''
    # Initialize
    print("Initializing ...");

    # Generate a simple sphere point cloud without normals
    print("Generating sphere point cloud without normals ...");
    sphere = tri.creation.icosphere(subdivisions=2, radius=1.0);
    spherePointCloud = tri.points.PointCloud(sphere.vertices);
    print("\t Sphere point cloud generated successfully.");

    # Show point cloud information
    print("Point cloud information:");
    print(f"\t Number of points: {len(spherePointCloud.vertices)}");
    has_normals = hasattr(spherePointCloud, 'vertex_normals') and spherePointCloud.vertex_normals is not None;
    print(f"\t Point cloud has normals: {'Yes' if has_normals else 'No'}");

    # Visualize the point cloud without normals
    print("Visualizing point cloud without normals ...");
    spherePointCloud.show();
    print("\t Point cloud visualization completed.");

    # Compute normals for the point cloud using the original sphere mesh
    print("Computing normals for the point cloud ...");
    ## Since PointCloud doesn't have estimate_normals, use the original sphere's vertex normals
    sphere.vertex_normals  # This computes vertex normals for the mesh
    ## Transfer normals from the original sphere to the point cloud
    spherePointCloud.vertex_normals = sphere.vertex_normals;
    print("\t Normals computed successfully.");

    # Show point cloud information after computing normals
    print("Point cloud information after computing normals:");
    print(f"\t Number of points: {len(spherePointCloud.vertices)}");
    has_normals = hasattr(spherePointCloud, 'vertex_normals') and spherePointCloud.vertex_normals is not None;
    print(f"\t Point cloud has normals: {'Yes' if has_normals else 'No'}");

    # Visualize the point cloud with normal vectors as lines
    print("Visualizing point cloud with normal vectors as lines ...");
    ## Create a new scene for visualization with normal vectors
    scene = tri.Scene();    
    ## Add the point cloud to the scene
    scene.add_geometry(spherePointCloud);    
    ## Create normal vector lines using Path3D correctly
    normal_scale = 0.1;  # Scale factor for normal vector length
    arrow_origins = spherePointCloud.vertices;
    arrow_directions = spherePointCloud.vertex_normals * normal_scale;
    arrow_endpoints = arrow_origins + arrow_directions;    
    ## Create vertices for all line segments
    all_vertices = [];
    line_entities = [];
    vertex_count = 0;    
    for i in range(len(arrow_origins)):
        ## Add start and end points
        all_vertices.extend([arrow_origins[i], arrow_endpoints[i]]);
        ## Create line entity connecting these two points
        line_entities.append(tri.path.entities.Line([vertex_count, vertex_count + 1]));
        vertex_count += 2;    
    ## Create Path3D object with all lines
    normal_lines = tri.path.Path3D(entities=line_entities, vertices=np.array(all_vertices));    
    ## Add normal vector lines to the scene
    scene.add_geometry(normal_lines);    
    ## Show the scene with normal vectors
    scene.show();
    print("\t Point cloud with normal vectors visualization completed.");

    # Finished processing
    print("Finished processing.");

#******************************************************************************************************************#
#**********************************************MAIN FUNCTION*******************************************************#
#******************************************************************************************************************#
def main():
    os.system("cls");
    computeNormalsForAPointCloud_usingPyVista();
if __name__ == "__main__":
    main()