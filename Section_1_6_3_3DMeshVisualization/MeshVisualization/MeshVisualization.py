#******************************************************************************************************************#
#**********************************************PROJECTION INFORMATION**********************************************#
#******************************************************************************************************************#
# This project will show the readers how to visualize a 3D triangle surface mesh using Python with varous libraries
# such as PyMesh, PyMeshLab, and PyVista. The tutorials include how to read and visualize mesh in wireframe, solid, and
# surface modes. It also show how to compute normals and visualize normals with the mesh.

#******************************************************************************************************************#
#**********************************************SUPPORTING LIBRARIES************************************************#
#******************************************************************************************************************#
import os;
import numpy as np;

import open3d as o3d;
import pyvista as pv;
import trimesh;
from PIL import Image;
import matplotlib.pyplot as plt;

#******************************************************************************************************************#
#**********************************************SUPPORTING BUFFERS**************************************************#
#******************************************************************************************************************#
dataFolder = "../../../Data/Section_1_6_3_3DMeshVisualization";

#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def visualizeSurfaceMesh_usingOpen3D():
    # Initialize
    print("Initializing ...");

    # Load the mesh from file
    print("Loading mesh from file ...");
    ## Forming file path
    meshFilePath = os.path.join(dataFolder, "FemaleHeadMesh_TriMesh.ply");
    ## Checking file path
    if not os.path.exists(meshFilePath):
        print(f"Mesh file not found at {meshFilePath}");
        return;
    ## Reading mesh
    mesh = o3d.io.read_triangle_mesh(meshFilePath);

    # Pring mesh information
    print("Mesh information:");
    print(f"Number of vertices: {len(mesh.vertices)}");
    print(f"Number of triangles: {len(mesh.triangles)}");
    print(f"Has vertex normals: {mesh.has_vertex_normals()}");

    # Compute normals if they are not present
    print("Checking if vertex normals are present ...");
    if not mesh.has_vertex_normals():
        print("Computing vertex normals ...");
        mesh.compute_vertex_normals();
        print("Vertex normals computed.");

    # Visualize the mesh
    print("Visualizing the mesh using the surface mode...");
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False, mesh_show_wireframe=False);

    # Visualize the mesh in wireframe mode
    print("Visualizing the mesh in wireframe mode...");
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=False, mesh_show_wireframe=True);

    # Visualize the mesh using vertex only mode, by getting the vertices and visualize them with red color
    print("Visualizing the mesh using vertex only mode...");
    vertices = o3d.geometry.PointCloud();
    vertices.points = mesh.vertices;
    vertices.paint_uniform_color([1, 0, 0]);  # Red color for vertices
    o3d.visualization.draw_geometries([vertices]);

    # Finished processing
    print("Finished processing.");
def computeNormalsForSurfaceMesh_usingOpen3D():
    # Initialize
    print("Initializing ...");

    # Forming sphere mesh with low resolution
    print("Forming sphere mesh with low resolution ...");
    sphereMesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10);

    # Checking if the sphere mesh has vertex normals if not compute them
    print("Checking if the sphere mesh has vertex normals if not compute them ...");
    if not sphereMesh.has_vertex_normals():
        print("\t Sphere mesh does not have vertex normals. Computing vertex normals ...");
        sphereMesh.compute_vertex_normals();
        print("\t Vertex normals computed.");

    # Visualizing the sphere mesh with vertex normals with 640x480 screen size
    print("Visualizing the sphere mesh with vertex normals ...");
    o3d.visualization.draw_geometries([sphereMesh], width=640, height=480);

    # Visualize mesh and normals together
    print("Visualizing the sphere mesh and normals together ...");
    ## Manually create a LineSet to visualize normals for older Open3D versions
    vertices = np.asarray(sphereMesh.vertices);
    normals = np.asarray(sphereMesh.vertex_normals);
    normalScale = 0.2;
    ## Create points for the lines (start = vertex, end = vertex + normal * scale)
    linePoints = [];
    for i in range(len(vertices)):
        linePoints.append(vertices[i]);
        linePoints.append(vertices[i] + normals[i] * normalScale);
    ## Create lines between each pair of points
    lines = [[i, i + 1] for i in range(0, len(linePoints), 2)];
    ## Create the LineSet object
    normalLines = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(linePoints),
        lines=o3d.utility.Vector2iVector(lines),
    );
    ## Color the normals to distinguish them from the mesh
    normalLines.paint_uniform_color([1.0, 0.0, 0.0]); # Red color for normals
    ## Visualize the mesh and the normal lines together
    o3d.visualization.draw_geometries([sphereMesh, normalLines], width=640, height=480);

    # Finished processing
    print("Finished processing.");
def visualizeColorMapOnSurfaceMesh_usingOpen3D():
    # Initialize
    print("Initializing ...");

    # Generate a sphere mesh with higher resolution
    print("Generating a sphere mesh with higher resolution ...");
    ## Generate a sphere mesh with radius 1.0 and resolution 20
    sphereMesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20);
    ## Check if the sphere mesh has vertex normals, if not compute them
    if not sphereMesh.has_vertex_normals():
        sphereMesh.compute_vertex_normals();
    
    # Compute the vertex colors based on the z-coordinate with rainbow colormap
    print("Computing vertex colors based on the z-coordinate with rainbow colormap ...");
    ## Get the z-coordinates of the vertices
    zCoords = np.asarray(sphereMesh.vertices)[:, 2];
    ## Normalize the z-coordinates to the range [0, 1]
    zCoordsNormalized = (zCoords - np.min(zCoords)) / (np.max(zCoords) - np.min(zCoords));
    ## Create a colormap using matplotlib's viridis colormap
    colormap = plt.get_cmap('viridis');
    ## Map the normalized z-coordinates to colors
    vertexColors = colormap(zCoordsNormalized)[:, :3];  # Get RGB values

    # Visualize the sphere mesh with vertex colors
    print("Visualizing the sphere mesh with vertex colors ...");
    ## Set the vertex colors to the mesh
    sphereMesh.vertex_colors = o3d.utility.Vector3dVector(vertexColors);
    ## Generate coordinates system for better visualization
    coordinateFrame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2, origin=[0, 0, 0]);
    ## Visualize the mesh with vertex colors
    o3d.visualization.draw_geometries([sphereMesh, coordinateFrame], height=480, width=640);

    # Finished processing
    print("Finished processing.");
def textureMappingSurfaceMesh_usingPyVista():
    # Initialize
    print("Initializing ...");

    # Create a sphere mesh
    print("Creating a sphere mesh ...");
    sphere = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60);

    # Generate UV texture coordinates
    print("Generating UV texture coordinates ...");
    sphere.texture_map_to_sphere(inplace=True);

    # Load texture image from file
    print("Loading texture image from file ...");
    textureImageFilePath = os.path.join(dataFolder, "ChessBoardImage.jpg");
    if not os.path.exists(textureImageFilePath):
        print(f"\t Texture image not found at {textureImageFilePath}");
        return
    texture = pv.read_texture(textureImageFilePath);

    # Plot the textured sphere
    print("Plotting the textured sphere ...");
    plotter = pv.Plotter();
    plotter.add_mesh(sphere, texture=texture);
    plotter.window_size = [640, 480];
    plotter.set_background("white");
    plotter.show();

    # Finished processing
    print("Finished processing.");

#******************************************************************************************************************#
#**********************************************MAIN FUNCTION*******************************************************#
#******************************************************************************************************************#
def main():
    os.system("cls");
    textureMappingSurfaceMesh_usingPyVista();
if __name__ == "__main__":
    main()