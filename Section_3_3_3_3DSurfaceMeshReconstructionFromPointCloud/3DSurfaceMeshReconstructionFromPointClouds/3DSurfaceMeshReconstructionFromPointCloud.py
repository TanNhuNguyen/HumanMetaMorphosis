#*********************************************************************************************************************#
#***************************************************SUPPORTING LIBRARIES**********************************************#
#*********************************************************************************************************************#
import os;
import numpy as np;
import open3d as o3d;
import trimesh;
from scipy.spatial import Delaunay;
from skimage import measure;
from scipy.spatial import ConvexHull
from trimesh.scene import scene;
import alphashape;

#*********************************************************************************************************************#
#***************************************************SUPPORTING BUFFERS************************************************#
#*********************************************************************************************************************#
mainFolder = "../../../Data/Section_3_3_3_3DSurfaceMeshReconstructionFromPointCloud";

#*********************************************************************************************************************#
#***************************************************SUPPORTING FUNCTIONS**********************************************#
#*********************************************************************************************************************#

#*********************************************************************************************************************#
#***************************************************PROCESSING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def meshReconstructionUsingPoisson():
    # Initialize
    print("Initializing ...");

    # Reading the raw point cloud data
    print("Reading the raw point cloud data ...");
    ## Readding the brain point cloud from file
    brainPointCloud = o3d.io.read_point_cloud(os.path.join(mainFolder, "BrainPointCloud.ply"));
    ## Add color as red color to the point cloud
    brainPointCloud.paint_uniform_color([0.5, 0.0, 0.0]);
    ## Compute normals for the point cloud
    brainPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30));

    # Poisson surface reconstruction
    print("Poisson surface reconstruction ...");
    ## Perform Poisson surface reconstruction
    brainMesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(brainPointCloud, depth=9);
    ## Convert the brain mesh to trimesh to better visualization
    brainMesh.compute_vertex_normals();
    trimeshBrainMesh = trimesh.Trimesh(np.asarray(brainMesh.vertices), np.asarray(brainMesh.triangles));
    ## Set mesh vertex color as the real bone color (white color)
    trimeshBrainMesh.visual.vertex_colors = np.array([0.96, 0.92, 0.86, 1.0]) * 255;
    ## Visualize the reconstructed mesh using trimesh
    trimeshBrainMesh.show();

    # Save the reconstructed mesh to a file
    print("Save the reconstructed mesh to a file ...");
    o3d.io.write_triangle_mesh(os.path.join(mainFolder, "BrainMesh_Poisson.ply"), brainMesh);

    # Finished processing.
    print("Finished processing.");
def meshReconstructionUsingBallPivoting():
    # Initialize
    print("Initializing ...");

    # Reading the raw point cloud data
    print("Reading the raw point cloud data ...");
    ## Readding the brain point cloud from file
    brainPointCloud = o3d.io.read_point_cloud(os.path.join(mainFolder, "BrainPointCloud.ply"));
    ## Add color as red color to the point cloud
    brainPointCloud.paint_uniform_color([0.5, 0.0, 0.0]);
    ## Compute normals for the point cloud
    brainPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30));

    # Ball pivoting surface reconstruction
    print("Ball pivoting surface reconstruction ...");
    ## Define the radii for the ball pivoting algorithm
    distances = brainPointCloud.compute_nearest_neighbor_distance();
    radii = np.mean(distances);
    ## Perform Ball pivoting surface reconstruction
    brainMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        brainPointCloud, o3d.utility.DoubleVector([radii, radii * 2]));
    ## Convert the brain mesh to trimesh to better visualization
    brainMesh.compute_vertex_normals();
    trimeshBrainMesh = trimesh.Trimesh(np.asarray(brainMesh.vertices), np.asarray(brainMesh.triangles));
    ## Set mesh vertex color as the real bone color (white color)
    trimeshBrainMesh.visual.vertex_colors = np.array([0.96, 0.92, 0.86, 1.0]) * 255;
    ## Visualize the reconstructed mesh using trimesh
    trimeshBrainMesh.show();

    # Save the reconstructed mesh to a file using trimesh
    print("Save the reconstructed mesh to a file ...");
    trimeshBrainMesh.export(os.path.join(mainFolder, "BrainMesh_BallPivoting.ply"));

    # Finished processing.
    print("Finished processing.");
def meshReconstructionUsingDelaunayTriangulation():
    # Initialize
    print("Initializing ...");

    # Reading the raw point cloud data
    print("Reading the raw point cloud data ...");
    ## Readding the brain point cloud from file
    brainPointCloud = o3d.io.read_point_cloud(os.path.join(mainFolder, "BrainPointCloud.ply"));
    ## Add color as red color to the point cloud
    brainPointCloud.paint_uniform_color([0.5, 0.0, 0.0]);
    ## Compute normals for the point cloud
    brainPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30));

    # Delauney triangulation surface reconstruction using the scipy.spatial.Delaunay function
    print("Delauney triangulation surface reconstruction ...");
    ## Convert the point cloud to numpy array
    points = np.asarray(brainPointCloud.points);
    ## Perform Delauney triangulation using scipy.spatial.Delaunay
    delaunay = Delaunay(points[:, :2]);
    ## Create the brain mesh using the Delauney triangulation
    trimeshBrainMesh = trimesh.Trimesh(vertices=points, faces=delaunay.simplices);
    ## Set mesh vertex color as the real bone color (white color)
    trimeshBrainMesh.visual.vertex_colors = np.array([0.96, 0.92, 0.86, 1.0]) * 255;
    ## Visualize the reconstructed mesh using trimesh
    trimeshBrainMesh.show();

    # Save the reconstructed mesh to a file using trimesh
    print("Save the reconstructed mesh to a file ...");
    trimeshBrainMesh.export(os.path.join(mainFolder, "BrainMesh_Delaunay.ply"));

    # Finished processing.
    print("Finished processing.");
def meshReconstructionUsingMarchingCubes():
    # Initialize
    print("Initializing ...");

    # Reading the raw point cloud data
    print("Reading the raw point cloud data ...");
    ## Readding the brain point cloud from file
    brainPointCloud = o3d.io.read_point_cloud(os.path.join(mainFolder, "BrainPointCloud.ply"));
    ## Add color as red color to the point cloud
    brainPointCloud.paint_uniform_color([0.5, 0.0, 0.0]);
    ## Compute normals for the point cloud
    brainPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30));

    # Marching cubes surface reconstruction
    print("Marching cubes surface reconstruction ...");
    ## Convert the point cloud to a voxel grid
    voxel_size = 10.0;
    brainVoxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(brainPointCloud, voxel_size)
    voxels = brainVoxelGrid.get_voxels()
    if not voxels:
        print("No voxels found in the grid.")
        return
    voxel_indices = np.array([v.grid_index for v in voxels])
    min_indices = voxel_indices.min(axis=0)
    max_indices = voxel_indices.max(axis=0)
    grid_shape = max_indices - min_indices + 1
    volume = np.zeros(grid_shape, dtype=np.uint8)
    for idx in voxel_indices:
        volume[tuple(idx - min_indices)] = 1
    # Marching cubes surface reconstruction using skimage
    verts, faces, normals, values = measure.marching_cubes(volume, level=0.5)
    # Transform verts back to world coordinates
    verts = verts + min_indices  # shift to original grid
    verts = verts * voxel_size + np.asarray(brainVoxelGrid.origin)
    # Create mesh and visualize
    trimeshBrainMesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimeshBrainMesh.visual.vertex_colors = np.array([0.96, 0.2, 0.2, 1.0]) * 255
    trimeshBrainMesh.show()

    # Save the reconstructed mesh to a file using trimesh
    print("Save the reconstructed mesh to a file ...");
    trimeshBrainMesh.export(os.path.join(mainFolder, "BrainMesh_MarchingCubes.ply"));

    # Finished processing.
    print("Finished processing.");
def meshReconstructionUsingConvexHull():
    # Initialize
    print("Initializing ...");

    # Reading the point cloud data
    print("Reading the point cloud data ...");
    ## Readding the brain point cloud from file

    brainPointCloud = o3d.io.read_point_cloud(os.path.join(mainFolder, "BrainPointCloud.ply"));
    ## Add color as red color to the point cloud
    brainPointCloud.paint_uniform_color([0.5, 0.0, 0.0]);
    ## Compute normals for the point cloud    
    brainPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30));
    points = np.asarray(brainPointCloud.points);

    # Convex hull surface reconstruction using scipy.spatial.ConvexHull
    print("Convex hull surface reconstruction ...");
    ## Perform Convex hull surface reconstruction using scipy.spatial.ConvexHull
    hull = ConvexHull(points);
    ## Convert convex hull to trimesh for better visualization
    trimeshConvexHull = trimesh.Trimesh(vertices=points, faces=hull.simplices);
    ## Set mesh convex hull color as brown red color
    trimeshConvexHull.visual.vertex_colors = np.array([0.65, 0.16, 0.16, 1.0]) * 255;
    ## Fixing the normal for visualization
    trimeshConvexHull.fix_normals();
    ## Visualize the mesh as edge only
    scene = trimesh.Scene(trimeshConvexHull);
    scene.show(wireframe=True);

    # Save the reconstructed mesh to a file using trimesh
    print("Save the reconstructed mesh to a file ...");
    trimeshConvexHull.export(os.path.join(mainFolder, "BrainMesh_ConvexHull.ply"));

    # Finished processing.
    print("Finished processing.");
def meshReconstructionUsingConcaveHull():
    # Initialize
    print("Initializing ...");

    # Load the point cloud for computing the concave hull
    print("Load the point cloud for computing the concave hull ...");
    ## Readding the brain point cloud from file
    print("\t Reading the brain point cloud from file ...");
    brainPointCloud = o3d.io.read_point_cloud(os.path.join(mainFolder, "BrainPointCloud.ply"));
    ## Add color as red color to the point cloud
    print("\t Add color as red color to the point cloud ...");
    brainPointCloud.paint_uniform_color([0.5, 0.0, 0.0]);
    ## Compute normals for the point cloud
    print("\t Compute normals for the point cloud ...");
    brainPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30));

    # Compute the concave hull using alphashape
    print("Compute the concave hull using alphashape ...");
    ## Downsample the point cloud for faster computation
    print("\t Downsample the point cloud for faster computation ...");
    brainPointCloud = brainPointCloud.voxel_down_sample(voxel_size=1.0);
    ## Compute the alphashape
    print("\t Compute the alphashape ...");
    optimalAlpha = 50.0;
    brainAlphaShape = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(brainPointCloud, optimalAlpha);
    ## Convert the brain alpha shape to trimesh for better visualization
    print("\t Convert the brain alpha shape to trimesh for better visualization ...");
    trimeshBrainAlphaShape = trimesh.Trimesh(np.asarray(brainAlphaShape.vertices), np.asarray(brainAlphaShape.triangles));
    trimeshBrainAlphaShape.fix_normals();
    ## Set mesh vertex color brown red color
    print("\t Set mesh vertex color brown red color ...");
    trimeshBrainAlphaShape.visual.vertex_colors = np.array([0.65, 0.16, 0.16, 1.0]) * 255;
    ## Visualize the reconstructed mesh using trimesh
    print("\t Visualize the reconstructed mesh using trimesh ...");
    trimeshBrainAlphaShape.show();

    # Finished processing.
    print("Finished processing.");
def meshReconstructionUsingScreenedPoisson():
    # Initialize
    print("Initializing ...");

    # Finished processing.
    print("Finished processing.");
def meshReconstructionUsingGreedyTriangulation():
    # Initialize
    print("Initializing ...");

    # Reading point cloud for processing
    print("Reading point cloud for processing ...");
    ## Readding the brain point cloud from file
    brainPointCloud = o3d.io.read_point_cloud(os.path.join(mainFolder, "BrainPointCloud.ply"));
    ## Add color as red color to the point cloud
    brainPointCloud.paint_uniform_color([0.5, 0.0, 0.0]);
    ## Compute normals for the point cloud
    brainPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30));

    # Greedy triangulation surface reconstruction
    print("Greedy triangulation surface reconstruction ...");
    ## Compute radius for the greedy triangulation
    print("\t Compute radius for the greedy triangulation ...");
    distances = brainPointCloud.compute_nearest_neighbor_distance();
    avgDistance = np.mean(distances);
    radius = 0.7 * avgDistance;
    ## Conduct the greedy triangulation
    print("\t Conduct the greedy triangulation ...");
    brainMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        brainPointCloud, o3d.utility.DoubleVector([radius, radius * 2]));
    ## Convert the brain mesh to trimesh to better visualization
    print("\t Convert the brain mesh to trimesh to better visualization ...");
    brainMesh.compute_vertex_normals();
    trimeshBrainMesh = trimesh.Trimesh(np.asarray(brainMesh.vertices), np.asarray(brainMesh.triangles));
    trimeshBrainMesh.fix_normals();
    ## Set mesh vertex color as the real bone color (white color)
    print("\t Set mesh vertex color as the real bone color (white color) ...");
    trimeshBrainMesh.visual.vertex_colors = np.array([0.96, 0.92, 0.86, 1.0]) * 255;
    ## Visualize the reconstructed mesh using trimesh
    print("\t Visualize the reconstructed mesh using trimesh ...");
    trimeshBrainMesh.show();

    # Finished processing.
    print("Finished processing.");

#*********************************************************************************************************************#
#***************************************************MAIN FUNCTIONS****************************************************#
#*********************************************************************************************************************#
def main():
    os.system("cls");
    meshReconstructionUsingGreedyTriangulation();
if __name__ == "__main__":
    main()