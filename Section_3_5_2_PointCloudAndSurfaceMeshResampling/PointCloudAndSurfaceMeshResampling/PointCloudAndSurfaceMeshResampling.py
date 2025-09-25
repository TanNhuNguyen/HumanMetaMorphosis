#*********************************************************************************************************************#
#***************************************************SUPPORTING LIBRARIES**********************************************#
#*********************************************************************************************************************#
import os;
import numpy as np;
import open3d as o3d;
from pyvista.core import grid
import tetgen;
import pyvista as pv;
import pymeshfix;

#*********************************************************************************************************************#
#***************************************************SUPPORTING BUFFERS************************************************#
#*********************************************************************************************************************#
mainFolder = "../../../Data/Section_3_5_2_PointCloudAndSurfaceMeshResampling";
pointCloudFolder = mainFolder + "/PointClouds";
surfaceMeshFolder = mainFolder + "/SurfaceMeshes";

#*********************************************************************************************************************#
#***************************************************SUPPORTING FUNCTIONS**********************************************#
#*********************************************************************************************************************#

#*********************************************************************************************************************#
#***************************************************PROCESSING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def pointCloudDownSampling():
    # Initialize
    print("Initializing ...");

    # Reading the high-resolution point cloud
    print("Reading the high-resolution point cloud ...");
    ## Reading the point cloud
    pointCloud = o3d.io.read_point_cloud(pointCloudFolder + "/FemaleHeadPointCloud.ply");
    ## Compute normals
    pointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30));
    ## Set point cloud colors
    pointCloud.paint_uniform_color([0.5, 0.15, 0.15]);
    ## Print information
    print("\t Finished loading point cloud, some information: ");
    print("\t\t- Number of points: " + str(len(pointCloud.points)));
    ## Visualize the point cloud
    o3d.visualization.draw_geometries([pointCloud], 
                                      window_name="High-resolution Point Cloud", 
                                      width=800, height=600);

    # Down-sampling the point cloud with the voxel size of 0.2
    print("Down-sampling the point cloud with the voxel size of 0.2 ...");
    ## Voxel down-sampling
    voxelDownSampledPointCloud = pointCloud.voxel_down_sample(voxel_size=0.2);
    ## Print information
    print("\t Finished down-sampling point cloud, some information: ");
    print("\t\t- Number of points: " + str(len(voxelDownSampledPointCloud.points)));
    ## Visualize the point cloud
    o3d.visualization.draw_geometries([voxelDownSampledPointCloud], 
                                      window_name="Voxel Down-sampled Point Cloud (voxel size = 0.2)", 
                                      width=800, height=600);
    
    # Downsampling the point cloud with the voxel size of 0.5
    print("Down-sampling the point cloud with the voxel size of 0.5 ...");
    ## Voxel down-sampling
    voxelDownSampledPointCloud = pointCloud.voxel_down_sample(voxel_size=0.5);
    ## Print information
    print("\t Finished down-sampling point cloud, some information: ");
    print("\t\t- Number of points: " + str(len(voxelDownSampledPointCloud.points)));
    ## Visualize the point cloud
    o3d.visualization.draw_geometries([voxelDownSampledPointCloud], 
                                      window_name="Voxel Down-sampled Point Cloud (voxel size = 0.5)", 
                                      width=800, height=600);

    # Finished processing
    print("Finished processing.");
def surfaceMeshUniformResampling():
    # Initialize
    print("Initializing ...");

    # Reading the original high-resolution surface mesh
    print("Reading the original high-resolution surface mesh ...");
    ## Reading the surface mesh
    surfaceMesh = o3d.io.read_triangle_mesh(surfaceMeshFolder + "/FemaleHeadMesh_TriMesh.ply");
    ## Compute vertex normals
    surfaceMesh.compute_vertex_normals();
    ## Set the vertex colors as skin color (red-ish)
    surfaceMesh.paint_uniform_color([0.9, 0.7, 0.6]);
    ## Print information
    print("\t Finished loading surface mesh, some information: ");
    print("\t\t- Number of vertices: " + str(len(surfaceMesh.vertices)));
    print("\t\t- Number of triangles: " + str(len(surfaceMesh.triangles)));
    ## Visualize the surface mesh
    o3d.visualization.draw_geometries([surfaceMesh], 
                                      window_name="Original High-resolution Surface Mesh", 
                                      width=800, height=600);

    # Uniformly sampling the surface mesh to get the point cloud with 5000 points
    print("Uniformly sampling the surface mesh to get the point cloud with 5000 points ...");
    ## Uniformly sampling the surface mesh
    uniformlySampledPointCloud = surfaceMesh.sample_points_uniformly(number_of_points=5000);
    ## Print information
    print("\t Finished uniformly sampling the surface mesh, some information: ");
    print("\t\t- Number of points: " + str(len(uniformlySampledPointCloud.points)));
    ## Visualize the point cloud
    o3d.visualization.draw_geometries([uniformlySampledPointCloud], 
                                      window_name="Uniformly Sampled Point Cloud (5000 points)", 
                                      width=800, height=600);
    
    # Uniformly sampling the surface mesh to get the point cloud with 10000 points
    print("Uniformly sampling the surface mesh to get the point cloud with 10000 points ...");
    ## Uniformly sampling the surface mesh
    uniformlySampledPointCloud = surfaceMesh.sample_points_uniformly(number_of_points=10000);
    ## Print information
    print("\t Finished uniformly sampling the surface mesh, some information: ");
    print("\t\t- Number of points: " + str(len(uniformlySampledPointCloud.points)));
    ## Visualize the point cloud
    o3d.visualization.draw_geometries([uniformlySampledPointCloud],
                                        window_name="Uniformly Sampled Point Cloud (10000 points)", 
                                        width=800, height=600);

    # Finished processing
    print("Finished processing.");
def surfaceMeshPoissonDiskSampling():
    # Initialize
    print("Initializing ...");

    # Reading the original high-resolution surface mesh
    print("Reading the original high-resolution surface mesh ...");
    ## Reading the surface mesh
    surfaceMesh = o3d.io.read_triangle_mesh(surfaceMeshFolder + "/FemaleHeadMesh_TriMesh.ply");
    ## Compute vertex normals
    surfaceMesh.compute_vertex_normals();
    ## Set the vertex colors as skin
    surfaceMesh.paint_uniform_color([0.9, 0.7, 0.6]);
    ## Print information
    print("\t Finished loading surface mesh, some information: ");
    print("\t\t- Number of vertices: " + str(len(surfaceMesh.vertices)));
    print("\t\t- Number of triangles: " + str(len(surfaceMesh.triangles)));
    ## Visualize the surface mesh
    o3d.visualization.draw_geometries([surfaceMesh], 
                                      window_name="Original High-resolution Surface Mesh", 
                                      width=800, height=600);
    
    # Poisson disk sampling the surface mesh to get the point cloud with 5000 points
    print("Poisson disk sampling the surface mesh to get the point cloud with 5000 points ...");
    ## Poisson disk sampling the surface mesh
    poissonDiskSampledPointCloud = surfaceMesh.sample_points_poisson_disk(number_of_points=5000);
    ## Print information
    print("\t Finished poisson disk sampling the surface mesh, some information: ");
    print("\t\t- Number of points: " + str(len(poissonDiskSampledPointCloud.points)));
    ## Visualize the point cloud
    o3d.visualization.draw_geometries([poissonDiskSampledPointCloud],
                                      window_name="Poisson Disk Sampled Point Cloud (5000 points)", 
                                      width=800, height=600);
    
    # Poisson disk sampling the surface mesh to get the point cloud with 10000 points
    print("Poisson disk sampling the surface mesh to get the point cloud with 10000 points ...");
    ## Poisson disk sampling the surface mesh
    poissonDiskSampledPointCloud = surfaceMesh.sample_points_poisson_disk(number_of_points=10000);
    ## Print information
    print("\t Finished poisson disk sampling the surface mesh, some information: ");
    print("\t\t- Number of points: " + str(len(poissonDiskSampledPointCloud.points)));
    ## Visualize the point cloud
    o3d.visualization.draw_geometries([poissonDiskSampledPointCloud],
                                      window_name="Poisson Disk Sampled Point Cloud (10000 points)", 
                                      width=800, height=600);

    # Finished processing
    print("Finished processing.");

#*********************************************************************************************************************#
#***************************************************MAIN FUNCTIONS****************************************************#
#*********************************************************************************************************************#
def main():
    os.system("cls");
    surfaceMeshPoissonDiskSampling();
if __name__ == "__main__":
    main()