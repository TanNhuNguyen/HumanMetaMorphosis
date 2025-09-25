#*********************************************************************************************************************#
#***************************************************SUPPORTING LIBRARIES**********************************************#
#*********************************************************************************************************************#
import os;
import trimesh;
import open3d as o3d;
import numpy as np;
import copy;

#*********************************************************************************************************************#
#***************************************************SUPPORTING BUFFERS************************************************#
#*********************************************************************************************************************#
mainFolder = "../../../Data/Section_3_7_2_MeshUnifiyingAndRefining";

#*********************************************************************************************************************#
#***************************************************SUPPORTING CLASSES************************************************#
#*********************************************************************************************************************#

#*********************************************************************************************************************#
#***************************************************PROCESSING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def meshUnifyingWithTriMesh():
    # Initialize
    print("Initializing ...");

    # Reading the parts of the mesh
    print("Reading the parts of the mesh ...");
    ## Reading lung lower left
    lungLowerLeft = trimesh.load(mainFolder + "/011736_lunglowerlobeleft.stl");
    ## Set red mesh color
    lungLowerLeft.visual.face_colors = [255, 0, 0, 100];
    ## Reading lung lower right
    lungLowerRight = trimesh.load(mainFolder + "/011736_lunglowerloberight.stl");
    ## Set green mesh color
    lungLowerRight.visual.face_colors = [0, 255, 0, 100];
    ## Reading lung upper left
    lungUpperLeft = trimesh.load(mainFolder + "/011736_lungupperlobeleft.stl");
    ## Set blue mesh color
    lungUpperLeft.visual.face_colors = [0, 0, 255, 100];
    ## Reading lung upper right
    lungUpperRight = trimesh.load(mainFolder + "/011736_lungupperloberight.stl");
    ## Set yellow mesh color
    lungUpperRight.visual.face_colors = [255, 255, 0, 100];

    # Unifying the parts of the mesh
    print("Unifying the parts of the mesh ...");
    unifiedMesh = trimesh.util.concatenate([lungLowerLeft, lungLowerRight, lungUpperLeft, lungUpperRight]);

    # View the unifying mesh
    print("Viewing the unifying mesh ...");
    unifiedMesh.show();

    # Finished processing
    print("Finished processing.");
def meshUniformRefinement():
    # Initialize
    print("Initializing ...");

    # Generate sphere mesh using open3d
    print("Generating sphere mesh using open3d ...");
    ## Generate sphere mesh
    lowResolutionSphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10);
    ## Compute vertex normals
    lowResolutionSphere.compute_vertex_normals();
    ## Visualize the mesh with wireframe
    o3d.visualization.draw_geometries([lowResolutionSphere], mesh_show_wireframe=True);

    # Mesh uniform refinement using open3d
    print("Mesh uniform refinement using open3d ...");
    ## Refine the mesh
    highResolutionSphere = lowResolutionSphere.subdivide_midpoint(number_of_iterations=2);
    ## Compute vertex normals
    highResolutionSphere.compute_vertex_normals();
    ## Visualize the mesh with wireframe
    o3d.visualization.draw_geometries([highResolutionSphere], mesh_show_wireframe=True);

    # Finished processing
    print("Finished processing.");
def meshSamplingAndConductRemeshing():
    # Initialize
    print("Initializing ...");

    # Generate the sphere mesh using open3d
    print("Generating the sphere mesh using open3d ...");
    ## Generate sphere mesh
    lowResolutionSphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10);
    ## Compute vertex normals
    lowResolutionSphere.compute_vertex_normals();
    ## Print the number of vertices and triangles
    print("\t Number of vertices: ", len(lowResolutionSphere.vertices));
    print("\t Number of triangles: ", len(lowResolutionSphere.triangles));
    ## Visualize the mesh with wireframe
    o3d.visualization.draw_geometries([lowResolutionSphere], mesh_show_wireframe=True);
    ## Save the mesh
    o3d.io.write_triangle_mesh(mainFolder + "/lowResolutionSphere.ply", lowResolutionSphere);

    # Sample the mesh using the Poisson disk sampling
    print("Sampling the mesh using the Poisson disk sampling ...");
    ## Sample the mesh with the point number double the original vertex number
    numberOfSampledPoints = 2 * len(lowResolutionSphere.vertices);
    sampledPointCloud = lowResolutionSphere.sample_points_poisson_disk(number_of_points=numberOfSampledPoints);
    ## Print the number of sampled points
    print("\t Number of sampled points: ", len(sampledPointCloud.points));
    ## Compute normals for the sampled point cloud
    sampledPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30));
    ## Add color red to the sampled point cloud
    sampledPointCloud.paint_uniform_color([1.0, 0.0, 0.0]);
    ## Save the sampled point cloud
    o3d.io.write_point_cloud(mainFolder + "/sampledPointCloud.ply", sampledPointCloud);
    ## Visualize the sampled point cloud
    o3d.visualization.draw_geometries([sampledPointCloud]);

    # Conduct remeshing using the screened Poisson surface reconstruction
    print("Conducting remeshing using the screened Poisson surface reconstruction ...");
    ## Conduct remeshing using point pivoting
    remeshedSphere, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(sampledPointCloud, depth=8);
    ## Compute vertex normals
    remeshedSphere.compute_vertex_normals();
    ## Print the number of vertices and triangles
    print("\t Number of vertices: ", len(remeshedSphere.vertices));
    print("\t Number of triangles: ", len(remeshedSphere.triangles));
    ## Visualize the mesh with wireframe
    o3d.visualization.draw_geometries([remeshedSphere], mesh_show_wireframe=True);
    ## Save the remeshed mesh
    o3d.io.write_triangle_mesh(mainFolder + "/remeshedSphere.ply", remeshedSphere);

    # Isotropic remesh the meshed sphere
    print("Isotropic remeshing the meshed sphere ...");
    ## Set target edge length
    targetEdgeLength = 0.1;
    ## Set number of iterations
    numberOfIterations = 100;
    ## Conduct isotropic remeshing
    isotropicallyRemeshedSphere = remeshedSphere.simplify_vertex_clustering(voxel_size=targetEdgeLength, contraction=o3d.geometry.SimplificationContraction.Average);
    for i in range(numberOfIterations - 1):
        isotropicallyRemeshedSphere = isotropicallyRemeshedSphere.simplify_vertex_clustering(voxel_size=targetEdgeLength, contraction=o3d.geometry.SimplificationContraction.Average);
    ## Compute vertex normals
    isotropicallyRemeshedSphere.compute_vertex_normals();
    ## Print the number of vertices and triangles
    print("\t Number of vertices: ", len(isotropicallyRemeshedSphere.vertices));
    print("\t Number of triangles: ", len(isotropicallyRemeshedSphere.triangles));
    ## Visualize the mesh with wireframe
    o3d.visualization.draw_geometries([isotropicallyRemeshedSphere], mesh_show_wireframe=True);
    ## Save the isotropically remeshed mesh
    o3d.io.write_triangle_mesh(mainFolder + "/isotropicallyRemeshedSphere.ply", isotropicallyRemeshedSphere);

    # Finished processing
    print("Finished processing.");

#*********************************************************************************************************************#
#***************************************************MAIN FUNCTIONS****************************************************#
#*********************************************************************************************************************#
def main():
    os.system("cls");
    meshSamplingAndConductRemeshing();
    
if __name__ == "__main__":
    main()