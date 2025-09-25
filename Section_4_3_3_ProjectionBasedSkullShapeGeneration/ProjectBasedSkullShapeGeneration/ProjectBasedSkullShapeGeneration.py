#*********************************************************************************************************************#
#***************************************************SUPPORTING LIBRARIES**********************************************#
#*********************************************************************************************************************#
import os;
import trimesh;
import open3d as o3d;
import numpy as np;
import copy;
from scipy.spatial import cKDTree;
import pymeshlab;

#*********************************************************************************************************************#
#***************************************************SUPPORTING BUFFERS************************************************#
#*********************************************************************************************************************#
mainFolder = "../../../Data/Section_4_3_3_ProjectionBasedSkullShapeGeneration";

#*********************************************************************************************************************#
#***************************************************SUPPORTING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def loadO3DMesh(meshFilePath):
    # Checking file path
    print("loadO3DMesh:: Checking file path...");
    if not os.path.exists(meshFilePath):
        print("File path does not exist: " + meshFilePath);
        return None;

    # Load mesh using open3d mesh
    print("loadO3DMesh:: Loading mesh...");
    mesh = o3d.io.read_triangle_mesh(meshFilePath);
    if mesh.is_empty():
        print("Failed to load mesh: " + meshFilePath);
        return None;

    # Return loaded mesh
    print("loadO3DMesh:: Successfully loaded mesh.");
    return mesh;
def fixMeshToRemoveVerticesWithoutFaces(inO3DMesh):
    # Convert Open3D mesh to Trimesh
    print("fixMeshToRemoveVerticesWithoutFaces:: Converting Open3D mesh to Trimesh...");
    tempMesh = trimesh.Trimesh(vertices=np.asarray(inO3DMesh.vertices), faces=np.asarray(inO3DMesh.triangles));

    # Remove unreferenced vertices
    print("fixMeshToRemoveVerticesWithoutFaces:: Removing unreferenced vertices...");
    tempMesh.remove_unreferenced_vertices();

    # Convert back to Open3D mesh
    print("fixMeshToRemoveVerticesWithoutFaces:: Converting back to Open3D mesh...");
    fixedMesh = o3d.geometry.TriangleMesh();
    fixedMesh.vertices = o3d.utility.Vector3dVector(tempMesh.vertices);
    fixedMesh.triangles = o3d.utility.Vector3iVector(tempMesh.faces);

    # Recompute vertex normals
    print("fixMeshToRemoveVerticesWithoutFaces:: Recomputing vertex normals...");
    fixedMesh.compute_vertex_normals();

    # Return the fixed mesh
    print("fixMeshToRemoveVerticesWithoutFaces:: Finished fixing the mesh.");
    return fixedMesh;
def estimateNearestPointsFromPoints(inSourcePoints, inTargetPoints):
    # Create KDTree from target points
    print("estimateNearestPointsFromPoints:: Creating KDTree...");
    kdtree = cKDTree(inTargetPoints);

    # Query nearest neighbors for source points
    print("estimateNearestPointsFromPoints:: Querying nearest neighbors...");
    distances, indices = kdtree.query(inSourcePoints);

    # Retrieve nearest points from target points using the indices
    print("estimateNearestPointsFromPoints:: Retrieving nearest points...");
    nearestPoints = inTargetPoints[indices];

    # Return the nearest points
    print("estimateNearestPointsFromPoints:: Finished retrieving nearest points.");
    return nearestPoints;
def formO3DMesh(inVertices, inTriangles):
    # Create an Open3D TriangleMesh object
    print("formO3DMesh:: Creating Open3D mesh...");
    mesh = o3d.geometry.TriangleMesh();

    # Set vertices and triangles
    print("formO3DMesh:: Setting vertices and triangles...");
    mesh.vertices = o3d.utility.Vector3dVector(inVertices);
    mesh.triangles = o3d.utility.Vector3iVector(inTriangles);

    # Compute vertex normals for better visualization
    print("formO3DMesh:: Computing vertex normals...");
    mesh.compute_vertex_normals();

    # Return the formed mesh
    print("formO3DMesh:: Finished creating mesh.");
    return mesh;
def isotropicRemeshO3DMeshWithResampling(inO3DMesh, inNumOfTargetVertices):
    # Poisson disk sampling to get target number of vertices
    print("isotropicRemeshO3DMeshWithResampling:: Sampling points...");
    sampledPointCloud = inO3DMesh.sample_points_poisson_disk(number_of_points=inNumOfTargetVertices);

    # Surface reconstruction using point pivot algorithm
    print("isotropicRemeshO3DMeshWithResampling:: Reconstructing mesh using BPA...");
    distances = sampledPointCloud.compute_nearest_neighbor_distance();
    avgDist = np.mean(distances);
    radius = 3 * avgDist;
    bpaMesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(sampledPointCloud, o3d.utility.DoubleVector([radius, radius * 2]));

    # Fixing the mesh
    print("isotropicRemeshO3DMeshWithResampling:: Fixing the mesh...");
    bpaMesh.remove_duplicated_vertices();
    bpaMesh.remove_degenerate_triangles();
    bpaMesh.remove_duplicated_triangles();
    bpaMesh.remove_non_manifold_edges();
    bpaMesh.compute_vertex_normals();

    # Convert to pymeshlab to close holes more effectively
    print("isotropicRemeshO3DMeshWithResampling:: Converting to pymeshlab for hole filling...");
    pymeshlabMesh = pymeshlab.MeshSet();
    tempMesh = pymeshlab.Mesh(np.asarray(bpaMesh.vertices), np.asarray(bpaMesh.triangles));
    pymeshlabMesh.add_mesh(tempMesh, "tempMesh");

    # Fill holes
    print("isotropicRemeshO3DMeshWithResampling:: Filling holes...");
    pymeshlabMesh.meshing_close_holes(maxholesize=1000);
    tempMesh = pymeshlabMesh.current_mesh();

    # Convert back to Open3D mesh
    print("isotropicRemeshO3DMeshWithResampling:: Converting back to Open3D mesh...");
    bpaMesh = formO3DMesh(tempMesh.vertex_matrix(), tempMesh.face_matrix());

    # Return the remeshed mesh
    print("isotropicRemeshO3DMeshWithResampling:: Finished.");
    return bpaMesh;
def estimateConvexHullO3DMesh(inO3DMesh):
    # Compute the convex hull of the mesh
    print("estimateConvexHullO3DMesh:: Computing convex hull...");
    hull, _ = inO3DMesh.compute_convex_hull();
    hull.compute_vertex_normals();

    # Return the convex hull mesh
    print("estimateConvexHullO3DMesh:: Finished computing convex hull.");
    return hull;
def scaleO3DMesh(inMesh, inScaleFactor):
    # Scale the mesh
    print("scaleO3DMesh:: Scaling mesh...");
    scaledMesh = copy.deepcopy(inMesh);
    scaledMesh.scale(inScaleFactor, center=scaledMesh.get_center());

    # Return the scaled mesh
    print("scaleO3DMesh:: Finished scaling mesh.");
    return scaledMesh;
def computeAverageEdgeLength(inO3DMesh):
    # Convert triangles and vertices to numpy arrays
    print("computeAverageEdgeLength:: Computing average edge length...");
    triangles = np.asarray(inO3DMesh.triangles);
    vertices = np.asarray(inO3DMesh.vertices);

    # Collect all edges from triangles
    print("computeAverageEdgeLength:: Collecting all edges from triangles...");
    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ]);

    # Remove duplicate edges
    print("computeAverageEdgeLength:: Removing duplicate edges...");
    edges = np.sort(edges, axis=1);
    edges = np.unique(edges, axis=0);

    # Compute edge lengths
    print("computeAverageEdgeLength:: Computing edge lengths...");
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1);

    # Return the average edge length
    print("computeAverageEdgeLength:: Finished computing average edge length.");
    return np.mean(edge_lengths);
def saveO3DMesh(inO3DMesh, inFilePath):
    # Save the mesh to the specified file path
    print("saveO3DMesh:: Saving mesh to file...");
    success = o3d.io.write_triangle_mesh(inFilePath, inO3DMesh);
    if not success:
        print("Failed to save mesh to file: " + inFilePath);
        return False;

    # Return success status
    print("saveO3DMesh:: Successfully saved mesh to file.");
    return True;

#*********************************************************************************************************************#
#***************************************************PROCESSING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def generateSkullShapeFromSkullMesh():
    # Initialize
    print("Initializing...");

    # Load skull mesh
    print("Loading skull mesh...");
    ## Load the skull mesh
    print("\t Loading skull mesh...");
    skullMesh = loadO3DMesh(os.path.join(mainFolder, "109833-SkullMesh.ply"));
    ## Print the number of vertices and triangles
    print("\t Number of vertices: " + str(len(skullMesh.vertices)) + ", number of triangles: " + str(len(skullMesh.triangles)));
    ## Fix the mesh to remove vertices without faces
    print("\t Fixing the mesh to remove vertices without faces...");
    skullMesh = fixMeshToRemoveVerticesWithoutFaces(skullMesh);
    ## Print the number of vertices and triangles after fixing
    print("\t Number of vertices after fixing: " + str(len(skullMesh.vertices)) + ", number of triangles: " + str(len(skullMesh.triangles)));
    ## Scale the skull mesh to meter
    print("\t Scaling skull mesh to meter...");
    skullMesh = scaleO3DMesh(skullMesh, 0.001);
    ## Estimate vertex normals
    print("\t Estimating vertex normals...");
    skullMesh.compute_vertex_normals();
    ## Set color as gray
    print("\t Setting color as gray...");
    skullMesh.paint_uniform_color([0.5, 0.5, 0.5]);
    ## Visualize the skull mesh
    print("\t Visualizing the skull mesh...");
    o3d.visualization.draw_geometries([skullMesh], mesh_show_back_face=True);

    # Estimate convex hull
    print("Estimating convex hull...");
    ## Estimate convex hull
    print("\t Estimating convex hull...");
    convexHullMesh = estimateConvexHullO3DMesh(skullMesh);
    ## Compute normals
    print("\t Computing normals...");
    convexHullMesh.compute_vertex_normals();
    ## Set color as light blue
    print("\t Setting color as light blue...");
    convexHullMesh.paint_uniform_color([0.678, 0.847, 0.902]);
    ## Visualize the convex hull with wireframe mode
    print("\t Visualizing the convex hull with wireframe mode...");
    o3d.visualization.draw_geometries([convexHullMesh], mesh_show_back_face=True, mesh_show_wireframe=True);
    ## Save the convex hull mesh
    print("\t Saving the convex hull mesh...");
    saveO3DMesh(convexHullMesh, os.path.join(mainFolder, "109833-ConvexHullMesh.ply"));

    # Isotropic remesh the convex hull
    print("Isotropic remesh the convex hull...");
    ## Read the convex hull mesh
    print("\t Reading the convex hull mesh...");
    convexHullMesh = loadO3DMesh(os.path.join(mainFolder, "109833-ConvexHullMesh.ply"));
    ## Isotropic remesh the convex hull
    print("\t Isotropic remeshing the convex hull ...");
    remeshedConvexHullMesh = isotropicRemeshO3DMeshWithResampling(convexHullMesh, inNumOfTargetVertices=100000);
    ## Set color as light green
    print("\t Setting color as light green ...");
    remeshedConvexHullMesh.paint_uniform_color([0.564, 0.933, 0.564]);
    ## Visualize the remeshed convex hull with wireframe mode
    print("\t Visualizing the remeshed convex hull with wireframe mode...");
    o3d.visualization.draw_geometries([remeshedConvexHullMesh], mesh_show_back_face=True, mesh_show_wireframe=True);
    ## Save the remeshed convex hull mesh
    print("\t Saving the remeshed convex hull mesh...");
    saveO3DMesh(remeshedConvexHullMesh, os.path.join(mainFolder, "109833-RemeshedConvexHullMesh.ply"));

    # Project the vertices of the remeshed convex hull onto the skull mesh
    print("Projecting the vertices of the remeshed convex hull onto the skull mesh...");
    ## Read the remeshed convex hull mesh
    print("\t Reading the remeshed convex hull mesh...");
    remeshedConvexHullMesh = loadO3DMesh(os.path.join(mainFolder, "109833-RemeshedConvexHullMesh.ply"));
    ## Get vertices and triangles of the remeshed convex hull mesh
    print("\t Getting vertices and triangles of the remeshed convex hull mesh...");
    remeshedConvexHullVertices = np.asarray(remeshedConvexHullMesh.vertices);
    remeshedConvexHullTriangles = np.asarray(remeshedConvexHullMesh.triangles);
    ## Get vertices of the skull mesh
    print("\t Getting vertices of the skull mesh...");
    skullMeshVertices = np.asarray(skullMesh.vertices);
    ## Estimate nearest points from the skull mesh vertices
    print("\t Estimating nearest points from the skull mesh vertices...");
    projectedVertices = estimateNearestPointsFromPoints(remeshedConvexHullVertices, skullMeshVertices);
    ## Form a new mesh using the projected vertices and the triangles of the remeshed convex
    print("\t Forming a new mesh using the projected vertices and the triangles of the remeshed convex hull...");
    projectedMesh = formO3DMesh(projectedVertices, remeshedConvexHullTriangles);
    ## Compute normals
    print("\t Computing normals...");
    projectedMesh.compute_vertex_normals();
    ## Set color as light coral
    print("\t Setting color as light coral...");
    projectedMesh.paint_uniform_color([0.941, 0.502, 0.502]);
    ## Visualize the projected mesh with wireframe mode
    print("\t Visualizing the projected mesh with wireframe mode...");
    o3d.visualization.draw_geometries([projectedMesh], mesh_show_back_face=True, mesh_show_wireframe=True);
    ## Save the projected mesh
    print("\t Saving the projected mesh...");
    saveO3DMesh(projectedMesh, os.path.join(mainFolder, "109833-ProjectedMesh.ply"));

    # Isotropic remesh the projected mesh
    print("Isotropic remesh the projected mesh...");
    ## Read the projected mesh
    print("\t Reading the projected mesh...");
    projectedMesh = loadO3DMesh(os.path.join(mainFolder, "109833-ProjectedMesh.ply"));
    ## Isotropic remesh with the target number of vertices as 100000
    print("\t Isotropic remeshing the projected mesh...");
    remeshedProjectedMesh = isotropicRemeshO3DMeshWithResampling(projectedMesh, inNumOfTargetVertices=100000);
    ## Set color as light yellow
    print("\t Setting color as light yellow...");
    remeshedProjectedMesh.paint_uniform_color([1.0, 1.0, 0.878]);
    ## Visualize the remeshed projected mesh with wireframe mode
    print("\t Visualizing the remeshed projected mesh with wireframe mode...");
    o3d.visualization.draw_geometries([remeshedProjectedMesh], mesh_show_back_face=True, mesh_show_wireframe=True);
    ## Save the remeshed projected mesh as the skull shape
    print("\t Saving the remeshed projected mesh as the skull shape...");
    saveO3DMesh(remeshedProjectedMesh, os.path.join(mainFolder, "109833-SkullShape.ply"));

    # Finished processing
    print("Finished processing.");

#*********************************************************************************************************************#
#***************************************************MAIN FUNCTIONS****************************************************#
#*********************************************************************************************************************#
def main():
    os.system("cls");
    generateSkullShapeFromSkullMesh();    
if __name__ == "__main__":
    main()