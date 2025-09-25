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
mainFolder = "../../../Data/Section_3_4_3_VoxelisationOf3DShape";

#*********************************************************************************************************************#
#***************************************************SUPPORTING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def meshFix(inOpen3DMesh):
    # Convert Open3D mesh to numpy arrays
    vertices = np.asarray(inOpen3DMesh.vertices);
    faces = np.asarray(inOpen3DMesh.triangles);

    # Use PyMeshFix to repair the mesh
    meshfix = pymeshfix.MeshFix(vertices, faces);
    meshfix.repair(verbose=True, joincomp=True, remove_smallest_components=True);
    repairedVertices, repairedFaces = meshfix.v, meshfix.f;

    # Create a new Open3D mesh from repaired data
    fixedMesh = o3d.geometry.TriangleMesh();
    fixedMesh.vertices = o3d.utility.Vector3dVector(repairedVertices);
    fixedMesh.triangles = o3d.utility.Vector3iVector(repairedFaces);
    fixedMesh.compute_vertex_normals();
    return fixedMesh;

#*********************************************************************************************************************#
#***************************************************PROCESSING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def volumeMeshReconstructionUsingVoxelGrid():
    # Initialize
    print("Initializing ...");

    # Read the triangle mesh
    print("Reading the triangle mesh ...");
    ## Reading mesh file
    liverMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "008738_liver.stl"));
    ## Compute the vertex normals of the triangle mesh
    liverMesh.compute_vertex_normals();
    ## Set mesh color as the liver color (reddish color)
    liverMesh.paint_uniform_color([1.0, 0.75, 0.79]);
    ## Print the triangle mesh information
    print(liverMesh);
    
    # Create the voxel grid from the triangle mesh
    print("Creating the voxel grid from the triangle mesh ...");
    voxel_size = 5.0;  # Adjust resolution here
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(liverMesh, voxel_size=voxel_size);
        
    # Draw the geometries of the voxel grid
    print("Drawing the geometries of the voxel grid ...");
    o3d.visualization.draw_geometries([voxel_grid]);

    # Finished processing
    print("Finished processing.");
def volumeMeshReconstructionUsingTetrahedralization():
    # Initialize
    print("Initializing ...");

    # Loading surface mesh
    print("Loading surface mesh ...");
    ## Reading mesh file
    print("\t Reading the mesh file ...");
    liverMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "008738_liver.ply"));
    ## Fixing the mesh
    print("\t Fixing the mesh ...");
    liverMesh = meshFix(liverMesh);
    ## Compute the vertex normals of the triangle mesh
    liverMesh.compute_vertex_normals();
    ## Set mesh color as the liver color (reddish color)
    liverMesh.paint_uniform_color([1.0, 0.75, 0.79]);
    ## Print the triangle mesh information
    print(liverMesh);

    # Tetrahedralization of the liverMesh
    print("Tetrahedralization of the liverMesh ...");
    ## Extract the vertices and triangles 
    vertices = np.asarray(liverMesh.vertices);
    triangles = np.asarray(liverMesh.triangles);
    ## Convert triangles to pvista face format
    faces = np.hstack([np.full((triangles.shape[0], 1), 3), triangles]).flatten();
    ## Create a pyvista surface mesh
    surfaceMesh = pv.PolyData(vertices, faces);
    ## Perform tetrahedralization using TetGen
    tet = tetgen.TetGen(surfaceMesh);
    tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5);
    ## Get the volumetric mesh
    volumeMesh = tet.grid;

    # Slice the volumetric mesh to see the internal structure
    print("Slice the volumetric mesh to see the internal structure ...");
    slicedVolumeMesh = volumeMesh.slice(normal='z', origin=volumeMesh.center);

    # Visualize the generated tetrahedral mesh
    print("Visualize the generated tetrahedral mesh ...");
    plotter = pv.Plotter();
    plotter.add_mesh(slicedVolumeMesh, show_edges=True, color="lightblue", opacity=0.7);
    plotter.show();

    # Finished processing
    print("Finished processing.");

#*********************************************************************************************************************#
#***************************************************MAIN FUNCTIONS****************************************************#
#*********************************************************************************************************************#
def main():
    os.system("cls");
    volumeMeshReconstructionUsingTetrahedralization();
if __name__ == "__main__":
    main()