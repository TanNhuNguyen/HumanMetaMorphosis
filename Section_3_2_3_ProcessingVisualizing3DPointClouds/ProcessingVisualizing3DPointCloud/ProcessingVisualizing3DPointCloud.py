#*********************************************************************************************************************#
#***************************************************SUPPORTING LIBRARIES**********************************************#
#*********************************************************************************************************************#
import os;
import open3d as o3d;
import numpy as np;
import matplotlib.pyplot as plt;

#*********************************************************************************************************************#
#***************************************************SUPPORTING BUFFERS************************************************#
#*********************************************************************************************************************#
mainFolder = "../../../Data/Section_3_2_3_ProcessingVisualizing3DPointClouds";

#*********************************************************************************************************************#
#***************************************************SUPPORTING FUNCTIONS**********************************************#
#*********************************************************************************************************************#

#*********************************************************************************************************************#
#***************************************************PROCESSING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def readPointCloudFromFile():
    # Initializing
    print("Initializing ...");

    # Reading point cloud using the open3d library
    print("Reading point cloud using the open3d library ...");
    brainPointCloudFilePath = mainFolder + "/BrainPointCloud.ply";
    brainPointCloud = o3d.io.read_point_cloud(brainPointCloudFilePath);
    print("\t Point cloud read successfully.");

    # Print some information of the point cloud
    print("Printing some information of the point cloud ...");
    print("\t Number of points: ", np.asarray(brainPointCloud.points).shape[0]);
    print("\t Point cloud bounds: ", brainPointCloud.get_axis_aligned_bounding_box());
    print("\t Point cloud colors: ", np.asarray(brainPointCloud.colors).shape[0]);
    print("\t Point cloud normals: ", np.asarray(brainPointCloud.normals).shape[0]);

    # Finished processing.
    print("Finished processing.");
def generatePointClouds():
    # Initializing
    print("Initializing ...");

    # Generate the sphere point cloud using the open3d
    print("Generating the sphere point cloud using the open3d ...");
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0);
    spherePointCloud = o3d.geometry.PointCloud();
    spherePointCloud.points = sphere.vertices;
    spherePointCloud.colors = sphere.vertex_colors;
    spherePointCloud.normals = sphere.vertex_normals;
    print("\t Sphere point cloud generated successfully.");

    # Print the information of the sphere point cloud
    print("Printing the information of the sphere point cloud ...");
    print("\t Number of points: ", np.asarray(spherePointCloud.points).shape[0]);
    print("\t Point cloud bounds: ", spherePointCloud.get_axis_aligned_bounding_box());
    print("\t Point cloud colors: ", np.asarray(spherePointCloud.colors).shape[0]);
    print("\t Point cloud normals: ", np.asarray(spherePointCloud.normals).shape[0]);

    # Compute the center of the sphere
    print("Computing the center of the sphere ...");
    center = spherePointCloud.get_center();
    print("\t Center of the sphere: ", center);

    # Compute the sphere radius
    print("Computing the radius of the sphere ...");
    points = np.asarray(spherePointCloud.points);
    distances = np.linalg.norm(points - center, axis=1);
    radius = np.max(distances);
    print("\t Radius of the sphere: ", radius);

    # Finished processing.
    print("Finished processing.");
def visualizePointCloud():
    # Initializing
    print("Initializing ...");

    # Read point cloud from file
    print("Reading point cloud from file ...");
    fullBodyPointCloudFilePath = mainFolder + "/FullBodyPointCloud.ply";
    fullBodyPointCloud = o3d.io.read_point_cloud(fullBodyPointCloudFilePath);
    print("\t Point cloud read successfully.");

    # Visualize the point cloud using the open3d support
    print("Visualizing the point cloud using the open3d support ...");
    o3d.visualization.draw_geometries([fullBodyPointCloud]);
    print("\t Point cloud visualization completed.");

    # Finished processing.
    print("Finished processing.");
def addColorToPointCloud():
    # Initializing
    print("Initializing ...");

    # Read liver point cloud
    print("Reading liver point cloud ...");
    liverPointCloudFilePath = mainFolder + "/LiverPointCloud.ply";
    liverPointCloud = o3d.io.read_point_cloud(liverPointCloudFilePath);
    print("\t Point cloud read successfully.");

    # Set color of all points in the point cloud as the liver color
    print("Setting color of all points in the point cloud as the liver color ...");
    liverColor = [0.8, 0.2, 0.2];  # RGB color for liver
    liverPointCloud.colors = o3d.utility.Vector3dVector([liverColor] * np.asarray(liverPointCloud.points).shape[0]);
    print("\t Color added to liver point cloud.");

    # Visualize the point cloud
    print("Visualizing the point cloud ...");
    o3d.visualization.draw_geometries([liverPointCloud]);
    print("\t Point cloud visualization completed.");

    # Finished processing.
    print("Finished processing.");
def estimatePointCloudNormals():
    # Initializing
    print("Initializing ...");

    # Read the lung point cloud
    print("Reading lung point cloud ...");
    lungPointCloudFilePath = mainFolder + "/LungPointCloud.ply";
    lungPointCloud = o3d.io.read_point_cloud(lungPointCloudFilePath);
    print("\t Point cloud read successfully.");

    # Compute the normals for the point cloud as nearest points
    print("Computing normals for the point cloud ...");
    lungPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30));
    print("\t Normals computed successfully.");

    # Set the color of the point cloud of the lung as the lung color
    print("Setting color of the point cloud of the lung as the lung color ...");
    lungColor = [0.2, 0.2, 0.8];  # RGB color for lung
    lungPointCloud.colors = o3d.utility.Vector3dVector([lungColor] * np.asarray(lungPointCloud.points).shape[0]);
    print("\t Color added to lung point cloud.");

    # Visualize the lung point cloud
    print("Visualizing the lung point cloud ...");
    o3d.visualization.draw_geometries([lungPointCloud]);
    print("\t Lung point cloud visualization completed.");

    # Finished processing.
    print("Finished processing.");
def pointCloudClusteringAndSegmentation():
    # Initializing
    print("Initializing ...");

    # Reading full body point cloud
    print("Reading full body point cloud ...");
    fullBodyPointCloudFilePath = mainFolder + "/FullBodyPointCloud.ply";
    fullBodyPointCloud = o3d.io.read_point_cloud(fullBodyPointCloudFilePath);
    print("\t Point cloud read successfully.");

    # Compute normals for the full body point cloud
    print("Computing normals for the full body point cloud ...");
    fullBodyPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30));
    print("\t Normals computed successfully.");

    # Set the color of the body point cloud as skin color
    print("Setting color of the point cloud of the body as the skin color ...");
    skinColor = [0.9, 0.7, 0.6];  # RGB color for skin
    fullBodyPointCloud.colors = o3d.utility.Vector3dVector([skinColor] * np.asarray(fullBodyPointCloud.points).shape[0]);
    print("\t Color added to full body point cloud.");

    # Visualize the point cloud
    print("Visualizing the point cloud ...");
    o3d.visualization.draw_geometries([fullBodyPointCloud]);
    print("\t Point cloud visualization completed.");

    # Conduct point cloud clustering based on the height of the bounding box
    print("Conducting point cloud clustering based on the height of the bounding box ...");
    ## Estimating the heights of the fullBodyPointCloud
    points = np.asarray(fullBodyPointCloud.points)
    yMin = np.min(points[:, 1]);
    yMax = np.max(points[:, 1]);
    height = yMax - yMin;
    print(f"\t Height of the point cloud: {height:.3f}")
    ## Compute boundaries for 5 segments
    segment_height = height / 5
    boundaries = [yMin + i * segment_height for i in range(6)]
    print(f"\t Height boundaries for segments: {boundaries}")
    ## Assign labels based on the height segment
    labels = np.digitize(points[:, 1], boundaries) - 1  # labels: 0 to 4
    labels[labels == 5] = 4  # Handle edge case where z == z_max
    print("\t Clustering completed.");

    # Change the color of each labels position as different colors
    print("Changing the color of each label position as different colors ...");
    max_label = labels.max();
    colors = plt.get_cmap("jet")(labels / max_label)[:, :3];  # Get colors from colormap
    fullBodyPointCloud.colors = o3d.utility.Vector3dVector(colors);
    print("\t Colors changed successfully.");

    # Visualize the clustered point cloud
    print("Visualizing the clustered point cloud ...");
    o3d.visualization.draw_geometries([fullBodyPointCloud]);
    print("\t Clustered point cloud visualization completed.");

    # Finished processing.
    print("Finished processing.");
def voxelBasedDownSamplingPointCloud():
    # Initializing
    print("Initializing ...");

    # Read the full resolution of the pelvis point cloud
    print("Reading full resolution of the pelvis point cloud ...");
    pelvisPointCloudFilePath = mainFolder + "/PelvisPointCloud.ply";
    pelvisPointCloud = o3d.io.read_point_cloud(pelvisPointCloudFilePath);
    print("\t Point cloud read successfully.");

    # Downsample the point cloud using voxel grid filtering
    print("Downsampling the point cloud using voxel grid filtering ...");
    voxel_size = 0.05;
    downSampledPelvisPointCloud = pelvisPointCloud.voxel_down_sample(voxel_size);
    print("\t Point cloud downsampled successfully.");

    # Translate the downsampled point cloud to visualize both of them
    print("Translating the downsampled point cloud to visualize both of them ...");
    downSampledPelvisPointCloud.translate((0, 0, -0.5));
    print("\t Translation completed.");

    # Visualize the two point clouds
    print("Visualizing the two point clouds ...");
    o3d.visualization.draw_geometries([pelvisPointCloud, downSampledPelvisPointCloud]);
    print("\t Visualization completed.");

    # Finished processing.
    print("Finished processing.");
def pointCloudMerging():
    # Initializing
    print("Initializing ...");

    # Read the brain point cloud
    print("Reading the brain point cloud ...");
    ## Reading the point cloud from file
    brainPointCloudFilePath = mainFolder + "/BrainPointCloud.ply";
    brainPointCloud = o3d.io.read_point_cloud(brainPointCloudFilePath);
    ## Setting the color of the brain as brain color
    brainColor = [0.8, 0.2, 0.2];  # RGB color for brain
    brainPointCloud.colors = o3d.utility.Vector3dVector([brainColor] * np.asarray(brainPointCloud.points).shape[0]);
    ## Computing the point cloud normals
    brainPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30));
    ## print some information of the brain point cloud
    print("\t Brain point cloud read successfully.");
    print("\t\t Number of points: ", np.asarray(brainPointCloud.points).shape[0]);
    print("\t\t Point cloud bounds: ", brainPointCloud.get_axis_aligned_bounding_box());
    print("\t\t Point cloud colors: ", np.asarray(brainPointCloud.colors).shape[0]);
    print("\t\t Point cloud normals: ", np.asarray(brainPointCloud.normals).shape[0]);

    # Read the pelvis point cloud
    print("Reading the pelvis point cloud ...");
    ## Reading the point cloud from file
    pelvisPointCloudFilePath = mainFolder + "/PelvisPointCloud.ply";
    pelvisPointCloud = o3d.io.read_point_cloud(pelvisPointCloudFilePath);
    ## Setting the color of the pelvis as pelvis color 
    pelvisColor = [0.2, 0.2, 0.8];  # RGB color for pelvis
    pelvisPointCloud.colors = o3d.utility.Vector3dVector([pelvisColor] * np.asarray(pelvisPointCloud.points).shape[0]);
    ## Computing the point cloud normals
    pelvisPointCloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30));
    ## print some information of the pelvis point cloud
    print("\t Pelvis point cloud read successfully.");
    print("\t\t Number of points: ", np.asarray(pelvisPointCloud.points).shape[0]);
    print("\t\t Point cloud bounds: ", pelvisPointCloud.get_axis_aligned_bounding_box());
    print("\t\t Point cloud colors: ", np.asarray(pelvisPointCloud.colors).shape[0]);
    print("\t\t Point cloud normals: ", np.asarray(pelvisPointCloud.normals).shape[0]);

    # Merge the two point clouds
    print("Merging the two point clouds ...");
    mergedPointCloud = brainPointCloud + pelvisPointCloud;
    print("\t Point clouds merged successfully.");
    print("\t\t Number of points: ", np.asarray(mergedPointCloud.points).shape[0]);
    print("\t\t Point cloud bounds: ", mergedPointCloud.get_axis_aligned_bounding_box());
    print("\t\t Point cloud colors: ", np.asarray(mergedPointCloud.colors).shape[0]);
    print("\t\t Point cloud normals: ", np.asarray(mergedPointCloud.normals).shape[0]);

    # Visualize the merged point cloud
    print("Visualizing the merged point cloud ...");
    o3d.visualization.draw_geometries([mergedPointCloud]);
    print("\t Visualization completed.");

    # Scale the pelvis point cloud to fit the brain point cloud
    print("Scaling the pelvis point cloud to fit the brain point cloud ...");
    ## Compute the bounding boxes of the two point clouds
    brainBoundingBox = brainPointCloud.get_axis_aligned_bounding_box();
    pelvisBoundingBox = pelvisPointCloud.get_axis_aligned_bounding_box();
    ## Compute the scaling factor based on the height (y-axis) of the bounding boxes
    brainHeight = brainBoundingBox.get_extent()[1];
    pelvisHeight = pelvisBoundingBox.get_extent()[1];
    scalingFactor = brainHeight / pelvisHeight;
    ## Scale the pelvis point cloud
    pelvisPointCloud.scale(scalingFactor, center=pelvisBoundingBox.get_center());
    print(f"\t Scaling factor: {scalingFactor:.3f}");
    print("\t Scaling completed.");

    # Merge again the two point clouds
    print("Merging again the two point clouds ...");
    mergedPointCloud = brainPointCloud + pelvisPointCloud;
    print("\t Point clouds merged successfully.");
    print("\t\t Number of points: ", np.asarray(mergedPointCloud.points).shape[0]);
    print("\t\t Point cloud bounds: ", mergedPointCloud.get_axis_aligned_bounding_box());
    print("\t\t Point cloud colors: ", np.asarray(mergedPointCloud.colors).shape[0]);
    print("\t\t Point cloud normals: ", np.asarray(mergedPointCloud.normals).shape[0]);
    
    # Visualize the merged point cloud
    print("Visualizing the merged point cloud ...");
    o3d.visualization.draw_geometries([mergedPointCloud]);
    print("\t Visualization completed.");

    # Finished processing.
    print("Finished processing.");

#*********************************************************************************************************************#
#***************************************************MAIN FUNCTIONS****************************************************#
#*********************************************************************************************************************#
def main():
    os.system("cls");
    pointCloudMerging();
if __name__ == "__main__":
    main()
