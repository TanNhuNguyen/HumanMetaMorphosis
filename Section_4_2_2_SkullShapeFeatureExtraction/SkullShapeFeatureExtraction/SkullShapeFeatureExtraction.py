#*********************************************************************************************************************#
#***************************************************SUPPORTING LIBRARIES**********************************************#
#*********************************************************************************************************************#
import os;
import trimesh;
import open3d as o3d;
import numpy as np;
import copy;
from scipy.spatial import cKDTree;

#*********************************************************************************************************************#
#***************************************************SUPPORTING BUFFERS************************************************#
#*********************************************************************************************************************#
mainFolder = "../../../Data/Section_4_2_2_SkullShapeFeatureExtraction";

#*********************************************************************************************************************#
#***************************************************SUPPORTING CLASSES************************************************#
#*********************************************************************************************************************#

#*********************************************************************************************************************#
#***************************************************PROCESSING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def intrinsicShapeSignatureFeatureExtraction():
    # Initialize
    print("Initializing ...");

    # Reading the skull shape
    print("Reading the skull shape ...");
    ## Reading the skull shape
    print("\t Reading the skull shape ...");
    skullShape = o3d.io.read_triangle_mesh(mainFolder + "/TemplateSkullShape.off");
    ## Compute skull shape vertex normals
    print("\t Computing skull shape vertex normals ...");
    skullShape.compute_vertex_normals();
    ## Paint the skull shape with gray color
    print("\t Painting the skull shape with gray color ...");
    skullShape.paint_uniform_color([0.7, 0.7, 0.7]);

    # Converting to point cloud
    print("Converting to point cloud ...");
    skullShapePointCloud = skullShape.sample_points_uniformly(number_of_points = 100000);
    
    # Estimating normals
    print("Estimating normals ...");
    skullShapePointCloud.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 0.01, max_nn = 30));

    # Computing ISS keypoints
    print("Computing ISS keypoints ...");
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(skullShapePointCloud, salient_radius = 0.015, non_max_radius = 0.012,
                                                            gamma_21 = 0.975, gamma_32 = 0.975, min_neighbors = 8);
    print("Number of ISS keypoints: " + str(len(keypoints.points)));
    o3d.io.write_point_cloud(mainFolder + "/ISSKeypoints.ply", keypoints);

    # Visualize the keypoints as sphere with red color
    print("Visualizing the keypoints as sphere with red color ...");
    ## Create a copy of the keypoints to paint them
    print("\t Creating a copy of the keypoints to paint them ...");
    keypointsPainted = copy.deepcopy(keypoints);
    ## Paint the keypoints with red color
    print("\t Painting the keypoints with red color ...");
    keypointsPainted.paint_uniform_color([1, 0, 0]);
    ## Visualize with the big size of points
    print("\t Visualizing with the big size of points ...");
    vis = o3d.visualization.Visualizer();
    vis.create_window(window_name="ISS Keypoints");
    vis.add_geometry(skullShape);
    vis.add_geometry(keypointsPainted);
    render_option = vis.get_render_option();
    render_option.point_size = 20.0;  # Increase point size (default is 1.0)
    vis.run();
    vis.destroy_window();

    # Finished processing
    print("Finished processing.");
def registerTwoSkullShapesUsingKeyPointFeatureExtractorAndFPFHDescriptor():
    # Initialize
    print("Initializing ...")

    # Reading the first and second skull shapes
    print("Reading the first and second skull shapes ...")
    skullShape1 = o3d.io.read_triangle_mesh(mainFolder + "/FirstSkullShape.ply")
    skullShape2 = o3d.io.read_triangle_mesh(mainFolder + "/SecondSkullShape.ply")

    # Convert to point cloud
    print("\t Converting to point cloud ...")
    skullShapePointCloud1 = skullShape1.sample_points_uniformly(number_of_points=100000)
    skullShapePointCloud2 = skullShape2.sample_points_uniformly(number_of_points=100000)

    # Estimate normals
    print("\t Estimating normals ...")
    skullShapePointCloud1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    skullShapePointCloud2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    # Compute ISS keypoints
    print("\t Computing ISS keypoints ...")
    keypoints1 = o3d.geometry.keypoint.compute_iss_keypoints(
        skullShapePointCloud1,
        salient_radius=0.015, non_max_radius=0.012,
        gamma_21=0.975, gamma_32=0.975, min_neighbors=8
    )
    keypoints2 = o3d.geometry.keypoint.compute_iss_keypoints(
        skullShapePointCloud2,
        salient_radius=0.015, non_max_radius=0.012,
        gamma_21=0.975, gamma_32=0.975, min_neighbors=8
    )
    print("\t Number of ISS keypoints on the first skull shape:", len(keypoints1.points))
    print("\t Number of ISS keypoints on the second skull shape:", len(keypoints2.points))

    # Compute FPFH descriptors for the full point cloud
    print("Computing FPFH descriptors ...")
    radius_feature = 0.025  # 2.5 cm, suitable for skull features (unit: meter)
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(skullShapePointCloud1, search_param)
    fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(skullShapePointCloud2, search_param)

    # Find indices of keypoints in the full point cloud
    print("\t Finding indices of keypoints in the full point cloud ...")
    def find_indices(point_cloud, keypoints):
        pc_points = np.asarray(point_cloud.points)
        kp_points = np.asarray(keypoints.points)
        indices = []
        for kp in kp_points:
            idx = np.where(np.all(np.isclose(pc_points, kp, atol=1e-8), axis=1))[0]
            if len(idx) > 0:
                indices.append(idx[0])
        return indices

    indices1 = find_indices(skullShapePointCloud1, keypoints1)
    indices2 = find_indices(skullShapePointCloud2, keypoints2)

    # Extract FPFH descriptors at keypoint locations
    fpfh1_keypoints = fpfh1.data[:, indices1]
    fpfh2_keypoints = fpfh2.data[:, indices2]
    print("\t Size of FPFH descriptors on the first skull shape:", fpfh1_keypoints.shape)
    print("\t Size of FPFH descriptors on the second skull shape:", fpfh2_keypoints.shape)

    # Simple nearest neighbor matching
    print("Matching the descriptors ...")
    fpfh1_data = fpfh1_keypoints.T
    fpfh2_data = fpfh2_keypoints.T
    tree = cKDTree(fpfh2_data)
    matches = []
    for i, feat in enumerate(fpfh1_data):
        dist, idx = tree.query(feat, k=1)
        matches.append((i, idx))
    print("\t Number of matches:", len(matches))

    # Visualize the matches
    print("Visualizing the matches ...")
    matched_keypoints1 = o3d.geometry.PointCloud()
    matched_keypoints2 = o3d.geometry.PointCloud()
    for match in matches:
        matched_keypoints1.points.append(keypoints1.points[match[0]])
        matched_keypoints2.points.append(keypoints2.points[match[1]])
    matched_keypoints1.paint_uniform_color([1, 0, 0])
    matched_keypoints2.paint_uniform_color([1, 0, 0])

    # Add lines between the matched keypoints
    lines = [[i, i] for i in range(len(matches))]
    colors = [[0, 1, 0] for _ in range(len(lines))]  # Green lines
    all_points = np.vstack((np.asarray(matched_keypoints1.points), np.asarray(matched_keypoints2.points)))
    lines_o3d = o3d.utility.Vector2iVector([[line[0], line[1] + len(matched_keypoints1.points)] for line in lines])
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=lines_o3d
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Matched Keypoints")
    vis.add_geometry(skullShape1)
    vis.add_geometry(skullShape2)
    vis.add_geometry(matched_keypoints1)
    vis.add_geometry(matched_keypoints2)
    vis.add_geometry(line_set)
    render_option = vis.get_render_option()
    render_option.point_size = 10.0  # Increase point size (default is 1.0)
    vis.run()
    vis.destroy_window()

    print("Finished processing.")
def registerTwoSkullShapesUsingFPFHRANSAC():
    # Initialize
    print("Initializing ...")

    # Reading the first and second skull shapes
    print("Reading the first and second skull shapes ...")
    skullShape1 = o3d.io.read_triangle_mesh(mainFolder + "/FirstSkullShape.ply")
    skullShape2 = o3d.io.read_triangle_mesh(mainFolder + "/SecondSkullShape.ply")

    # Convert to point cloud
    print("\t Converting to point cloud ...")
    skullShapePointCloud1 = skullShape1.sample_points_uniformly(number_of_points=100000)
    skullShapePointCloud2 = skullShape2.sample_points_uniformly(number_of_points=100000)

    # Estimate normals
    print("\t Estimating normals ...")
    skullShapePointCloud1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    skullShapePointCloud2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    # Compute FPFH descriptors for the full point cloud
    print("Computing FPFH descriptors ...")
    radius_feature = 0.025  # 2.5 cm, suitable for skull features (unit: meter)
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    fpfh1 = o3d.pipelines.registration.compute_fpfh_feature(skullShapePointCloud1, search_param)
    fpfh2 = o3d.pipelines.registration.compute_fpfh_feature(skullShapePointCloud2, search_param)

    # RANSAC registration
    print("Running FPFH-based RANSAC registration ...")
    distance_threshold = 0.03  # 3 cm, adjust as needed
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        skullShapePointCloud1, skullShapePointCloud2, fpfh1, fpfh2, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # RANSAC correspondence set size
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )

    print("RANSAC registration finished.")
    print("Transformation matrix:")
    print(result.transformation)

    # Visualize the aligned point clouds
    print("Visualizing the aligned skull shapes ...")
    skullShapePointCloud1.paint_uniform_color([1, 0.706, 0])
    skullShapePointCloud2.paint_uniform_color([0, 0.651, 0.929])
    skullShapePointCloud1.transform(result.transformation)
    o3d.visualization.draw_geometries([skullShapePointCloud1, skullShapePointCloud2])

    print("Finished processing.")

#*********************************************************************************************************************#
#***************************************************MAIN FUNCTIONS****************************************************#
#*********************************************************************************************************************#
def main():
    os.system("cls");
    registerTwoSkullShapesUsingFPFHRANSAC();
    
if __name__ == "__main__":
    main()