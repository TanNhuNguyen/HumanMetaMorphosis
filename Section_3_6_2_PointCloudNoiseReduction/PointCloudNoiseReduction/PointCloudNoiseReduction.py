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
from skimage.restoration import denoise_bilateral;
import pandas as pd;

import torch;
import torch.nn as nn;
import torch.optim as optim;

#*********************************************************************************************************************#
#***************************************************SUPPORTING BUFFERS************************************************#
#*********************************************************************************************************************#
mainFolder = "../../../Data/Section_3_6_2_PointCloudNoiseReduction";
selectedPointCloudWithNoiseFolder = mainFolder + "/SelectedPointCloudWithNoise";
trainingDatasetFolder = mainFolder + "/pointCleanNetOutliersTrainingSet";

#*********************************************************************************************************************#
#***************************************************SUPPORTING CLASSES************************************************#
#*********************************************************************************************************************#
class PointCloudOutlierNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=2):
        super(PointCloudOutlierNet, self).__init__()
        # Now input_dim is 4: x, y, z, density
        self.model = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for density
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

class OutlierClassifier:
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=2, device='cpu'):
        self.device = device
        self.net = PointCloudOutlierNet(input_dim, hidden_dim, num_classes).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def compute_density(self, points, radius=0.05):
        # Compute density for each point: number of neighbors within radius
        from sklearn.neighbors import KDTree
        tree = KDTree(points)
        densities = tree.query_radius(points, r=radius, count_only=True)
        return densities.astype(np.float32).reshape(-1, 1)

    def train(self, points, labels, epochs=10):
        """
        points: numpy array of shape (N, 3)
        labels: numpy array of shape (N,) with 0 (not outlier) or 1 (outlier)
        """
        self.net.train();
        # Compute density and concatenate as feature
        densities = self.compute_density(points)
        features = np.concatenate([points, densities], axis=1)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device);
        totalLoss = 0;
        for epoch in range(epochs):
            outputs = self.net(features)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            totalLoss += loss.item();
        avgLoss = totalLoss / epochs;
        return avgLoss;       

    def predict(self, points):
        """
        points: numpy array of shape (N, 3)
        returns: numpy array of shape (N,) with predicted labels (0 or 1)
        """
        self.net.eval()
        densities = self.compute_density(points)
        features = np.concatenate([points, densities], axis=1)
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.net(features)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def evaluate(self, points, predicted_labels, ground_truth_labels):
        """
        points: numpy array of shape (N, 3)
        predicted_labels: numpy array of shape (N,) with predicted labels
        ground_truth_labels: numpy array of shape (N,) with ground truth labels
        returns: accuracy, precision, recall, f1_score
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(ground_truth_labels, predicted_labels)
        precision = precision_score(ground_truth_labels, predicted_labels)
        recall = recall_score(ground_truth_labels, predicted_labels)
        f1 = f1_score(ground_truth_labels, predicted_labels)
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        return accuracy, precision, recall, f1

class PointCloudDataset(torch.utils.data.Dataset):
        def __init__(self, file_list, root_dir):
            self.file_list = file_list;
            self.root_dir = root_dir;

        def __len__(self):
            return len(self.file_list);

        def __getitem__(self, idx):
            file_name = self.file_list[idx][0];
            pointFilePath = os.path.join(self.root_dir, file_name);
            labelFilePath = os.path.join(self.root_dir, file_name.replace('.xyz', '.outliers'));
            pointData = np.loadtxt(pointFilePath);
            points = pointData[:, :3];
            labelData = np.loadtxt(labelFilePath);
            labels = labelData.flatten().astype(int);
            return points, labels;

def printProgressBar(current, total, bar_length=40, info=""):
    percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\rProgress: [{arrow}{spaces}] {int(percent * 100)}% {info}', end='')
    if current == total:
        print()  # Move to next line when done           

#*********************************************************************************************************************#
#***************************************************PROCESSING FUNCTIONS**********************************************#
#*********************************************************************************************************************#
def statisticalOutlierRemoval():
    # Initialize
    print("Initializing ...");

    # Reading the noisy point cloud.
    print("Reading the noisy point cloud ...");
    ## Reading the point cloud
    noisyPointCloud = o3d.io.read_point_cloud(selectedPointCloudWithNoiseFolder + "/bunny140k_outliers10%.xyz");
    ## Print information for the point cloud
    print("\t Noisy Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(noisyPointCloud.points).shape[0]);
    ## Set point cloud color
    noisyPointCloud.paint_uniform_color([1, 0.706, 0]);
    ## Visualize the noisy point cloud
    o3d.visualization.draw_geometries([noisyPointCloud]);

    # Statistical Outlier Removal
    print("Statistical Outlier Removal ...");
    ## Removing the outliers
    cl, ind = noisyPointCloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0);
    ## Select inlier points
    inlierPointCloud = noisyPointCloud.select_by_index(ind);
    ## Print information for the inlier point cloud
    print("\t Inlier Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(inlierPointCloud.points).shape[0]);
    ## Set point cloud color
    inlierPointCloud.paint_uniform_color([0, 0.651, 0.929]);
    ## Visualize the inlier point cloud
    o3d.visualization.draw_geometries([inlierPointCloud]);

    # Reduce the number of neighbors to 10
    print("Reducing the number of neighbors to 10 ...");
    ## Removing the outliers
    cl, ind = noisyPointCloud.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0);
    ## Select inlier points
    inlierPointCloud = noisyPointCloud.select_by_index(ind);
    ## Print information for the inlier point cloud
    print("\t Inlier Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(inlierPointCloud.points).shape[0]);
    ## Set point cloud color
    inlierPointCloud.paint_uniform_color([0, 0.651, 0.929]);
    ## Visualize the inlier point cloud
    o3d.visualization.draw_geometries([inlierPointCloud]);

    # Finished processing.
    print("Finished processing.");
def radiusOutlierRemoval():
    # Initialize
    print("Initializing ...");

    # Reading the noisy point cloud.
    print("Reading the noisy point cloud ...");
    ## Reading the point cloud
    noisyPointCloud = o3d.io.read_point_cloud(selectedPointCloudWithNoiseFolder + "/bunny140k_outliers10%.xyz");
    ## Print information for the point cloud
    print("\t Noisy Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(noisyPointCloud.points).shape[0]);
    ## Set point cloud color
    noisyPointCloud.paint_uniform_color([1, 0.706, 0]);
    ## Visualize the noisy point cloud
    o3d.visualization.draw_geometries([noisyPointCloud]);

    # Radius Outlier Removal
    print("Radius Outlier Removal ...");
    ## Removing the outliers
    cl, ind = noisyPointCloud.remove_radius_outlier(nb_points=16, radius=0.05);
    ## Select inlier points
    inlierPointCloud = noisyPointCloud.select_by_index(ind);
    ## Print information for the inlier point cloud
    print("\t Inlier Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(inlierPointCloud.points).shape[0]);
    ## Set point cloud color
    inlierPointCloud.paint_uniform_color([0, 0.651, 0.929]);
    ## Visualize the inlier point cloud
    o3d.visualization.draw_geometries([inlierPointCloud]);

    # Increase the radius to 0.1
    print("Increasing the radius to 0.1 ...");
    ## Removing the outliers
    cl, ind = noisyPointCloud.remove_radius_outlier(nb_points=16, radius=0.1);
    ## Select inlier points
    inlierPointCloud = noisyPointCloud.select_by_index(ind);
    ## Print information for the inlier point cloud
    print("\t Inlier Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(inlierPointCloud.points).shape[0]);
    ## Set point cloud color
    inlierPointCloud.paint_uniform_color([0, 0.651, 0.929]);
    ## Visualize the inlier point cloud
    o3d.visualization.draw_geometries([inlierPointCloud]);

    # Finished processing.
    print("Finished processing.");
def biliateralFilter():
    # Initialize
    print("Initializing ...");

    # Reading the noisy point cloud.
    print("Reading the noisy point cloud ...");
    ## Reading the point cloud
    noisyPointCloud = o3d.io.read_point_cloud(selectedPointCloudWithNoiseFolder + "/bunny140k_outliers10%.xyz");
    ## Print information for the point cloud
    print("\t Noisy Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(noisyPointCloud.points).shape[0]);
    ## Set point cloud color
    noisyPointCloud.paint_uniform_color([1, 0.706, 0]);
    ## Visualize the noisy point cloud
    o3d.visualization.draw_geometries([noisyPointCloud]);
    
    # Biliateral Filter
    print("Biliateral Filter ...");
    ## Get the point data from the point cloud
    points = np.asarray(noisyPointCloud.points);
    ## Apply the bilateral filter
    filtered_points = np.zeros_like(points)
    for i in range(3):
        filtered_points[:, i] = denoise_bilateral(points[:, i], sigma_color=0.1, sigma_spatial=0.5, bins=1000);
    ## Create a new point cloud with the filtered points
    filteredPointCloud = o3d.geometry.PointCloud();
    filteredPointCloud.points = o3d.utility.Vector3dVector(filtered_points);
    ## Print information for the filtered point cloud
    print("\t Filtered Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(filteredPointCloud.points).shape[0]);
    ## Set point cloud color
    filteredPointCloud.paint_uniform_color([0, 0.651, 0.929]);
    ## Visualize the filtered point cloud
    o3d.visualization.draw_geometries([filteredPointCloud]);

    # Finished processing.
    print("Finished processing.");
def voxelGridDownsampling():
    # Initialize
    print("Initializing ...");

    # Reading the noisy point cloud.
    print("Reading the noisy point cloud ...");
    ## Reading the point cloud
    noisyPointCloud = o3d.io.read_point_cloud(selectedPointCloudWithNoiseFolder + "/bunny140k_outliers10%.xyz");
    ## Print information for the point cloud
    print("\t Noisy Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(noisyPointCloud.points).shape[0]);
    ## Set point cloud color
    noisyPointCloud.paint_uniform_color([1, 0.706, 0]);
    ## Visualize the noisy point cloud
    o3d.visualization.draw_geometries([noisyPointCloud]);

    # Voxel Grid Downsampling
    print("Voxel Grid Downsampling ...");
    ## Downsample the point cloud
    downsampledPointCloud = noisyPointCloud.voxel_down_sample(voxel_size=0.05);
    ## Print information for the downsampled point cloud
    print("\t Downsampled Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(downsampledPointCloud.points).shape[0]);
    ## Set point cloud color
    downsampledPointCloud.paint_uniform_color([0, 0.651, 0.929]);
    ## Visualize the downsampled point cloud
    o3d.visualization.draw_geometries([downsampledPointCloud]);

    # Finished processing.
    print("Finished processing.");
def deepLearningBasedDenoising_trainTestSplit():
    # Initialize
    print("Initializing ...");

    # Get the train test split for the dataset
    print("Getting the train test split for the dataset ...");
    ## List all files having .xyz extension
    allFiles = [f for f in os.listdir(trainingDatasetFolder) if f.endswith('.xyz')];
    ## Shuffle the files
    np.random.shuffle(allFiles);
    ## Split the files into train and test sets
    trainFiles = allFiles[:int(0.8 * len(allFiles))];
    testFiles = allFiles[int(0.8 * len(allFiles)):];
    ## Save the train and test files to csv files
    pd.DataFrame(trainFiles).to_csv(trainingDatasetFolder + "/train_files.csv", index=False, header=False);
    pd.DataFrame(testFiles).to_csv(trainingDatasetFolder + "/test_files.csv", index=False, header=False);

    # Finished processing.
    print("Finished processing.");
def deepLearningBasedOutlierRemoval_train():
    # Initialize
    print("Initializing ...");

    # Reading train and test file names
    print("Reading train and test file names ...");
    ## Reading the train files
    trainFiles = pd.read_csv(trainingDatasetFolder + "/train_files.csv", header=None);
    ## Reading the test files
    testFiles = pd.read_csv(trainingDatasetFolder + "/test_files.csv", header=None);

    # Forming the training dataset using the torch data framework
    print("Forming the training dataset using the torch data framework ...");
    trainDataset = PointCloudDataset(trainFiles.values, trainingDatasetFolder);

    # Creating the data loader for the training dataset
    print("Creating the data loader for the training dataset ...");
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=1, shuffle=True);

    # Initialize the outlier classifier
    print("Initializing the outlier classifier ...");
    device = 'cuda';
    outlierClassifier = OutlierClassifier(device=device);
    # Training the outlier classifier
    print("Training the outlier classifier ...");
    for epoch in range(5):  # Number of epochs
        print(f"#*************************** Global Epoch {epoch+1}/5");
        totalBatches = len(trainDataLoader);
        totalLoss = 0;
        for i, (points, labels) in enumerate(trainDataLoader):
            points = points.squeeze(0).numpy();  # Shape: (N, 3)
            labels = labels.squeeze(0).numpy();  # Shape: (N,)
            avgLoss = outlierClassifier.train(points, labels, epochs=1);
            totalLoss += avgLoss;
            printProgressBar(i + 1, totalBatches, info=f"Avg Loss: {avgLoss:.4f}");
        epochLoss = totalLoss / totalBatches;
        print(f"Epoch {epoch+1} Loss: {epochLoss:.4f}");
    
    # Save the trained model
    print("Saving the trained model ...");
    torch.save(outlierClassifier.net.state_dict(), mainFolder + "/outlier_classifier.pth");

    # Finished processing.
    print("Finished processing.");
def deepLearningBasedOutlierRemoval_test():
    # Initialize
    print("Initializing ...");

    # Load the trained model
    print("Loading the trained model ...");
    device = 'cuda';
    outlierClassifier = OutlierClassifier(device=device);
    outlierClassifier.net.load_state_dict(torch.load(mainFolder + "/outlier_classifier.pth"));
    outlierClassifier.net.eval();

    # Reading the test file names
    print("Reading the test file names ...");
    ## Reading the test files
    testFiles = pd.read_csv(trainingDatasetFolder + "/test_files.csv", header=None);
    testDataset = PointCloudDataset(testFiles.values, trainingDatasetFolder);
    testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False);
    totalTestBatches = len(testDataLoader);

    # Evaluating the model on the test dataset
    print("Evaluating the model on the test dataset ...");
    allPredictedLabels = [];
    allGroundTruthLabels = [];
    for i, (points, labels) in enumerate(testDataLoader):
        points = points.squeeze(0).numpy();
        labels = labels.squeeze(0).numpy();
        predictedLabels = outlierClassifier.predict(points);
        allPredictedLabels.append(predictedLabels);
        allGroundTruthLabels.append(labels);
        printProgressBar(i + 1, totalTestBatches, info=f"Evaluated {i+1}/{totalTestBatches}");
    allPredictedLabels = np.concatenate(allPredictedLabels);
    allGroundTruthLabels = np.concatenate(allGroundTruthLabels);
    accuracy, precision, recall, f1 = outlierClassifier.evaluate(None, allPredictedLabels, allGroundTruthLabels);
    print(f"Test Set - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}");

    # Finished processing.
    print("Finished processing.");
def deepLearningBasedDenoising_caseIllustration():
    # Initialize
    print("Initializing ...");

    # Reading the noisy point cloud.
    print("Reading the noisy point cloud ...");
    ## Reading the point cloud
    noisyPointCloud = o3d.io.read_point_cloud(trainingDatasetFolder + "/armadillo140k_outliers10%.xyz");
    ## Print information for the point cloud
    print("\t Noisy Point Cloud: ");
    print("\t\t Number of points: ", np.asarray(noisyPointCloud.points).shape[0]);
    ## Set point cloud color
    noisyPointCloud.paint_uniform_color([1, 0.706, 0]);
    ## Visualize the noisy point cloud
    o3d.visualization.draw_geometries([noisyPointCloud]);
    
    # Deep Learning Based Outlier Removal
    print("Deep Learning Based Outlier Removal ...");
    ## Load the trained model
    device = 'cuda';
    outlierClassifier = OutlierClassifier(device=device);
    outlierClassifier.net.load_state_dict(torch.load(mainFolder + "/outlier_classifier.pth"));
    outlierClassifier.net.eval();
    ## Get the point data from the point cloud
    points = np.asarray(noisyPointCloud.points);
    ## Predict the outlier labels
    predictedLabels = outlierClassifier.predict(points);
    ## Select inlier points
    inlierPoints = points[predictedLabels == 0];
    ## Print information for the inlier points
    print("\t Inlier Points: ");
    print("\t\t Number of points: ", inlierPoints.shape[0]);
    ## Visualize the inlier points
    inlierPointCloud = o3d.geometry.PointCloud();
    inlierPointCloud.points = o3d.utility.Vector3dVector(inlierPoints);
    inlierPointCloud.paint_uniform_color([0, 1, 0]);
    o3d.visualization.draw_geometries([inlierPointCloud]);

    # Finished processing.
    print("Finished processing.");

#*********************************************************************************************************************#
#***************************************************MAIN FUNCTIONS****************************************************#
#*********************************************************************************************************************#
def main():
    os.system("cls");
    deepLearningBasedOutlierRemoval_train();
if __name__ == "__main__":
    main()