#***********************************************************************************************************************************************#
#************************************************************SUPPORTING LIBRARIES***************************************************************#
#***********************************************************************************************************************************************#
import os;
import numpy as np;
import open3d as o3d;
from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
import vtk;
from scipy.stats import pearsonr;
import matplotlib.pyplot as plt;

#***********************************************************************************************************************************************#
#*************************************************************VISUALIZER CLASS******************************************************************#
#***********************************************************************************************************************************************#

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING BUFFERS*****************************************************************#
#***********************************************************************************************************************************************#
mainFolder = "../../../Data/Section_5_2_2_HeadandSkullShapeCorrelationAnalyses";

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
def readListOfStringsFromTextFile(filePath):
    # Checking file path
    if not os.path.isfile(filePath):
        raise FileNotFoundError(f"readListOfStringsFromTextFile:: The file at {filePath} does not exist.");

    # Initialize
    listOfStrings = [];

    # Read the text file
    with open(filePath, 'r') as file:
        for line in file:
            listOfStrings.append(line.strip());

    # Return the list of strings
    return listOfStrings;
def saveListOfStringsToTextFile(listOfStrings, filePath):
    # Checking the directory
    directory = os.path.dirname(filePath);
    if not os.path.exists(directory):
        os.makedirs(directory);

    # Save the list of strings to the text file
    with open(filePath, 'w') as file:
        for string in listOfStrings:
            file.write(f"{string}\n");
def visualizeProgressBar(progress, total, barLength=40):
    percent = float(progress) / total;
    arrow = '-' * int(round(percent * barLength)-1) + '>';
    spaces = ' ' * (barLength - len(arrow));

    print(f"Progress: [{arrow}{spaces}] {int(round(percent * 100))}%", end='\r');
    if progress == total:
        print();
def readO3DMesh(inFilePath):
    # Checking the input file path
    if not os.path.isfile(inFilePath):
        raise FileNotFoundError(f"readO3DMesh:: The input file at {inFilePath} does not exist.");
    # Read the mesh using Open3D
    mesh = o3d.io.read_triangle_mesh(inFilePath);

    # Return the mesh
    return mesh;
def computePointsToPointsDistance(inPointsA, inPointsB):
    # Checking inputs
    if inPointsA is None or len(inPointsA) == 0:
        raise ValueError("computePointsToPointsDistance:: The input points A is None or empty.");
    if inPointsB is None or len(inPointsB) == 0:
        raise ValueError("computePointsToPointsDistance:: The input points B is None or empty.");
    if inPointsA.shape != inPointsB.shape:
        raise ValueError("computePointsToPointsDistance:: The input points A and B have different shapes.");

    # Compute the distances
    distances = np.linalg.norm(inPointsA - inPointsB, axis=1);

    # Return the distances
    return distances;
def drawFusionMatrix(headPCAData, skullPCAData):
    """
    head_pca_data: numpy array of shape (n_samples, n_components)
    skull_pca_data: numpy array of shape (n_samples, n_components)
    Assumes both have the same number of samples and components (e.g., 20).
    """
    # Prepare buffers
    numComps = headPCAData.shape[1];
    fusion_matrix = np.zeros((numComps, numComps));

    # Compute the fusion matrix
    for i in range(numComps):
        for j in range(numComps):
            # Pearson correlation between head component i and skull component j
            corr, _ = pearsonr(headPCAData[:, i], skullPCAData[:, j]);
            fusion_matrix[j, i] = corr;  # rows: skull, columns: head

    # Plot the fusion matrix
    plt.figure(figsize=(8, 6));
    plt.imshow(fusion_matrix, cmap='coolwarm', vmin=-1, vmax=1);
    plt.colorbar(label='Pearson Correlation');
    plt.xlabel('Head PCA Component');
    plt.ylabel('Skull PCA Component');
    plt.title('Fusion Matrix: Head vs Skull PCA Components');
    plt.xticks(np.arange(numComps), [f'H{i+1}' for i in range(numComps)]);
    plt.yticks(np.arange(numComps), [f'S{j+1}' for j in range(numComps)]);

    # Annotate each cell with the correlation value
    for i in range(numComps):
        for j in range(numComps):
            plt.text(i, j, f"{fusion_matrix[j, i]:.2f}", ha='center', va='center', color='black', fontsize=8);

    # Tight layout and show
    plt.tight_layout();
    plt.show();

#***********************************************************************************************************************************************#
#************************************************************PROCESSING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
def trainTestIDSplit():
    # Initialize
    print("Splitting the dataset into training and test sets...");

    # Reading full list of IDs
    print("Reading full list of IDs...");
    subjectIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "SubjectIDs.txt"));

    # Training and test split
    print("Splitting into training and test sets...");
    np.random.seed(42);
    np.random.shuffle(subjectIDs);
    splitIndex = int(0.8 * len(subjectIDs));
    trainingIDs = subjectIDs[:splitIndex];
    testIDs = subjectIDs[splitIndex:];
    print(f"Training set size: {len(trainingIDs)}");
    print(f"Test set size: {len(testIDs)}");

    # Save the splits
    print("Saving the training and test IDs...");
    saveListOfStringsToTextFile(trainingIDs, os.path.join(mainFolder, "TrainingIDs.txt"));
    saveListOfStringsToTextFile(testIDs, os.path.join(mainFolder, "TestIDs.txt"));

    # Finished processing.
    print("Finished processing.");
def buildPCAModelsOfHeadAndSkullShapes():
    # Initialize
    print("Building PCA models of head and skull shapes...");

    # Reading training IDs
    print("Reading training IDs...");
    trainingIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TrainingIDs.txt"));

    # Forming head training data
    print("Forming head training data...");
    headData = [];
    for subjectID in trainingIDs:
        # Print progress
        visualizeProgressBar(len(headData)+1, len(trainingIDs));
        # Read head mesh
        headShape = readO3DMesh(os.path.join(mainFolder, "AlignedFullHeadShapes", f"{subjectID}-PersonalizedHeadShape.ply"));
        # Convert to numpy array and flatten
        headPoints = np.asarray(headShape.vertices).flatten();
        headData.append(headPoints);
    headData = np.array(headData);

    # Forming skull training data
    print("Forming skull training data...");
    skullData = [];
    for subjectID in trainingIDs:
        # Print progress
        visualizeProgressBar(len(skullData)+1, len(trainingIDs));
        # Read skull mesh
        skullShape = readO3DMesh(os.path.join(mainFolder, "AlignedFullSkullShapes", f"{subjectID}-PersonalizedSkullShape.ply"));
        # Convert to numpy array and flatten
        skullPoints = np.asarray(skullShape.vertices).flatten();
        skullData.append(skullPoints);
    skullData = np.array(skullData);

    # Train the PCA shape model of the head shapes
    print("Training the PCA shape model of the head shapes...");
    numOfComponents = 20;
    headScaler = StandardScaler();
    headDataScaled = headScaler.fit_transform(headData);
    headPCA = PCA(n_components=numOfComponents);
    headPCA.fit(headDataScaled);

    # Train the PCA shape model of the skull shapes
    print("Training the PCA shape model of the skull shapes...");
    skullScaler = StandardScaler();
    skullDataScaled = skullScaler.fit_transform(skullData);
    skullPCA = PCA(n_components=numOfComponents);
    skullPCA.fit(skullDataScaled);

    # Save the trained model to files
    print("Saving the trained PCA models...");
    np.save(os.path.join(mainFolder, "TrainedModels/TrainedHeadPCAModel.npy"), headPCA);
    np.save(os.path.join(mainFolder, "TrainedModels/TrainedSkullPCAModel.npy"), skullPCA);
    np.save(os.path.join(mainFolder, "TrainedModels/TrainedHeadScaler.npy"), headScaler);
    np.save(os.path.join(mainFolder, "TrainedModels/TrainedSkullScaler.npy"), skullScaler);

    # Finished processing
    print("Finished processing.");
def correlationAnalysisBetweenHeadAndSkullShapes():
    # Initialize
    print("Performing correlation analysis between head and skull shapes...");

    # Load the trained pca models of the head shape and skull shape
    print("Loading the trained PCA models...");
    headPCA = np.load(os.path.join(mainFolder, "TrainedModels/TrainedHeadPCAModel.npy"), allow_pickle=True).item();
    skullPCA = np.load(os.path.join(mainFolder, "TrainedModels/TrainedSkullPCAModel.npy"), allow_pickle=True).item();
    headScaler = np.load(os.path.join(mainFolder, "TrainedModels/TrainedHeadScaler.npy"), allow_pickle=True).item();
    skullScaler = np.load(os.path.join(mainFolder, "TrainedModels/TrainedSkullScaler.npy"), allow_pickle=True).item();

    # Reading training IDs
    print("Reading training IDs...");
    trainingIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TrainingIDs.txt"));
    
    # Forming head and skull data
    print("Forming head and skull data...");
    headData = [];
    skullData = [];
    for subjectID in trainingIDs:
        # Print progress
        visualizeProgressBar(len(headData)+1, len(trainingIDs));
        # Read head mesh
        headShape = readO3DMesh(os.path.join(mainFolder, "AlignedFullHeadShapes", f"{subjectID}-PersonalizedHeadShape.ply"));
        # Convert to numpy array and flatten
        headPoints = np.asarray(headShape.vertices).flatten();
        headData.append(headPoints);
        # Read skull mesh
        skullShape = readO3DMesh(os.path.join(mainFolder, "AlignedFullSkullShapes", f"{subjectID}-PersonalizedSkullShape.ply"));
        # Convert to numpy array and flatten
        skullPoints = np.asarray(skullShape.vertices).flatten();
        skullData.append(skullPoints);
    headData = np.array(headData);
    skullData = np.array(skullData);

    # Scale the data
    print("Scaling the data...");
    skullDataScaled = skullScaler.transform(skullData);
    headDataScaled = headScaler.transform(headData);

    # Compute head and skull parameter data
    print("Computing head and skull parameter data...");
    headParamData = headPCA.transform(headDataScaled);
    skullParamData = skullPCA.transform(skullDataScaled);

    # Draw the fusion matrix
    print("Drawing the fusion matrix...");
    drawFusionMatrix(headParamData, skullParamData);

    # Finished processing.
    print("Finished processing.");

#***********************************************************************************************************************************************#
#************************************************************MAIN FUNCTIONS*********************************************************************#
#***********************************************************************************************************************************************#
def main():
    os.system("cls");
    correlationAnalysisBetweenHeadAndSkullShapes();
if __name__ == "__main__":
    main()