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
from sklearn.linear_model import LinearRegression;

#***********************************************************************************************************************************************#
#*************************************************************VISUALIZER CLASS******************************************************************#
#***********************************************************************************************************************************************#
class Visualizer:
    # Supporting functions
    def __init__(self):
        self.renderer = None
        self.render_window = None
        self.interactor = None
        self.mesh_actors = {}
        self.camera = None
        self.isInitialized = False
    def _convertO3DMeshToVTKPolyData(self, inO3DMesh):
        if not isinstance(inO3DMesh, o3d.geometry.TriangleMesh):
            raise ValueError("The input mesh is not an Open3D TriangleMesh.");
        vertices = np.asarray(inO3DMesh.vertices);
        faces = np.asarray(inO3DMesh.triangles);
        points = vtk.vtkPoints()
        for v in vertices:
            points.InsertNextPoint(v)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        triangles = vtk.vtkCellArray()
        for f in faces:
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, f[0])
            triangle.GetPointIds().SetId(1, f[1])
            triangle.GetPointIds().SetId(2, f[2])
            triangles.InsertNextCell(triangle)
        polydata.SetPolys(triangles)
        return polydata

    # For general interface functions
    def initializeRendering(self):
        # Set flag
        self.isInitialized = False;

        # Create renderer and set background to gray
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.5, 0.5, 0.5)

        # Create render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetSize(1600, 900)
        self.render_window.AddRenderer(self.renderer)

        # Create interactor and set trackball camera style
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Store camera for reset
        self.camera = self.renderer.GetActiveCamera()

        # Set flag
        self.isInitialized = True
    def render(self):
        # Checking initialization
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");
    
        # Render the scene
        if self.render_window:
            self.render_window.Render()
    def startInteractionWindows(self):
        # Checking initialization
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");
    
        # Start the interaction
        if self.interactor:
            self.interactor.Initialize()
            self.interactor.Start()
    def removeAllRenderingObjects(self):
        """
        Removes all mesh actors and slider widgets from the scene and refreshes the view.
        """
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");
        # Remove mesh actors
        for actor in list(self.mesh_actors.values()):
            self.renderer.RemoveActor(actor);
        self.mesh_actors.clear();
        # Disable and clear sliders if present
        if hasattr(self, 'slider_widgets'):
            for w in self.slider_widgets:
                try:
                    w.EnabledOff();
                except:
                    pass
            self.slider_widgets = [];
        self.render();
    
    # For camera functions
    def resetCamera(self):
        # Checking initialization
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");
        
        # Reset the camera and render
        if self.renderer:
            self.renderer.ResetCamera()
            self.render()

    # For mesh functions
    def addO3DMesh(self, inMeshName, inO3DMesh, inRGBColor):
        # Check initialization
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");

        # Convert inO3DMesh to vtkPolyData
        meshPolyData = self._convertO3DMeshToVTKPolyData(inO3DMesh)
        
        # mesh: expects a vtkPolyData object
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(meshPolyData)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*inRGBColor)
        self.renderer.AddActor(actor)
        self.mesh_actors[inMeshName] = actor
    def updateO3DMeshVertices(self, inMeshName, newMeshVertices):
        # Checking initialization
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");
    
        # Check is name exists
        if inMeshName not in self.mesh_actors:
            raise ValueError(f"Visualizer:: No mesh with the name {inMeshName} exists.");
    
        # newMeshVertices: numpy array of shape (N, 3)
        actor = self.mesh_actors.get(inMeshName)
        if actor is not None:
            polydata = actor.GetMapper().GetInput()
            points = vtk.vtkPoints()
            for v in newMeshVertices:
                points.InsertNextPoint(v)
            polydata.SetPoints(points)
            polydata.Modified()
            self.render()

    # Slider bar functions
    def addVTKSliderBar(self, sliderName, minValue, maxValue, initialValue, callback, position=(0.1, 0.1, 0.4, 0.1)):
        """
        Adds a VTK slider widget to the render window.
        callback: function to call with the slider value when it changes.
        position: (xmin, ymin, xmax, ymax) in normalized display coordinates.
        """
        slider_rep = vtk.vtkSliderRepresentation2D();
        slider_rep.SetMinimumValue(minValue);
        slider_rep.SetMaximumValue(maxValue);
        slider_rep.SetValue(initialValue);
        slider_rep.SetTitleText(sliderName);
        slider_rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay();
        slider_rep.GetPoint1Coordinate().SetValue(position[0], position[1]);
        slider_rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay();
        slider_rep.GetPoint2Coordinate().SetValue(position[2], position[3]);
        slider_rep.SetSliderLength(0.02);
        slider_rep.SetSliderWidth(0.03);
        slider_rep.SetEndCapLength(0.01);
        slider_rep.SetEndCapWidth(0.03);
        slider_rep.SetTubeWidth(0.005);
        slider_rep.SetLabelFormat("%0.2f");
        slider_rep.SetTitleHeight(0.02);
        slider_rep.SetLabelHeight(0.02);

        slider_widget = vtk.vtkSliderWidget();
        slider_widget.SetInteractor(self.interactor);
        slider_widget.SetRepresentation(slider_rep);
        slider_widget.SetAnimationModeToAnimate();
        slider_widget.EnabledOn();

        def slider_callback(obj, event):
            value = obj.GetRepresentation().GetValue();
            callback(value);

        slider_widget.AddObserver("InteractionEvent", slider_callback);

        # Store reference to prevent garbage collection
        if not hasattr(self, 'slider_widgets'):
            self.slider_widgets = [];
        self.slider_widgets.append(slider_widget);

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING BUFFERS*****************************************************************#
#***********************************************************************************************************************************************#
mainFolder = "../../../Data/Section_5_3_3_HeadShapePredictionfromtheSkullShape";

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
def trainHeadShapeToSkullShapeRelationship():
    # Initialize
    print("Training the relationship between head and skull shapes...");

    # Reading the training IDs
    print("Reading training IDs...");
    trainingIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TrainingIDs.txt"));

    # Train the head shape and skull shape pca models
    print("Train the head shape and skull shape pca models...");
    ## Load the head and skull shape training data
    print("\t Loading the head and skull shape training data...");
    headShapeData = []; skullShapeData = [];
    for subjectID in trainingIDs:
        # Print progress
        visualizeProgressBar(len(headShapeData)+1, len(trainingIDs));
        # Read head shape
        headShape = readO3DMesh(os.path.join(mainFolder, "AlignedFullHeadShapes", f"{subjectID}-PersonalizedHeadShape.ply"));
        headShapeVertices = np.asarray(headShape.vertices).flatten();
        headShapeData.append(headShapeVertices);
        # Read skull shape
        skullShape = readO3DMesh(os.path.join(mainFolder, "AlignedFullSkullShapes", f"{subjectID}-PersonalizedSkullShape.ply"));
        skullShapeVertices = np.asarray(skullShape.vertices).flatten();
        skullShapeData.append(skullShapeVertices);
    headShapeData = np.array(headShapeData);
    skullShapeData = np.array(skullShapeData);
    ## Scale the head and skull shape data
    print("\t Scale the head and skull shape data ...");
    headShapeScaler = StandardScaler();
    headShapeDataScaled = headShapeScaler.fit_transform(headShapeData);
    skullShapeScaler = StandardScaler();
    skullShapeDataScaled = skullShapeScaler.fit_transform(skullShapeData);
    ## Train the head and skull shape pca models
    print("\t Train the head and skull shape pca models ...");
    numOfComponents = 20;
    headShapePCA = PCA(n_components=numOfComponents);
    headShapePCA.fit(headShapeDataScaled);
    skullShapePCA = PCA(n_components=numOfComponents);
    skullShapePCA.fit(skullShapeDataScaled);
    ## Generate the head and skull shape pca data
    print("\t Generate the head and skull shape pca data ...");
    headPCAData = headShapePCA.transform(headShapeDataScaled);
    skullPCAData = skullShapePCA.transform(skullShapeDataScaled);
    ## Train linear regression model
    print("\t Train linear regression model ...");
    regressionModel = LinearRegression();
    regressionModel.fit(headPCAData, skullPCAData);
    ## Save the trained regression model to file
    print("\t Save the trained regression model to file ...");
    np.save(os.path.join(mainFolder, "TrainedModels/TrainedHeadToSkullShapeRegressionModel.npy"), regressionModel);
    np.save(os.path.join(mainFolder, "TrainedModels/TrainedHeadScaler.npy"), headShapeScaler);
    np.save(os.path.join(mainFolder, "TrainedModels/TrainedSkullScaler.npy"), skullShapeScaler);
    np.save(os.path.join(mainFolder, "TrainedModels/TrainedHeadPCAModel.npy"), headShapePCA);
    np.save(os.path.join(mainFolder, "TrainedModels/TrainedSkullPCAModel.npy"), skullShapePCA);

    # Finished processing.
    print("Finished processing.");
def testHeadShapeToSkullShapeRelationship():
    # Initialize
    print("Testing the relationship between head and skull shapes...");

    # Read the template skull shape and head shape
    print("Read the template skull shape and head shape...");
    templateHeadShape = readO3DMesh(os.path.join(mainFolder, "TemplateHeadShape.ply"));
    templateSkullShape = readO3DMesh(os.path.join(mainFolder, "TemplateSkullShape.ply"));
    templateHeadShapeVertices = np.asarray(templateHeadShape.vertices).flatten();
    templateSkullShapeVertices = np.asarray(templateSkullShape.vertices).flatten();
    templateHeadShapeFaces = np.asarray(templateHeadShape.triangles);
    templateSkullShapeFaces = np.asarray(templateSkullShape.triangles);

    # Read the trained models
    print("Read the trained models...");
    headScaler = np.load(os.path.join(mainFolder, "TrainedModels/TrainedHeadScaler.npy"), allow_pickle=True).item();
    skullScaler = np.load(os.path.join(mainFolder, "TrainedModels/TrainedSkullScaler.npy"), allow_pickle=True).item();
    headPCA = np.load(os.path.join(mainFolder, "TrainedModels/TrainedHeadPCAModel.npy"), allow_pickle=True).item();
    skullPCA = np.load(os.path.join(mainFolder, "TrainedModels/TrainedSkullPCAModel.npy"), allow_pickle=True).item();
    regressionModel = np.load(os.path.join(mainFolder, "TrainedModels/TrainedSkullToHeadShapeRegressionModel.npy"), allow_pickle=True).item();

    # Reading the test IDs
    print("Reading test IDs...");
    testIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TestIDs.txt"));
    print(f"\t Test set size: {len(testIDs)}");

    # Forming testing data
    print("Forming testing data...");
    skullTestData = []; headTestData = [];
    for subjectID in testIDs:
        # Print progress
        visualizeProgressBar(len(skullTestData)+1, len(testIDs));
        # Read skull shape
        skullShape = readO3DMesh(os.path.join(mainFolder, "AlignedFullSkullShapes", f"{subjectID}-PersonalizedSkullShape.ply"));
        skullShapeVertices = np.asarray(skullShape.vertices).flatten();
        skullTestData.append(skullShapeVertices);
        # Read head shape
        headShape = readO3DMesh(os.path.join(mainFolder, "AlignedFullHeadShapes", f"{subjectID}-PersonalizedHeadShape.ply"));
        headShapeVertices = np.asarray(headShape.vertices).flatten();
        headTestData.append(headShapeVertices);
    skullTestData = np.array(skullTestData);
    headTestData = np.array(headTestData);

    # Predicting head shapes from skull shapes
    print("Predicting head shapes from skull shapes...");
    predictedSkullShapes = [];
    for i in range(len(headTestData)):
        # Print progress
        visualizeProgressBar(i+1, len(headTestData));
        # Get the head shape vertex data
        headShapeVertexData = headTestData[i, :];
        # Scale the head shape vertex data
        headShapeVertexDataScaled = headScaler.transform(headShapeVertexData.reshape(1, -1));
        # Project the head shape vertex data to the head shape PCA space
        headShapeVertexDataPCA = headPCA.transform(headShapeVertexDataScaled);
        # Predict the head shape using the regression model
        predictedSkullShapeVertexDataPCA = regressionModel.predict(headShapeVertexDataPCA);
        # Reconstruct the skull shape vertex data from the head shape PCA space
        predictedSkullShapeVertexDataScaled = skullPCA.inverse_transform(predictedSkullShapeVertexDataPCA);
        # Unscale the predicted skull shape vertex data
        predictedSkullShapeVertexData = skullScaler.inverse_transform(predictedSkullShapeVertexDataScaled);
        # Reshape to (N, 3)
        predictedSkullShapeVertices = predictedSkullShapeVertexData.reshape(-1, 3);
        # Forming skull shape mesh
        predictedSkullShape = o3d.geometry.TriangleMesh();
        predictedSkullShape.vertices = o3d.utility.Vector3dVector(predictedSkullShapeVertices);
        predictedSkullShape.triangles = o3d.utility.Vector3iVector(templateSkullShapeFaces);
        # Append to the list
        predictedSkullShapes.append(predictedSkullShape);
    
    # Compute the testing errors
    print("Compute the testing errors...");
    testingErrors = [];
    for i in range(len(predictedSkullShapes)):
        # Print progress
        visualizeProgressBar(i+1, len(predictedSkullShapes));
        # Get the predicted skull shape
        predictedSkullShape = predictedSkullShapes[i];
        predictedSkullShapeVertices = np.asarray(predictedSkullShape.vertices);
        # Get the ground truth skull shape
        groundTruthSkullShapeVertices = skullTestData[i, :].reshape(-1, 3);
        # Compute the point-to-point distances
        distances = computePointsToPointsDistance(predictedSkullShapeVertices, groundTruthSkullShapeVertices);
        meanDistance = np.mean(distances);
        testingErrors.append(meanDistance);
    testingErrors = np.array(testingErrors);

    # Report the testing errors
    print("Report the testing errors...");
    meanError = np.mean(testingErrors);
    stdError = np.std(testingErrors);
    print(f"Mean Testing Error: {meanError}");
    print(f"Standard Deviation of Testing Errors: {stdError}");

    # Visualize the first predicted skull shape vs ground truth
    print("Visualize the first predicted skull shape vs ground truth...");
    ## Preparing data for visualization
    print("\t Preparing data for visualization ...");
    groundTruthSkullShapeVertices = o3d.geometry.TriangleMesh();
    groundTruthSkullShapeVertices.vertices = o3d.utility.Vector3dVector(skullTestData[0, :].reshape(-1, 3));
    groundTruthSkullShapeVertices.triangles = o3d.utility.Vector3iVector(templateSkullShapeFaces);
    groundTruthSkullShapeVertices.compute_vertex_normals();
    predictedSkullShape = predictedSkullShapes[0];
    predictedSkullShape.compute_vertex_normals();
    ## Initialize visualizer
    print("\t Initialize visualizer ...");
    visualizer = Visualizer();
    visualizer.initializeRendering();
    ## Visualize the predicted skull shape first
    print("\t Visualize the predicted skull shape first ...");
    visualizer.addO3DMesh("predictedSkullShape", predictedSkullShape, (0.7, 0.7, 0.7));
    visualizer.render();
    visualizer.resetCamera();
    visualizer.startInteractionWindows();
    ## Remove all rendering objects
    print("\t Remove all rendering objects ...");
    visualizer.removeAllRenderingObjects();
    ## Visualizer the ground truth skull shape with the predicted skull shape
    visualizer.addO3DMesh("predictedSkullShape", predictedSkullShape, (0.7, 0.7, 0.7));
    visualizer.addO3DMesh("groundTruthSkullShape", groundTruthSkullShapeVertices, (0.7, 0.0, 0.0));
    visualizer.render();
    visualizer.resetCamera();
    visualizer.startInteractionWindows();

    # Finished processing.
    print("Finished processing.");

#***********************************************************************************************************************************************#
#************************************************************MAIN FUNCTIONS*********************************************************************#
#***********************************************************************************************************************************************#
def main():
    os.system("cls");
    testHeadShapeToSkullShapeRelationship();
if __name__ == "__main__":
    main()