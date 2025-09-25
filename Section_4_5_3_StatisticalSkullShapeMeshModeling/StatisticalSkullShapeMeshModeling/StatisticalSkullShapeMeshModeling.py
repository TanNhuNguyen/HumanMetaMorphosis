#***********************************************************************************************************************************************#
#************************************************************SUPPORTING LIBRARIES***************************************************************#
#***********************************************************************************************************************************************#
import os;
import numpy as np;
import open3d as o3d;
from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
import vtk;

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

    # Interface functions
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
    def resetCamera(self):
        # Checking initialization
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");
        
        # Reset the camera and render
        if self.renderer:
            self.renderer.ResetCamera()
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
mainFolder = "../../../Data/Section_4_5_3_StatisticalSkullShapeMeshModeling";

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

#***********************************************************************************************************************************************#
#************************************************************PROCESSING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
def trainTestSplit():
    # Initialize
    print("Initializing ...");

    # Read the full subject IDs
    print("Reading the full subject IDs ...");
    fullSubjectIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "SubjectIDs.txt"));

    # Split the subject IDs into training and testing sets
    print("Splitting the subject IDs into training and testing sets ...");
    trainingSubjectIDs = fullSubjectIDs[:int(0.8 * len(fullSubjectIDs))];
    testingSubjectIDs = fullSubjectIDs[int(0.8 * len(fullSubjectIDs)):];

    # Save the training and testing subject IDs to text files
    print("Saving the training and testing subject IDs to text files ...");
    saveListOfStringsToTextFile(trainingSubjectIDs, os.path.join(mainFolder, "TrainingSubjectIDs.txt"));
    saveListOfStringsToTextFile(testingSubjectIDs, os.path.join(mainFolder, "TestingSubjectIDs.txt"));

    # Finished processing.
    print("Finished processing.");
def buildSkullShapePCAModels():
    # Initializing
    print("Initializing ...");

    # Reading the training and testing subject IDs
    print("Reading the training and testing subject IDs ...");
    ## Reading subject IDs
    trainingSubjectIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TrainingSubjectIDs.txt"));
    testingSubjectIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TestingSubjectIDs.txt"));
    ## Checking the reading
    if len(trainingSubjectIDs) == 0:
        print("\t No training subjects found.");
        return;
    if len(testingSubjectIDs) == 0:
        print("\t No testing subjects found.");
        return;
    ## Checking the number of subjects
    print(f"\t Number of training subjects: {len(trainingSubjectIDs)}");
    print(f"\t Number of testing subjects: {len(testingSubjectIDs)}");

    # Forming data for PCA
    print("Forming data for PCA ...");
    trainingXData = [];
    for subjectIndex in range(len(trainingSubjectIDs)):
        visualizeProgressBar(subjectIndex+1, len(trainingSubjectIDs));
        ## Reading the subject's mesh
        subjectMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "SkullShapes", 
                                                             f"{trainingSubjectIDs[subjectIndex]}-PersonalizedSkullShape.ply"));
        subjectVertices = np.asarray(subjectMesh.vertices);
        ## Flattening the vertices and adding to the data
        trainingXData.append(subjectVertices.flatten());
    trainingXData = np.array(trainingXData);

    # Train the PCA with the number of components of 1
    print("Training the PCA with the number of components of 1 ...");
    ## Scale the data using stadardization
    scaler = StandardScaler();
    trainingXData = scaler.fit_transform(trainingXData);
    ## Train the PCA
    numComps = 1;
    pca = PCA(n_components=numComps);
    pca.fit(trainingXData);
    ## Save the PCA model and the scaler
    np.save(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), scaler);
    np.save(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), pca);

    # Train the PCA with the number of components of 5
    print("Training the PCA with the number of components of 5 ...");
    ## Train the PCA
    numComps = 5;
    pca = PCA(n_components=numComps);
    pca.fit(trainingXData);
    ## Save the PCA model and the scaler
    np.save(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), scaler);
    np.save(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), pca);

    # Train the PCA with the number of components of 10
    print("Training the PCA with the number of components of 10 ...");
    ## Train the PCA
    numComps = 10;
    pca = PCA(n_components=numComps);
    pca.fit(trainingXData);
    ## Save the PCA model and the scaler
    np.save(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), scaler);
    np.save(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), pca);

    # Train the PCA with the number of components of 100
    print("Training the PCA with the number of components of 100 ...");
    ## Train the PCA
    numComps = 100;
    pca = PCA(n_components=numComps);
    pca.fit(trainingXData);
    ## Save the PCA model and the scaler
    np.save(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), scaler);
    np.save(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), pca);

    # Finished processing.
    print("Finished processing.");
def buildSkullMeshPCAModels():
    # Initializing
    print("Initializing ...");

    # Reading the training and testing subject IDs
    print("Reading the training and testing subject IDs ...");
    ## Reading subject IDs
    trainingSubjectIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TrainingSubjectIDs.txt"));
    testingSubjectIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TestingSubjectIDs.txt"));
    ## Checking the reading
    print("Finished reading subject IDs.");
    if len(trainingSubjectIDs) == 0:
        print("\t No training subjects found.");
        return;
    if len(testingSubjectIDs) == 0:
        print("\t No testing subjects found.");
        return;
    ## Checking the number of subjects
    print(f"\t Number of training subjects: {len(trainingSubjectIDs)}");
    print(f"\t Number of testing subjects: {len(testingSubjectIDs)}");

    # Forming data for PCA
    print("Forming data for PCA ...");
    trainingXData = [];
    for subjectIndex in range(len(trainingSubjectIDs)):
        visualizeProgressBar(subjectIndex+1, len(trainingSubjectIDs));
        ## Reading the subject's mesh
        subjectMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "SkullMeshes", 
                                                             f"{trainingSubjectIDs[subjectIndex]}-PersonalizedSkullMesh.ply"));
        subjectVertices = np.asarray(subjectMesh.vertices);
        ## Flattening the vertices and adding to the data
        trainingXData.append(subjectVertices.flatten());
    trainingXData = np.array(trainingXData);

    # Train the PCA with the number of components of 1
    print("Training the PCA with the number of components of 1 ...");
    ## Scale the data using stadardization
    scaler = StandardScaler();
    trainingXData = scaler.fit_transform(trainingXData);
    ## Train the PCA
    numComps = 1;
    pca = PCA(n_components=numComps);
    pca.fit(trainingXData);
    ## Save the PCA model and the scaler
    np.save(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), scaler);
    np.save(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), pca);

    # Train the PCA with the number of components of 5
    print("Training the PCA with the number of components of 5 ...");
    ## Train the PCA
    numComps = 5;
    pca = PCA(n_components=numComps);
    pca.fit(trainingXData);
    ## Save the PCA model and the scaler
    np.save(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), scaler);
    np.save(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), pca);
    # Train the PCA with the number of components of 10
    print("Training the PCA with the number of components of 10 ...");
    ## Train the PCA
    numComps = 10;
    pca = PCA(n_components=numComps);
    pca.fit(trainingXData);
    ## Save the PCA model and the scaler
    np.save(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), scaler);
    np.save(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), pca);

    # Train the PCA with the number of components of 100
    print("Training the PCA with the number of components of 100 ...");
    ## Train the PCA
    numComps = 100;
    pca = PCA(n_components=numComps);
    pca.fit(trainingXData);
    ## Save the PCA model and the scaler
    np.save(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), scaler);
    np.save(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), pca);
    
    # Finished processing.
    print("Finished processing.");
def computeStatisticsForSkullShapePCAModels():
    # Initializing
    print("Initializing ...");

    # Reading the template skull shape
    print("Reading the template skull shape ...");
    templateSkullShapeMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "TemplateSkullShape.ply"));
    templateFaces = np.asarray(templateSkullShapeMesh.triangles);

    # Computing statistics for skull shape PCA models with 1 number of components
    print("Computing statistics for skull shape PCA models with 1 number of components ...");
    ## Load the scaler and the PCA model
    numComps = 1;
    scaler = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Mean shape
    meanShapeData = pca.mean_;
    meanShapeData = meanShapeData.reshape(1, -1);
    ## Inverse scale the mean shape to get the vertices
    meanShapeVertices = scaler.inverse_transform(meanShapeData).reshape(-1, 3);
    ## Print some statistics for illustrating the quality of the model
    print(f"\t Explained variance ratio: {pca.explained_variance_ratio_}");
    print(f"\t Singular values: {pca.singular_values_}");
    print(f"\t Mean shape data size: {meanShapeData.shape}");
    print(f"\t Mean shape vertices size: {meanShapeVertices.shape}");
    ## Visualize the mean shape
    meanShapeMesh = o3d.geometry.TriangleMesh();
    meanShapeMesh.vertices = o3d.utility.Vector3dVector(meanShapeVertices);
    meanShapeMesh.triangles = o3d.utility.Vector3iVector(templateFaces);
    meanShapeMesh.compute_vertex_normals();
    o3d.visualization.draw_geometries([meanShapeMesh]);

    # Compute statistics for skull shape PCA models with 100 components
    print("Computing statistics for skull shape PCA models with 100 number of components ...");
    ## Load the scaler and the PCA model
    numComps = 100;
    scaler = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Mean shape
    meanShapeData = pca.mean_;
    meanShapeData = meanShapeData.reshape(1, -1);
    ## Inverse scale the mean shape to get the vertices
    meanShapeVertices = scaler.inverse_transform(meanShapeData).reshape(-1, 3);
    ## Print some statistics for illustrating the quality of the model
    print(f"\t Explained variance ratio: {pca.explained_variance_ratio_}");
    print(f"\t Singular values: {pca.singular_values_}");
    print(f"\t Mean shape data size: {meanShapeData.shape}");
    print(f"\t Mean shape vertices size: {meanShapeVertices.shape}");
    ## Visualize the mean shape
    meanShapeMesh = o3d.geometry.TriangleMesh();
    meanShapeMesh.vertices = o3d.utility.Vector3dVector(meanShapeVertices);
    meanShapeMesh.triangles = o3d.utility.Vector3iVector(templateFaces);
    meanShapeMesh.compute_vertex_normals();
    o3d.visualization.draw_geometries([meanShapeMesh]);

    # Finished processing.
    print("Finished processing.");
def computeStatisticsForSkullMeshPCAModels():
    # Initialize
    print("Initializing ...");

    # Reading the template skull mesh
    print("Reading the template skull mesh ...");
    templateSkullMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "TemplateSkullMesh.ply"));
    templateFaces = np.asarray(templateSkullMesh.triangles);

    # Computing statistics for skull mesh PCA models with 1 number of components
    print("Computing statistics for skull mesh PCA models with 1 number of components ...");
    ## Load the scaler and the PCA model
    numComps = 1;
    scaler = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Mean shape
    meanShapeData = pca.mean_;
    meanShapeData = meanShapeData.reshape(1, -1);
    ## Inverse scale the mean shape to get the vertices
    meanShapeVertices = scaler.inverse_transform(meanShapeData).reshape(-1, 3);
    ## Print some statistics for illustrating the quality of the model
    print(f"\t Explained variance ratio: {pca.explained_variance_ratio_}");
    print(f"\t Singular values: {pca.singular_values_}");
    print(f"\t Mean shape data size: {meanShapeData.shape}");
    print(f"\t Mean shape vertices size: {meanShapeVertices.shape}");
    ## Visualize the mean shape
    meanShapeMesh = o3d.geometry.TriangleMesh();
    meanShapeMesh.vertices = o3d.utility.Vector3dVector(meanShapeVertices);
    meanShapeMesh.triangles = o3d.utility.Vector3iVector(templateFaces);
    meanShapeMesh.compute_vertex_normals();
    o3d.visualization.draw_geometries([meanShapeMesh]);

    # Compute statistics for skull mesh PCA models with 100 components
    print("Computing statistics for skull mesh PCA models with 100 number of components ...");
    ## Load the scaler and the PCA model
    numComps = 100;
    scaler = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Mean shape
    meanShapeData = pca.mean_;
    meanShapeData = meanShapeData.reshape(1, -1);
    ## Inverse scale the mean shape to get the vertices
    meanShapeVertices = scaler.inverse_transform(meanShapeData).reshape(-1, 3);
    ## Print some statistics for illustrating the quality of the model
    print(f"\t Explained variance ratio: {pca.explained_variance_ratio_}");
    print(f"\t Singular values: {pca.singular_values_}");
    print(f"\t Mean shape data size: {meanShapeData.shape}");
    print(f"\t Mean shape vertices size: {meanShapeVertices.shape}");
    ## Visualize the mean shape
    meanShapeMesh = o3d.geometry.TriangleMesh();
    meanShapeMesh.vertices = o3d.utility.Vector3dVector(meanShapeVertices);
    meanShapeMesh.triangles = o3d.utility.Vector3iVector(templateFaces);
    meanShapeMesh.compute_vertex_normals();
    o3d.visualization.draw_geometries([meanShapeMesh]);

    # Finished processing.
    print("Finished processing.");
def computeTestingErrorsForSkullShapePCAModels():
    # Initializing
    print("Initializing ...");
    def computeTestingErrorsPCAModel(inScaler, inPCAMode, inTestingData):
        # Checking inputs
        if inScaler is None:
            raise ValueError("computeTestingErrorsPCAModel:: The input scaler is None.");
        if inPCAMode is None:
            raise ValueError("computeTestingErrorsPCAModel:: The input PCA model is None.");
        if inTestingData is None or len(inTestingData) == 0:
            raise ValueError("computeTestingErrorsPCAModel:: The input testing data is None or empty.");

        # Initialize buffers
        averageDistances = [];
        numTestingSamples = len(inTestingData);
    
        # Iterate for each testing sample
        for sampleIndex in range(numTestingSamples):
            visualizeProgressBar(sampleIndex+1, numTestingSamples);
            ## Getting the sample data
            sampleData = inTestingData[sampleIndex].reshape(1, -1);
            ## Scaling the sample data
            sampleDataScaled = inScaler.transform(sampleData);
            ## Projecting to the PCA space
            sampleDataPCA = inPCAMode.transform(sampleDataScaled);
            ## Reconstructing back to the original space
            sampleDataReconstructedScaled = inPCAMode.inverse_transform(sampleDataPCA);
            ## Inverse scaling to get the original data
            sampleDataReconstructed = inScaler.inverse_transform(sampleDataReconstructedScaled);
            ## Computing the average distance between the original and reconstructed data
            averageDistance = np.mean(np.linalg.norm(sampleData - sampleDataReconstructed, axis=1));
            averageDistances.append(averageDistance);
    
        # Compute grand average distance
        grandAverageDistance = np.mean(averageDistances);
        return grandAverageDistance;

    # Reading testing subject IDs
    print("Reading testing subject IDs ...");
    testingSubjectIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TestingSubjectIDs.txt"));
    if len(testingSubjectIDs) == 0:
        print("\t No testing subjects found.");
        return;
    print(f"\t Number of testing subjects: {len(testingSubjectIDs)}");

    # Reading the template skull shape
    print("Reading the template skull shape ...");
    templateSkullShapeMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "TemplateSkullShape.ply"));
    templateFaces = np.asarray(templateSkullShapeMesh.triangles);

    # Forming the testing data
    print("Forming the testing data ...");
    testingXData = [];
    for subjectIndex in range(len(testingSubjectIDs)):
        visualizeProgressBar(subjectIndex+1, len(testingSubjectIDs));
        ## Reading the subject's mesh
        subjectMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "SkullShapes", 
                                                             f"{testingSubjectIDs[subjectIndex]}-PersonalizedSkullShape.ply"));
        subjectVertices = np.asarray(subjectMesh.vertices);
        ## Flattening the vertices and adding to the data
        testingXData.append(subjectVertices.flatten());

    # Evaluating the 1-component PCA model
    print("Evaluating the 1-component PCA model ...");
    ## Load the scaler and the PCA model
    numComps = 1;
    scaler = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Calling evaluator
    averageDistance = computeTestingErrorsPCAModel(scaler, pca, testingXData);
    ## Grand average distance
    print(f"\t Grand average distance for {numComps} components: {averageDistance}");

    # Evaluating the 5-component PCA model
    print("Evaluating the 5-component PCA model ...");
    ## Load the scaler and the PCA model
    numComps = 5;
    scaler = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Calling evaluator
    averageDistance = computeTestingErrorsPCAModel(scaler, pca, testingXData);
    ## Grand average distance
    print(f"\t Grand average distance for {numComps} components: {averageDistance}");

    # Evaluating the 10-component PCA model
    print("Evaluating the 10-component PCA model ...");
    ## Load the scaler and the PCA model
    numComps = 10;
    scaler = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Calling evaluator
    averageDistance = computeTestingErrorsPCAModel(scaler, pca, testingXData);
    ## Grand average distance
    print(f"\t Grand average distance for {numComps} components: {averageDistance}");

    # Evaluating the 100-component PCA model
    print("Evaluating the 100-component PCA model ...");
    ## Load the scaler and the PCA model
    numComps = 100;
    scaler = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Calling evaluator
    averageDistance = computeTestingErrorsPCAModel(scaler, pca, testingXData);
    ## Grand average distance
    print(f"\t Grand average distance for {numComps} components: {averageDistance}");
    
    # Finished processing.
    print("Finished processing.");
def computeTestingErrorsForSkullMeshPCAModels():
    # Initializing
    print("Initializing ...");
    def computeTestingErrorsPCAModel(inScaler, inPCAMode, inTestingData):
        # Checking inputs
        if inScaler is None:
            raise ValueError("computeTestingErrorsPCAModel:: The input scaler is None.");
        if inPCAMode is None:
            raise ValueError("computeTestingErrorsPCAModel:: The input PCA model is None.");
        if inTestingData is None or len(inTestingData) == 0:
            raise ValueError("computeTestingErrorsPCAModel:: The input testing data is None or empty.");

        # Initialize buffers
        averageDistances = [];
        numTestingSamples = len(inTestingData);
    
        # Iterate for each testing sample
        for sampleIndex in range(numTestingSamples):
            visualizeProgressBar(sampleIndex+1, numTestingSamples);
            ## Getting the sample data
            sampleData = inTestingData[sampleIndex].reshape(1, -1);
            ## Scaling the sample data
            sampleDataScaled = inScaler.transform(sampleData);
            ## Projecting to the PCA space
            sampleDataPCA = inPCAMode.transform(sampleDataScaled);
            ## Reconstructing back to the original space
            sampleDataReconstructedScaled = inPCAMode.inverse_transform(sampleDataPCA);
            ## Inverse scaling to get the original data
            sampleDataReconstructed = inScaler.inverse_transform(sampleDataReconstructedScaled);
            ## Computing the average distance between the original and reconstructed data
            averageDistance = np.mean(np.linalg.norm(sampleData - sampleDataReconstructed, axis=1));
            averageDistances.append(averageDistance);
    
        # Compute grand average distance
        grandAverageDistance = np.mean(averageDistances);
        return grandAverageDistance;

    # Reading testing subject IDs
    print("Reading testing subject IDs ...");
    testingSubjectIDs = readListOfStringsFromTextFile(os.path.join(mainFolder, "TestingSubjectIDs.txt"));
    if len(testingSubjectIDs) == 0:
        print("\t No testing subjects found.");
        return;
    print(f"\t Number of testing subjects: {len(testingSubjectIDs)}");

    # Reading the template skull mesh
    print("Reading the template skull mesh ...");
    templateSkullMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "TemplateSkullMesh.ply"));
    templateFaces = np.asarray(templateSkullMesh.triangles);

    # Forming the testing data
    print("Forming the testing data ...");
    testingXData = [];
    for subjectIndex in range(len(testingSubjectIDs)):
        visualizeProgressBar(subjectIndex+1, len(testingSubjectIDs));
        ## Reading the subject's mesh
        subjectMesh = o3d.io.read_triangle_mesh(os.path.join(mainFolder, "SkullMeshes", 
                                                             f"{testingSubjectIDs[subjectIndex]}-PersonalizedSkullMesh.ply"));
        subjectVertices = np.asarray(subjectMesh.vertices);
        ## Flattening the vertices and adding to the data
        testingXData.append(subjectVertices.flatten());

    # Evaluating the 1-component PCA model
    print("Evaluating the 1-component PCA model ...");
    ## Load the scaler and the PCA model
    numComps = 1;
    scaler = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Calling evaluator
    averageDistance = computeTestingErrorsPCAModel(scaler, pca, testingXData);
    ## Grand average distance
    print(f"\t Grand average distance for {numComps} components: {averageDistance}");

    # Evaluating the 5-component PCA model
    print("Evaluating the 5-component PCA model ...");
    ## Load the scaler and the PCA model
    numComps = 5;
    scaler = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Calling evaluator
    averageDistance = computeTestingErrorsPCAModel(scaler, pca, testingXData);
    ## Grand average distance
    print(f"\t Grand average distance for {numComps} components: {averageDistance}");

    # Evaluating the 10-component PCA model
    print("Evaluating the 10-component PCA model ...");
    ## Load the scaler and the PCA model
    numComps = 10;
    scaler = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Calling evaluator
    averageDistance = computeTestingErrorsPCAModel(scaler, pca, testingXData);
    ## Grand average distance
    print(f"\t Grand average distance for {numComps} components: {averageDistance}");

    # Evaluating the 100-component PCA model
    print("Evaluating the 100-component PCA model ...");
    ## Load the scaler and the PCA model
    numComps = 100;
    scaler = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();
    ## Calling evaluator
    averageDistance = computeTestingErrorsPCAModel(scaler, pca, testingXData);
    ## Grand average distance
    print(f"\t Grand average distance for {numComps} components: {averageDistance}");

    # Finished processing.
    print("Finished processing.");
def visualizeSkullShapeModel():
    # Initializing
    print("Initializing ...");
    def updateSkullShapeModel(value, componentIndex, inScaler, inPCA, inTemplateFaces, inVisualizer):
        # Checking inputs
        if inScaler is None:
            raise ValueError("updateSkullShapeModel:: The input scaler is None.");
        if inPCA is None:
            raise ValueError("updateSkullShapeModel:: The input PCA model is None.");
        if inTemplateFaces is None or len(inTemplateFaces) == 0:
            raise ValueError("updateSkullShapeModel:: The input template faces is None or empty.");
        if inVisualizer is None:
            raise ValueError("updateSkullShapeModel:: The input visualizer is None.");

        # Initialize the PCA coefficients
        numComponents = inPCA.n_components_;
        pcaCoeffs = np.zeros((1, numComponents));
        pcaCoeffs[0, componentIndex] = value;

        # Generate the skull shape from the PCA model
        skullShapeData = inPCA.inverse_transform(pcaCoeffs);
        skullShapeData = inScaler.inverse_transform(skullShapeData);
        skullShapeVertices = skullShapeData.reshape(-1, 3);

        # Update the visualizer
        inVisualizer.updateO3DMeshVertices("SkullShape", skullShapeVertices);
        inVisualizer.render();

    # Reading template skull shape
    print("Reading template skull shape ...");
    templateSkullShape = readO3DMesh(os.path.join(mainFolder, "TemplateSkullShape.ply"));
    templateFaces = np.asarray(templateSkullShape.triangles);

    # Load the scaler and the PCA model with the 1 components
    print("Loading the scaler and the PCA model with the 1 components ...");
    numComps = 1;
    scaler = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullShapePCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();

    # Try to determin the min and max value of the pca coefficients
    print("Trying to determine the min and max value of the PCA coefficients ...");
    pcaCoeffsMin = -10 * np.sqrt(pca.explained_variance_);
    pcaCoeffsMax = 10 * np.sqrt(pca.explained_variance_);
    print(f"\t PCA coefficients min: {pcaCoeffsMin}");
    print(f"\t PCA coefficients max: {pcaCoeffsMax}");

    # Initialize the visualizer
    print("Initializing the visualizer ...");
    visualizer = Visualizer();
    visualizer.initializeRendering();
    visualizer.addO3DMesh("SkullShape", templateSkullShape, (0.8, 0.8, 0.8));
    visualizer.resetCamera();
    visualizer.render();

    # Adding sliders for controlling the PCA components
    print("Adding sliders for controlling the PCA components ...");
    visualizer.addVTKSliderBar("PC 1", pcaCoeffsMin, pcaCoeffsMax, 0.0, lambda value: updateSkullShapeModel(value, 0, scaler, pca, templateFaces, visualizer));
    
    # Start interaction windows
    print("Starting interaction windows ...");
    visualizer.startInteractionWindows();

    # Finished processing.
    print("Finished processing.");
def visualizeSkullMeshModel():
    # Initializing
    print("Initializing ...");
    def updateSkullMeshModel(value, componentIndex, inScaler, inPCA, inTemplateFaces, inVisualizer):
        # Checking inputs
        if inScaler is None:
            raise ValueError("updateSkullMeshModel:: The input scaler is None.");
        if inPCA is None:
            raise ValueError("updateSkullMeshModel:: The input PCA model is None.");
        if inTemplateFaces is None or len(inTemplateFaces) == 0:
            raise ValueError("updateSkullMeshModel:: The input template faces is None or empty.");
        if inVisualizer is None:
            raise ValueError("updateSkullMeshModel:: The input visualizer is None.");

        # Initialize the PCA coefficients
        numComponents = inPCA.n_components_;
        pcaCoeffs = np.zeros((1, numComponents));
        pcaCoeffs[0, componentIndex] = value;

        # Generate the skull mesh from the PCA model
        skullMeshData = inPCA.inverse_transform(pcaCoeffs);
        skullMeshData = inScaler.inverse_transform(skullMeshData);
        skullMeshVertices = skullMeshData.reshape(-1, 3);

        # Update the visualizer
        inVisualizer.updateO3DMeshVertices("SkullMesh", skullMeshVertices);
        inVisualizer.render();

    # Reading template skull mesh
    print("Reading template skull mesh ...");
    templateSkullMesh = readO3DMesh(os.path.join(mainFolder, "TemplateSkullMesh.ply"));
    templateFaces = np.asarray(templateSkullMesh.triangles);

    # Load the scaler and the PCA model with the 1 components
    print("Loading the scaler and the PCA model with the 1 components ...");
    numComps = 1;
    scaler = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/Scaler_Comp_{numComps}.npy"), allow_pickle=True).item();
    pca = np.load(os.path.join(mainFolder, f"SkullMeshPCAModels/PCAModel_Comp_{numComps}.npy"), allow_pickle=True).item();

    # Try to determin the min and max value of the pca coefficients
    print("Trying to determine the min and max value of the PCA coefficients ...");
    pcaCoeffsMin = -10 * np.sqrt(pca.explained_variance_);
    pcaCoeffsMax = 10 * np.sqrt(pca.explained_variance_);
    print(f"\t PCA coefficients min: {pcaCoeffsMin}");
    print(f"\t PCA coefficients max: {pcaCoeffsMax}");

    # Initialize the visualizer
    print("Initializing the visualizer ...");
    visualizer = Visualizer();
    visualizer.initializeRendering();
    visualizer.addO3DMesh("SkullMesh", templateSkullMesh, (0.8, 0.8, 0.8));
    visualizer.resetCamera();
    visualizer.render();

    # Adding sliders for controlling the PCA components
    print("Adding sliders for controlling the PCA components ...");
    visualizer.addVTKSliderBar("PC 1", pcaCoeffsMin, pcaCoeffsMax, 0.0, 
                               lambda value: updateSkullMeshModel(value, 0, scaler, pca, templateFaces, visualizer));

    # Start interaction windows
    print("Starting interaction windows ...");
    visualizer.startInteractionWindows();

    # Finished processing.
    print("Finished processing.");

#***********************************************************************************************************************************************#
#************************************************************MAIN FUNCTIONS*********************************************************************#
#***********************************************************************************************************************************************#
def main():
    os.system("cls");
    visualizeSkullShapeModel();
if __name__ == "__main__":
    main()