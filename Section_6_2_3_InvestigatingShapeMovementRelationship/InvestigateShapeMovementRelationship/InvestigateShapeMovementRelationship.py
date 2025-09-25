#***********************************************************************************************************************************************#
#************************************************************SUPPORTING LIBRARIES***************************************************************#
#***********************************************************************************************************************************************#
import os;
import numpy as np;
import re;
import vtk;
import open3d as o3d;
import time;
import copy;
from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
import pandas as pd;
from scipy.spatial import cKDTree;
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

        # Add keypress event handler for quitting
        def _keypress_callback(obj, event):
            key = obj.GetKeySym()
            if key.lower() == 'q':
                print("Quitting visualization (pressed 'q').")
                self.interactor.TerminateApp()
                exit();
        self.interactor.AddObserver("KeyPressEvent", _keypress_callback)

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
    def processEvents(self):
        """
        Pumps pending VTK/OS events so the window stays responsive
        during manual animation loops (without calling Start()).
        Call this each loop iteration after render().
        """
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");
        if self.interactor:
            # Process pending UI events
            self.interactor.ProcessEvents()

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

    # For sphere functions
    def addSphere(self, sphereName, center, radius, inRGBColor):
        # Check initialization
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");

        # Create sphere source
        sphereSource = vtk.vtkSphereSource()
        sphereSource.SetCenter(center)
        sphereSource.SetRadius(radius)
        sphereSource.SetThetaResolution(5)
        sphereSource.SetPhiResolution(5)
        sphereSource.Update()

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphereSource.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*inRGBColor)

        # Add actor to renderer and store reference
        self.renderer.AddActor(actor)
        self.mesh_actors[sphereName] = actor
    def updateSphereCenter(self, sphereName, newCenter):
        # Fast: just set actor position (no geometry rebuild, no per-call render)
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.")
        if sphereName not in self.mesh_actors:
            raise ValueError(f"Visualizer:: No sphere with the name {sphereName} exists.")
        actor = self.mesh_actors.get(sphereName)
        if actor is not None:
            actor.SetPosition(*newCenter)
    def updateSphereColor(self, sphereName, newRGBColor):
        # Checking initialization
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");
    
        # Check is name exists
        if sphereName not in self.mesh_actors:
            raise ValueError(f"Visualizer:: No sphere with the name {sphereName} exists.");
    
        # Update sphere color
        actor = self.mesh_actors.get(sphereName)
        if actor is not None:
            actor.GetProperty().SetColor(*newRGBColor)
            self.render()
    def updateSphereRadius(self, sphereName, newRadius):
        # Checking initialization
        if not self.isInitialized:
            raise RuntimeError("Visualizer:: The visualizer is not initialized. Call initializeRendering() first.");
    
        # Check is name exists
        if sphereName not in self.mesh_actors:
            raise ValueError(f"Visualizer:: No sphere with the name {sphereName} exists.");
    
        # Update sphere radius
        actor = self.mesh_actors.get(sphereName)
        if actor is not None:
            # Assuming the actor's mapper input is a vtkSphereSource
            sphereSource = actor.GetMapper().GetInputConnection(0, 0).GetProducer()
            if isinstance(sphereSource, vtk.vtkSphereSource):
                sphereSource.SetRadius(newRadius)
                sphereSource.Update()
                self.render()
            else:
                raise ValueError(f"Visualizer:: The actor associated with {sphereName} is not a sphere source.");
        
    # Slider bar functions
    def addVTKSliderBar(self, sliderName, minValue, maxValue, initialValue, callback, position=(0.1, 0.1, 0.4, 0.1)):
        """
        Adds a VTK slider widget to the render window.
        Places the label as a vtkTextActor at the left end of the slider bar.
        callback: function to call with the slider value when it changes.
        position: (xmin, ymin, xmax, ymax) in normalized display coordinates.
        """
        slider_rep = vtk.vtkSliderRepresentation2D();
        slider_rep.SetMinimumValue(minValue);
        slider_rep.SetMaximumValue(maxValue);
        slider_rep.SetValue(initialValue);
        slider_rep.SetTitleText("");  # No title on the slider itself
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

        # Add a vtkTextActor at the left end of the slider
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(sliderName)
        text_prop = text_actor.GetTextProperty()
        text_prop.SetFontSize(18)
        text_prop.SetColor(1, 1, 1)
        text_prop.SetJustificationToLeft()
        text_prop.SetVerticalJustificationToCentered()
 
        # Convert normalized display coordinates to pixel coordinates
        if self.render_window:
            window_width, window_height = self.render_window.GetSize()
            x_pixel = int(position[0] * window_width) - 60;
            y_pixel = int(position[1] * window_height)
            text_actor.SetDisplayPosition(x_pixel, y_pixel)

            self.renderer.AddActor2D(text_actor)
            if not hasattr(self, 'slider_text_actors'):
                self.slider_text_actors = []
            self.slider_text_actors.append(text_actor)

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING BUFFERS*****************************************************************#
#***********************************************************************************************************************************************#
mainFolder = "../../../Data/Section_6_2_3_InvestigatingShapeMovementRelationship";

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
def parseMarkerNamesFromHeader(header_line):
    """
    Parse a Vicon-style header line to extract ordered marker base names.
    Header pattern example (tokens after splitting on whitespace):
      Frames Subject0005:ARIEL X Subject0005:ARIEL Y Subject0005:ARIEL Z Subject0005:LFHD X ...
    We look for repeating pattern: <Subj:MARKER> X <Subj:MARKER> Y <Subj:MARKER> Z
    Returns: list of marker names in order.
    """
    tokens = re.split(r'\s+', header_line.strip())
    marker_names = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "Frames":
            i += 1
            continue
        # Need at least 6 tokens for one marker pattern
        if (i + 5 < len(tokens) and
            ':' in tokens[i] and
            tokens[i+1] in ('X','Y','Z') and
            tokens[i+2] == tokens[i] and
            tokens[i+3] in ('X','Y','Z') and
            tokens[i+4] == tokens[i] and
            tokens[i+5] in ('X','Y','Z')):
            base_full = tokens[i]  # e.g. Subject0005:LFHD
            base = base_full.split(':')[-1]  # strip subject prefix
            marker_names.append(base)
            i += 6
        else:
            # Fallback: advance one token to avoid infinite loop
            i += 1
    return marker_names
def readNumPy2DArrayFromTextFile(filePath):
    """
    Reads numbers from a text file separated by spaces.
    Returns a 2D numpy array (each line is a row).
    """
    rows = []
    with open(filePath, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                row = [float(token) for token in tokens]
                rows.append(row)
    return np.array(rows)
def readIndicesFromTextFile(inFilePath):
    """
    Read a list of integer indices from a text file.
    Args:
        inFilePath: Input file path.
    Returns:
        indices: List of integer indices.
    """
    indices = [];
    with open(inFilePath, 'r') as f:
        for line in f:
            line = line.strip();
            if line:
                indices.append(int(line));
    return indices;
def saveNumPy2DArrayToTextFile(filePath, array2D, fmt='%.6f'):
    """
    Saves a 2D numpy array to a text file with specified format.
    Each row is written on a new line, values separated by spaces.
    """
    np.savetxt(filePath, array2D, fmt=fmt);
def save3DPointsToOFFFile(inFilePath, inPoints):
    """
    Save 3D points to an OFF file.
    Args:
        inFilePath: Output OFF file path.
        inPoints: Nx3 numpy array of 3D points.
    """
    num_points = inPoints.shape[0];
    with open(inFilePath, 'w') as f:
        f.write("OFF\n");
        f.write(f"{num_points} 0 0\n");
        for p in inPoints:
            f.write(f"{p[0]} {p[1]} {p[2]}\n");
def load3DPointsFromOFFFile(inFilePath):
    """
    Load 3D points from an OFF file.
    Args:
        inFilePath: Input OFF file path.
    Returns:
        points: Nx3 numpy array of 3D points.
    """
    with open(inFilePath, 'r') as f:
        lines = f.readlines();
    if lines[0].strip() != "OFF":
        raise ValueError("Not a valid OFF file.");
    header = lines[1].strip().split();
    num_points = int(header[0]);
    points = [];
    for i in range(2, 2 + num_points):
        coords = list(map(float, lines[i].strip().split()));
        points.append(coords);
    return np.array(points);
def estimateNearestIndicesFromPointsToPoints(sourcePoints, targetPoints):
    """
    For each point in sourcePoints, find the index of the nearest point in targetPoints.
    Args:
        sourcePoints: Nx3 numpy array of source points.
        targetPoints: Mx3 numpy array of target points.
    Returns:
        indices: List of length N with indices of nearest target points.
    """    
    tree = cKDTree(targetPoints);
    distances, indices = tree.query(sourcePoints);
    return np.array(indices);
def estimateRigidSVDTransform(inSourcePoints, inTargetPoints):
    # Buffer
    sourcePoints = np.array(inSourcePoints);
    targetPoints = np.array(inTargetPoints);
    
    # Compute centroids of source and target points
    sourceCentroid = np.mean(sourcePoints, axis=0);
    targetCentroid = np.mean(targetPoints, axis=0);

    # Center the points around the centroids
    sourceCenter = sourcePoints - sourceCentroid;
    targetCenter = targetPoints - targetCentroid;

    # Compute the cross-covariance matrix
    H = np.dot(sourceCenter.T, targetCenter);

    # Compute the Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H);

    # Compute rotation matrix R
    R = np.dot(Vt.T, U.T);

    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
       Vt[2,:] *= -1;
       R = np.dot(Vt.T, U.T);

    # Compute translation vector t
    t = targetCentroid.T - np.dot(R, sourceCentroid.T);

    # Construct the transformation matrix
    transformMatrix = np.identity(4);
    transformMatrix[:3, :3] = R;
    transformMatrix[:3, 3] = t;
    return transformMatrix;
def transform3DPoints(inPoints, transformMatrix):
    """
    Apply a 4x4 transformation matrix to a set of 3D points.
    Args:
        inPoints: Nx3 numpy array of 3D points.
        transformMatrix: 4x4 numpy array representing the transformation.
    Returns:
        outPoints: Nx3 numpy array of transformed 3D points.
    """
    num_points = inPoints.shape[0];
    homogeneous_points = np.hstack((inPoints, np.ones((num_points, 1))));
    transformed_homogeneous = (transformMatrix @ homogeneous_points.T).T;
    outPoints = transformed_homogeneous[:, :3];
    return outPoints;
def pause():
    """
    Utility to pause execution until user presses Enter.
    """
    input("Press Enter to continue...");

#***********************************************************************************************************************************************#
#************************************************************PROCESSING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
def readAndSaveFirstPoseToOFFFile():
    # Initialize
    print("Initializing ...");

    # Reading all poses from file
    print("Loading skeleton poses ...");
    filePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001.txt");
    poseValues = readNumPy2DArrayFromTextFile(filePath);
    poseValues = poseValues[:, 1:]; # Remove the first column (frame index)
    print("\t The shape of poseValues:", poseValues.shape);

    # Save the first pose to file
    print("Saving the first pose to file ...");
    firstPose = poseValues[0].reshape(-1, 3);

    # Save to OFF file
    print("Saving first pose to OFF file ...");
    outputFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Normalized_FirstPose.off");
    save3DPointsToOFFFile(outputFilePath, firstPose);

    # Finished processing.
    print("Finished processing.");
def estimatePelvicFeaturesIndices():
    # Initialize
    print("Initializing ...");

    # Load the full skeleton pose
    print("Loading skeleton poses ...");
    firstPoseFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_FirstPose.off");
    firstPose = load3DPointsFromOFFFile(firstPoseFilePath);
    print(f"\t Loaded {firstPose.shape[0]} points.");

    # Load the pelvic feature points
    print("Loading pelvic feature points ...");
    pelvicPoseFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_FirstPose_PelvicFeatures.off");
    pelvicPose = load3DPointsFromOFFFile(pelvicPoseFilePath);
    print(f"\t Loaded {pelvicPose.shape[0]} points.");

    # Estimate the pelvic pose indices
    print("Estimating pelvic feature indices ...");
    pelvicPoseIndices = estimateNearestIndicesFromPointsToPoints(pelvicPose, firstPose);
    print(f"\t Estimated {len(pelvicPoseIndices)} indices.");

    # Save the indices to a text file
    print("Saving pelvic feature indices to file ...");
    outputFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_PelvicFeatureIndices.txt");
    np.savetxt(outputFilePath, pelvicPoseIndices, fmt='%d');

    # Finished processing.
    print("Finished processing");
def estimateHandFeatureIndices():
    # Initialize
    print("Initializing ...");

    # Load the full skeleton pose
    print("Loading skeleton poses ...");
    firstPoseFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_FirstPose.off");
    firstPose = load3DPointsFromOFFFile(firstPoseFilePath);
    print(f"\t Loaded {firstPose.shape[0]} points.");

    # Load the left hand feature points
    print("Loading left hand feature points ...");
    leftHandPoseFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_FirstPose_LeftHandPoint.off");
    leftHandPose = load3DPointsFromOFFFile(leftHandPoseFilePath);
    print(f"\t Loaded {leftHandPose.shape[0]} points.");

    # Estimate the left hand pose indices
    print("Estimating left hand feature indices ...");
    leftHandPoseIndices = estimateNearestIndicesFromPointsToPoints(leftHandPose, firstPose);
    print(f"\t Estimated {len(leftHandPoseIndices)} indices.");

    # Save the left hand indices to a text file
    print("Saving left hand feature indices to file ...");
    outputFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_LeftHandFeatureIndices.txt");
    np.savetxt(outputFilePath, leftHandPoseIndices, fmt='%d');

    # Finished processing.
    print("Finished processing.");
def normalizePoseToFirstFrameUsingPelvicFeatures():
    # Initialize
    print("Initializing ...");

    # Reading original poses from file
    print("Loading skeleton poses ...");
    originalPoseFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001.txt");
    poseValues = readNumPy2DArrayFromTextFile(originalPoseFilePath);
    poseValues = poseValues[:, 1:]; # Remove the first column (frame index)
    print("\t The shape of poseValues:", poseValues.shape);

    # Read the pelvic feature indices
    print("Loading pelvic feature indices ...");
    pelvicIndicesFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_PelvicFeatureIndices.txt");
    pelvicFeatureIndices = readIndicesFromTextFile(pelvicIndicesFilePath);
    print(f"\t Loaded {len(pelvicFeatureIndices)} indices.");

    # Getting the first pose as reference
    print("Normalizing poses to first frame using pelvic features ...");
    referencePose = poseValues[0].reshape(-1, 3);
    referencePelvicPoints = referencePose[pelvicFeatureIndices];

    # For each pose normalize using the pelvic features
    print("Processing each pose ...");
    normalizedPoseData = [];
    for pose in poseValues:
        # Forming the current posePoints
        posePoints = pose.reshape(-1, 3);
        currentPelvicPoints = posePoints[pelvicFeatureIndices];
    
        # Estimate rigid transform using pelvic features
        svdTransform = estimateRigidSVDTransform(currentPelvicPoints, referencePelvicPoints);
    
        # Transform the current pose
        normalizedPose = transform3DPoints(posePoints, svdTransform);
        normalizedPoseData.append(normalizedPose.reshape(-1));
    normalizedPoseData = np.array(normalizedPoseData);
    print("\t The shape of normalizedPoseData:", normalizedPoseData.shape);
    
    # Save the normalized poses to a new file
    print("Saving normalized poses to file ...");
    outputFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Normalized.txt");
    saveNumPy2DArrayToTextFile(outputFilePath, normalizedPoseData, fmt='%.6f');
    
    # Finished processing.
    print("Finished processing.");
def loadAndVisualizeSkeletonPoses():
    # Initialize
    print("Initializing ...");

    # Loading skeleton poses
    print("Loading skeleton poses ...");
    filePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Normalized.txt");
    poseValues = readNumPy2DArrayFromTextFile(filePath);
    print("\t The shape of poseValues:", poseValues.shape);
    
    # Initialize visualizer
    print("Initializing visualizer ...");
    visualizer = Visualizer();
    visualizer.initializeRendering();

    # Add the first pose as spheres
    print("Adding first pose as spheres ...");
    sphereRadius = 25; # in meters
    firstPose = poseValues[0].reshape(-1, 3);
    markerNames = [];
    for i in range(len(firstPose)):
        position = firstPose[i];
        markerName = f"Marker_{i}";
        markerNames.append(markerName);
        visualizer.addSphere(markerName, position, sphereRadius, (1.0, 0.0, 0.0));
    visualizer.resetCamera();
    visualizer.render();

    # Repeat for all poses with the frame rates of 30 frames per second
    print("Animating through poses ...");
    frameDelay = 1.0 / 60.0; # seconds
    for poseIdx, pose in enumerate(poseValues):
        # Timing
        startTime = time.time();

        # Update sphere positions
        pose = pose.reshape(-1, 3);
        for i in range(len(pose)):
            position = pose[i];
            visualizer.updateSphereCenter(markerNames[i], position);
        visualizer.render();        
        visualizer.processEvents();

        # Delay to maintain target frame rate
        elapsed = time.time() - startTime;
        remaining = frameDelay - elapsed;
        if remaining > 0:
            time.sleep(remaining);

    # Finished processing.
    print("Finished processing.");
def trainPCAModelForSkeletonPoses():
    # Initialize
    print("Initializing ...");

    # Load the normalized skeleton poses
    print("Loading normalized skeleton poses ...");
    filePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Normalized.txt");
    poseValues = readNumPy2DArrayFromTextFile(filePath);
    print("\t The shape of poseValues:", poseValues.shape);

    # Perform PCA training coupling with standard scaler
    print("Training PCA model ...");
    ## Scale the data before training
    scaler = StandardScaler();
    scaledPoseValues = scaler.fit_transform(poseValues);
    ## Train PCA model
    pca = PCA(n_components=1);
    pca.fit(scaledPoseValues);
    
    # Save the trained PCA model and scaler to files
    print("Saving PCA model and scaler to files ...");
    pcaFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_PCAModel.pkl");
    scalerFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Scaler.pkl");
    pd.to_pickle(pca, pcaFilePath);
    pd.to_pickle(scaler, scalerFilePath);

    # Finished processing.
    print("Finished processing.");
def reconstructPoseMovementFromPCAComponents():
    # Initialize
    print("Initializing ...");

    # Load the PCA model and the scaler from file
    print("Loading PCA model and scaler from files ...");
    ## Load the trained pca model and the scaler
    pcaFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_PCAModel.pkl");
    scalerFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Scaler.pkl");
    pca = pd.read_pickle(pcaFilePath);
    scaler = pd.read_pickle(scalerFilePath);
    ## Print information of the pca and scaler 
    print(f"\t PCA components shape: {pca.components_.shape}");
    print(f"\t PCA explained variance ratio: {pca.explained_variance_ratio_}");
    print(f"\t Scaler mean shape: {scaler.mean_.shape}");
    print(f"\t Scaler var shape: {scaler.var_.shape}");

    # Initialize the visualizer
    print("Initializing visualizer ...");
    ## Initialize visualizer buffer
    visualizer = Visualizer();
    ## Initialize rendering window
    visualizer.initializeRendering();
    ## Get the mean shape of the pose pca model
    meanPose = scaler.mean_.reshape(-1, 3);
    ## Add the sphere for the points of the mean pose
    sphereRadius = 25; # in millimeters
    for i in range(len(meanPose)):
        position = meanPose[i];
        markerName = f"Marker_{i}";
        visualizer.addSphere(markerName, position, sphereRadius, (1.0, 0.0, 0.0));
    ## Reset camera and render
    visualizer.resetCamera();
    visualizer.render();

    # Add slider bar to control the first PCA component
    print("Adding slider bar to control the first PCA component ...");
    ## Initialize visualizer for calling from slider bar
    def onSliderChange(value):
        # Reconstruct the pose from PCA components
        pcaComponents = np.array([value]).reshape(1, -1);
        reconstructedPoseScaled = pca.inverse_transform(pcaComponents);
        reconstructedPose = scaler.inverse_transform(reconstructedPoseScaled);
        reconstructedPose = reconstructedPose.reshape(-1, 3);
        # Update the sphere positions
        for i in range(len(reconstructedPose)):
            position = reconstructedPose[i];
            visualizer.updateSphereCenter(f"Marker_{i}", position);
        visualizer.render();
        visualizer.processEvents();
    ## Add slider bar to control the first PCA component
    visualizer.addVTKSliderBar("PCA Component 1", -3.0, 3.0, 0.0, onSliderChange, position=(0.1, 0.05, 0.4, 0.05));

    # Start interaction windows
    print("Starting interaction windows ...");
    visualizer.startInteractionWindows();

    # Finished processing.
    print("Finished processing.");
def estimateTheOptimalNumberOfPCAComponentsForPoseReconstruction():
    # Initialize
    print("Initializing ...");

    # Load the normalized skeleton poses
    print("Loading normalized skeleton poses ...");
    filePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Normalized.txt");
    poseValues = readNumPy2DArrayFromTextFile(filePath);
    print("\t The shape of poseValues:", poseValues.shape);

    # Split the data into the training and testing sets
    print("Splitting data into training and testing sets ...");
    numPoses = poseValues.shape[0];
    numTrain = int(numPoses * 0.8);
    trainPoses = poseValues[:numTrain, :];
    testPoses = poseValues[numTrain:, :];
    print(f"\t Training set shape: {trainPoses.shape}");
    print(f"\t Testing set shape: {testPoses.shape}");

    # Perform the PCA training with the number of components from 1 to 3000 and compute testing errors
    # The training stop when the testing errors start to increase
    print("Training PCA models with varying number of components and computing testing errors ...");
    maxComponents = min(3000, trainPoses.shape[0], trainPoses.shape[1]);
    print(f"\t Maximum number of components: {maxComponents}");
    previousError = float('inf');
    optimalNumComponents = 1;
    trainingErrors = [];
    testingErrors = [];
    delta = 1e-4;
    maxConsecutive = 2;
    for numComponents in range(1, maxComponents + 1):
        # Debugging
        print(f"\t The number of components: {numComponents}.", end='', flush=True);

        # Scale the training data
        scaler = StandardScaler();
        scaledTrainPoses = scaler.fit_transform(trainPoses);
        
        # Train PCA model
        pca = PCA(n_components=numComponents);
        pca.fit(scaledTrainPoses);
        
        # Compute training error
        reconstructedTrainScaled = pca.inverse_transform(pca.transform(scaledTrainPoses));
        reconstructedTrain = scaler.inverse_transform(reconstructedTrainScaled);
        trainError = np.mean(np.linalg.norm(trainPoses - reconstructedTrain, axis=1));
        trainingErrors.append(trainError);
        
        # Scale the testing data using the same scaler
        scaledTestPoses = scaler.transform(testPoses);
        
        # Compute testing error
        reconstructedTestScaled = pca.inverse_transform(pca.transform(scaledTestPoses));
        reconstructedTest = scaler.inverse_transform(reconstructedTestScaled);
        testError = np.mean(np.linalg.norm(testPoses - reconstructedTest, axis=1));
        testingErrors.append(testError);
        
        print(f" -> Train Error: {trainError:.4f}, Test Error: {testError:.4f}");
        
        # Stricter stopping: require increase by delta, and consecutive increases
        if testError > previousError + delta:
            consecutive_increases += 1
            print(f"\t Testing error increased ({consecutive_increases}/{maxConsecutive}).")
            if consecutive_increases >= maxConsecutive:
                print(f"\t Stopping at {numComponents - maxConsecutive} components due to consecutive increases.")
                break
        else:
            consecutive_increases = 0
            previousError = testError
            optimalNumComponents = numComponents

    # Train again with the number of optimal components and save the model
    print(f"Training final PCA model with optimal number of components: {optimalNumComponents} ...");
    # Scale the training data
    scaler = StandardScaler();
    scaledTrainPoses = scaler.fit_transform(trainPoses);
    # Train PCA model
    pca = PCA(n_components=optimalNumComponents);
    pca.fit(scaledTrainPoses);
    # Save the trained PCA model and scaler to files
    print("Saving PCA model and scaler to files ...");
    pcaFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Optimal_PCAModel.pkl");
    scalerFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Optimal_Scaler.pkl");
    pd.to_pickle(pca, pcaFilePath);
    pd.to_pickle(scaler, scalerFilePath);

    # Finished processing.
    print("Finished processing.");
def reconstructPoseMovementFromOptimalPCAComponents():
    # Initialize
    print("Initializing ...");

    # Load the optimal PCA model and the scaler from file
    print("Loading optimal PCA model and scaler from files ...");
    ## Load the trained pca model and the scaler
    pcaFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Optimal_PCAModel.pkl");
    scalerFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Optimal_Scaler.pkl");
    pca = pd.read_pickle(pcaFilePath);
    scaler = pd.read_pickle(scalerFilePath);

    # Initialize the visualizer
    print("Initializing visualizer ...");
    ## Initialize visualizer buffer
    visualizer = Visualizer();
    ## Initialize rendering window
    visualizer.initializeRendering();
    ## Get the mean shape of the pose pca model
    meanPose = scaler.mean_.reshape(-1, 3);
    ## Add the sphere for the points of the mean pose
    sphereRadius = 25; # in millimeters
    for i in range(len(meanPose)):
        position = meanPose[i];
        markerName = f"Marker_{i}";
        visualizer.addSphere(markerName, position, sphereRadius, (1.0, 0.0, 0.0));
    ## Reset camera and render
    visualizer.resetCamera();
    visualizer.render();

    # Add slider bars to control only the first 10 PCA components
    print("Adding slider bars to control the first 10 PCA components ...");
    numComponents = pca.n_components_;
    numSliders = min(10, numComponents)
    pcaValues = np.zeros((numComponents, ));
    firstTimeMove = True;
    def onSliderChangeFactory(index):
        def onSliderChange(value):
            pcaValues[index] = value;
            # Reconstruct the pose from PCA components
            pcaComponents = pcaValues.reshape(1, -1);
            reconstructedPoseScaled = pca.inverse_transform(pcaComponents);
            reconstructedPose = scaler.inverse_transform(reconstructedPoseScaled);
            reconstructedPose = reconstructedPose.reshape(-1, 3);

            # Update the sphere positions
            for i in range(len(reconstructedPose)):
                position = reconstructedPose[i];
                visualizer.updateSphereCenter(f"Marker_{i}", position);
            visualizer.render();
            visualizer.processEvents();
        
            # Reset the camera only the first time
            nonlocal firstTimeMove;
            if firstTimeMove:
                visualizer.resetCamera();
                firstTimeMove = False;
        return onSliderChange
    ## Add slider bars for only the first 10 PCA components
    for i in range(numSliders):
        visualizer.addVTKSliderBar(
            f"C{i+1}", 
            -3.0, 3.0, 0.0, 
            onSliderChangeFactory(i), 
            position=(0.1, 0.05 + i*0.08, 0.4, 0.05 + i*0.08)
        );
    
    # Start interaction windows
    print("Starting interaction windows ...");
    visualizer.startInteractionWindows();

    # Finished processing.
    print("Finished processing.");
def relationBetweenHandMotionAndFullBodyPoseMovement():
    # Initialize
    print("Initializing ...");

    # Forming training and testing data
    print("Forming training and testing data ...");
    ## Load the normalized skeleton poses
    filePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Normalized.txt");
    poseValues = readNumPy2DArrayFromTextFile(filePath);
    ## Load the hand marker indices
    handIndicesFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_LeftHandFeatureIndices.txt");
    handMarkerIndices = readIndicesFromTextFile(handIndicesFilePath);
    ## Forming the hand motion data
    handMotionData = poseValues[:, np.array(handMarkerIndices)*3];
    print(f"\t Hand motion data shape: {handMotionData.shape}");
    ## Forming the full body pose data
    fullBodyPoseData = poseValues;
    print(f"\t Full body pose data shape: {fullBodyPoseData.shape}");

    # Split the data into the training and testing sets
    print("Splitting data into training and testing sets ...");
    numSamples = handMotionData.shape[0];
    numTrain = int(numSamples * 0.8);
    trainHandMotion = handMotionData[:numTrain, :];
    testHandMotion = handMotionData[numTrain:, :];
    trainFullBodyPose = fullBodyPoseData[:numTrain, :];
    testFullBodyPose = fullBodyPoseData[numTrain:, :];
    print(f"\t Training hand motion shape: {trainHandMotion.shape}");
    print(f"\t Training full body pose shape: {trainFullBodyPose.shape}");
    print(f"\t Testing hand motion shape: {testHandMotion.shape}");
    print(f"\t Testing full body pose shape: {testFullBodyPose.shape}");

    # Perform regression using linear regression model and evaluate the testing error
    # The number of components will be from 1 to 300 and the training will stop when the testing error starts to increase
    # The parameterization will be conducted only on the full body pose data
    print("Training regression models with varying number of PCA components and computing testing errors ...");
    maxFullPoseComponents = min(300, trainFullBodyPose.shape[0], trainFullBodyPose.shape[1]);
    maxHandMotionComponents = min(300, trainHandMotion.shape[0], trainHandMotion.shape[1]);
    print(f"\t Maximum number of components: {maxFullPoseComponents}");
    previousError = float('inf');
    optimalNumComponents = 1;
    trainingErrors = [];
    testingErrors = [];
    delta = 1e-4;
    maxConsecutive = 2;
    for numComponents in range(1, maxFullPoseComponents + 1):
        # Debugging
        print(f"\t The number of components: {numComponents}.", end='', flush=True);

        # Scale the training data of hand motion
        handMotionScaler = StandardScaler();
        scaledTrainHandMotion = handMotionScaler.fit_transform(trainHandMotion);

        # Train PCA model on the hand motion
        pcaHand = PCA(n_components=min(numComponents, maxHandMotionComponents));
        pcaHand.fit(scaledTrainHandMotion);

        # Scale the training data of full body pose
        fullPoseScaler = StandardScaler();
        scaledTrainFullBodyPose = fullPoseScaler.fit_transform(trainFullBodyPose);

        # Train PCA model on the full body pose
        pcaFullPose = PCA(n_components=numComponents);
        pcaFullPose.fit(scaledTrainFullBodyPose);

        # Transform the training hand motion to PCA components
        trainHandMotionPCA = pcaHand.transform(scaledTrainHandMotion);
        
        # Transform the training full body pose to PCA components
        trainFullBodyPosePCA = pcaFullPose.transform(scaledTrainFullBodyPose);
        
        # Train linear regression model to map from hand motion to full body pose PCA components
        reg = LinearRegression();
        reg.fit(trainHandMotionPCA, trainFullBodyPosePCA);
        
        # Compute training error
        predictedTrainFullBodyPosePCA = reg.predict(trainHandMotionPCA);
        reconstructedTrainScaled = pcaFullPose.inverse_transform(predictedTrainFullBodyPosePCA);
        reconstructedTrain = fullPoseScaler.inverse_transform(reconstructedTrainScaled);
        trainError = np.mean(np.linalg.norm(trainFullBodyPose - reconstructedTrain, axis=1));
        trainingErrors.append(trainError);
        
        # Compute testing error
        scaledTestHandMotion = handMotionScaler.transform(testHandMotion);
        testHandMotionPCA = pcaHand.transform(scaledTestHandMotion);
        predictedTestFullBodyPosePCA = reg.predict(testHandMotionPCA);
        reconstructedTestScaled = pcaFullPose.inverse_transform(predictedTestFullBodyPosePCA);
        reconstructedTest = fullPoseScaler.inverse_transform(reconstructedTestScaled);
        testError = np.mean(np.linalg.norm(testFullBodyPose - reconstructedTest, axis=1));
        testingErrors.append(testError);
        
        print(f" -> Train Error: {trainError:.4f}, Test Error: {testError:.4f}");
        
        # Stricter stopping: require increase by delta, and consecutive increases
        if testError > previousError + delta:
            consecutive_increases += 1
            print(f"\t Testing error increased ({consecutive_increases}/{maxConsecutive}).")
            if consecutive_increases >= maxConsecutive:
                print(f"\t Stopping at {numComponents - maxConsecutive}.")
                break
        else:
            consecutive_increases = 0

        # Update previous error
        optimalNumComponents = numComponents;
        previousError = testError;
    
    # Train again with the number of optimal components and save the models
    print(f"Training final regression model with optimal number of components: {optimalNumComponents} ...");
    # Scale the training data of hand motion
    handMotionScaler = StandardScaler();
    scaledTrainHandMotion = handMotionScaler.fit_transform(trainHandMotion);
    # Train PCA model on the hand motion
    pcaHand = PCA(n_components=min(optimalNumComponents, maxHandMotionComponents));
    pcaHand.fit(scaledTrainHandMotion);
    # Scale the training data of full body pose
    fullPoseScaler = StandardScaler();
    scaledTrainFullBodyPose = fullPoseScaler.fit_transform(trainFullBodyPose);
    # Train PCA model on the full body pose
    pcaFullPose = PCA(n_components=optimalNumComponents);
    pcaFullPose.fit(scaledTrainFullBodyPose);
    # Transform the training hand motion to PCA components
    trainHandMotionPCA = pcaHand.transform(scaledTrainHandMotion);
    # Transform the training full body pose to PCA components
    trainFullBodyPosePCA = pcaFullPose.transform(scaledTrainFullBodyPose);
    # Train linear regression model to map from hand motion to full body pose PCA components
    reg = LinearRegression();
    reg.fit(trainHandMotionPCA, trainFullBodyPosePCA);

    # Save the trained models to files
    print("Saving regression model, PCA model, and scaler to files ...");
    regFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_HandToFullBody_RegressionModel.pkl");
    pcaHandFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Hand_PCAModel.pkl");
    pcaFullPoseFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_FullBody_PCAModel.pkl");
    handMotionScalerFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Hand_Scaler.pkl");
    fullPoseScalerFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_FullBody_Scaler.pkl");
    pd.to_pickle(reg, regFilePath);
    pd.to_pickle(pcaHand, pcaHandFilePath);
    pd.to_pickle(pcaFullPose, pcaFullPoseFilePath);
    pd.to_pickle(fullPoseScaler, fullPoseScalerFilePath);
    pd.to_pickle(handMotionScaler, handMotionScalerFilePath);

    # Finished processing.
    print("Finished processing.");
def visualizeHandToFullBodyPoseMovementRelationship():
    # Initialize
    print("Initializing ...");

    # Load the optimal PCA model and the scaler from file
    print("Loading regression model, PCA models, and scalers from files ...");
    ## Load the trained regression model
    regFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_HandToFullBody_RegressionModel.pkl");
    pcaHandFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Hand_PCAModel.pkl");
    pcaFullPoseFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_FullBody_PCAModel.pkl");
    handMotionScalerFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_Hand_Scaler.pkl");
    fullPoseScalerFilePath = os.path.join(mainFolder, "MotionAnalyses/0005_Walking001_FullBody_Scaler.pkl");
    reg = pd.read_pickle(regFilePath);
    pcaHand = pd.read_pickle(pcaHandFilePath);
    pcaFullPose = pd.read_pickle(pcaFullPoseFilePath);
    handMotionScaler = pd.read_pickle(handMotionScalerFilePath);
    fullPoseScaler = pd.read_pickle(fullPoseScalerFilePath);

    # Initialize the visualizer
    print("Initializing visualizer ...");
    ## Initialize visualizer buffer
    visualizer = Visualizer();
    ## Initialize rendering window
    visualizer.initializeRendering();
    ## Get the mean shape of the pose pca model
    meanPose = fullPoseScaler.mean_.reshape(-1, 3);
    ## Add the sphere for the points of the mean pose
    sphereRadius = 25; # in millimeters
    for i in range(len(meanPose)):
        position = meanPose[i];
        markerName = f"Marker_{i}";
        visualizer.addSphere(markerName, position, sphereRadius, (1.0, 0.0, 0.0));
    ## Reset camera and render
    visualizer.resetCamera();
    visualizer.render();

    # Add slider bars to control only the first 10 PCA components of hand motion
    print("Adding slider bars to control the first 10 PCA components of hand motion ...");
    numComponents = pcaHand.n_components_;
    numSliders = min(10, numComponents);
    handPCAValues = np.zeros((numComponents, ));
    firstTimeMove = True;
    def onSliderChangeFactory(index):
        def onSliderChange(value):
            # Update the hand PCA values
            handPCAValues[index] = value;

            # Reconstruct the pose from PCA components
            handPCAComponents = handPCAValues.reshape(1, -1);
            predictedFullBodyPosePCA = reg.predict(handPCAComponents);
            reconstructedPoseScaled = pcaFullPose.inverse_transform(predictedFullBodyPosePCA);
            reconstructedPose = fullPoseScaler.inverse_transform(reconstructedPoseScaled);
            reconstructedPose = reconstructedPose.reshape(-1, 3);

            # Update the sphere positions
            for i in range(len(reconstructedPose)):
                position = reconstructedPose[i];
                visualizer.updateSphereCenter(f"Marker_{i}", position);
            visualizer.render();
            visualizer.processEvents();
        
            # Reset the camera only the first time
            nonlocal firstTimeMove;
            if firstTimeMove:
                visualizer.resetCamera();
                firstTimeMove = False;
        return onSliderChange
    ## Add slider bars for only the first 10 PCA components
    for i in range(numSliders):
        visualizer.addVTKSliderBar(
            f"C{i+1}", 
            -3.0, 3.0, 0.0, 
            onSliderChangeFactory(i), 
            position=(0.1, 0.05 + i*0.08, 0.4, 0.05 + i*0.08)
        );

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
    visualizeHandToFullBodyPoseMovementRelationship();
if __name__ == "__main__":
    main()