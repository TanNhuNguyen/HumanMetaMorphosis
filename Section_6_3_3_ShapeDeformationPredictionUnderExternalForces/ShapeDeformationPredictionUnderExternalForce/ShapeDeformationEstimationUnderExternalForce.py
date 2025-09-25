#***********************************************************************************************************************************************#
#************************************************************SUPPORTING LIBRARIES***************************************************************#
#***********************************************************************************************************************************************#
import os;
import numpy as np;
import vtk;
import open3d as o3d;
from sklearn.decomposition import PCA;
from sklearn.preprocessing import StandardScaler;
from sklearn.linear_model import RidgeCV;
import pyvista as pv
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

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
mainFolder = "../../../Data/Section_6_3_3_ShapeDeformationPredictionUnderExternalForces";

#***********************************************************************************************************************************************#
#************************************************************SUPPORTING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
def pause():
    """
    Utility to pause execution until user presses Enter.
    """
    input("Press Enter to continue...");
def generateSphereMesh(center, radius, resolution=20):
    """
    Generates a sphere mesh using Open3D.
    center: (x, y, z) coordinates of the sphere center.
    radius: radius of the sphere.
    resolution: number of segments along latitude and longitude.
    Returns an Open3D TriangleMesh object.
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution);
    sphere.translate(center);
    sphere.compute_vertex_normals();
    return sphere;
def generatePlaneMesh(center, normal, size=1.0, thickness=0.01):
    """
    Generates a flat plane mesh using Open3D.
    center: (x, y, z) coordinates of the plane center.
    normal: (nx, ny, nz) normal vector of the plane.
    size: length of the plane sides.
    thickness: thickness of the plane.
    Returns an Open3D TriangleMesh object.
    """
    # Create a grid in the XY plane
    plane = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=thickness);
    plane.translate((-size/2, -size/2, 0));
    
    # Rotate the plane to align with the given normal
    normal = np.array(normal);
    normal = normal / np.linalg.norm(normal);
    z_axis = np.array([0, 0, 1]);
    if np.allclose(normal, z_axis):
        R = np.eye(3);
    elif np.allclose(normal, -z_axis):
        R = o3d.geometry.get_rotation_matrix_from_axis_angle([1, 0, 0], np.pi);
    else:
        v = np.cross(z_axis, normal);
        c = np.dot(z_axis, normal);
        s = np.linalg.norm(v);
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]]);
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2));
    plane.rotate(R, center=(0, 0, 0));
    
    # Translate to the desired center
    plane.translate(center);
    
    plane.compute_vertex_normals();
    return plane;

#***********************************************************************************************************************************************#
#************************************************************PROCESSING FUNCTIONS***************************************************************#
#***********************************************************************************************************************************************#
def generateSphereDeformationUsingMassSpringModel():
    # Initialize
    print("Initializing...");
    dataFolder = os.path.join(mainFolder, "Data");
    if not os.path.exists(dataFolder):
        os.makedirs(dataFolder);

    print("1. Generating sphere mesh...")
    sphere = pv.Sphere(radius=1.0, theta_resolution=20, phi_resolution=20)
    vertices = sphere.points.copy()
    n_vertices = vertices.shape[0]
    # Extract edges from faces
    faces = sphere.faces.reshape(-1, 4)[:, 1:]
    edge_set = set()
    for tri in faces:
        edge_set.add(tuple(sorted([tri[0], tri[1]])))
        edge_set.add(tuple(sorted([tri[1], tri[2]])))
        edge_set.add(tuple(sorted([tri[2], tri[0]])))
    edges = np.array(list(edge_set))

    print("2. Initializing mass-spring system...")
    velocities = np.zeros_like(vertices)
    rest_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    mass = 1.0
    k = 50.0
    damp = 0.5
    dt = 0.01
    steps = 500
    external_force = np.array([0, 0, -10.0])  # Stronger force for visible deformation

    # Identify fixed (bottom) vertices and force-applied (top) vertices
    z_coords = vertices[:, 2]
    fixed_indices = np.where(z_coords < -0.95)[0]  # Bottom cap
    force_indices = np.where(z_coords > 0.95)[0]   # Top cap

    print("3. Visualizing...")
    plotter = pv.Plotter()
    mesh = pv.PolyData(vertices, sphere.faces)
    plotter.add_mesh(mesh, color='lightblue')
    plotter.show(auto_close=False)

    print("4. Running simulation...")
    for step in range(steps):
        forces = np.zeros_like(vertices)

        # Spring forces
        vec = vertices[edges[:, 1]] - vertices[edges[:, 0]]
        lengths = np.linalg.norm(vec, axis=1)
        mask = lengths > 1e-8
        directions = np.zeros_like(vec)
        directions[mask] = vec[mask] / lengths[mask][:, None]
        spring_forces = k * (lengths - rest_lengths)[:, None] * directions

        # Accumulate spring forces to vertices
        for i, (a, b) in enumerate(edges):
            forces[a] += spring_forces[i]
            forces[b] -= spring_forces[i]

        # Damping
        forces -= damp * velocities

        # External force: only apply to top vertices
        forces[force_indices] += external_force

        # Fix bottom vertices
        velocities[fixed_indices] = 0
        forces[fixed_indices] = 0

        # Update
        velocities += (forces / mass) * dt
        vertices += velocities * dt

        # Update mesh
        mesh.points = vertices
        plotter.render()

        # Save the sphere mesh for each step for later training
        mesh.save(os.path.join(dataFolder, f"sphere_deformation_step_{step:03d}.ply"));
        
        # Save the force information for later training
        np.save(os.path.join(dataFolder, f"sphere_forces_step_{step:03d}.npy"), forces)

    plotter.close()
def trainForceToShapeDeformation_MultivariateRegression():
    # Initialize
    print("Initializing ...");

    # Reading data for training
    print("Reading data for training ...");
    ## Data buffer
    print("\t Define data buffers ...");
    dataFolder = os.path.join(mainFolder, "Data");
    num_steps = 501  # 000 to 500 inclusive

    # Load force and shape data by index
    forceData = []
    shapeData = []
    shapeFiles = []
    for i in range(num_steps):
        force_path = os.path.join(dataFolder, f"sphere_forces_step_{i:03d}.npy")
        shape_path = os.path.join(dataFolder, f"sphere_deformation_step_{i:03d}.ply")
        if os.path.exists(force_path) and os.path.exists(shape_path):
            force = np.load(force_path)
            forceData.append(force.flatten())
            shape = pv.read(shape_path)
            shapeData.append(shape.points.flatten())
            shapeFiles.append(f"sphere_deformation_step_{i:03d}.ply")
        else:
            print(f"Warning: Missing file for step {i:03d}")

    forceData = np.array(forceData)
    shapeData = np.array(shapeData)
    print("\t Force data shape:", forceData.shape)
    print("\t Shape data shape:", shapeData.shape)

    # Getting the template faces of the shape data
    print("\t Getting the template faces of the shape data ...");
    if shapeFiles:
        templateFaces = pv.read(os.path.join(dataFolder, shapeFiles[0])).faces
    else:
        templateFaces = None

    # Split the training and testing data
    print("Splitting the training and testing data ...");
    numOfSamples = forceData.shape[0];
    numOfTrains = int(0.8 * numOfSamples);
    forceTrainData = forceData[:numOfTrains];
    forceTestData = forceData[numOfTrains:];
    shapeTrainData = shapeData[:numOfTrains];
    shapeTestData = shapeData[numOfTrains:];

    # With the number of components from 1 to 200, train and test the model of prediction.
    print("Training and testing the model with different number of PCA components ...");
    tolerance = 3  # Number of consecutive increases allowed before stopping
    increase_count = 0
    prev_error = None

    for numOfComponents in range(1, 201):
        ## Standardize the force and shape data
        forceScaler = StandardScaler();
        shapeScaler = StandardScaler();
        forceTrainDataScaled = forceScaler.fit_transform(forceTrainData);
        shapeTrainDataScaled = shapeScaler.fit_transform(shapeTrainData);
        forceTestDataScaled = forceScaler.transform(forceTestData);
        shapeTestDataScaled = shapeScaler.transform(shapeTestData);

        ## Parameterize the force and shape data using PCA
        forcePCA = PCA(n_components=numOfComponents);
        shapePCA = PCA(n_components=numOfComponents);
        forceTrainDataPCA = forcePCA.fit_transform(forceTrainDataScaled);
        shapeTrainDataPCA = shapePCA.fit_transform(shapeTrainDataScaled);

        ## Form the X and Y data for regression
        XTrain = np.hstack((forceTrainDataPCA[1:], shapeTrainDataPCA[:(len(shapeTrainDataPCA)-1)]));
        YTrain = shapeTrainDataPCA[1:];
        if YTrain.ndim == 1:
            YTrain = YTrain.reshape(-1, 1)

        ### Train the relation using a multivariate linear regression model
        model = RidgeCV(
            alphas=np.logspace(-3, 3, 13),
            scoring='neg_mean_absolute_error',
            fit_intercept=True
        )
        model.fit(XTrain, YTrain)

        ## Test the model
        forceTestPCAData = forcePCA.transform(forceTestDataScaled);
        shapeTestPCAData = shapePCA.transform(shapeTestDataScaled);
        XTest = np.hstack((forceTestPCAData[1:], shapeTestPCAData[:(len(shapeTestPCAData)-1)]));
        XPred = model.predict(XTest);
        if XPred.ndim == 1:
            XPred = XPred.reshape(-1, 1)

        shapePredScaledData = shapePCA.inverse_transform(XPred);
        shapePredData = shapeScaler.inverse_transform(shapePredScaledData);

        shapeTestDataValid = shapeTestData[1:];
        meanAbsError = np.mean(np.abs(shapePredData - shapeTestDataValid));

        print(f"\t Number of PCA components: {numOfComponents}, Mean Absolute Error: {meanAbsError}");

        # Early stopping logic
        if prev_error is not None and meanAbsError > prev_error:
            increase_count += 1
            if increase_count >= tolerance:
                print(f"Early stopping: Testing error increased for {tolerance} consecutive steps.")
                break
        else:
            increase_count = 0
        prev_error = meanAbsError

        # After early stopping, retrain with the optimal number of components
    print("Retraining with optimal number of PCA components and saving models ...")
    optimal_num = numOfComponents - tolerance if increase_count >= tolerance else numOfComponents
    print(f"Optimal number of PCA components: {optimal_num}")

    # Prepare data and models again
    forceScaler = StandardScaler()
    shapeScaler = StandardScaler()
    forceTrainDataScaled = forceScaler.fit_transform(forceTrainData)
    shapeTrainDataScaled = shapeScaler.fit_transform(shapeTrainData)

    forcePCA = PCA(n_components=optimal_num)
    shapePCA = PCA(n_components=optimal_num)
    forceTrainDataPCA = forcePCA.fit_transform(forceTrainDataScaled)
    shapeTrainDataPCA = shapePCA.fit_transform(shapeTrainDataScaled)

    XTrain = np.hstack((forceTrainDataPCA[1:], shapeTrainDataPCA[:(len(shapeTrainDataPCA)-1)]))
    YTrain = shapeTrainDataPCA[1:]
    if YTrain.ndim == 1:
        YTrain = YTrain.reshape(-1, 1)

    model = RidgeCV(
        alphas=np.logspace(-3, 3, 13),
        scoring='neg_mean_absolute_error',
        fit_intercept=True
    )
    model.fit(XTrain, YTrain)

    # Save models and scalers as .pkl files
    save_folder = os.path.join(mainFolder, "TrainedModels")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    joblib.dump(forceScaler, os.path.join(save_folder, "force_scaler.pkl"))
    joblib.dump(shapeScaler, os.path.join(save_folder, "shape_scaler.pkl"))
    joblib.dump(forcePCA, os.path.join(save_folder, "force_pca.pkl"))
    joblib.dump(shapePCA, os.path.join(save_folder, "shape_pca.pkl"))
    joblib.dump(model, os.path.join(save_folder, "regressor.pkl"))
    print(f"Saved optimal models and scalers to {save_folder}")

    # Finished processing.
    print("Finished processing.");
def trainForceToShapeDeformation_LongShortTermMemory():
    print("Initializing LSTM training ...")

    # Reading data for training
    dataFolder = os.path.join(mainFolder, "Data")
    saveFolder = os.path.join(mainFolder, "TrainedModels")
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    num_steps = 501  # 000 to 500 inclusive

    forceData = []
    shapeData = []
    for i in range(num_steps):
        force_path = os.path.join(dataFolder, f"sphere_forces_step_{i:03d}.npy")
        shape_path = os.path.join(dataFolder, f"sphere_deformation_step_{i:03d}.ply")
        if os.path.exists(force_path) and os.path.exists(shape_path):
            force = np.load(force_path)
            forceData.append(force.flatten())
            shape = pv.read(shape_path)
            shapeData.append(shape.points.flatten())
        else:
            print(f"Warning: Missing file for step {i:03d}")

    forceData = np.array(forceData)
    shapeData = np.array(shapeData)

    numOfSamples = forceData.shape[0]
    numOfTrains = int(0.8 * numOfSamples)
    forceTrainData = forceData[:numOfTrains]
    forceTestData = forceData[numOfTrains:]
    shapeTrainData = shapeData[:numOfTrains]
    shapeTestData = shapeData[numOfTrains:]

    # Standardize only (no PCA)
    forceScaler = StandardScaler()
    shapeScaler = StandardScaler()
    forceTrainDataScaled = forceScaler.fit_transform(forceTrainData)
    shapeTrainDataScaled = shapeScaler.fit_transform(shapeTrainData)
    forceTestDataScaled = forceScaler.transform(forceTestData)
    shapeTestDataScaled = shapeScaler.transform(shapeTestData)

    # Prepare LSTM input: [force_t, shape_t] -> shape_{t+1}
    XTrain = np.hstack((forceTrainDataScaled[:-1], shapeTrainDataScaled[:-1]))
    YTrain = shapeTrainDataScaled[1:]
    XTest = np.hstack((forceTestDataScaled[:-1], shapeTestDataScaled[:-1]))
    YTest = shapeTestDataScaled[1:]

    # Reshape for LSTM: (samples, seq_len, features)
    XTrain = XTrain.reshape((XTrain.shape[0], 1, XTrain.shape[1]))
    XTest = XTest.reshape((XTest.shape[0], 1, XTest.shape[1]))

    # Convert to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    XTrain_torch = torch.tensor(XTrain, dtype=torch.float32).to(device)
    YTrain_torch = torch.tensor(YTrain, dtype=torch.float32).to(device)
    XTest_torch = torch.tensor(XTest, dtype=torch.float32).to(device)
    YTest_torch = torch.tensor(YTest, dtype=torch.float32).to(device)

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]  # Take last time step
            out = self.fc(out)
            return out

    input_size = XTrain.shape[2]
    hidden_size = 256
    output_size = YTrain.shape[1]
    model = LSTMModel(input_size, hidden_size, output_size).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop with early stopping
    epochs = 1000
    batch_size = 32
    tolerance = 3  # Number of consecutive increases allowed before stopping
    increase_count = 0
    prev_val_loss = None
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None

    n_train = XTrain_torch.shape[0]
    n_test = XTest_torch.shape[0]
    print("Training LSTM model ...")
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            batch_X = XTrain_torch[idx]
            batch_Y = YTrain_torch[idx]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= n_train

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(XTest_torch)
            val_loss = criterion(val_outputs, YTest_torch).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()

        # Early stopping logic
        if prev_val_loss is not None and val_loss > prev_val_loss:
            increase_count += 1
            if increase_count >= tolerance:
                print(f"Early stopping: Validation loss increased for {tolerance} consecutive epochs.")
                break
        else:
            increase_count = 0
        prev_val_loss = val_loss

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model from epoch {best_epoch+1} with validation loss {best_val_loss:.6f}")

    # Save model and scalers
    torch.save(model.state_dict(), os.path.join(saveFolder, "lstm_model.pt"))
    joblib.dump(forceScaler, os.path.join(saveFolder, "force_scaler.pkl"))
    joblib.dump(shapeScaler, os.path.join(saveFolder, "shape_scaler.pkl"))
    print(f"Saved LSTM model and scalers to {saveFolder}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        YPred = model(XTest_torch).cpu().numpy()
    shapePredData = shapeScaler.inverse_transform(YPred)
    shapeTestDataValid = shapeTestData[1:]
    meanAbsError = np.mean(np.abs(shapePredData - shapeTestDataValid))
    print(f"LSTM Mean Absolute Error: {meanAbsError}")
def testForceToShapeDeformation_MultivariateRegression():
    print("Testing and visualizing shape deformation prediction using trained multivariate regression model...")

    # Paths
    dataFolder = os.path.join(mainFolder, "Data")
    modelFolder = os.path.join(mainFolder, "TrainedModels")

    # Load models and scalers
    forceScaler = joblib.load(os.path.join(modelFolder, "force_scaler.pkl"))
    shapeScaler = joblib.load(os.path.join(modelFolder, "shape_scaler.pkl"))
    forcePCA = joblib.load(os.path.join(modelFolder, "force_pca.pkl"))
    shapePCA = joblib.load(os.path.join(modelFolder, "shape_pca.pkl"))
    regressor = joblib.load(os.path.join(modelFolder, "regressor.pkl"))

    # Load force and shape data
    num_steps = 500
    forceData = []
    shapeData = []
    for i in range(num_steps):
        force_path = os.path.join(dataFolder, f"sphere_forces_step_{i:03d}.npy")
        shape_path = os.path.join(dataFolder, f"sphere_deformation_step_{i:03d}.ply")
        if os.path.exists(force_path) and os.path.exists(shape_path):
            force = np.load(force_path)
            forceData.append(force.flatten())
            shape = pv.read(shape_path)
            shapeData.append(shape.points.flatten())
        else:
            print(f"Warning: Missing file for step {i:03d}")

    forceData = np.array(forceData)
    shapeData = np.array(shapeData)

    # Standardize all force and shape data
    forceDataScaled = forceScaler.transform(forceData)
    shapeDataScaled = shapeScaler.transform(shapeData)

    # Transform to PCA space
    forceDataPCA = forcePCA.transform(forceDataScaled)
    shapeDataPCA = shapePCA.transform(shapeDataScaled)

    # Visualization setup
    print("Setting up PyVista visualization...")
    initial_shape = pv.read(os.path.join(dataFolder, f"sphere_deformation_step_000.ply"))
    mesh = pv.PolyData(initial_shape.points.copy(), initial_shape.faces)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue')
    plotter.show(auto_close=False)

    # Predict and visualize sequence
    prev_shape_pca = shapeDataPCA[0]  # Initial shape (first frame)
    for t in range(1, num_steps):
        force_pca = forceDataPCA[t].reshape(1, -1)
        prev_shape_pca = prev_shape_pca.reshape(1, -1)
        X_input = np.hstack((force_pca, prev_shape_pca))
        next_shape_pca = regressor.predict(X_input)[0]
        predicted_shape = shapeScaler.inverse_transform(shapePCA.inverse_transform(next_shape_pca.reshape(1, -1)))[0]
        # Update mesh vertices
        mesh.points = predicted_shape.reshape(-1, 3)
        plotter.render()
        prev_shape_pca = next_shape_pca

    print("Visualization finished.")
    plotter.close()
def testForceToShapeDeformation_LongShortTermMemory():
    print("Testing and visualizing shape deformation prediction using trained LSTM model...")

    # Paths
    dataFolder = os.path.join(mainFolder, "Data")
    modelFolder = os.path.join(mainFolder, "TrainedModels")

    # Load scalers
    forceScaler = joblib.load(os.path.join(modelFolder, "force_scaler.pkl"))
    shapeScaler = joblib.load(os.path.join(modelFolder, "shape_scaler.pkl"))

    # Load force and shape data
    num_steps = 500
    forceData = []
    shapeData = []
    for i in range(num_steps):
        force_path = os.path.join(dataFolder, f"sphere_forces_step_{i:03d}.npy")
        shape_path = os.path.join(dataFolder, f"sphere_deformation_step_{i:03d}.ply")
        if os.path.exists(force_path) and os.path.exists(shape_path):
            force = np.load(force_path)
            forceData.append(force.flatten())
            shape = pv.read(shape_path)
            shapeData.append(shape.points.flatten())
        else:
            print(f"Warning: Missing file for step {i:03d}")

    forceData = np.array(forceData)
    shapeData = np.array(shapeData)

    # Standardize all force and shape data
    forceDataScaled = forceScaler.transform(forceData)
    shapeDataScaled = shapeScaler.transform(shapeData)

    # Visualization setup
    print("Setting up PyVista visualization...")
    initial_shape = pv.read(os.path.join(dataFolder, f"sphere_deformation_step_000.ply"))
    mesh = pv.PolyData(initial_shape.points.copy(), initial_shape.faces)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='lightblue')
    plotter.show(auto_close=False)

    # Load LSTM model definition
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.fc(out)
            return out

    input_size = forceDataScaled.shape[1] + shapeDataScaled.shape[1]
    hidden_size = 256
    output_size = shapeDataScaled.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load(os.path.join(modelFolder, "lstm_model.pt"), map_location=device))
    model.eval()

    # Predict and visualize sequence
    prev_shape_scaled = shapeDataScaled[0]  # Initial shape (first frame)
    for t in range(1, num_steps):
        force_scaled = forceDataScaled[t].reshape(1, 1, -1)
        prev_shape_scaled = prev_shape_scaled.reshape(1, 1, -1)
        X_input = np.concatenate((force_scaled, prev_shape_scaled), axis=2)
        X_input_torch = torch.tensor(X_input, dtype=torch.float32).to(device)
        with torch.no_grad():
            next_shape_scaled = model(X_input_torch).cpu().numpy()[0]
        predicted_shape = shapeScaler.inverse_transform(next_shape_scaled.reshape(1, -1))[0]
        # Update mesh vertices
        mesh.points = predicted_shape.reshape(-1, 3)
        plotter.render()
        prev_shape_scaled = next_shape_scaled

    print("Visualization finished.")
    plotter.close()

#***********************************************************************************************************************************************#
#************************************************************MAIN FUNCTIONS*********************************************************************#
#***********************************************************************************************************************************************#
def main():
    os.system("cls");
    testForceToShapeDeformation_MultivariateRegression();

if __name__ == "__main__":
    main()