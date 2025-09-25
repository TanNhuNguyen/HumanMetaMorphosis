#******************************************************************************************************************#
#**********************************************PROJECTION INFORMATION**********************************************#
#******************************************************************************************************************#

#******************************************************************************************************************#
#**********************************************SUPPORTING LIBRARIES************************************************#
#******************************************************************************************************************#
import os;
import numpy as np;
import pydicom;
import cv2 as cv;
import pyvista as pv;


#******************************************************************************************************************#
#**********************************************SUPPORTING BUFFERS**************************************************#
#******************************************************************************************************************#
dataFolder = "../../../Data/Section_2_2_6_BasicMRIImageVisualization";
dicomSeries = [];

#******************************************************************************************************************#
#**********************************************SUPPORTING FUNCTIONS************************************************#
#*******************************************************************************************************************#
def updateSlice(index):
    # Update the displayed slice based on the slider value
    sliceIndex = cv.getTrackbarPos("Slice", "MRI Slice Viewer");
    if sliceIndex < 0 or sliceIndex >= len(dicomSeries):
        print(f"Slice index {sliceIndex} is out of bounds. Please select a valid slice index.");
        return;
    # Get the selected slice
    selectedSlice = dicomSeries[sliceIndex];
    # Extract pixel data from the selected slice
    pixelData = selectedSlice.pixel_array;
    # Normalize the pixel data
    normalizedData = (pixelData - np.min(pixelData)) \
        / (np.max(pixelData) - np.min(pixelData)) * 255;
    normalizedData = normalizedData.astype(np.uint8);
    # Show the selected slice
    cv.imshow("MRI Slice Viewer", normalizedData);

#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def loadDICOMSeriesAndShowMetaData():
    # Initialize
    print("Initializing ...");

    # Load DICOM series
    print("Loading DICOM series ...");
    ## Form folder path
    folderPath = os.path.join(dataFolder, "MRI-Head", "Sequence_1");
    ## Check if folder exists
    if not os.path.exists(folderPath):
        print(f"Folder {folderPath} does not exist.");
        return;
    ## Load DICOM files
    dicomFiles = [os.path.join(folderPath, f) for f in os.listdir(folderPath)];
    ## Load DICOM files into a list
    dicomSeries = [pydicom.dcmread(f) for f in dicomFiles];
    ## Check if DICOM series is empty
    if not dicomSeries:
        print("No DICOM files found in the folder.");
        return;
    # Show the number of DICOM files loaded
    print(f"\t Loaded {len(dicomSeries)} DICOM files from the folder.");

    # Show metadata of the first DICOM file
    print("Showing metadata of the first DICOM file ...");
    firstDicom = dicomSeries[0];
    print(f"\t Patient Name: {firstDicom.PatientName}");
    print(f"\t Patient ID: {firstDicom.PatientID}");
    print(f"\t Study Date: {firstDicom.StudyDate}");
    print(f"\t Modality: {firstDicom.Modality}");
    print(f"\t Image Position: {firstDicom.ImagePositionPatient}");
    print(f"\t Image Orientation: {firstDicom.ImageOrientationPatient}");
    print(f"\t Pixel Spacing: {firstDicom.PixelSpacing}");
    print(f"\t Slice Thickness: {firstDicom.SliceThickness}");
    print(f"\t Number of slices: {len(dicomSeries)}");
    
    # Finished processing
    print("Finished processing.");
def visualizeSpecificSlice():
    # Initialize
    print("Initializing ...");
    def sliceNormalize(slice):
        # Normalize the slice to the range [0, 255]
        slice = slice - np.min(slice);
        slice = slice / np.max(slice);
        slice = (slice * 255).astype(np.uint8);
        return slice;

    # Load dicom series
    print("Loading DICOM series ...");
    ## Form folder path
    folderPath = os.path.join(dataFolder, "MRI-Head", "Sequence_1");
    ## Check if folder exists
    if not os.path.exists(folderPath):
        print(f"Folder {folderPath} does not exist.");
        return;
    ## Load DICOM files
    dicomFiles = [os.path.join(folderPath, f) for f in os.listdir(folderPath)];
    ## Load DICOM files into a list
    dicomSeries = [pydicom.dcmread(f) for f in dicomFiles];
    ## Check if DICOM series is empty
    if not dicomSeries:
        print("No DICOM files found.");
        return;
    # Show the number of DICOM files loaded
    print(f"\t Loaded {len(dicomSeries)} DICOM files from the folder.");

    # Select a specific slice
    print("Selecting a specific slice ...");
    sliceIndex = 0;  # Change this index to select a different slice
    if sliceIndex < 0 or sliceIndex >= len(dicomSeries):
        print(f"Slice index {sliceIndex} is out of bounds. Please select a valid slice index.");
        return;
    # Get the selected slice
    selectedSlice = dicomSeries[sliceIndex];
    # Extract pixel data from the selected slice
    pixelData = selectedSlice.pixel_array;
    # Normalize the pixel data
    normalizedData = sliceNormalize(pixelData);

    # Show the selected slice
    print("Showing the selected slice ...");
    cv.imshow(f"Slice {sliceIndex + 1}", normalizedData);
    cv.waitKey(0);
    cv.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def changeSliceDynamically():
    # Initialize
    print("Initializing ...");
    global dicomSeries;

    # Load dicom series
    print("Loading DICOM series ...");
    ## Form folder path
    folderPath = os.path.join(dataFolder, "MRI-Head", "Sequence_1");
    ## Check if folder exists
    if not os.path.exists(folderPath):
        print(f"Folder {folderPath} does not exist.");
        return;
    ## Load DICOM files
    dicomFiles = [os.path.join(folderPath, f) for f in os.listdir(folderPath)];
    ## Load DICOM files into a list
    dicomSeries = [pydicom.dcmread(f) for f in dicomFiles];
    ## Check if DICOM series is empty
    if not dicomSeries:
        print("No DICOM files found.");
        return;
    # Show the number of DICOM files loaded
    print(f"\t Loaded {len(dicomSeries)} DICOM files from the folder.");

    # Create a window to display the slices and a slider to change the slice dynamically
    print("Creating a window to display the slices ...");
    cv.namedWindow("MRI Slice Viewer", cv.WINDOW_NORMAL);
    cv.resizeWindow("MRI Slice Viewer", 800, 600);
    cv.createTrackbar("Slice", "MRI Slice Viewer", 0, len(dicomSeries) - 1, updateSlice);

    # Show the first slice
    print("Showing the first slice ...");
    firstSlice = dicomSeries[0];
    firstPixelData = firstSlice.pixel_array;
    normalizedFirstSlice = (firstPixelData - np.min(firstPixelData)) \
        / (np.max(firstPixelData) - np.min(firstPixelData)) * 255;
    normalizedFirstSlice = normalizedFirstSlice.astype(np.uint8);
    cv.imshow("MRI Slice Viewer", normalizedFirstSlice);
    cv.waitKey(0);
    cv.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def saveASliceAsImage():
    # Initialize
    print("Initializing ...");
def visualize3DVoxelData():
    # Initialize
    print("Initializing ...");

    # Load dicom series
    print("Loading DICOM series ...");
    ## Form folder path
    folderPath = os.path.join(dataFolder, "MRI-Head", "Sequence_1");
    ## Check if folder exists
    if not os.path.exists(folderPath):
        print(f"Folder {folderPath} does not exist.");
        return;
    ## Load DICOM files
    dicomFiles = [os.path.join(folderPath, f) for f in os.listdir(folderPath)];
    ## Load DICOM files into a list
    dicomSeries = [pydicom.dcmread(f) for f in dicomFiles];
    ## Check if DICOM series is empty
    if not dicomSeries:
        print("No DICOM files found.");
        return;
    # Show the number of DICOM files loaded
    print(f"\t Loaded {len(dicomSeries)} DICOM files from the folder.");

    # Extract pixel data from the DICOM series
    print("Extracting pixel data from the DICOM series ...");
    volume = np.array([ds.pixel_array for ds in dicomSeries]);

    # Convert pixel data to volume for rendering
    print("Converting pixel data to volume for rendering ...");
    pixelData = volume.astype(np.float32);
    pixelData = pixelData - np.min(pixelData);
    pixelData = pixelData / np.max(pixelData);
    pixelData = (pixelData * 255).astype(np.uint8);
    print(f"Pixel data shape: {pixelData.shape}");

    # Visualize 3-D volume using pyvista
    print("Visualizing 3-D volume ...");
    ## Forming volume data from DICOM series
    print("\t Forming volume data from DICOM series ...");
    volume = np.stack([s.pixel_array for s in dicomSeries], axis=-1)  # shape: (rows, cols, num_slices)
    ## Create a pyvista grid
    print("\t Creating a pyvista grid ...");
    grid = pv.UniformGrid();
    spacing = [
        float(dicomSeries[0].PixelSpacing[0]),  # x spacing
        float(dicomSeries[0].PixelSpacing[1]),  # y spacing
        float(dicomSeries[0].SliceThickness)    # z spacing
    ];
    grid.dimensions = volume.shape;
    grid.spacing = spacing;
    grid.origin = (0, 0, 0);
    ## Add the volume data to the grid
    print("\t Adding the volume data to the grid ...");
    grid.point_data["values"] = volume.flatten(order="F");
    ## Visualize the grid
    print("\t Visualizing the grid ...");
    plotter = pv.Plotter();
    plotter.add_volume(grid, cmap="gray");
    plotter.show();

    # Finished processing
    print("Finished processing.");

#******************************************************************************************************************#
#**********************************************MAIN FUNCTION*******************************************************#
#******************************************************************************************************************#
def main():
    os.system("cls");
    visualize3DVoxelData();
if __name__ == "__main__":
    main()