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
dataFolder = "../../../Data/Section_2_2_4_BasicCTImageVisualization";
sliceIndex = 0;  # Default slice index for visualization
dicomSeries = [];  # Placeholder for DICOM series

#******************************************************************************************************************#
#**********************************************SUPPORTING BUFFERS**************************************************#
#******************************************************************************************************************#
def updateSlice(index):
    global sliceIndex, dicomSeries;
    sliceIndex = index;
    if 0 <= sliceIndex < len(dicomSeries):
        specificSlice = dicomSeries[sliceIndex];
        sliceImage = specificSlice.pixel_array;
        if sliceImage is not None:
            sliceImage = sliceImage.astype(np.uint8);
            cv.imshow("CT Slice Viewer", sliceImage);
        else:
            print("Failed to read pixel data from the DICOM slice.");
    else:
        print(f"Slice index {sliceIndex} is out of range. Valid range: 0 to {len(dicomSeries) - 1}");

#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def loadDICOMSeriesAndShowMetaData():
    # Initializing
    print("Initializing ...");

    # Load DICOM from folder
    print("Loading DICOM series from folder ...");
    ## Forming folder path
    folderPath = os.path.join(dataFolder, "CT-HeadNeck");
    ## Checking folder existence
    if not os.path.exists(folderPath):
        print("Folder does not exist: " + folderPath);
        return;
    ## Reading DICOM files using pydicom
    dicomFiles = [os.path.join(folderPath, f) for f in os.listdir(folderPath) if f.endswith('.dcm')];
    if not dicomFiles:
        print("No DICOM files found in folder: " + folderPath);
        return;
    dicomSeries = [pydicom.dcmread(f) for f in dicomFiles];
    print(f"Loaded {len(dicomSeries)} DICOM files from {folderPath}");

    # Displaying metadata of the first DICOM file
    print("Displaying metadata of the first DICOM file ...");
    firstDICOM = dicomSeries[0];
    print("Patient Name:", firstDICOM.PatientName);
    print("Patient ID:", firstDICOM.PatientID);
    print("Study Date:", firstDICOM.StudyDate);
    print("Modality:", firstDICOM.Modality);
    print("Image Size:", firstDICOM.Rows, "x", firstDICOM.Columns);
    print("Pixel Spacing:", firstDICOM.PixelSpacing);
    print("Slice Thickness:", firstDICOM.SliceThickness);
    print("Number of Slices:", len(dicomSeries));

    # Finished processing
    print("Finished processing ...");
def visualizeSpecificSlice():
    # Initialize
    print("Initializing ...");

    # Load DICOM series
    print("Loading DICOM series ...");
    ## Forming folder path
    folderPath = os.path.join(dataFolder, "CT-Liver");
    ## Checking folder existence
    if not os.path.exists(folderPath):
        print("Folder does not exist: " + folderPath);
        return;
    ## Reading DICOM files
    dicomFiles = [os.path.join(folderPath, f) for f in os.listdir(folderPath) if f.endswith('.dcm')];
    if not dicomFiles:
        print("No DICOM files found in folder: " + folderPath);
        return;
    dicomSeries = [pydicom.dcmread(f) for f in dicomFiles];

    # Displaying a specific slice
    print("Displaying a specific slice ...");
    ## Define slice index
    sliceIndex = 10;  # Example slice index
    ## Check if slice index is valid
    if sliceIndex < 0 or sliceIndex >= len(dicomSeries):
        print(f"Slice index {sliceIndex} is out of range. Valid range: 0 to {len(dicomSeries) - 1}");
        return;
    ## Extract slice dicom image
    specificSlice = dicomSeries[sliceIndex];
    ## Get pixel data from the slice
    sliceImage = specificSlice.pixel_array;
    if sliceImage is None:
        print("Failed to read pixel data from the DICOM slice.");
        return;
    ## Convert pixel data to uint8 for visualization
    sliceImage = sliceImage.astype(np.uint8);

    # Visualize the slice
    print("Visualizing the slice ...");
    cv.imshow(f"Slice {sliceIndex}", sliceImage);
    cv.waitKey(0);  # Wait for a key press to close the window
    cv.destroyAllWindows();

    # Finished processing
    print("Finished processing ...");
def changeSliceDynamically():
    # Initialize
    print("Initializing ...");
    global sliceIndex, dicomSeries;

    # Load DICOM series
    print("Loading DICOM series ...");
    ## Forming folder path
    folderPath = os.path.join(dataFolder, "CT-Lung");
    ## Checking folder existence
    if not os.path.exists(folderPath):
        print("Folder does not exist: " + folderPath);
        return;
    ## Reading DICOM files
    dicomFiles = [os.path.join(folderPath, f) for 
                  f in os.listdir(folderPath) if f.endswith('.dcm')];
    if not dicomFiles:
        print("No DICOM files found in folder: " + folderPath);
        return;
    dicomSeries = [pydicom.dcmread(f) for f in dicomFiles];
    print(f"Loaded {len(dicomSeries)} DICOM files from {folderPath}");

    # Initialize visual windows using opencv and slider for changing slices
    print("Initializing visualization window with slider ...");
    cv.namedWindow("CT Slice Viewer", cv.WINDOW_NORMAL);
    cv.resizeWindow("CT Slice Viewer", 800, 600);
    cv.createTrackbar("Slice Index", "CT Slice Viewer", 0, 
                      len(dicomSeries) - 1, updateSlice);
    print("Use the slider to change slices ...");

    # Visualize the first slice in the window
    print("Visualizing the first slice ...");
    ## Getting the first slice
    sliceIndex = 0;
    if (sliceIndex < 0 or sliceIndex >= len(dicomSeries)):
        print(f"Slice index {sliceIndex} is out of range. Valid range: 0 to \
              {len(dicomSeries) - 1}");
        return;
    specificSlice = dicomSeries[sliceIndex];
    ## Extract pixel data
    sliceImage = specificSlice.pixel_array;
    if sliceImage is None:
        print("Failed to read pixel data from the DICOM slice.");
        return;
    ## Convert pixel data to uint8 for visualization
    sliceImage = sliceImage.astype(np.uint8);
    cv.imshow("CT Slice Viewer", sliceImage);
    cv.waitKey(0);  # Wait for a key press to close the window
    cv.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def saveSliceAsPNGOrJPEG():
    # Initialize
    print("Initializing ...");
    def normalizeImage(image):
        """Normalize image to 0-255 range for visualization."""
        normImage = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX);
        return normImage.astype(np.uint8);

    # Load DICOM series
    print("Loading DICOM series ...");
    ## Forming folder path
    folderPath = os.path.join(dataFolder, "CT-SpinopelvicStructure");
    ## Checking folder existence
    if not os.path.exists(folderPath):
        print("Folder does not exist: " + folderPath);
        return;
    ## Reading DICOM files
    dicomFiles = [os.path.join(folderPath, f) 
                  for f in os.listdir(folderPath) if f.endswith('.dcm')];
    if not dicomFiles:
        print("No DICOM files found in folder: " + folderPath);
        return;
    dicomSeries = [pydicom.dcmread(f) for f in dicomFiles];
    print(f"Loaded {len(dicomSeries)} DICOM files from {folderPath}");

    # Save a specific slice as PNG or JPEG
    print("Saving a specific slice as PNG or JPEG ...");
    ## Define slice index
    sliceIndex = 5;  # Example slice index
    ## Check if slice index is valid
    if sliceIndex < 0 or sliceIndex >= len(dicomSeries):
        print(f"Slice index {sliceIndex} is out of range. Valid range: 0 to \
               {len(dicomSeries) - 1}");
        return;
    ## Extract slice dicom image
    specificSlice = dicomSeries[sliceIndex]; 
    ## Normalize image for better visualization
    normalizedImage = normalizeImage(specificSlice.pixel_array);

    # Save the image as PNG
    print("Saving image as PNG ...");
    ## Create output folder if it does not exist
    outputFolder = os.path.join(dataFolder, "Output");
    os.makedirs(outputFolder, exist_ok=True);
    ## Define output file path
    outputFilePath = os.path.join(outputFolder, f"slice_{sliceIndex}.png");
    ## Save the image
    cv.imwrite(outputFilePath, normalizedImage);
    print(f"\t Image saved as: {outputFilePath}");

    # Save the image as JPEG
    print("Saving image as JPEG ...");
    ## Define output file path for JPEG
    jpegOutputFilePath = os.path.join(outputFolder, f"slice_{sliceIndex}.jpg");
    ## Save the image
    cv.imwrite(jpegOutputFilePath, normalizedImage);
    print(f"\t Image saved as: {jpegOutputFilePath}");

    # Finished processing
    print("Finished processing.");
def visualize3DVoxelData():
    # Information: This function load dicom series and visualize the slices as 3-D voxel data using the pyvista library.
    # Note: This function requires the pyvista library to be installed.
    # Initialize
    print("Initializing ...");

    # Load DICOM series
    print("Loading DICOM series ...");
    ## Forming folder path
    folderPath = os.path.join(dataFolder, "CT-SpinopelvicStructure");
    ## Checking folder existence
    if not os.path.exists(folderPath):
        print("Folder does not exist: " + folderPath);
        return;
    ## Reading DICOM files
    dicomFiles = [os.path.join(folderPath, f)
                    for f in os.listdir(folderPath) if f.endswith('.dcm')];
    if not dicomFiles:
        print("No DICOM files found in folder: " + folderPath);
        return;
    dicomSeries = [pydicom.dcmread(f) for f in dicomFiles];
    print(f"Loaded {len(dicomSeries)} DICOM files from {folderPath}");

    # Visualize 3-D volume using pyvista
    print("Visualizing 3-D volume ...");
    ## Forming volume data from DICOM series
    volume = np.stack([s.pixel_array for s in dicomSeries], axis=-1)  # shape: (rows, cols, num_slices)
    ## Create a pyvista grid
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
    grid.point_data["values"] = volume.flatten(order="F");
    ## Visualize the grid
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