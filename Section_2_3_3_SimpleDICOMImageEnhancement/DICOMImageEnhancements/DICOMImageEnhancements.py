#******************************************************************************************************************#
#**********************************************PROJECTION INFORMATION**********************************************#
#******************************************************************************************************************#

#******************************************************************************************************************#
#**********************************************SUPPORTING LIBRARIES************************************************#
#******************************************************************************************************************#
import os;
import numpy as np;
import pydicom;
import cv2;

#******************************************************************************************************************#
#**********************************************SUPPORTING BUFFERS**************************************************#
#******************************************************************************************************************#
dataFolder = "../../../Data/Section_2_3_3_SimpleDICOMImageEnhancement";
dicomSeries = []; # List to hold DICOM images
pixelData = None;  # Placeholder for pixel data
contrastValue = 50;  # Default contrast value
brightnessValue = 50;  # Default brightness value
noiseReductionIndex = 0;  # Default noise reduction index

#******************************************************************************************************************#
#**********************************************SUPPORTING FUNCTIONS************************************************#
#******************************************************************************************************************#
def normalizePixelDataFromMinToMax(inLowerThreshold, inUpperThreshold, 
                                       pixelData, dtype=cv2.CV_32F):
    # Define the min max value
    minValue = inLowerThreshold;
    maxValue = inUpperThreshold;
    # Normalize the data using the opencv function
    if pixelData is None or len(pixelData) == 0:
        print("No pixel data to normalize.");
        return None;
    if minValue >= maxValue:
        print("Invalid range for normalization: minValue should be less than maxValue.");
        return None;
    # Normalize pixel data to the range minValue to maxValue
    normalizedData = cv2.normalize(pixelData, None, alpha=minValue, beta=maxValue, norm_type=cv2.NORM_MINMAX, dtype=dtype);

    # Return the normalized pixel data
    return normalizedData;

#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def intensityRescaling():
    # Initialize
    print("Initializing ...");    
    def normalizePixelDataFromMinToMax(inLowerThreshold, 
                                       inUpperThreshold, pixelData, dtype=cv2.CV_32F):
        # Define the min max value
        minValue = inLowerThreshold;
        maxValue = inUpperThreshold;
        # Normalize the data using the opencv function
        if pixelData is None or len(pixelData) == 0:
            print("No pixel data to normalize.");
            return None;
        if minValue >= maxValue:
            print("Invalid range for normalization: minValue should be less than maxValue.");
            return None;
        # Normalize pixel data to the range minValue to maxValue
        normalizedData = cv2.normalize(pixelData, None, alpha=minValue, beta=maxValue, norm_type=cv2.NORM_MINMAX, dtype=dtype);

        # Return the normalized pixel data
        return normalizedData;
    def updateSlice(sliceIndex):
        # Update the displayed slice
        global pixelData;
        if dicomSeries and 0 <= sliceIndex < len(dicomSeries):
            pixelData = dicomSeries[sliceIndex].pixel_array;
    
        # Normalize the pixel data for visualization
        lowerValue = 0.0; upperValue = 1.0;
        pixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue, pixelData, dtype=cv2.CV_32F);
        
        # Update the visualization window
        cv2.imshow("DICOM Image - Intensity Rescaling", pixelData);
    
    # Load DICOM images
    print("Loading DICOM images ...");
    ## Forming dicom series folder path
    dicomSeriesFolder = os.path.join(dataFolder, "CT-Lung");
    ## Checking if the folder exists
    if not os.path.exists(dicomSeriesFolder):
        print("DICOM series folder does not exist: ", dicomSeriesFolder);
        return;
    ## Getting dicom file paths
    dicomFiles = [os.path.join(dicomSeriesFolder, f) 
                  for f in os.listdir(dicomSeriesFolder) if f.endswith('.dcm')];
    ## Checking if any DICOM files are found
    if not dicomFiles:
        print("No DICOM files found in the folder: ", dicomSeriesFolder);
        return;
    ## Loading DICOM images
    for file in dicomFiles:
        try:
            dicomImage = pydicom.dcmread(file);
            dicomSeries.append(dicomImage);
        except Exception as e:
            print(f"Error reading {file}: {e}");
    ## Checking if any images were loaded
    if not dicomSeries:
        print("No DICOM images were loaded.");
        return;
    print(f"Loaded {len(dicomSeries)} DICOM images.");

    # Print some pixel value information for all dicom slices
    print("Pixel value information for all DICOM slices:");
    globalMinDICOMValue = np.min([np.min(slice.pixel_array) for slice in dicomSeries]);
    globalMaxDICOMValue = np.max([np.max(slice.pixel_array) for slice in dicomSeries]);
    print(f"\t Global Min pixel value: {globalMinDICOMValue}");
    print(f"\t Global Max pixel value: {globalMaxDICOMValue}");

    # Initialize the visual for the first slice
    print("Initializing the visual for the first slice ...");
    ## Getting first slice
    firstSlice = dicomSeries[0];
    ## Getting pixel data
    pixelData = firstSlice.pixel_array;
    ## Normalize the pixel data for visualization
    lowerValue = 0.0; upperValue = 1.0;
    pixelData = normalizePixelDataFromMinToMax(lowerValue, 
                                               upperValue, pixelData, dtype=cv2.CV_32F);
    ## Visualize window using opencv
    cv2.imshow("DICOM Image - Intensity Rescaling", pixelData);
    ## Setting one slider for changing slices
    cv2.createTrackbar("Slice", "DICOM Image - Intensity Rescaling", 0, 
                       len(dicomSeries) - 1, updateSlice);
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def contrastEnhancement():
    # Initialize
    print("Initializing ...");    
    def normalizePixelDataFromMinToMax(inLowerThreshold, inUpperThreshold, pixelData, dtype=cv2.CV_32F):
        # Define the min max value
        minValue = inLowerThreshold;
        maxValue = inUpperThreshold;
        # Normalize the data using the opencv function
        if pixelData is None or len(pixelData) == 0:
            print("No pixel data to normalize.");
            return None;
        if minValue >= maxValue:
            print("Invalid range for normalization: minValue should be less than maxValue.");
            return None;
        # Normalize pixel data to the range minValue to maxValue
        normalizedData = cv2.normalize(pixelData, None, alpha=minValue, beta=maxValue, norm_type=cv2.NORM_MINMAX, dtype=dtype);

        # Return the normalized pixel data
        return normalizedData;
    def updateContrastBrightness(inContrastValue, inBrightnessValue, pixelData, dtype=cv2.CV_32F):
        # Change the contrast value of the pixel data
        if pixelData is None or len(pixelData) == 0:
            print("No pixel data to adjust contrast and brightness.");
            return None;
        if inContrastValue < 0 or inBrightnessValue < 0:
            print("Contrast and brightness values should be non-negative.");
            return None;

        # Adjust the contrast and brightness not using cv2.convertScaleAbs
        adjustedPixelData = pixelData * (inContrastValue / 50.0) + (inBrightnessValue - 50.0) / 50.0;

        # Return the adjusted pixel data
        return adjustedPixelData;
    def updateSlice(sliceIndex):
        # Update the displayed slice
        global pixelData;
        if dicomSeries and 0 <= sliceIndex < len(dicomSeries):
            pixelData = dicomSeries[sliceIndex].pixel_array;
    
        # Normalize the pixel data for visualization
        lowerValue = 0.0; upperValue = 1.0;
        adjustedPixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue, pixelData, dtype=cv2.CV_32F);
        
        # Update the contrast and brightness values
        global contrastValue, brightnessValue;
        adjustedPixelData = updateContrastBrightness(contrastValue, brightnessValue, adjustedPixelData);

        # Update the visualization window
        cv2.imshow("DICOM Image - Intensity Rescaling", adjustedPixelData);
    def updateContrast(sliceIndex):
        # Update the contrast value
        global contrastValue;
        contrastValue = sliceIndex;
    
        # Update the brightness value to maintain a consistent range
        if pixelData is not None:
            # Normalize before visualization
            lowerValue = 0.0; upperValue = 1.0;
            adjustedPixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue, pixelData, dtype=cv2.CV_32F);

            # Update the pixel data with new contrast value
            adjustedPixelData = updateContrastBrightness(contrastValue, brightnessValue, adjustedPixelData);

            # Update the visualization window
            cv2.imshow("DICOM Image - Intensity Rescaling", adjustedPixelData);
    def updateBrightness(sliceIndex):
        # Update the brightness value
        global brightnessValue;
        brightnessValue = sliceIndex;

        # Update the contrast value to maintain a consistent range
        if pixelData is not None:
            # Normalize before visualization
            lowerValue = 0.0; upperValue = 1.0;
            adjustedPixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue, pixelData, dtype=cv2.CV_32F);

            # Update the pixel data with new brightness value
            adjustedPixelData = updateContrastBrightness(contrastValue, brightnessValue, adjustedPixelData);

            # Update the visualization window
            cv2.imshow("DICOM Image - Intensity Rescaling", adjustedPixelData);
    
    # Load DICOM images
    print("Loading DICOM images ...");
    ## Forming dicom series folder path
    dicomSeriesFolder = os.path.join(dataFolder, "CT-Lung");
    ## Checking if the folder exists
    if not os.path.exists(dicomSeriesFolder):
        print("DICOM series folder does not exist: ", dicomSeriesFolder);
        return;
    ## Getting dicom file paths
    dicomFiles = [os.path.join(dicomSeriesFolder, f) for f in os.listdir(dicomSeriesFolder) if f.endswith('.dcm')];
    ## Checking if any DICOM files are found
    if not dicomFiles:
        print("No DICOM files found in the folder: ", dicomSeriesFolder);
        return;
    ## Loading DICOM images
    for file in dicomFiles:
        try:
            dicomImage = pydicom.dcmread(file);
            dicomSeries.append(dicomImage);
        except Exception as e:
            print(f"Error reading {file}: {e}");
    ## Checking if any images were loaded
    if not dicomSeries:
        print("No DICOM images were loaded.");
        return;
    print(f"Loaded {len(dicomSeries)} DICOM images.");

    # Print some pixel value information for all dicom slices
    print("Pixel value information for all DICOM slices:");
    globalMinDICOMValue = np.min([np.min(slice.pixel_array) for slice in dicomSeries]);
    globalMaxDICOMValue = np.max([np.max(slice.pixel_array) for slice in dicomSeries]);
    print(f"\t Global Min pixel value: {globalMinDICOMValue}");
    print(f"\t Global Max pixel value: {globalMaxDICOMValue}");

    # Initialize the visual for the first slice
    print("Initializing the visual for the first slice ...");
    ## Getting first slice
    firstSlice = dicomSeries[0];
    ## Getting pixel data
    pixelData = firstSlice.pixel_array;
    ## Normalize the pixel data for visualization
    lowerValue = 0.0; upperValue = 1.0;
    pixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue, pixelData, dtype=cv2.CV_32F);
    ## Visualize window using opencv
    cv2.imshow("DICOM Image - Intensity Rescaling", pixelData);
    ## Setting one slider for changing slices
    cv2.createTrackbar("Slice", "DICOM Image - Intensity Rescaling", 0, len(dicomSeries) - 1, updateSlice);
    ## Setting trackbar for constrast enhancement
    cv2.createTrackbar("Contrast", "DICOM Image - Intensity Rescaling", contrastValue, 100, updateContrast);
    ## Setting trackbar for brightness adjustment
    cv2.createTrackbar("Brightness", "DICOM Image - Intensity Rescaling", brightnessValue, 100, updateBrightness);
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def noiseReduction():
    # Initialize
    print("Initializing ...");
    global dicomSeries, pixelData, noiseReductionIndex; 
    def normalizePixelDataFromMinToMax(inLowerThreshold, inUpperThreshold, 
                                       pixelData, dtype=cv2.CV_32F):
        # Define the min max value
        minValue = inLowerThreshold;
        maxValue = inUpperThreshold;
        # Normalize the data using the opencv function
        if pixelData is None or len(pixelData) == 0:
            print("No pixel data to normalize.");
            return None;
        if minValue >= maxValue:
            print("Invalid range for normalization: minValue should be less than maxValue.");
            return None;
        # Normalize pixel data to the range minValue to maxValue
        normalizedData = cv2.normalize(pixelData, None, alpha=minValue, beta=maxValue, norm_type=cv2.NORM_MINMAX, dtype=dtype);

        # Return the normalized pixel data
        return normalizedData;
    def applyNoiseReductionMethod(pixelData, methodIndex):
        # Apply the selected noise reduction method
        if methodIndex == 0:
            print("No noise reduction applied.");
            return pixelData;
        elif methodIndex == 1:
            print("Applying Gaussian Blur for noise reduction.");
            return cv2.GaussianBlur(pixelData, (11, 11), 0);
        elif methodIndex == 2:
            print("Applying Median Blur for noise reduction.");
            return cv2.medianBlur(pixelData, 5);
        elif methodIndex == 3:
            print("Applying Bilateral Filter for noise reduction.");
            return cv2.bilateralFilter(pixelData, 9, 75, 75);
        elif methodIndex == 4:
            print("Applying blur in OpenCV for noise reduction.");
            return cv2.blur(pixelData, (11, 11));
        elif methodIndex == 5:
            print("Applying box filter for noise reduction.");
            return cv2.boxFilter(pixelData, -1, (11, 11));
        else:
            print("Invalid noise reduction option selected.");
            return pixelData;
    def updateSlice(sliceIndex):
        # Update the displayed slice
        global pixelData;
        if dicomSeries and 0 <= sliceIndex < len(dicomSeries):
            pixelData = dicomSeries[sliceIndex].pixel_array;
    
        # Normalize the pixel data for visualization
        lowerValue = 0.0; upperValue = 1.0;
        adjustedPixelData = normalizePixelDataFromMinToMax(lowerValue, 
                                                           upperValue, pixelData, dtype=cv2.CV_32F);
        
        # Apply the selected noise reduction method
        global noiseReductionIndex;
        adjustedPixelData = applyNoiseReductionMethod(adjustedPixelData, 
                                                      noiseReductionIndex);

        # Update the visualization window
        cv2.imshow("DICOM Image", adjustedPixelData);
    def updateNoiseReduction(sliceIndex):
        # Set noise reduction index
        global noiseReductionIndex, pixelData;
        noiseReductionIndex = sliceIndex;

        # Check if pixel data is available
        if pixelData is None or len(pixelData) == 0:
            print("No pixel data to apply noise reduction.");
            return;

        # Normalize the pixel data for visualization
        lowerValue = 0.0; upperValue = 1.0;
        adjustedPixelData = normalizePixelDataFromMinToMax(lowerValue, 
                                                           upperValue, pixelData, dtype=cv2.CV_32F);            
        adjustedPixelData = applyNoiseReductionMethod(adjustedPixelData, 
                                                      noiseReductionIndex);

        # Visualize the updated pixel data
        cv2.imshow("DICOM Image", adjustedPixelData);

    # Load DICOM images
    print("Loading DICOM images ...");
    ## Forming dicom series folder path
    dicomSeriesFolder = os.path.join(dataFolder, "CT-Lung");
    ## Checking if the folder exists
    if not os.path.exists(dicomSeriesFolder):
        print("DICOM series folder does not exist: ", dicomSeriesFolder);
        return;
    ## Getting dicom file paths
    dicomFiles = [os.path.join(dicomSeriesFolder, f) 
                  for f in os.listdir(dicomSeriesFolder) if f.endswith('.dcm')];
    ## Checking if any DICOM files are found
    if not dicomFiles:
        print("No DICOM files found in the folder: ", dicomSeriesFolder);
        return;
    ## Loading DICOM images
    for file in dicomFiles:
        try:
            dicomImage = pydicom.dcmread(file);
            dicomSeries.append(dicomImage);
        except Exception as e:
            print(f"Error reading {file}: {e}");
    ## Checking if any images were loaded
    if not dicomSeries:
        print("No DICOM images were loaded.");
        return;
    print(f"Loaded {len(dicomSeries)} DICOM images.");

    # Initialize the visual for the first slice
    print("Initializing the visual for the first slice ...");
    ## Getting first slice
    firstSlice = dicomSeries[0];
    ## Getting pixel data
    pixelData = firstSlice.pixel_array;
    ## Normalize the pixel data for visualization
    lowerValue = 0.0; upperValue = 1.0;
    pixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue, 
                                               pixelData, dtype=cv2.CV_32F);
    ## Visualize window using opencv
    cv2.imshow("DICOM Image", pixelData);
    ## Setting one slider for changing slices
    cv2.createTrackbar("Slice", "DICOM Image", 0, 
                       len(dicomSeries) - 1, updateSlice);
    ## Setting slider for setting noise reduction function
    cv2.createTrackbar("Noise Reduction Option", "DICOM Image", 
                       0, 5, updateNoiseReduction);
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def croppingAndPadding():
    # Initialize
    print("Initializing ...");
    
    # Load DICOM images
    print("Loading DICOM images ...");
    ## Forming dicom series folder path
    dicomSeriesFolder = os.path.join(dataFolder, "CT-Lung");
    ## Checking if the folder exists
    if not os.path.exists(dicomSeriesFolder):
        print("DICOM series folder does not exist: ", dicomSeriesFolder);
        return;
    ## Getting dicom file paths
    dicomFiles = [os.path.join(dicomSeriesFolder, f)
                  for f in os.listdir(dicomSeriesFolder) if f.endswith('.dcm')];
    ## Checking if any DICOM files are found
    if not dicomFiles:
        print("No DICOM files found in the folder: ", dicomSeriesFolder);
        return;
    ## Loading DICOM images
    for file in dicomFiles:
        try:
            dicomImage = pydicom.dcmread(file);
            dicomSeries.append(dicomImage);
        except Exception as e:
            print(f"Error reading {file}: {e}");
    ## Checking if any images were loaded
    if not dicomSeries:
        print("No DICOM images were loaded.");
        return;
    print(f"Loaded {len(dicomSeries)} DICOM images.");

    # Get the middle slice
    print("Getting the middle slice ...");
    middleSliceIndex = len(dicomSeries) // 2;
    middleSlice = dicomSeries[middleSliceIndex];
    ## Getting pixel data
    pixelData = middleSlice.pixel_array;
    ## Normalize the pixel data for visualization
    lowerValue = 0.0; upperValue = 1.0;
    pixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue,
                                               pixelData, dtype=cv2.CV_32F);
    
    # Show original image
    print("Showing original DICOM image ...");
    cv2.imshow("Original DICOM Image", pixelData);

    # Test cropping image
    print("Testing cropping image ...");
    ## Define crop rectangle (x, y, width, height)
    cropX = 50; cropY = 50; cropWidth = 200; cropHeight = 200;
    cropRect = (cropX, cropY, cropWidth, cropHeight);
    ## Crop the image using numpy slicing
    croppedImage = pixelData[cropY:cropY + cropHeight, cropX:cropX + cropWidth];
    ## Normalize the cropped image for visualization
    croppedImage = normalizePixelDataFromMinToMax(lowerValue, upperValue,
                                                   croppedImage, dtype=cv2.CV_32F);
    ## Visualize cropped image using opencv
    cv2.imshow("Cropped DICOM Image", croppedImage);

    # Test padding image
    print("Testing padding image ...");
    ## Define padding values (top, bottom, left, right)
    padTop = 100; padBottom = 100; padLeft = 100; padRight = 100;
    ## Create a padded image using numpy
    paddedImage = np.pad(pixelData, ((padTop, padBottom), (padLeft, padRight)), 
                         mode='constant', constant_values=0);
    ## Visualize window using opencv
    cv2.imshow("Padded DICOM Image", paddedImage);    

    # Wait for key press and close windows
    print("Press any key to close the windows ...");
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def histogramEqualization():
    # Initialize
    print("Initializing ...");

    # Load DICOM images
    print("Loading DICOM images ...");
    dicomSeriesFolder = os.path.join(dataFolder, "CT-Lung");
    if not os.path.exists(dicomSeriesFolder):
        print("DICOM series folder does not exist: ", dicomSeriesFolder);
        return;
    dicomFiles = [os.path.join(dicomSeriesFolder, f) 
                  for f in os.listdir(dicomSeriesFolder) if f.endswith('.dcm')];
    if not dicomFiles:
        print("No DICOM files found in the folder: ", dicomSeriesFolder);
        return;
    for file in dicomFiles:
        try:
            dicomImage = pydicom.dcmread(file);
            dicomSeries.append(dicomImage);
        except Exception as e:
            print(f"Error reading {file}: {e}");
    if not dicomSeries:
        print("No DICOM images were loaded.");
        return;
    print(f"Loaded {len(dicomSeries)} DICOM images.");

    # Get the middle slice
    print("Getting the middle slice ...");
    middleSliceIndex = len(dicomSeries) // 2;
    middleSlice = dicomSeries[middleSliceIndex];
    pixelData = middleSlice.pixel_array;
    lowerValue = 0.0; upperValue = 1.0;
    pixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue, 
                                               pixelData, dtype=cv2.CV_32F);

    # Show original image
    print("Showing original DICOM image ...");
    cv2.imshow("Original DICOM Image", pixelData);

    # Apply histogram equalization
    print("Applying histogram equalization ...");
    ## Convert pixel data to uint8 for histogram equalization
    pixelDataUint8 = cv2.normalize(pixelData, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U);
    ## Apply histogram equalization
    equalizedImage = cv2.equalizeHist(pixelDataUint8);
    ## Normalize the equalized image for visualization
    equalizedImage = normalizePixelDataFromMinToMax(lowerValue, upperValue,
                                                    equalizedImage, dtype=cv2.CV_32F);
    ## Visualize equalized image using opencv
    cv2.imshow("Histogram Equalized DICOM Image", equalizedImage);

    # Wait for key press and close windows
    print("Press any key to close the windows ...");
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def edgeEnhancement():
    # Initialize
    print("Initializing ...");

    # Load DICOM images
    print("Loading DICOM images ...");
    dicomSeriesFolder = os.path.join(dataFolder, "CT-Lung");
    if not os.path.exists(dicomSeriesFolder):
        print("DICOM series folder does not exist: ", dicomSeriesFolder);
        return;
    dicomFiles = [os.path.join(dicomSeriesFolder, f) 
                  for f in os.listdir(dicomSeriesFolder) if f.endswith('.dcm')];
    if not dicomFiles:
        print("No DICOM files found in the folder: ", dicomSeriesFolder);
        return;
    for file in dicomFiles:
        try:
            dicomImage = pydicom.dcmread(file);
            dicomSeries.append(dicomImage);
        except Exception as e:
            print(f"Error reading {file}: {e}");
    if not dicomSeries:
        print("No DICOM images were loaded.");
        return;
    print(f"Loaded {len(dicomSeries)} DICOM images.");

    # Get the middle slice
    print("Getting the middle slice ...");
    middleSliceIndex = len(dicomSeries) // 2;
    middleSlice = dicomSeries[middleSliceIndex];
    pixelData = middleSlice.pixel_array;
    lowerValue = 0.0; upperValue = 1.0;
    pixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue, 
                                               pixelData, dtype=cv2.CV_32F);

    # Show original image
    print("Showing original DICOM image ...");
    cv2.imshow("Original DICOM Image", pixelData);

    # Apply edge enhancement
    print("Applying edge enhancement ...");
    ## Convert pixel data to uint8 for edge enhancement
    pixelDataUint8 = cv2.normalize(pixelData, None, 0, 255, 
                                   cv2.NORM_MINMAX, dtype=cv2.CV_8U);
    ## Apply Canny edge detection
    edges = cv2.Canny(pixelDataUint8, 100, 200);
    ## Normalize the edges for visualization
    edges = normalizePixelDataFromMinToMax(lowerValue, upperValue, 
                                           edges, dtype=cv2.CV_32F);
    ## Visualize edges using opencv
    cv2.imshow("Edge Enhanced DICOM Image", edges);

    # Wait for key press and close windows
    print("Press any key to close the windows ...");
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def frequencyDomainFiltering():
    # Initialize
    print("Initializing ...");

    # Load DICOM images
    print("Loading DICOM images ...");
    dicomSeriesFolder = os.path.join(dataFolder, "CT-Lung");
    if not os.path.exists(dicomSeriesFolder):
        print("DICOM series folder does not exist: ", dicomSeriesFolder);
        return;
    dicomFiles = [os.path.join(dicomSeriesFolder, f) 
                  for f in os.listdir(dicomSeriesFolder) if f.endswith('.dcm')];
    if not dicomFiles:
        print("No DICOM files found in the folder: ", dicomSeriesFolder);
        return;
    for file in dicomFiles:
        try:
            dicomImage = pydicom.dcmread(file);
            dicomSeries.append(dicomImage);
        except Exception as e:
            print(f"Error reading {file}: {e}");
    if not dicomSeries:
        print("No DICOM images were loaded.");
        return;
    print(f"Loaded {len(dicomSeries)} DICOM images.");
    
    # Get the middle slice
    print("Getting the middle slice ...");
    middleSliceIndex = len(dicomSeries) // 2;
    middleSlice = dicomSeries[middleSliceIndex];
    pixelData = middleSlice.pixel_array;
    lowerValue = 0.0; upperValue = 1.0;
    pixelData = normalizePixelDataFromMinToMax(lowerValue, upperValue,
                                                  pixelData, dtype=cv2.CV_32F);
    
    # Show original image
    print("Showing original DICOM image ...");
    cv2.imshow("Original DICOM Image", pixelData);
    
    # Apply frequency domain filtering
    print("Applying frequency domain filtering ...");
    ## Step 1: Compute the DFT (Discrete Fourier Transform)
    dft = np.fft.fft2(pixelData)
    dftShift = np.fft.fftshift(dft)
    ## Step 2: Create a low-pass filter mask
    rows, cols = pixelData.shape
    crow, ccol = rows // 2 , cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # 60x60 square at center
    ## Step 3: Apply the mask
    filteredDft = dftShift * mask
    ## Step 4: Inverse DFT to get the filtered image
    idftShift = np.fft.ifftshift(filteredDft)
    imgBack = np.fft.ifft2(idftShift)
    imgBack = np.abs(imgBack)
    ## Step 5: Normalize the filtered image for visualization
    imgBack = normalizePixelDataFromMinToMax(lowerValue, upperValue, imgBack, 
                                              dtype=cv2.CV_32F)
    ## Visualize filtered image using opencv
    cv2.imshow("Frequency Domain Filtered DICOM Image", imgBack);

    # Wait for key press and close windows
    print("Press any key to close the windows ...");
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Finished processing
    print("Finished processing.");

#******************************************************************************************************************#
#**********************************************MAIN FUNCTION*******************************************************#
#******************************************************************************************************************#
def main():
    os.system("cls");
    frequencyDomainFiltering();
if __name__ == "__main__":
    main()