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
dataFolder = "../../../Data/Section_2_4_3_ImageProcessingBasedSegmentation";

#******************************************************************************************************************#
#**********************************************SUPPORTING FUNCTIONS************************************************#
#******************************************************************************************************************#

#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def segmentImagesUsingAvailableMask():
    # Initialize
    print("Initializing ...");

    # Read an example image
    print("Reading an example image ...");
    imageFilePath = os.path.join(dataFolder, "BrainTumorData", "images", "1.png");
    image = cv2.imread(imageFilePath, cv2.IMREAD_GRAYSCALE);
    if image is None:
        raise FileNotFoundError(f"Image file not found: {imageFilePath}");
    print("Image shape:", image.shape);

    # Read the corresponding mask
    print("Reading the corresponding mask ...");
    maskFilePath = os.path.join(dataFolder, "BrainTumorData", "masks", "1.png");
    mask = cv2.imread(maskFilePath, cv2.IMREAD_GRAYSCALE);
    if mask is None:
        raise FileNotFoundError(f"Mask file not found: {maskFilePath}");
    print("Mask shape:", mask.shape);

    # Segment the image using the mask
    print("Segmenting the image using the mask ...");
    segmentedImage = cv2.bitwise_and(image, image, mask=mask);

    # Draw the red color with opacity on the original image
    print("Drawing the red color with opacity on the original image ...");
    ## Create a color image for the overlay
    imageColor = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR);
    overlay = imageColor.copy();
    overlay[mask > 0] = [0, 0, 255];  # Red color (BGR format)
    overlayResult = cv2.addWeighted(overlay, 0.5, imageColor, 0.5, 0);  # Fixed the assignment

    # Show the original image
    print("Displaying the original image ...");
    cv2.imshow("Original Image", image);

    # Show the mask
    print("Displaying the mask ...");
    cv2.imshow("Mask", mask);

    # Show the segmented image
    print("Displaying the segmented image ...");
    cv2.imshow("Segmented Image", segmentedImage);

    # Show the overlay image
    print("Displaying the overlay image ...");
    cv2.imshow("Overlay Image", overlayResult);

    # Wait for a key press and close the windows
    print("Press any key to close the windows ...");
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Finished processing
    print("Finished processing.");
def segmentImageSlicesUsingUpperLowerThresholds():
    # Initialize
    print("Initializing ...");
    ## Initialize variables for saving global data
    global dicomImages, numOfSlices, minValue, maxValue, sliceIndex, sliceMask;
    ## Define function for update dicom image
    def updateDicomImage(inValue):
        global sliceIndex;
        sliceIndex = inValue;
        if 0 <= sliceIndex < numOfSlices:
            # Get the current image slice
            currentSlice = dicomImages[sliceIndex];

            # Generate the mask for image slice
            sliceMask = np.zeros_like(dicomImages[sliceIndex], dtype=np.uint8);    

            # Generate the mask using the current slice and threshold values
            sliceMask[(currentSlice >= minValue) & (currentSlice <= maxValue)] = 255;

            # Show the mask as red region on the current window,
            # but having the alpha value to show both the original and the mask
            overlay = cv2.cvtColor(currentSlice, cv2.COLOR_GRAY2BGR)
            redMask = np.zeros_like(overlay)
            redMask[sliceMask > 0] = [0, 0, 255]  # Red color (BGR)
            alpha = 0.25  # Adjust alpha for transparency
            blended = cv2.addWeighted(overlay, 1 - alpha, redMask, alpha, 0)
            cv2.imshow("Dicom Image", blended)
        else:
            print("Invalid slice index.");
    def upperThresholdSliderAction(inValue):
        # Define global variable
        global sliceMask, minValue, maxValue, sliceIndex;
        maxValue = inValue;

        # Get the current image slice
        currentSlice = dicomImages[sliceIndex];

        # Generate the mask for image slice
        sliceMask = np.zeros_like(dicomImages[sliceIndex], dtype=np.uint8);    

        # Generate the mask using the current slice and threshold values
        sliceMask[(currentSlice >= minValue) & (currentSlice <= maxValue)] = 255;

        # Show the mask as red region on the current window,
        # but having the alpha value to show both the original and the mask
        overlay = cv2.cvtColor(currentSlice, cv2.COLOR_GRAY2BGR)
        redMask = np.zeros_like(overlay)
        redMask[sliceMask > 0] = [0, 0, 255]  # Red color (BGR)
        alpha = 0.25  # Adjust alpha for transparency
        blended = cv2.addWeighted(overlay, 1 - alpha, redMask, alpha, 0)
        cv2.imshow("Dicom Image", blended)
    def lowerThresholdSliderAction(inValue):
        # Define global variable
        global sliceMask, minValue, maxValue, sliceIndex;
        minValue = inValue;

        # Get the current image slice
        currentSlice = dicomImages[sliceIndex];

        # Generate the mask for image slice
        sliceMask = np.zeros_like(dicomImages[sliceIndex], dtype=np.uint8);    

        # Generate the mask using the current slice and threshold values
        sliceMask[(currentSlice >= minValue) & (currentSlice <= maxValue)] = 255;

        # Show the mask as red region on the current window,
        # but having the alpha value to show both the original and the mask
        overlay = cv2.cvtColor(currentSlice, cv2.COLOR_GRAY2BGR)
        redMask = np.zeros_like(overlay)
        redMask[sliceMask > 0] = [0, 0, 255]  # Red color (BGR)
        alpha = 0.25  # Adjust alpha for transparency
        blended = cv2.addWeighted(overlay, 1 - alpha, redMask, alpha, 0)
        cv2.imshow("Dicom Image", blended)
    
    # Reading dicom slices
    print("Reading dicom slices ...");
    dicomFolderPath = os.path.join(dataFolder, "CT-SpinopelvicStructure");

    # Loading all dicom slices as images
    print("Loading all dicom slices as images ...");
    dicomImages = [];
    for filename in os.listdir(dicomFolderPath):
        if filename.endswith(".dcm"):
            dicomFilePath = os.path.join(dicomFolderPath, filename);
            dicomImage = pydicom.dcmread(dicomFilePath).pixel_array;
            dicomImages.append(dicomImage);
    numOfSlices = len(dicomImages);
    print("\t Loaded", numOfSlices, "dicom slices.");

    # Estimate the global min and max value of all dicom images
    print("Estimating global min and max values of the dicom images ...");
    allPixels = np.concatenate([img.flatten() for img in dicomImages]);
    globalMin = np.min(allPixels);
    globalMax = np.max(allPixels);
    print(f"\tGlobal min value: {globalMin}");
    print(f"\tGlobal max value: {globalMax}");

    # Normalize all image slice into 0 255 of the opencv
    print("Normalizing all images ...");
    for i in range(numOfSlices):
        dicomImages[i] = dicomImages[i].astype(np.float32);
        dicomImages[i] = (dicomImages[i] - globalMin) / (globalMax - globalMin);
        dicomImages[i] = (dicomImages[i] * 255).astype(np.uint8);
    print("Normalization complete.");

    # Estimate the min and max value of the dicom image value
    print("Estimating min and max values of the dicom image ...");
    minValue = float("inf");
    maxValue = float("-inf");
    for dicomImage in dicomImages:
        minValue = min(minValue, np.min(dicomImage));
        maxValue = max(maxValue, np.max(dicomImage));
    print("\tMin value:", minValue);
    print("\tMax value:", maxValue);

    # Initialize the visual windows with slider, visualize the middle dicom image
    print("Initializing visual windows ...");
    ## Initialize the empty windows named "Dicom Image"
    cv2.namedWindow("Dicom Image");
    ## Initialize the slider for slice selection
    cv2.createTrackbar("Slice", "Dicom Image", 
                       0, numOfSlices - 1, updateDicomImage);
    ## Initialize the sliders for upper and lower threshold
    cv2.createTrackbar("Upper", "Dicom Image", 
                       int(minValue), int(maxValue), 
                       upperThresholdSliderAction);
    cv2.createTrackbar("Lower", "Dicom Image", 
                       int(minValue), int(maxValue), 
                       lowerThresholdSliderAction);    
    ## Initialize the image visualization
    sliceIndex = int(numOfSlices / 2);
    ## Set the current value of the upper and lower threshold
    cv2.setTrackbarPos("Slice", "Dicom Image", int(sliceIndex));
    cv2.setTrackbarPos("Upper", "Dicom Image", int(maxValue));
    cv2.setTrackbarPos("Lower", "Dicom Image", int(minValue));
    cv2.waitKey(0);

    # Finished processing
    print("Finished processing.");

#******************************************************************************************************************#
#**********************************************MAIN FUNCTION*******************************************************#
#******************************************************************************************************************#
def main():
    os.system("cls");
    segmentImageSlicesUsingUpperLowerThresholds();
if __name__ == "__main__":
    main()