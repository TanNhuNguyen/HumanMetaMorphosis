#******************************************************************************************************************#
#**********************************************PROJECTION INFORMATION**********************************************#
#******************************************************************************************************************#

#******************************************************************************************************************#
#**********************************************SUPPORTING LIBRARIES************************************************#
#******************************************************************************************************************#
import os;
import numpy as np;
import cv2 as cv;

#******************************************************************************************************************#
#**********************************************SUPPORTING BUFFERS**************************************************#
#******************************************************************************************************************#
dataFolder = "../../../Data/Section_2_2_2_BasicXRayImageVisualization";

#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def loadAndDisplayXRayImage():
    # Information: This function will instruct the reader to load and visualize the x-ray image. The x-ray image was
    # processed and stored in the common image format, such as png and jpg. The reading and visualizing of the image
    # will be supported by OpenCV, which is a powerful library for image processing.
    # Initialize
    print("Initializing ...");

    # Loading imaging from file
    print("Loading x-ray image from file ...");
    ## Forming file path
    filePath = os.path.join(dataFolder, "XRayChestImages", "00000001_000.png");
    ## Checking file existence
    if not os.path.exists(filePath):
        print("File does not exist: " + filePath);
        return;
    ## Reading image
    xRayImage = cv.imread(filePath, cv.IMREAD_GRAYSCALE);
    if xRayImage is None:
        print("Failed to read image from file: " + filePath);
        return;

    # Displaying image
    print("Displaying x-ray image ...");
    cv.imshow("X-Ray Image", xRayImage);
    cv.waitKey(0);  # Wait for a key press to close the window
    cv.destroyAllWindows();

    # Finished processing
    print("Finished processing ...");
def printBasicImageInformation():
    # Initializing
    print("Initializing ...");

    # Loading image from file
    print("Loading x-ray image from file ...");
    ## Forming file path
    filePath = os.path.join(dataFolder, "XRayChestImages", "00000001_001.png");
    ## Checking file existence
    if not os.path.exists(filePath):
        print("File does not exist: " + filePath);
        return;
    ## Reading image
    xRayImage = cv.imread(filePath, cv.IMREAD_GRAYSCALE);
    if xRayImage is None:
        print("Failed to read image from file: " + filePath);
        return;

    # Displaying basic image information
    print("Displaying basic image information ...");
    print("Image shape: ", xRayImage.shape);
    print("Image size: ", xRayImage.size);
    print("Image data type: ", xRayImage.dtype);

    # Finished processing
    print("Finished processing ...");
def zoomIntoRegionsOfInterest():
    # Initializing
    print("Initializing ...");

    # Loading image from file
    print("Loading x-ray image from file ...");
    imageFilePath = os.path.join(dataFolder, "XRayChestImages", "00000001_002.png");
    if not os.path.exists(imageFilePath):
        print("File does not exist: " + imageFilePath);
        return;
    xRayImage = cv.imread(imageFilePath, cv.IMREAD_GRAYSCALE);
    if xRayImage is None:
        print("Failed to read image from file: " + imageFilePath);
        return;

    # Defining regions of interest (ROIs)
    print("Defining regions of interest (ROIs) ...");
    rois = [(300, 300, 200, 200)]; # Example ROI (x, y, width, height)
    print("\t ROIs defined: ", rois);

    # Getting the image in the ROI using OpenCV
    print("Getting the image in the ROI ...");
    roiImage = xRayImage[rois[0][1]:rois[0][1]+rois[0][3], rois[0][0]:rois[0][0]+rois[0][2]];

    # Visualize the ROI
    print("Visualizing the ROI ...");
    cv.imshow("Region of Interest", roiImage);
    cv.waitKey(0);  # Wait for a key press to close the window
    cv.destroyAllWindows();

    # Finished processing
    print("Finished processing ...");
def applyBasicImageEnhancements():
    # Initializing
    print("Initializing ...");

    # Loading image from file
    print("Loading x-ray image from file ...");
    imageFilePath = os.path.join(dataFolder, "XRayChestImages", "00000003_003.png");
    if not os.path.exists(imageFilePath):
        print("File does not exist: " + imageFilePath);
        return;
    xRayImage = cv.imread(imageFilePath, cv.IMREAD_GRAYSCALE);
    if xRayImage is None:
        print("Failed to read image from file: " + imageFilePath);
        return;

    # Applying basic image enhancements
    print("Applying basic image enhancements ...");
    # Example: Histogram Equalization
    print("\t Conducting histogram equalization ...");
    enhancedImage = cv.equalizeHist(xRayImage);
    # Example: Gaussian Blur
    print("\t Applying Gaussian Blur ...");
    blurredImage = cv.GaussianBlur(enhancedImage, (5, 5), 0);
    # Example: Canny Edge Detection
    print("\t Applying Canny Edge Detection ...");
    edgesImage = cv.Canny(blurredImage, 100, 200);

    # Displaying the processed images with the original image
    # The original image is on the top left, and the processed images are arranged in a grid
    print("Displaying the processed images ...");
    ## Generate the visual image
    firstRowImage = np.hstack((xRayImage, enhancedImage));
    secondRoundImage = np.hstack((blurredImage, edgesImage));
    visualImage = np.vstack((firstRowImage, secondRoundImage));
    ## Resize image to visualize it better
    visualImage = cv.resize(visualImage, (900, 900));
    ## Display the visual image
    cv.imshow("Processed X-Ray Image", visualImage);
    cv.waitKey(0);  # Wait for a key press to close the window
    cv.destroyAllWindows();

    # Finished processing
    print("Finished processing ...");
def saveProcessedImage():
    # Initializing
    print("Initializing ...");

    # Load image from file
    print("Loading x-ray image from file ...");
    imageFilePath = os.path.join(dataFolder, "XRayDentals", "Image_0.jpg");
    if not os.path.exists(imageFilePath):
        print("File does not exist: " + imageFilePath);
        return;
    xRayImage = cv.imread(imageFilePath, cv.IMREAD_GRAYSCALE);
    if xRayImage is None:
        print("Failed to read image from file: " + imageFilePath);
        return;

    # Processing the image using threshold
    print("Processing the image using threshold ...");
    _, processedImage = cv.threshold(xRayImage, 127, 255, cv.THRESH_BINARY);

    # Saving the processed image
    print("Saving the processed image ...");
    ## Create processed image folder if it does not exist
    processedImageFolder = os.path.join(dataFolder, "ProcessedImages");
    if not os.path.exists(processedImageFolder):
        os.makedirs(processedImageFolder);
    ## Forming output file path
    outputFilePath = os.path.join(dataFolder, "ProcessedImages", "Processed_XRay_Image.png");
    ## Saving the image
    cv.imwrite(outputFilePath, processedImage);

    # Finished processing
    print("Finished processing ...");
#******************************************************************************************************************#
#**********************************************MAIN FUNCTION*******************************************************#
#******************************************************************************************************************#
def main():
    os.system("cls");
    saveProcessedImage();
if __name__ == "__main__":
    main()