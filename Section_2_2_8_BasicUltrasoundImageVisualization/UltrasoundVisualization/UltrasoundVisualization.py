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
dataFolder = "../../../Data/Section_2_2_8_BasicUltrasoundImageVisualization";
dicomSeries = [];

#******************************************************************************************************************#
#**********************************************SUPPORTING FUNCTIONS************************************************#
#******************************************************************************************************************#


#******************************************************************************************************************#
#**********************************************PROCESSING FUNCTIONS************************************************#
#******************************************************************************************************************#
def segmentUltrasoundImageWithPreDefinedMask():
    # Initialize
    print("Initializing ...");

    # Reading ultrasound image
    print("Reading ultrasound image ...");
    ## Forming file path
    ultrasoundImagePath = dataFolder + "/Ultrasound_malignant.png";
    ## Checking if the file exists
    if not os.path.exists(ultrasoundImagePath):
        print("Error: Ultrasound image file does not exist at the specified path: "\
               + ultrasoundImagePath);
        return;
    ## Reading the image
    ultrasoundImage = cv.imread(ultrasoundImagePath, cv.IMREAD_GRAYSCALE);

    # Print information about the image
    print("Image information:");
    print("\t Image shape: " + str(ultrasoundImage.shape));
    print("\t Image data type: " + str(ultrasoundImage.dtype));
    print("\t Image size: " + str(ultrasoundImage.size));
    print("\t Image pixel value range: " + str(np.min(ultrasoundImage)) + " to " + \
          str(np.max(ultrasoundImage)));
    print("\t Image pixel value mean: " + str(np.mean(ultrasoundImage)));
    print("\t Image pixel value standard deviation: " + str(np.std(ultrasoundImage)));
    print("\t Image pixel value median: " + str(np.median(ultrasoundImage)));
    print("\t Image pixel value variance: " + str(np.var(ultrasoundImage)));

    # Display the ultrasound image
    print("Displaying ultrasound image ...");
    cv.imshow("Ultrasound Image", ultrasoundImage);
    cv.waitKey(0);
    cv.destroyAllWindows();

    # Reading mask image
    print("Reading mask image ...");
    ## Forming file path
    maskImagePath = dataFolder + "/Ultrasound_malignant_tumorMask.png";
    ## Checking if the file exists
    if not os.path.exists(maskImagePath):
        print("Error: Mask image file does not exist at the specified path: " + maskImagePath);
        return;
    ## Reading the mask image
    maskImage = cv.imread(maskImagePath, cv.IMREAD_GRAYSCALE);

    # Print information about the mask image
    print("Mask image information:");
    print("\t Mask image shape: " + str(maskImage.shape));
    print("\t Mask image data type: " + str(maskImage.dtype));
    print("\t Mask image size: " + str(maskImage.size));
    print("\t Mask image pixel value range: " + str(np.min(maskImage)) + " to " + \
          str(np.max(maskImage)));
    print("\t Mask image pixel value mean: " + str(np.mean(maskImage)));
    print("\t Mask image pixel value standard deviation: " + str(np.std(maskImage)));
    print("\t Mask image pixel value median: " + str(np.median(maskImage)));
    print("\t Mask image pixel value variance: " + str(np.var(maskImage)));

    # Display the mask image
    print("Displaying mask image ...");
    cv.imshow("Mask Image", maskImage);
    cv.waitKey(0);
    cv.destroyAllWindows();

    # Segmenting the ultrasound image using the mask
    print("Segmenting the ultrasound image using the mask ...");
    segmentedImage = cv.bitwise_and(ultrasoundImage, ultrasoundImage, mask=maskImage);
    cv.imshow("Segmented Image", segmentedImage);
    cv.waitKey(0);
    cv.destroyAllWindows();

    # Save the segmented image to file
    print("Saving the segmented image to file ...");
    ## Create output folder if it does not exist
    outputFolder = dataFolder + "/SegmentedImages";
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder);
    ## Forming output file path
    segmentedImagePath = outputFolder + "/Ultrasound_malignant_segmented.png";
    ## Saving the segmented image
    cv.imwrite(segmentedImagePath, segmentedImage);

    # Finished processing
    print("Finished processing.");
#******************************************************************************************************************#
#**********************************************MAIN FUNCTION*******************************************************#
#******************************************************************************************************************#
def main():
    os.system("cls");
    segmentUltrasoundImageWithPreDefinedMask();
if __name__ == "__main__":
    main()