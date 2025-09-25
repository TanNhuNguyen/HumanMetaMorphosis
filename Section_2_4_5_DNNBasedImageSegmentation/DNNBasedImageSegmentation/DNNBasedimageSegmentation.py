#**********************************************************************************#
#****************************** SUPPORTING LIBRARIES ******************************#
#**********************************************************************************#
import os;
import random
import sys;
import torch;
import torch.nn as nn;
import torch.optim as optim;
from torch.utils.data import Dataset;
from torch.utils.data import DataLoader;
from PIL import Image, ImageDraw;
import numpy as np;
from torchsummary import summary;
import cv2;

#**********************************************************************************#
#****************************** SUPPORTING BUFFERS ********************************#
#**********************************************************************************#
mainFolder = "../../../Data/Section_2_4_5_DNNBasedImageSegmentation";
rawDataFolder = mainFolder + "/BrainTumorImagesSegments";
imageFolder = rawDataFolder + "/images";
maskFolder = rawDataFolder + "/masks";
syntheticDataFolder = mainFolder + "/SyntheticImageSegments";
syntheticImageFolder = syntheticDataFolder + "/Images";
syntheticMaskFolder = syntheticDataFolder + "/Masks";
trainValidTestSplitFolder = mainFolder + "/TrainValidTestSplits";
crossValidationFoldFolder = mainFolder + "/CrossValidations";
networkGraphFolder = mainFolder + "/NetworkGraphs";

#**********************************************************************************#
#****************************** SUPPORTING CLASSES ********************************#
#**********************************************************************************#
# The simple UNET class
class SimpleUnet(nn.Module):
    def __init__(self):
        super(SimpleUnet, self).__init__();
        # Encoder
        self.enc1 = self.convBlock(1, 64);
        self.enc2 = self.convBlock(64, 128);
        self.enc3 = self.convBlock(128, 256);
        self.enc4 = self.convBlock(256, 512);
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2);
        self.dec3 = self.convBlock(512, 256);
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2);
        self.dec2 = self.convBlock(256, 128);
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2);
        self.dec1 = self.convBlock(128, 64);
        # Output
        self.final = nn.Conv2d(64, 1, kernel_size=1);
        self.sigmoid = nn.Sigmoid();
        self.pool = nn.MaxPool2d(2);
    def convBlock(self, inChannels, outChannels):
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        );
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x);
        e2 = self.enc2(self.pool(e1));
        e3 = self.enc3(self.pool(e2));
        e4 = self.enc4(self.pool(e3));
        # Decoder
        d3 = self.up3(e4);
        d3 = torch.cat([d3, e3], dim=1);
        d3 = self.dec3(d3);
        d2 = self.up2(d3);
        d2 = torch.cat([d2, e2], dim=1);
        d2 = self.dec2(d2);
        d1 = self.up1(d2);
        d1 = torch.cat([d1, e1], dim=1);
        d1 = self.dec1(d1);
        out = self.final(d1);
        out = self.sigmoid(out);
        return out;
    def trainModel(self, inTrainLoader, inNumEpochs=1, inLearningRate=1e-3, inDevice='cpu'):
        # Move model to the specified device
        self.to(inDevice);

        # Define loss function
        criterion = nn.BCELoss();

        # Define optimizer
        optimizer = optim.Adam(self.parameters(), lr=inLearningRate);

        # Training loop
        trainingLossValues = np.zeros((inNumEpochs, 1));
        for epoch in range(inNumEpochs):
            # Initializing
            self.train(); trainTotalLoss = 0.0;
            totalBatches = len(inTrainLoader);

            # Iterate over batches
            for batchIdx, (images, masks) in enumerate(inTrainLoader):
                # Move data to the specified device
                images = images.to(inDevice);
                masks = masks.to(inDevice);

                # Forward pass
                optimizer.zero_grad();
                outputs = self(images);
                loss = criterion(outputs, masks);

                # Backward pass
                loss.backward();
                optimizer.step();
                trainTotalLoss += loss.item() * images.size(0);

                # Progress bar
                progress = int(50 * (batchIdx + 1) / totalBatches);
                bar = '[' + '=' * progress + ' ' * (50 - progress) + ']';
                percent = 100 * (batchIdx + 1) / totalBatches;
                print(f"\rEpoch {epoch+1}/{inNumEpochs} Training {bar} {percent:6.2f}% - loss: {loss.item():.4f}", end='');

            # Compute average training loss
            avgTrainLoss = trainTotalLoss / len(inTrainLoader.dataset);
            trainingLossValues[epoch] = avgTrainLoss;
            print(f"\nSimpleUnet::trainModel:: Epoch [{epoch+1}/{inNumEpochs}], Train Loss: {avgTrainLoss:.4f}");

        # Return the training losses
        return trainingLossValues;
    def evaluateModel(self, inValidationLoader, inNumEpochs=1, inDevice='cpu'):
        # Move model to the specified device
        self.to(inDevice);

        # Define loss function
        criterion = nn.BCELoss();

        # Validation loop
        evaluatedLossValues = np.zeros((inNumEpochs, 1));
        for epoch in range(inNumEpochs):
            # Set model to evaluation mode
            self.eval(); evalTotalLoss = 0.0;

            # Disable gradient calculation
            totalValBatches = len(inValidationLoader);
            with torch.no_grad():
                # Iterate over validation batches
                for batchIdx, (images, masks) in enumerate(inValidationLoader):
                    # Move data to the specified device
                    images = images.to(inDevice);
                    masks = masks.to(inDevice);

                    # Forward pass
                    outputs = self(images);
                    loss = criterion(outputs, masks);
                    evalTotalLoss += loss.item() * images.size(0);

                    # Progress bar for validation
                    progress = int(50 * (batchIdx + 1) / totalValBatches);
                    bar = '[' + '=' * progress + ' ' * (50 - progress) + ']';
                    percent = 100 * (batchIdx + 1) / totalValBatches;
                    print(f"\rValidation {bar} {percent:6.2f}% - loss: {loss.item():.4f}", end='');

            # Compute average validation loss
            avgValLoss = evalTotalLoss / len(inValidationLoader.dataset);
            evaluatedLossValues[epoch] = avgValLoss;
            print(f"\nSimpleUnet::EvaluateModel:: Epoch [{epoch+1}/{inNumEpochs}], Val Loss: {avgValLoss:.4f}");

        # Return the validation losses
        return evaluatedLossValues;
    def predict(self, inImage, inDevice='cpu'):
        # Move model to the specified device
        self.to(inDevice);

        # Set model to evaluation mode
        self.eval();

        # Disable gradient calculation
        with torch.no_grad():
            # Move image to the specified device
            inImage = inImage.to(inDevice);

            # Forward pass
            output = self(inImage.unsqueeze(0));  # Add batch dimension

            # Apply sigmoid activation
            output = output.squeeze().cpu().numpy();

            # Create binary mask
            binaryMask = (output > 0.5).astype('uint8');
        return binaryMask;
    def saveModel(self, inModelPath):
        """Save the model weights to a file."""
        torch.save(self.state_dict(), inModelPath);
    def loadModel(self, inModelPath, inDevice='cpu'):
        """Load model weights from a file into this instance."""
        self.load_state_dict(torch.load(inModelPath, map_location=inDevice, weights_only=True));
        self.to(inDevice);
        self.eval();
    def diceScore(self, inPredictedMask, inTrueMask, inSmooth=1e-6):
        """
        Compute the Dice coefficient between the predicted mask and ground truth mask.
        Args:
            predMask (numpy.ndarray or torch.Tensor): Predicted binary mask.
            trueMask (numpy.ndarray or torch.Tensor): Ground truth binary mask.
            smooth (float): Smoothing factor to avoid division by zero.
        Returns:
            dice (float): Dice coefficient score.
        """
        if isinstance(inPredictedMask, torch.Tensor):
            inPredictedMask = inPredictedMask.cpu().numpy();
        if isinstance(inTrueMask, torch.Tensor):
            inTrueMask = inTrueMask.cpu().numpy();
        inPredictedMask = inPredictedMask.astype('bool');
        inTrueMask = inTrueMask.astype('bool');
        intersection = (inPredictedMask & inTrueMask).sum();
        dice = (2. * intersection + inSmooth) / (inPredictedMask.sum() + inTrueMask.sum() + inSmooth);
        return dice;

# The advanced UNET class
class AdvancedUnet(nn.Module):
    def __init__(self):
        super(AdvancedUnet, self).__init__()
        # Encoder with more filters and dropout
        self.enc1 = self.convBlock(1, 64)
        self.enc2 = self.convBlock(64, 128)
        self.enc3 = self.convBlock(128, 256)
        self.enc4 = self.convBlock(256, 512)
        self.enc5 = self.convBlock(512, 1024)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(0.5)

        # Decoder with attention and more layers
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.convBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.convBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.convBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.convBlock(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def convBlock(self, inChannels, outChannels):
        return nn.Sequential(
            nn.Conv2d(inChannels, outChannels, 3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChannels, outChannels, 3, padding=1),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e5 = self.dropout(e5)

        # Decoder
        d4 = self.up4(e5)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.final(d1)
        out = self.sigmoid(out)
        return out
    def trainModel(self, inTrainLoader, inNumEpochs=1, inLearningRate=1e-3, inDevice='cpu'):
        # Move model to the specified device
        self.to(inDevice);

        # Define loss function
        criterion = nn.BCELoss();

        # Define optimizer
        optimizer = optim.Adam(self.parameters(), lr=inLearningRate);

        # Training loop
        trainingLossValues = np.zeros((inNumEpochs, 1));
        for epoch in range(inNumEpochs):
            # Initializing
            self.train(); trainTotalLosses = 0.0;
            totalBatches = len(inTrainLoader);

            # Iterate over batches
            for batchIdx, (images, masks) in enumerate(inTrainLoader):
                # Move data to the specified device
                images = images.to(inDevice);
                masks = masks.to(inDevice);

                # Forward pass
                optimizer.zero_grad();
                outputs = self(images);
                loss = criterion(outputs, masks);

                # Backward pass
                loss.backward();
                optimizer.step();
                trainTotalLosses += loss.item() * images.size(0);

                # Progress bar
                progress = int(50 * (batchIdx + 1) / totalBatches);
                bar = '[' + '=' * progress + ' ' * (50 - progress) + ']';
                percent = 100 * (batchIdx + 1) / totalBatches;
                print(f"\rEpoch {epoch+1}/{inNumEpochs} Training {bar} {percent:6.2f}% - loss: {loss.item():.4f}", end='');

            # Compute average training loss
            avgTrainLoss = trainTotalLosses / len(inTrainLoader.dataset);
            trainingLossValues[epoch] = avgTrainLoss;
            print(f"\nAdvancedUnet::trainModel:: Epoch [{epoch+1}/{inNumEpochs}], Train Loss: {avgTrainLoss:.4f}");

        # Return the training losses
        return trainingLossValues;
    def evaluateModel(self, inValidationLoader, inNumEpochs=1, inDevice='cpu'):
        # Move model to the specified device
        self.to(inDevice);

        # Define loss function
        criterion = nn.BCELoss();

        # Validation loop
        validationLossValues = np.zeros((inNumEpochs, 1));
        for epoch in range(inNumEpochs):
            # Set model to evaluation mode
            self.eval(); evalTotalLosses = 0.0;

            # Disable gradient calculation
            totalValBatches = len(inValidationLoader);
            with torch.no_grad():
                # Iterate over validation batches
                for batchIdx, (images, masks) in enumerate(inValidationLoader):
                    # Move data to the specified device
                    images = images.to(inDevice);
                    masks = masks.to(inDevice);

                    # Forward pass
                    outputs = self(images);
                    loss = criterion(outputs, masks);
                    evalTotalLosses += loss.item() * images.size(0);

                    # Progress bar for validation
                    progress = int(50 * (batchIdx + 1) / totalValBatches);
                    bar = '[' + '=' * progress + ' ' * (50 - progress) + ']';
                    percent = 100 * (batchIdx + 1) / totalValBatches;
                    print(f"\rValidation {bar} {percent:6.2f}% - loss: {loss.item():.4f}", end='');

            # Compute average validation loss
            avgValLoss = evalTotalLosses / len(inValidationLoader.dataset);
            validationLossValues[epoch] = avgValLoss;
            print(f"\nAdvancedUnet::EvaluateModel:: Epoch [{epoch+1}/{inNumEpochs}], Val Loss: {avgValLoss:.4f}");

        # Return the validation losses
        return validationLossValues;
    def predict(self, inImage, inDevice='cpu'):
        # Move model to the specified device
        self.to(inDevice);

        # Set model to evaluation mode
        self.eval();

        # Disable gradient calculation
        with torch.no_grad():
            # Move image to the specified device
            inImage = inImage.to(inDevice);

            # Forward pass
            output = self(inImage.unsqueeze(0));  # Add batch dimension

            # Apply sigmoid activation
            output = output.squeeze().cpu().numpy();

            # Create binary mask
            binaryMask = (output > 0.5).astype('uint8');
        return binaryMask;
    def saveModel(self, inModelPath):
        torch.save(self.state_dict(), inModelPath)
    def loadModel(self, inModelPath, inDevice='cpu'):
        self.load_state_dict(torch.load(inModelPath, map_location=inDevice, weights_only=True));
        self.to(inDevice);
        self.eval();
    def diceScore(self, inPredictedMask, inTrueMask, inSmooth=1e-6):
        if isinstance(inPredictedMask, torch.Tensor):
            inPredictedMask = inPredictedMask.cpu().numpy()
        if isinstance(inTrueMask, torch.Tensor):
            inTrueMask = inTrueMask.cpu().numpy()
        inPredictedMask = inPredictedMask.astype('bool')
        inTrueMask = inTrueMask.astype('bool')
        intersection = (inPredictedMask & inTrueMask).sum()
        dice = (2. * intersection + inSmooth) / (inPredictedMask.sum() + inTrueMask.sum() + inSmooth)
        return dice

# The Brain Tumor Dataset
class BrainTumorDataset(Dataset):
    def __init__(self, fileNames, imageFolder, maskFolder, transform=None):
        self.fileNames = fileNames
        self.imageFolder = imageFolder
        self.maskFolder = maskFolder
        self.transform = transform
        self.targetSize = (512, 512)
    def __len__(self):
        return len(self.fileNames)
    def __getitem__(self, idx):
        imageName = self.fileNames[idx]
        imagePath = os.path.join(self.imageFolder, imageName)
        maskPath = os.path.join(self.maskFolder, imageName)

        # Load image and mask as grayscale
        image = Image.open(imagePath).convert('L').resize(self.targetSize, Image.BILINEAR)
        mask = Image.open(maskPath).convert('L').resize(self.targetSize, Image.BILINEAR)

        # Convert to numpy and normalize to [0,1]
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0

        # Add channel dimension
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        # To tensor
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

#**********************************************************************************#
#****************************** SUPPORTING FUNCTIONS ******************************#
#**********************************************************************************#
def saveListOfStringsToTXTFile(inFilePath, inListOfStrings):
    with open(inFilePath, "w") as f:
        for item in inListOfStrings:
            f.write("%s\n" % item);
def readListOfStringsFromTXTFile(inFilePath):
    with open(inFilePath, 'r') as f:
        return [line.strip() for line in f.readlines()];
def saveNumPyArrayToCSVFile(inFilePath, inArray):
    np.savetxt(inFilePath, inArray, delimiter=",");
def renderMaskOnImage(inImage, inMask, inAlpha=0.25):
    """
    test_image: PIL Image (mode 'L' or 'RGB')
    mask: numpy array (H, W), binary (0 or 1)
    alpha: opacity of the red mask
    """
    # Ensure test_image is RGB
    if inImage.mode != 'RGB':
        inImage = inImage.convert('RGB')

    # Create red mask image
    red_mask = Image.new('RGBA', inImage.size, (255, 0, 0, 0))
    mask_img = Image.fromarray((inMask * 255).astype(np.uint8)).resize(inImage.size)
    red_mask_np = np.array(red_mask)
    mask_np = np.array(mask_img)
    red_mask_np[..., 3] = (mask_np > 0) * int(255 * inAlpha)
    red_mask = Image.fromarray(red_mask_np, mode='RGBA')

    # Overlay red mask on the original image
    result = inImage.convert('RGBA')
    result = Image.alpha_composite(result, red_mask)

    # Show the result
    result.show()
    return result
def generatePieAndTriangleImages(inImageFolder, inMaskFolder, inNumImages=10):
    os.makedirs(inImageFolder, exist_ok=True)
    os.makedirs(inMaskFolder, exist_ok=True)
    size = 512

    for idx in range(inNumImages):
        # Create blank images
        img = Image.new('L', (size, size), 0)
        mask = Image.new('L', (size, size), 0)
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        # --- Draw full pie (full circle) ---
        while True:
            cx = random.randint(100, 412)
            cy = random.randint(100, 412)
            r = random.randint(60, 180)
            # Ensure the circle is fully inside the image
            if (cx - r >= 0) and (cy - r >= 0) and (cx + r < size) and (cy + r < size):
                break
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw_img.ellipse(bbox, fill=255)
        draw_mask.ellipse(bbox, fill=255)
        # Save the pie mask for overlap checking
        pie_mask = np.array(mask)

        # --- Draw full triangle (well-separated random points, no overlap with pie) ---
        def random_triangle(size, margin=50):
            pts = []
            while len(pts) < 3:
                pt = (random.randint(margin, size-margin), random.randint(margin, size-margin))
                if all(np.linalg.norm(np.array(pt)-np.array(p)) > 60 for p in pts):
                    pts.append(pt)
            return pts

        # Try triangles until one does not overlap the pie
        max_attempts = 100
        for attempt in range(max_attempts):
            tri = random_triangle(size)
            # Create a temp mask for the triangle
            temp_tri_mask = Image.new('L', (size, size), 0)
            ImageDraw.Draw(temp_tri_mask).polygon(tri, fill=255)
            temp_tri_mask_np = np.array(temp_tri_mask)
            # Check overlap
            overlap = np.logical_and(pie_mask > 0, temp_tri_mask_np > 0).any()
            if not overlap:
                break
        # Draw triangle on image only (not on mask)
        draw_img.polygon(tri, fill=255)

        # Save images
        filename = f"synthetic_{idx:03d}.png"
        img.save(os.path.join(inImageFolder, filename))
        mask.save(os.path.join(inMaskFolder, filename))

        print(f"Saved {filename} (image and mask)")

#**********************************************************************************#
#****************************** PROCESSING PROCEDURES *****************************#
#**********************************************************************************#
def generateSyntheticDataForTrainingValidatingAndTesting():
    # Initializing
    print("Initializing ...");
    numOfImages = 5000;

    # Generate the images and masks
    print("Generate the images and masks ...");
    generatePieAndTriangleImages(syntheticImageFolder, syntheticMaskFolder, numOfImages);

    # Finished processing
    print("Finished processing.");
def generateTrainValidTestSets():
    # Initializing
    print("Initializing train/valid/test sets generation ...");

    # Get all image file names
    print("Getting all image file names ...");
    ## Get all image file names and stored as lists
    imageFileNames = os.listdir(syntheticImageFolder);
    ## Sort the image file names as nature not alphabet
    imageFileNames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))) or 0)
    ## Save the image file names to file
    saveListOfStringsToTXTFile(trainValidTestSplitFolder + "/AllImageFileNames.txt", 
                               imageFileNames);
    ## Debugging to show the number of found image
    print("\t The number of images found:", len(imageFileNames));

    # Generate the train, valid, and test list in ten-fold cross-validation
    print("Generating train/valid/test splits ...");
    numOfValidations = 10; trainRate = 0.7; validRate = 0.2; testRate = 0.1;
    for v in range(0, numOfValidations):
        # Debugging
        print("\t Generating split for validation fold:", v);

        # Shuffle the image file names
        random.shuffle(imageFileNames);

        # Forming the train, valid, test
        numOfTrain = int(trainRate * len(imageFileNames));
        numOfValidation = int(validRate * len(imageFileNames));
        numOfTest = len(imageFileNames) - numOfTrain - numOfValidation;

        # Save the splits to files
        saveListOfStringsToTXTFile(trainValidTestSplitFolder + f"/TrainSet_Fold_{v}.txt", 
                                   imageFileNames[:numOfTrain]);
        saveListOfStringsToTXTFile(trainValidTestSplitFolder + f"/ValidSet_Fold_{v}.txt", 
                                   imageFileNames[numOfTrain:numOfTrain + numOfValidation]);
        saveListOfStringsToTXTFile(trainValidTestSplitFolder + f"/TestSet_Fold_{v}.txt", 
                                   imageFileNames[numOfTrain + numOfValidation:]);

    # Finished processing
    print("Finished processing.");
def trainAndValidateWithSimpleUNet():
    # Initializing
    print("Initializing training and validation...");
    if (len(sys.argv) < 3):
        print("Usage: python script.py <startValidationIndex> <endValidationIndex>");
        return;
    startValidationIndex = int(sys.argv[1]);
    endValidationIndex = int(sys.argv[2]);
    simpleUNETCrossValidationFolder = crossValidationFoldFolder + "/SimpleUNet";

    # Iterate for each fold
    print("Starting cross-validation...");
    numEpochs = 1; validationLossValues = []; numFolds = endValidationIndex + 1;
    for fold in range(startValidationIndex, endValidationIndex + 1):
        print(f"\n======================== Fold {fold+1}/{numFolds} ========================");

        # Read file names for this fold
        print("Reading file names for this fold...");
        trainFiles = readListOfStringsFromTXTFile(f"{trainValidTestSplitFolder}/TrainSet_Fold_{fold}.txt");
        validFiles = readListOfStringsFromTXTFile(f"{trainValidTestSplitFolder}/ValidSet_Fold_{fold}.txt");
        trainDataset = BrainTumorDataset(trainFiles, syntheticImageFolder, syntheticMaskFolder);
        validDataset = BrainTumorDataset(validFiles, syntheticImageFolder, syntheticMaskFolder);
        trainLoader = DataLoader(trainDataset, batch_size=2, shuffle=True);
        validLoader = DataLoader(validDataset, batch_size=2, shuffle=False);

        # Prepare the model
        print("Preparing the model...");
        model = SimpleUnet();

        # Train the model with the number of epoches of 1
        print("Training the model...");
        model.trainModel(trainLoader, inNumEpochs=numEpochs, inDevice='cuda');

        # Validate the model with the number of epoches of 1
        print("Validating the model...");
        lossValues = model.evaluateModel(validLoader, inNumEpochs=numEpochs, inDevice='cuda');
        validationLossValues.append(np.mean(lossValues));
    
        # Save the validation errors
        print("Saving validation errors...");
        saveNumPyArrayToCSVFile(f"{simpleUNETCrossValidationFolder}/ValidationLosses_Fold_{fold}.csv", np.array(lossValues));
    
        # Save the trained model
        print("Saving the trained model...");
        model.saveModel(f"{simpleUNETCrossValidationFolder}/TrainedModel_Fold_{fold}.pth");

    # Finished training and validation
    print("Cross-validation complete: ");
    print("\t Validation Losses for each fold:", validationLossValues);
    print("\t Average Validation Loss:", np.mean(validationLossValues));
def trainAndValidateWithAdvancedUnet():
    # Initializing
    print("Initializing training and validation...");
    if (len(sys.argv) < 3):
        print("Usage: python script.py <startValidationIndex> <endValidationIndex>");
        return;
    startValidationIndex = int(sys.argv[1]);
    endValidationIndex = int(sys.argv[2]);
    advancedUNETCrossValidationFolder = crossValidationFoldFolder + "/AdvancedUNet";

    # Iterate for each fold
    print("Starting cross-validation...");
    numEpochs = 1; validationLossValues = []; numFolds = endValidationIndex + 1;
    for fold in range(startValidationIndex, endValidationIndex + 1):
        print(f"\n======================== Fold {fold+1}/{numFolds} ========================");

        # Read file names for this fold
        print("Reading file names for this fold...");
        trainFiles = readListOfStringsFromTXTFile(f"{trainValidTestSplitFolder}/TrainSet_Fold_{fold}.txt");
        validFiles = readListOfStringsFromTXTFile(f"{trainValidTestSplitFolder}/ValidSet_Fold_{fold}.txt");
        trainDataset = BrainTumorDataset(trainFiles, syntheticImageFolder, syntheticMaskFolder);
        validDataset = BrainTumorDataset(validFiles, syntheticImageFolder, syntheticMaskFolder);
        trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=True);
        validLoader = DataLoader(validDataset, batch_size=1, shuffle=False);

        # Prepare the model
        print("Preparing the model...");
        model = AdvancedUnet();

        # Train the model with the number of epoches of 1
        print("Training the model...");
        model.trainModel(trainLoader, inNumEpochs=numEpochs, inDevice='cuda');

        # Validate the model with the number of epoches of 1
        print("Validating the model...");
        lossValues = model.evaluateModel(validLoader, inNumEpochs=numEpochs, inDevice='cuda');
        validationLossValues.append(np.mean(lossValues));
    
        # Save the validation errors
        print("Saving validation errors...");
        saveNumPyArrayToCSVFile(f"{advancedUNETCrossValidationFolder}/ValidationLosses_Fold_{fold}.csv", np.array(lossValues));

        # Save the trained model
        print("Saving the trained model...");
        model.saveModel(f"{advancedUNETCrossValidationFolder}/TrainedModel_Fold_{fold}.pth");

    # Finished training and validation
    print("Cross-validation complete: ");
    print("\t Validation Losses for each fold:", validationLossValues);
    print("\t Average Validation Loss:", np.mean(validationLossValues));
def printSummaryOfSimpleUnet():
    # Define the model to visualize
    print("Defining the model...");
    model = SimpleUnet();

    # Move model to CPU
    print("Moving model to CPU...");
    model.to('cpu');

    # Get model summary
    print("Getting model summary...");
    summary(model, input_size=(1, 512, 512), device='cpu');

    # Finished processing
    print("Finished processing.");
def printSummaryOfAdvancedUnet():
    # Define the model to visualize
    print("Defining the model...");
    model = AdvancedUnet();

    # Move model to CPU
    print("Moving model to CPU...");
    model.to('cpu');

    # Get model summary
    print("Getting model summary...");
    summary(model, input_size=(1, 512, 512), device='cpu');

    # Finished processing
    print("Finished processing.");
def testWithOptimalNetworkStructure():
    # Initialize
    print("Initializing ...");
    if (len(sys.argv) < 3):
        print("Usage: python script.py <startValidationIndex> <endValidationIndex>");
        return;
    startValidationIndex = int(sys.argv[1]);
    endValidationIndex = int(sys.argv[2]);
    modelCrossValidationFolder = crossValidationFoldFolder + "/SimpleUNet";

    # Iterate for each validation fold
    print("Starting testing ...");
    numOfFolds = endValidationIndex + 1;
    for fold in range(startValidationIndex, endValidationIndex + 1):
        # Debugging
        print(f"**************** Test for Fold {fold + 1}/{numOfFolds} ****************");

        # Prepare data loader for testing
        print("Preparing data loader for testing...");
        testFiles = readListOfStringsFromTXTFile(f"{trainValidTestSplitFolder}/TestSet_Fold_{fold}.txt");
        testDataset = BrainTumorDataset(testFiles, imageFolder, maskFolder);
        testLoader = DataLoader(testDataset, batch_size=2, shuffle=False);

        # Define the model
        print("Defining the model...");
        model = SimpleUnet();
    
        # Load the pre-trained model
        print("Loading the pre-trained model...");
        model.loadModel(f"{modelCrossValidationFolder}/TrainedModel_Fold_{fold}.pth");

        # Evaluate the model
        print("Evaluating the model...");
        evaluatedLossValues = model.evaluateModel(testLoader, inDevice='cuda');
        print("\t Evaluation metrics:", evaluatedLossValues);

    # Finished processing
    print("Finished processing.");
def testSliceSegmentation():
    # Initializing
    print("Initializing ...");
    if (len(sys.argv) < 2):
        print("Usage: python script.py <testIndex>");
        return;
    textIndex = int(sys.argv[1]);
    modelCrossValidationFolder = f"{crossValidationFoldFolder}/AdvancedUNet";

    # Load the pre-trained model
    print("Loading the pre-trained model...");
    optimalFold = 0;
    model = AdvancedUnet();
    model.loadModel(f"{modelCrossValidationFolder}/TrainedModel_Fold_{optimalFold}.pth");

    # Prepare data for segmentation
    print("Preparing data for segmentation...");
    testFiles = readListOfStringsFromTXTFile(f"{trainValidTestSplitFolder}/TestSet_Fold_{0}.txt");
    imageSize = (512, 512);
    imagePath = syntheticImageFolder + f"/{testFiles[textIndex]}";
    testImage = Image.open(imagePath).convert('L').resize(imageSize, Image.BILINEAR);
    image = np.array(testImage, dtype=np.float32) / 255.0;
    imageTensor = torch.from_numpy(image).unsqueeze(0);
    
    # Conduct prediction
    print("Conducting prediction...");
    outputImage = model.predict(imageTensor, inDevice='cpu');

    # Rendering prediction with test image as opacity rendering the mask with the test image
    print("Rendering prediction with the test image...");
    renderMaskOnImage(testImage, outputImage, inAlpha=0.25);

    # Finished processing.
    print("Finished processing.");
def testMultipleSliceSegmentation():
    # Initialize
    print("Initialize ...");
    if (len(sys.argv) < 2):
        print("Usage: python script.py <TrainingFoldIndex>");
        return;
    trainingFoldIndex = int(sys.argv[1]);
    imageSlices = []; currentIndex = 0; 
    imageSegmenter = AdvancedUnet();
    def updateSlice(inValue):
        # Get the current index
        currentIndex = inValue;

        # Get the image
        image = imageSlices[currentIndex];

        # Convert image to tensor for conducting segmentation
        floatImage = np.array(image, dtype=np.float32) / 255.0;
        imageTensor = torch.from_numpy(floatImage).unsqueeze(0);
        outMask = imageSegmenter.predict(imageTensor, inDevice='cuda');

        # Overlay the mask and the image
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        redMask = np.zeros_like(overlay)
        redMask[outMask > 0] = [0, 0, 255]  # Red color (BGR)
        alpha = 0.25  # Adjust alpha for transparency
        blended = cv2.addWeighted(overlay, 1 - alpha, redMask, alpha, 0)

        # Visualize image
        cv2.imshow("Slice Segmentation", blended);

    # Reading all slices on the testing set
    print("Reading all slices on the testing set...");
    ## Reading all file names
    testFiles = readListOfStringsFromTXTFile(f"{trainValidTestSplitFolder}/TestSet_Fold_{trainingFoldIndex}.txt");
    print("\t The number of testing slices is:", len(testFiles));
    ## Reading all slices and stored as images using opencv
    for testFile in testFiles:
        imageFilePath = syntheticImageFolder + f"/{testFile}";
        image = cv2.imread(imageFilePath, cv2.IMREAD_GRAYSCALE);
        image = cv2.resize(image, (512, 512));
        imageSlices.append(image);
    
    # Loading image segmenter
    print("Loading image segmenter...");
    modelCrossValidationFolder = crossValidationFoldFolder + "/AdvancedUNet";
    imageSegmenter.loadModel(f"{modelCrossValidationFolder}/TrainedModel_Fold_{trainingFoldIndex}.pth");

    # Initialize visualization
    print("Initializing visualization...");
    ## Setting up new windows
    cv2.namedWindow("Slice Segmentation", cv2.WINDOW_AUTOSIZE);
    ## Adding slider for controlling slices
    cv2.createTrackbar("Slice", "Slice Segmentation", 0, len(imageSlices) - 1, updateSlice);
    ## Set the current slice
    currentIndex = 0;
    updateSlice(currentIndex);
    ## Wait key for exiting
    cv2.waitKey(0);
    cv2.destroyAllWindows();

    # Finished processing
    print("Finished processing.");

#**********************************************************************************#
#****************************** MAIN PROCEDURE ************************************#
#**********************************************************************************#
def main():
    os.system("cls");
    testMultipleSliceSegmentation();
if __name__ == "__main__":
    main();

