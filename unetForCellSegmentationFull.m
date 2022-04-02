%% Data setup
% Goal: Create a u-net, train it with cell adhesion images separately with two 
% different labels of cell boundaries, and compare the prediction accuracy of 
% the two nets. Download the images of cell adhesions. Also download the two types 
% of the labels: a mask where the cell inside is not fully segmented; and another 
% mask where the interior part of a cell is filled. We want to ultimately obtain 
% cell-segmentation that has filled the hole from a thresholded image. Matlab 
% uses an image-management system called image datastore. Use this function to 
% create an image-access handle for images and labels. The data are located in 
% /pylon5/ /ac5pifp/sjhan. (?ac5pifp? is our project account name). 

dataSetDir = 'D:/Han Project/TrainingDataset';
imageDir = fullfile(dataSetDir,'Images');
labelDir = fullfile(dataSetDir,'Labels');

imds  = imageDatastore(imageDir);
%% Labels
% It is necessary to convert the ?numbered? label into categorical label:

classNames=["cell", "background"];
labelIDs = [1 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
% It is a good habit to ?see? your images
sampleImg = readimage(imds, 5);
sampleLabel = readimage(pxds,5);

imshowpair(sampleImg, uint8(sampleLabel), 'montage')
%% parameteers for NN building

filterSize = 3;
numFilters = 32;
numClasses = numel(classNames);
%% Create a randomPatchExtractionDatastore from the image datastore and the
% Pixel label datastore. Each mini-batch contains 16 patches of size 256-by-256 
% pixels. One thousand mini-batches are extracted at each iteration of the epoch. 
% The random patch extraction datastore dsTrain provides mini-batches of data 
% to the network at each iteration of the epoch. Preview the datastore to explore 
% the data. Otherwise we can use trainingData = pixelLabelImageDatastore(imds,pxds);

dsTrain = randomPatchExtractionDatastore(imds,pxds,[256,256],'PatchesPerImage',8);
inputTileSize = [256 256 1];

% To view one of the patches:
inputBatch = preview(dsTrain);
disp(inputBatch)
%% NN building
% Now, let?s create a semantic segmentation network. Students are required to 
% compare results from two different nets with a different architecture, i.e., 
% they have different layers, and different parameters. Let?s start with a simple 
% semantic segmentation network based on a downsampling and upsampling design 

layers = [
    imageInputLayer(inputTileSize)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer()];
% Set up training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',20, ...
    'MiniBatchSize',64, ...
    'L2Regularization',0.0001,...
    'LearnRateSchedule','piecewise',...
    'Shuffle','every-epoch',...
    'GradientThresholdMethod','l2norm',...
    'Plots','training-progress',...
    'VerboseFrequency',20,...
    'Momentum', 0.9);
%% Training
% Let?s train the network. It can take about 2 hours in a single P100 GPU.

[net, info] = trainNetwork(dsTrain,layers,opts);
save('net_trained.mat','net','info')
%% Prediction with test images
% Segment the test image and display the results.

imageFolderTest = fullfile(dataSetDir,'test/Images');
imdsTest = imageDatastore(imageFolderTest);
% You can test your network using multiple test data. Create an
% imageDatastore for the images. Create a pixelLabelDatastore for the
% ground truth pixel labels.  
labelFolderTest = fullfile(dataSetDir,'test/Labels');
pxdsTest = pixelLabelDatastore(labelFolderTest,classNames,labelIDs);

% Make predictions using the test data and trained network.
patchSize = size(imdsTest.readimage(1))/2;
[labelPred,location] = segmentImageDatastore(imdsTest,net,patchSize);
pxdsPred = pixelLabelDatastore(location,classNames,labelIDs);

save('predictedImages.mat','pxdsPred')
disp(['The prediction is done. The predicted data are stored in ' location '.'])

% Evaluate the prediction accuracy using evaluateSemanticSegmentation.

metrics = evaluateSemanticSegmentation(pxdsPred,pxdsTest);
save('metric_noAugmentation.mat','metrics')

% Filling holes and median filtering
labelFolderFilled = fullfile(dataSetDir,'SegOutputFilled');
if ~exist(labelFolderFilled,'dir')
    mkdir(labelFolderFilled)
end
for ii=1:numel(labelPred.Files)
    currMask = labelPred.readimage(ii)=='cell';
    currMask2 = imfill(currMask,'holes');
    currMask3 = medfilt2(currMask2);
    [~,curName] = fileparts(labelPred.Files{ii});    
    imwrite(currMask3,[labelFolderFilled filesep curName '.tiff'],'tif')
end
pxdsPredFilled = pixelLabelDatastore(labelFolderFilled,classNames,labelIDs);
% Evaluate again
metricsFilled = evaluateSemanticSegmentation(pxdsPredFilled,pxdsTest);
save('metric_noAugmentation_filled.mat','metricsFilled')

%% Augment Data While Training
% Create an imageDataAugmenter object to randomly rotate and mirror image data

augmenter = imageDataAugmenter('RandRotation',[-10 10],'RandXReflection',true)

% Create a pixelLabelImageDatastore object to train the network with augmented data.
dsTrainAug = randomPatchExtractionDatastore(imds,pxds,[256,256],'PatchesPerImage',8,'DataAugmentation',augmenter);

% Train the network with the same layers and the same option
[netAug, infoAug] = trainNetwork(dsTrainAug,layers,opts);
save('net_trained_with_augmentedData.mat','netAug','infoAug')

% Make predictions using the test data and the new trained network.
patchSize = size(imdsTest.readimage(1))/2;
[labelPredAug,locationAug] = segmentImageDatastore(imdsTest,netAug,patchSize);
pxdsPredAug = pixelLabelDatastore(locationAug,classNames,labelIDs);
save('predictedImagesFromAugmentedData.mat','pxdsPredAug')
disp(['The prediction is done. The predicted data are stored in ' locationAug '.'])

% Evaluate the prediction accuracy using evaluateSemanticSegmentation.

metricsAug = evaluateSemanticSegmentation(pxdsPredAug.readimage(1),pxdsTest.readimage(1));
save('metric_withAugmentation.mat','metricsAug')

% Filling holes and median filtering
labelFolderFilledAug = fullfile(dataSetDir,'SegOutputFilledFromAug');
if ~exist(labelFolderFilledAug,'dir')
    mkdir(labelFolderFilledAug)
end
for ii=1:numel(labelPredAug.Files)
    currMask = labelPredAug.readimage(ii)=='cell';
    currMask2 = imfill(currMask,'holes');
    currMask3 = medfilt2(currMask2);
    [~,curName] = fileparts(labelPredAug.Files{ii});    
    imwrite(currMask3,[labelFolderFilledAug filesep curName '.tiff'],'tif')
end
pxdsPredFilledAug = pixelLabelDatastore(labelFolderFilledAug,classNames,labelIDs);
% Evaluate again
metricsFilledAug = evaluateSemanticSegmentation(pxdsPredFilledAug,pxdsTest);
save('metric_noAugmentation_filled_aug.mat','metricsFilledAug')
%% 
% How is the accuracy differentt from one by the network without data augmentation?
% 
% After running this in Bridges, copy 'SegOutput' folders and other newly saved 
% files (e.g. net_trained.mat, net_trained_with_augmentatedData.mat, metric_noAugmentation.mat, 
% metric_withAugmentation.mat) to your home drive in Bridges then send them again 
% using Globus to your local hard drive. Don't forget to your 'myjob.XXXXXXX.out' 
% file to your hard drive too.
% 
% Then show the network accuracy and output image from your network.