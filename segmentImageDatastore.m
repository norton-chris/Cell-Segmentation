function [outds, outDir] = segmentImageDatastore(imds, net, patchSize)
%segmentImageDatastore performs patchwise semantic segmentation on the
%image datastore using segmentImage using the provided network.
%
%  [outds, outDir] = segmentImageDatastore(imds, net, patchSize) stores a
%  semantically segmented images, segmented using the network NET. The
%  segmentation is performed patches-wise on patches of size PATCHSIZE.
%  Segmentation output will be stored in a separate folder, registered as
%  imageDatastore, outds.
%
% December 2019 Sangyoon Han

% Output directory
outDir = [fileparts(fileparts(imds.Files{1})) filesep 'SegOutput'];
if ~exist(outDir,'dir')
    mkdir(outDir)
else
    k=0;
    while exist(outDir,'dir')
        k=k+1;
        folName = ['SegOutput' num2str(k)];
        outDir = [fileparts(fileparts(imds.Files{1})) filesep folName];
    end
    mkdir(outDir)
end

numImages = numel(imds.Files);

for ii=1:numImages
    curImg = imds.readimage(ii); 
    out = segmentImageOneChan(curImg, net, patchSize);
    % Save output image to output directory
    [~,curName] = fileparts(imds.Files{ii});
    imwrite(out,[outDir filesep curName '.tiff'],'tif')
end

outds = imageDatastore(outDir);

end
