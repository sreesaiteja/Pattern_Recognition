%% Using GoogleNet to train the network network
load('alexnet.mat')

dataDir= './data/wallpapers/';
checkpointDir = 'modelCheckpoints';

rng(1) % For reproducibility
Symmetry_Groups = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',...
    'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};

%train_folder = 'train';
%test_folder  = 'test';
% uncomment after you create the augmentation dataset
train_folder = 'train_aug';
test_folder  = 'test_aug';
fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames', 'ReadFcn', @inputImage);
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
% Split with validation set
[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames', 'ReadFcn',@inputImage);
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

%%
rng('default');
numEpochs = 20; % 5 for both learning rates
batchSize = 400;
nTraining = length(train.Labels);

% Define the Network Structure, To add more layers, copy and paste the
% lines such as the example at the bottom of the code
% Using Alexnet, appending our input layer and output layers. 
transferLayers = net.Layers(1:end-3);
layers = [
    transferLayers;
    fullyConnectedLayer(17,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer();
    classificationLayer();
    ];


if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end
% Set the training options
options = trainingOptions('sgdm','MaxEpochs',20,... 
    'InitialLearnRate',5e-4,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs);
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
    %'OutputFcn',@plotTrainingAccuracy,... 

% Train the network, info contains information about the training accuracy
% and loss
 t = tic;
[net1,info1] = trainNetwork(train,layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

% Test on the validation data
YTrainPred1        = classify(net1, train);
train_acc1         = mean(YTrainPred1 == train.Labels);
[trainOutputs1, ~] = grp2idx(YTrainPred1);
[trainTargets1, ~] = grp2idx(train.Labels);

figure(1)
plotConfusionMatrix(trainTargets1, trainOutputs1, 'Training');
savefig('TrainingConfusionMatrix_AlexNet1.fig')

YValPred1        = classify(net1, val);
val_acc1         = mean(YValPred1 == val.Labels);
[valOutputs1, ~] = grp2idx(YValPred1);
[valTargets1, ~] = grp2idx(val.Labels);

figure(2)
plotConfusionMatrix(valTargets1, valOutputs1, 'Validation');
savefig('validationConfusionMatrix_AlexNet1.fig')

% It seems like it isn't converging after looking at the graph but lets
%   try dropping the learning rate to show you how.  

options = trainingOptions('sgdm','MaxEpochs',20,...
    'InitialLearnRate',1e-4,... % learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs);
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
%     'OutputFcn',@plotTrainingAccuracy,...

 t = tic;
[net2,info2] = trainNetwork(train,net1.Layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));


% Test on the validation data
YTrainPred2        = classify(net2, train);
train_acc2         = mean(YTrainPred2 == train.Labels);
[trainOutputs2, ~] = grp2idx(YTrainPred2);
[trainTargets2, ~] = grp2idx(train.Labels);

figure(3)
plotConfusionMatrix(trainTargets2, trainOutputs2, 'Training');
savefig('TrainingConfusionMatrix_AlexNet2.fig')

YValPred2        = classify(net2, val);
val_acc2         = mean(YValPred2 == val.Labels);
[valOutputs2, ~] = grp2idx(YValPred2);
[valTargets2, ~] = grp2idx(val.Labels);

figure(4)
plotConfusionMatrix(valTargets2, valOutputs2, 'Validation');
savefig('validationConfusionMatrix_AlexNet2.fig')


% Test on the Testing data
YTestPred        = classify(net2, test);
test_acc         = mean(YTestPred == test.Labels);
[testOutputs, ~] = grp2idx(YTestPred);
[testTargets, ~] = grp2idx(test.Labels);

figure(5)
plotConfusionMatrix(testTargets, testOutputs, 'Test');
savefig('testConfusionMatrix_AlexNet2.fig')

figure(6)
plotTrainingAccuracy_All(info1,numEpochs);
savefig('higherLearningRate.fig')

figure(7)
plotTrainingAccuracy_All(info2,numEpochs);
savefig('lesserLearningRate.fig')

save('AlexNet.mat');

function img = inputImage(imgPath)
    img = imread(imgPath);
    img= imresize(img(:, :, [1 1 1]), [227 227]);
end
    
