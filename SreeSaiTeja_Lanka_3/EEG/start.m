clear all;
close all;
clc;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% load the EEG data  - DO NOT SUBMIT THIS PROJECT WITH EEG DATA!
%%%
%%% Data contain EEG features extracted from 25 different people over multiple sessions 
%%% EEG data is from the Wallpaper Groups P2, PMG, P4M.  You can find out
%%% more about how the data was collected and Wallpaper Groups in the paper
%%% below:  
%%%
%%% Representation of Maximally Regular Textures in Human Visual Cortex
%%% Peter Kohler, Alasdair Clarke, Alexandra Yakovleva, Yanxi Liu and Anthony Norcia
%%% The Journal of Neuroscience: The Official Journal of the Society of Neuroscience
%%% Volume 36(3)
%%% Pages 714-729
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('eeg_data.mat')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%the flow of your code should look like this
Dim = size(eeg_data,2)-1; %dimension of the feature
countfeat(Dim,2) = 0;
%%countfeat is a Mx2 matrix that keeps track of how many times a feature has been selected, where M is the dimension of the original feature space.
%%The first column of this matrix records how many times a feature has ranked within top 1% during 100 times of feature ranking.
%%The second column of this matrix records how many times a feature was selected by forward feature selection during 100 times.

%%%%%%%%%%%%%%%%%%%% test code %%%%%
%comment this out 
%tmp = randperm(Dim);
%topfeatures(:,1) = tmp(1:1000)';
%topfeatures(:,2) = 100*rand(1000,1);
%forwardselected = tmp(1:100)';
%%%%%%%%%%%%%%%%%%%%%%%%************

classes_count = length(unique(labels));
train_ConfMat_avg = zeros(classes_count,classes_count);
train_ClassMat_avg = zeros(classes_count,classes_count);
test_ConfMat_avg=zeros(classes_count,classes_count);
test_ClassMat_avg=zeros(classes_count,classes_count);
train_acc_avg = 0;
train_std_avg = 0;
test_acc_avg = 0;
test_std_avg = 0;

for i=1:100
    
    % randomly divide into train and test sets with 80%/20% split
    [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti([labels,eeg_data]);

    %start feature ranking
    topfeatures = rankingfeat(TrainMat, LabelTrain); 
    countfeat(topfeatures(:,1),1) =  countfeat(topfeatures(:,1),1) +1;
    
    %% visualize the variance ratio of the top 1% features
    if i==1
        %% colorbar indicates the correspondance between the variance ratio
        %% of the selected feature
       plotFeat(topfeatures,feature_names,20);
    end

    % start forward feature selection
    forwardselected = forwardselection(TrainMat, LabelTrain, topfeatures);
    countfeat(forwardselected,2) =  countfeat(forwardselected,2) +1;    
        
    train_featureVector=TrainMat(:,forwardselected');
    test_featureVector=TestMat(:,forwardselected');
    labels_train=categorical(LabelTrain);    
    labels_test=categorical(LabelTest);
    % start classification
    MdlLinear = fitcdiscr(train_featureVector,labels_train);

% Find the training accurracy (you will have to write testing 
%      function (the function for finding the class labels from a set of
%      features)
train_pred = predict(MdlLinear,train_featureVector);
% Create confusion matrix
train_ConfMat = confusionmat(labels_train,train_pred);
train_ConfMat_avg=train_ConfMat_avg+train_ConfMat;
% Create classification matrix (rows should sum to 1)
train_ClassMat = (train_ConfMat./(meshgrid(countcats(labels_train))'));
train_ClassMat_avg=train_ClassMat_avg+train_ClassMat;
% mean group accuracy and std
train_acc = mean(diag(train_ClassMat));
train_acc_avg=train_acc_avg+train_acc;
train_std = std(diag(train_ClassMat));
train_acc_avg=train_acc_avg+train_acc;

% Find the testing accurracy (you will have to write testing 
%      function (the function for finding the class labels from a set of
%      features)
test_pred = predict(MdlLinear,test_featureVector);
% Create confusion matrix
test_ConfMat = confusionmat(labels_test,test_pred);
test_ConfMat_avg = test_ConfMat_avg+test_ConfMat;
% Create classification matrix (rows should sum to 1)
test_ClassMat = (test_ConfMat./(meshgrid(countcats(labels_test))'));
test_ClassMat_avg=test_ClassMat_avg+test_ClassMat;
% mean group accuracy and std
test_acc = mean(diag(test_ClassMat));
test_acc_avg=test_acc_avg+test_acc;
test_std = std(diag(test_ClassMat));
test_std_avg = test_std_avg+test_std;
end
train_ConfMat_avg = 0.01* train_ConfMat_avg
train_ClassMat_avg = 0.01* train_ClassMat_avg
test_ConfMat_avg = 0.01* test_ConfMat_avg
test_ClassMat_avg = 0.01* test_ClassMat_avg
train_acc_avg = 0.01*  train_acc_avg
train_std_avg = 0.01* train_std_avg
test_acc_avg = 0.01* test_acc_avg
test_std_avg = 0.01* test_std_avg
%% visualize the features that have ranked within top 20 (or however many you can display) most during 100 times of feature ranking
data(:,1)=[1:Dim]';
data(:,2) = countfeat(:,1);
%% colorbar indicates the number of times a feature at that location was
%% ranked within top 1%
plotFeat(data,feature_names,20);
%% visualize the features that have been selected most during 100 times of
%% forward selection
data(:,2) = countfeat(:,2);
%% colorbar indicates the number of times a feature at that location was
%% selected by forward selection
plotFeat(data,feature_names,20);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

