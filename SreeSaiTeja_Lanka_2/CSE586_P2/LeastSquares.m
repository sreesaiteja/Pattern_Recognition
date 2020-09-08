%starter code for project 2: linear classification
%pattern recognition, CSE583/EE552
%Weina Ge, Aug 2008
%Christopher Funk, Jan 2017


close all;
clear all;
addpath export_fig
% Choose which dataset to use (choices wine, wallpaper, taiji) :
dataset = 'wallpaper';
[train_featureVector, train_labels, test_featureVector, test_labels] = loadDataset(dataset);

%% An example Linear Discriminant Classification


%  Classification here is based on 2 Features (featureA and feature B).  
%       You will be using all of the features but using 2 features makes it 
%       easier to visualize than the multidimensional hyperplane

featureA = 1;
featureB = 7;
feature_idx = [featureA,featureB];
numGroups = length(countcats(test_labels));
%Uncomment the following line to use all features
feature_idx = 1:size(train_featureVector,2);


train_featureVector = train_featureVector(:,feature_idx);
test_featureVector = test_featureVector(:,feature_idx);

%%  Classify the data and show statistics
%    This example is using Matlab's inbuilt Classifier.
%        You do not need to follow their data management style, this is
%        to allow you to see the result and show some visualizations
%        of the classification
% Train the model (you will have to write this function)
 %MdlLinear = fitcdiscr(train_featureVector,train_labels);
labels_inorder = unique(train_labels);
classes_count = length(labels_inorder);
pre_train_labels = cat_to_num(train_labels);

MdlLinear = leastSquaresError(train_featureVector, pre_train_labels, classes_count);
size(MdlLinear)
% Find the training accurracy (you will have to write testing 
%      function (the function for finding the class labels from a set of
%      features)
train_predict = predict(MdlLinear, train_featureVector);
post_train_labels = categorical(num_to_cat(train_predict,  labels_inorder));
% Create confusion matrix
train_ConfMat = confusionmat(train_labels,post_train_labels)
% Create classification matrix (rows should sum to 1)
train_ClassMat = train_ConfMat./(meshgrid(countcats(train_labels))')
% mean group accuracy and std
train_acc = mean(diag(train_ClassMat))
train_std = std(diag(train_ClassMat))

% Find the testing accurracy (you will have to write testing 
%      function (the function for finding the class labels from a set of
%      features)
test_predict = predict(MdlLinear,test_featureVector);
post_test_labels = categorical(num_to_cat(test_predict,  labels_inorder));
% Create confusion matrix
test_ConfMat = confusionmat(test_labels,post_test_labels)
% Create classification matrix (rows should sum to 1)
test_ClassMat = test_ConfMat./(meshgrid(countcats(test_labels))')
% mean group accuracy and std
test_acc = mean(diag(test_ClassMat))
test_std = std(diag(test_ClassMat))




%%  Display the linear discriminants and a set of features in two of the feature dimensions
%      You will need to modify this function for to use your LDA
%      classification boundries to work with your code.. Look at the code
%      for more details
figure(1)
visualizeBoundaries(MdlLinear,test_featureVector,test_labels,1,2)
title('{\bf Linear Discriminant Classification}')
export_fig linear_discriminant_example -png -transparent
%%  Display the classified regions of two of the feature dimensions  
%      You will need to modify this function for with your testing 
%      function (the function for finding the class labels from a set of
%      features).

figure(2)
h = visualizeBoundariesFill(MdlLinear,test_featureVector,test_labels,labels_inorder, 1,2);
title('{\bf Classification Area}')
export_fig classification_fill_example -png -transparent
