function topfeatures = rankingfeat(TrainMat, LabelTrain)
%% input: TrainMat - a NxM matrix that contains the full list of features
%% of training data. N is the number of training samples and M is the
%% dimension of the feature. So each row of this matrix is the face
%% features of a single person.
%%        LabelTrain - a Nx1 vector of the class labels of training data

%% output: topfeatures - a Kx2 matrix that contains the information of the
%% top 1% features of the highest variance ratio. K is the number of
%% selected feature (K = ceil(M*0.01)). The first column of this matrix is
%% the index of the selected features in the original feature list. So the
%% range of topfeatures(:,1) is between 1 and M. The second column of this
%% matrix is the variance ratio of the selected features.

labels = double(LabelTrain);
classes_count = length(unique(labels));
N = length(labels);
M=size(TrainMat,2);
K = ceil(M*0.01);
topfeatures=zeros(K,2);

class_mean = zeros(classes_count,M);
variance_inclass=zeros(classes_count,M);
% Compute class means, and within class variance for each class
i=1;
for class_labels = unique(labels')
    Xc=TrainMat(labels==class_labels,:);
    class_mean(i,:) = mean(Xc,1);
    variance_inclass(i,:)=var(Xc,1);
    i=i+1;
end
variance_crossclass=zeros(1,M);
difference_in_mean=zeros(classes_count ,M);
fraction=zeros(classes_count,M);
fraction_mean=zeros(1,M);
for j=1:M
    variance_crossclass(j)=var(class_mean(:,j));
    for k=1:classes_count
        mean_now = sort(abs(class_mean(k,j)- class_mean(:,j)));
        difference_in_mean(k,j)= mean_now(2);
        fraction(k,j)=variance_inclass(k,j)/difference_in_mean(k,j);
    end
    
    fraction_mean(j)=mean(fraction(:,j));
   
end

AVR=variance_crossclass./fraction_mean;      %Augmented Variance Ratio
[AVR_sort,index]=sort(AVR,'descend');
 % Selecting the top vectors
topfeatures(:,1)=index(1:K);
topfeatures(:,2)=AVR_sort(1:K);
end