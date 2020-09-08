function forwselc_features = forwardselection(TrainMat, LabelTrain, topfeatures)
LabelTrain  = categorical(LabelTrain);
C = length(unique(LabelTrain));
N=length(LabelTrain);
M = size(TrainMat,2);
K = size(topfeatures,1);
%% input: TrainMat - a NxM matrix that contains the full list of features
%% of training data. N is the number of training samples and M is the
%% dimension of the feature. So each row of this matrix is the face
%% features of a single person.
%%        LabelTrain - a Nx1 vector of the class labels of training data
%%        topfeatures - a Kx2 matrix that contains the information of the
%% top 1% features of the highest variance ratio. K is the number of
%% selected feature (K = ceil(M*0.01)). The first column of this matrix is
%% the index of the selected features in the original feature list. So the
%% range of topfeatures(:,1) is between 1 and M. The second column of this
%% matrix is the variance ratio of the selected features.

%% output: forwselc_features - a Px1 vector that contains the index of the 
%% selected features in the original feature list, where P is the number of
%% selected features. The range of forwselc_features is between 1 and M. 





forwselc_features=[];
not_selected=topfeatures(:,1)';
previous_top = 0;
while length(not_selected)>0
    result = zeros(1,length(not_selected));
    for v=1:length(not_selected)
        now = [forwselc_features not_selected(v)];
        featureVec_train=TrainMat(:,now);
       
        MdlLinear = fitcdiscr(featureVec_train,LabelTrain);
        train_predict = predict(MdlLinear,featureVec_train);
        %  confusion matrix
        train_ConfMat = confusionmat(LabelTrain,train_predict);
        %  classification matrix 
        train_ClassMat = train_ConfMat./(meshgrid(countcats(LabelTrain))');
        % mean group accuracy is the optimization criterion
        train_acc = mean(diag(train_ClassMat));
        result(v)=train_acc;
    end
    %result
    [top,top_index] = max(result);
    %conditon to stop
    if top<=previous_top 
        break
    end
    forwselc_features = [forwselc_features top_index];
    not_selected(top_index)=[];
    previous_top = top;
end
forwselc_features = forwselc_features';
end


