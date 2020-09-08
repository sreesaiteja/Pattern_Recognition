function [Weight,X_mat] = train_LDA(featureVector,labels,numGroups)

%Least Square fit
labels=double(labels);
count = length(labels);

%create t from labels
t =[];
for i=1:count
    a =zeros(1,num_of_groups);
    a(labels(i)+1)=a(labels(i)+1)+1;
    t=[t; a];
end

%solving for weight matrix
X_mat = [ones(count,1) featureVector];
num = X_mat'*X_mat;
denom = X_mat'*t;

Weight = num\denom;
end

