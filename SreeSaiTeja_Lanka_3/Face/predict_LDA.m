function [I] = predict_LDA(Weight,X_mat)
predict =(Weight'*X_mat')';
[M,temp] = max(predict');
temp = categorical(temp');
end
