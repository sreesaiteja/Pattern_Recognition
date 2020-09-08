function[predicted_label] = predict(weights_trained, feature_vector)
    length_fv = size(feature_vector, 1);
    %append 1's to X
    X = [ones(length_fv , 1) feature_vector];
    y = X*weights_trained;
    predicted_label = zeros(length_fv , 1);
    %predict
    for i = 1 : length_fv
        [a,P] = max(y(i,:));
        predicted_label(i) = P;
    end
end
        
    


