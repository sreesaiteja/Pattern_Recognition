function[w] =leastSquaresError(X,train,classes_count)
    n= size(X,1);
    %append 1's to X
    X_new = [ones(n,1) X];
    
    target = zeros(n, classes_count);
    for i = 1:n;
        target(i, train(i)) = 1;
    end     
    %find optimal w
    w = (X_new' * X_new)\(X_new'*target); 
    disp(w)
end 