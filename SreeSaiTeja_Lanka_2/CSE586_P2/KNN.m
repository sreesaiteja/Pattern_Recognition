function[knnarray] = KNN(w, X,Y,k, l)
%dimensionality reduction
new_X = X * w;
new_Y = Y * w;
%to initialize the distance parameters
[row1,col1] = size(new_X);
[row2,col2]= size(new_Y);

distance = zeros(row1,row2);
%knnarray = zeros(row1,row2);
for i=1:row2
for j=1:row1
      %formula for L2 norm
      distance(i,j)=sqrt(sum((new_X(j,:)-new_Y(i,:)).^2));
end
%sort and collect k values
[values,index]=sort(distance(i,:));
nearest_indx = index(1, 1:k);
temp(i) = mode(l(nearest_indx));
end
knnarray = temp';
end
