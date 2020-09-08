% Visualizing Boundaries: this is using matlab's internal code
%         you will have to conform it your data structures to use (you
%         don't have to have the same class structure)  You will have to 
%         change the predict function for your own code. 
%     This is visualizing the linear function defining the class boundaries
%          boundaries for each class comparison
%    MDlLinear is a matrix of discriminant functions
%    featureVector is the features to show
%    labels are the feature vector labels as a categorical array
%    featureA and FeatureB are the index of the features to visualize
function h = visualizeBoundariesFill(MdlLinear,featureVector,labels, labels_order,featureA,featureB)
%%
clf
nPoints = 1000;

cats = categories(labels);
numGroups = length(cats);
lim_info =  [min(featureVector(:,featureA)),max(featureVector(:,featureA)),...
    min(featureVector(:,featureB)),max(featureVector(:,featureB))  ];

x = linspace(lim_info(1),lim_info(2),nPoints);
y = linspace(lim_info(3),lim_info(4),nPoints);
[X,Y] = meshgrid(x,y);
xx=X(:);
yy=Y(:);
P = -inf*ones(nPoints,nPoints);
G = ones(nPoints,nPoints);
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);

featureVector_for_prediction = zeros(length(xx),size(featureVector,2));
featureVector_for_prediction(:,featureA) = xx;
featureVector_for_prediction(:,featureB) = yy;

% Add your testing function here (the function for finding the 
%      class labels from a set of features) and predict the group on 
%      featureVector_for_prediction for this to work
pred1 = predict(MdlLinear,featureVector_for_prediction);
pred = categorical(num_to_cat(pred1, labels_order));
G = reshape(pred,nPoints,nPoints);


hold on;
for i = 1:numGroups
    cc = bwconncomp(G==cats(i));
    for rp = regionprops(cc,'ConvexHull')'
        xx = x(min(round(rp.ConvexHull(:,1)),nPoints));
        yy = y(min(round(rp.ConvexHull(:,2)),nPoints));
        patch(xx,yy,colors(i,:),'FaceAlpha',.2)
    end
    
end

h = gscatter(featureVector(:,featureA),featureVector(:,featureB),labels,'','+o*');
for i = 1:numGroups
    h(i).LineWidth = 1;
    h(i).MarkerEdgeColor = colors(i,:);
    h(i).MarkerFaceColor = colors(i,:);
end
axis(lim_info)
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)