% Visualizing Boundaries: this is using matlab's internal code
%         you will have to confirm it with your data structures to use
%     This is visualizing the linear function defining the class boundaries
%          boundaries for each class comparison (eq 4.9)
%    MDlLinear is a matrix of discriminant functions
%    featureVector is the features to show
%    labels are the feature vector labels as a categorical array
%    featureA and FeatureB are the index of the features to visualize
function visualizeBoundaries(MdlLinear,featureVector,labels,featureA,featureB)

clf
category_names = categories(labels);
numGroups = length(category_names);
colors = jet(numGroups*10);
colors = colors(round(linspace(1,numGroups*10,numGroups)),:);
lim_info =  [min(featureVector(:,featureA)),max(featureVector(:,featureA)),...
    min(featureVector(:,featureB)),max(featureVector(:,featureB))];

h1 = gscatter(featureVector(:,featureA),featureVector(:,featureB),labels,'','+o*v^');
for i = 1:numGroups
    h1(i).LineWidth = 2;
    h1(i).MarkerEdgeColor = min(colors(i,:)*1.2,1);
end
hold on
x =  [min(featureVector(:,featureA)),max(featureVector(:,featureA)) ];
h2 = [];
for i = 1:numGroups
    for j = i+1:numGroups
        if i~=j
            % Get the linear discriminant function 
            % for the equation  0 = w_1 * x_1 * w_2 * x_2 * b
            %  You will need to motify this for how you define the
            %  descriminant functions
            w_0 = MdlLinear(1, i) - MdlLinear(1, j);
            w = [MdlLinear(featureA+1, i) - MdlLinear(featureA+1, j), MdlLinear(featureB+1, i) - MdlLinear(featureB+1, j)];
                
            h2 = cat(1,h2,plot(x,(-w_0 - w(1)*x) / w(2),'LineWidth',2,...
                'DisplayName',sprintf('Class Sep b/w %s,%s',category_names{i},category_names{j})));
        end
    end
end


axis(lim_info);
hold off
grid on;
set(gca,'FontWeight','bold','LineWidth',2)