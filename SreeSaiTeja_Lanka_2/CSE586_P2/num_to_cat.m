function [outlabels]  = num_to_cat(inlabels, labels_in_order)
    n = size(inlabels);
    %change to numericals
    for i = 1:n
        outlabels(i) = labels_in_order(inlabels(i));
    end
    
end