function [outlabels] = cat_to_num(inlabels)
    c = (inlabels);
    %use group to index to change to numericals back
    outlabels = grp2idx(c);
    %keys=unique(c, 'stable')
    %values=unique(outlabels, 'stable')
    %map = containers.Map(values, keys);
end