function scaled_image  = scaled(images)
    %scaling
sclaed_image = {};
for i = 1:length(images)
    s = 50:100;
    rand = randsample(s,1);
    scale_image  = imresize(images{i},rand * 0.01);
    scaled_image{i, 1} = imresize(scale_image, [128, 128]);
end
end