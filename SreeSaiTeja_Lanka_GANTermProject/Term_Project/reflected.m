function flipped_image = reflected(images)
flipped_image = {};
for i = 1:length(images)
    %flip/reflect
    I = flipdim(images{i} ,2);           %# horizontal flip
    I1 = flipdim(images{i} ,1);           %# vertical flip
    flip_image = flipdim(I1,2);
    flipped_image{i, 1} = imresize(flip_image, [128, 128]);
end
end
