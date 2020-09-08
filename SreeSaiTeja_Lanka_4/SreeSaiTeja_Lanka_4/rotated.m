function rotated_image = rotated(images)
rotated_image = {};
for i = 1:length(images)
        %rotation of image
    r = 360;
    rotate = randsample(r, 1);
    image = images{i};
    rotate_image = imrotate(images{i},rotate,'bilinear');
    [m,n] = size(rotate_image);
    a =64; %max half length of the center image
    %center of image = [m/2 n/2]
    x = m/2;
    y = n/2;
    croped_image = imcrop(rotate_image, [x-a y-a 128-1 128-1]);
    rotated_image{i, 1} = imresize(croped_image, [128, 128]);
    %b = size(find(rotated_image ==0));
end
end