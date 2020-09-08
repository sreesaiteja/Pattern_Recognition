function image_resize = resized(images)
image_resize = {};
for i = 1:length(images)
    image_resize{i, 1} = imresize(images{i}, [128, 128]);
end
end