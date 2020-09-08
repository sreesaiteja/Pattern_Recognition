function resize_translation = translated(images)
% translation
resize_translation = {};
for i = 1:length(images)
    n = 30;
    x = randsample(n,1);
    y = randsample(n,1);
    translated_image   = imtranslate(images{i},[x, y]);
    croped_translation = imcrop(translated_image, [x y 256-x 256-y]);
    resize_translation{i,1} = imresize(croped_translation, [128, 128]);
end