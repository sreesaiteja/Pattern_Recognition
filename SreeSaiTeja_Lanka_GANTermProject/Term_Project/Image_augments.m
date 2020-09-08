%******************augemntation part***********************
%***********************************************************

groups_list = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM','P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};
train_aug_path = 'data/wallpapers/train_aug';
test_aug_path = 'data/wallpapers/test_aug';
train_path = 'data/wallpapers/train';
test_path = 'data/wallpapers/test';

%*****creating the train augemntation************************************
for i = 1:length(groups_list)
    group = groups_list{i};
    concat_train_path = fullfile(train_path,group);
    concat_trainaug_path = fullfile(train_aug_path,group);
    mkdir(concat_trainaug_path);
    train_aug_all = imageDatastore((concat_train_path),'IncludeSubfolders',true,'LabelSource','foldernames');
    images = readall(train_aug_all);
    image_resize_result = resized(images);
    translate_result = translated(images);
    scale_result = scaled(images);
    reflect_result = reflected(images);
    rotate_result = rotated(images);
    for j=1:length(images)
        imwrite(image_resize_result{j},fullfile(concat_trainaug_path, strcat('train_', num2str(j), '.png')),'png');
        imwrite(translate_result{j}, fullfile(concat_trainaug_path, strcat('translate_',num2str(j), '.png')),'png');
        imwrite(scale_result{j}, fullfile(concat_trainaug_path, strcat('scale_',num2str(j), '.png')),'png');
        imwrite(reflect_result{j}, fullfile(concat_trainaug_path, strcat('reflect_',num2str(j), '.png')),'png');
        imwrite(rotate_result{j}, fullfile(concat_trainaug_path, strcat('rotate_',num2str(j), '.png')),'png');
    end
    
end

%********************creating the test augmentation************************
for i = 1:length(groups_list)
    group = groups_list{i};
    concat_test_path = fullfile(test_path,group);
    concat_testaug_path = fullfile(test_aug_path,group);
    mkdir(concat_testaug_path);
    test_aug_all = imageDatastore((concat_test_path),'IncludeSubfolders',true,'LabelSource','foldernames');
    images = readall(test_aug_all);
    image_resize_result = resized(images);
    translate_result = translated(images);
    scale_result = scaled(images);
    reflect_result = reflected(images);
    rotate_result = rotated(images);
    for j=1:length(images)
        imwrite(image_resize_result{j},fullfile(concat_testaug_path, strcat('test_', num2str(j), '.png')),'png');
        imwrite(translate_result{j}, fullfile(concat_testaug_path, strcat('translate_',num2str(j), '.png')),'png');
        imwrite(scale_result{j}, fullfile(concat_testaug_path, strcat('scale_',num2str(j), '.png')),'png');
        imwrite(reflect_result{j}, fullfile(concat_testaug_path, strcat('reflect_',num2str(j), '.png')),'png');
        imwrite(rotate_result{j}, fullfile(concat_testaug_path, strcat('rotate_',num2str(j), '.png')),'png');
    end
end
