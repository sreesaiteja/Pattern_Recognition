function Confusion_Matrix(original_labels, predicted_labels)
class_count = length(unique(original_labels));
target_class_mat = zeros(class_count, size(predicted_labels, 1));
output_class_mat = zeros(class_count, size(predicted_labels, 1));
%*******************TARGET*************************
    for a = 1 : class_count
        for b = 1:size(original_labels, 1)
            if original_labels(b, 1) == 6
                original_labels(b, 1) = 2;
            elseif original_labels(b, 1) == 10
                original_labels(b, 1) = 3;
            end
            %*********************************
            if (original_labels(b, 1) == a)
                target_class_mat(a, b) = 1;
            else
                target_class_mat(a, b) = 0;
            end
        end
    end
    
%*************output*****************
    for c = 1 : class_count
        for d = 1:size(original_labels, 1)
            if original_labels(d, 1) == 6
                original_labels(d, 1) = 2;
            elseif original_labels(d, 1) == 10
                original_labels(d, 1) = 3;
            end
            %***************************************
            if (original_labels(d, 1) == c)
                output_class_mat(c, d) = 1;
            else
                output_class_mat(a, b) = 0;
            end
        end
    end
  
%fprintf('Plotting Confusion Matrix...\n');
plotconfusion(target_class_mat, output_class_mat);

end