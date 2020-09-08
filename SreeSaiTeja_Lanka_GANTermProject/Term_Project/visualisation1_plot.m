load('./outputs/skinnynet/Skinnynetwork_output.mat')

activation_func = activations(net2, train, 10);
y_axis = tsne(activation_func);

figure(1)
gscatter(y_axis(:,1), y_axis(:,2), train.Labels)

savefig('visualisation1_skinny.fig');
save './outputs/skinnynet/Skinnynetwork_output.mat' 'activation_func' 'y_axis'