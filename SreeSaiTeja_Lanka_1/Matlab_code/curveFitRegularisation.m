%Christopher Funk, Jan 2018

addpath export_fig/

%load the data points
load data.mat
%% Start your curve fitting program here
m = 9; %assigning m value manually
t = t.'; %transposes the t vector
xmat = zeros(npts, m+1); %x matrix for the order m initialised to all zeros
for i = 1:m+1
    xmat(:,i)= x.^(i-1); % N by m+1 order of x matrix
end

lambda = exp(-18);%ln(lambda) = values gives, lambda = exp(value)
wreg = ((xmat'*xmat) + (lambda*eye(m+1)))\(xmat'* t) %optimised w after differentiation of Error fucntion along with lambda penalty
yreg = xmat * wreg; %calculating new y with new w

%plot the groud truth curve
figure(1)
clf
hold on;
% plot the x and y color the area around the line by err (here the std)
%c = @cmu.colors; % shortcut function handle

%h = shadedErrorBar(x, yderived, betaerrorvect,{'r-','color','r','LineWidth',2},0);
%plot the noisy observations
plot(x,t,'bo','MarkerSize',8,'LineWidth',1.5);
p = plot(x,yreg, 'r', 'LineWidth', 1.5);
%p.Marker='o';
p = plot(x,y, 'g', 'LineWidth', 1.5);
%p.Marker='o';
hold off; 
% Make it look good
grid on;
set(gca,'FontWeight','bold','LineWidth',2)
xlabel('x')
ylabel('t')
dim =  [0.7 0.865 0.07 0.05];
str = sprintf('N = 10 \nM = 9 \nln\\lambda = -18' ); 
annotation('textbox',dim,  'String',str, 'FontWeight', 'bold', 'FontSize', 10, 'FitBoxToText', 'on', 'EdgeColor', [0.5 0.5 0.5],'LineWidth', 1);

% Save the image into a decent resolution
export_fig sampleplot -png -transparent -r150