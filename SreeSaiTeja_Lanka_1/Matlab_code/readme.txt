Code contains the following MATLAB files: 

generateData.m
 -Code to generate noisy sample points
 -Code to save data
Run generateData.m will create a file named data.mat in your current working directory.
  
curveFitSSE.m
 -Code to plot points and curves for the Basic Error Minimization approach.
 -Run curveFit.m. which loads data points from generateData.m and plot figures based on N and m.
 -Plots are saved in the 'export_fig' directory under the sub directory 'Plots'.
 -Names of the images start with 'sse' followed by 'n10' for N=10 and 'n50' for N=50 and then followed by 'm(vaue)' for m= value.

curveFitRegularisation.m
 -Code to plot points and curves for the Basic Error Minimization with Regularisation approach.
 -Run curveFitRegularisation.m. which loads data points from generateData.m and plot figures based on N, m and lambda.
 -Plots are saved in the 'export_fig' directory under the sub directory 'Plots'.
 -Names of the images start with 'reg' followed by 'ln' for ln function and 'lm(value)' for labmda value.

curveFitLikelihood.m
Code to plot points and curves for the Bayesian MLE approach.
 -Run curveFitLikelihood.m. which loads data points from generateData.m and plot figures based on N and m.
 -Plots are saved in the 'export_fig' directory under the sub directory 'Plots'.
 -Names of the images start with 'lhd' followed by 'n10' for N=10 and then followed by 'm(vaue)' for m= value.

curveFitSSE.m
Code to plot points and curves for the Bayesian MAP approach.
 -Run curveFitPosterior.m. which loads data points from generateData.m and plot figures based on N, m, alpha and beta.
 -Plots are saved in the 'export_fig' directory under the sub directory 'Plots'.
 -Names of the images start with 'pstr' followed by 'n10' for N=10 and then followed by 'm(vaue)' for m= value.

Note: once the generateData.m file is executed with a certian N value, it should load the same data points always whenever we plot figured for the same N value for any aproach. Amd they are all saved in 'DataFiles' sub directory of the directory 'export_fig' with variable followed by N value. 




