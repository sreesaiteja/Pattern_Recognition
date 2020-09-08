Files:
cat_to_num: function for converting categorical labels to numericals.
num_to_cat: function for converting numericals to categorical.

*** Least Square Error (LSE) method with one-vs-all classifier ***
LeastSquares: this computes the weights using the least sqaures method and then
leastSquaresError: classifies the data using one vs all classifier. leastSq: this function computes the least square error based weight matrix
predict: function used to predict the labels of the test feature vectors given
trained weights using one-vs-all classifier.

*** Fischer Projection with KNN classifier ***
FisherDiscriminant: this is the top module to perform fischer projection followed by knn
classification
Fisher_linear_disc: This functions outputs the Weight matrix for fischer projection.
KNN: function for classifying the test data set using knn classification
approach.
---Extra credit
I have uncommented a part from FisherDiscriminant.m and LeastSquares.m 

*******************************************************************************
Parameters:
dataset: update dataset for all the executions.
K: Parameter for performing KNN classification.

********************************************************************************
Execution Process:
Run
LeastSquares #for LSE for all classes
FisherDiscriminant # for fischer for all classes

********************************************************************************
Experiments performed:

#LSE for 3 datasets. - 2 plots and 4 tables have been shown per dataset
#Fischer Projection for 3 datasets - 2 plots and 4 tables have been shown per dataset.