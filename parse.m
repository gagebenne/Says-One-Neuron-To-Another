% Dataset parsing
%% Iris dataset
% Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species
% Iris-setosa: 1, Iris-versicolor: 2, Iris-virginica: 3
M = readmatrix('iris.csv');
irisX = M(:,(2:5));
irisY = M(:,6);

%% Sin 'dataset'
sinX = rand(1000,1);
sinY = sin(Y);
