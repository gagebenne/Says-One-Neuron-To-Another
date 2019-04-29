M = readmatrix('../iris.csv');
irisX = M(:,(2:5));
irisY = M(:,6);

%% Sin 'dataset'
sinX = rand(1000,1);
sinY = sin(sinX);

X = [ 0 0 1; 0 1 1; 1 0 1; 1 1 1];
Y = [ 0; 1; 1; 0];

nn = BasicClass(irisX,irisY);
error = zeros(1000);

for i = 1:1000
   nn = nn.FeedForward();
   nn = nn.BackProp();
   error(i) = sum((nn.Y-nn.Output).^2);
end

plot(1:1000,error)

nn.Output
