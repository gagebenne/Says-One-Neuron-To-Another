X = [ 0 0 1; 0 1 1; 1 0 1; 1 1 1];
Y = [ 0 1 1 0];

nn = BasicClass(X,Y);

for i = 1:1000
   nn = nn.FeedForward();
   nn = nn.BackProp();
end

nn.Output