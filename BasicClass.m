classdef BasicClass
   properties
      Input
      Weights1
      Weights2
      Y
      Output
      Layer1
   end
    methods
        function obj = BasicClass(x,y)
            obj.Input = x;
            s = size(x);
            n = 5;
            obj.Weights1 = rand(s(2), 5);
            obj.Weights2 = rand(5, 1);
            obj.Y = y;
            obj.Output = zeros(size(y));
            obj.Layer1 = zeros(10,10);
        end
        function obj = FeedForward(obj)
            obj.Layer1 = arrayfun(@(x) Sigmoid(x),mtimes(obj.Input, obj.Weights1));
            obj.Output = arrayfun(@(x) Sigmoid(x),mtimes(obj.Layer1, obj.Weights2));

        end
        function pred = Predict(obj, input)
            obj.Layer1 = arrayfun(@(x) Sigmoid(x),mtimes(input, obj.Weights1));
            obj.Output = arrayfun(@(x) Sigmoid(x),mtimes(obj.Layer1, obj.Weights2));
            pred = obj.Output;
        end
        function obj = BackProp(obj)
            arg1 = transpose(obj.Layer1);
            arg2 = 2*(obj.Y - obj.Output);
            arg3 = arrayfun(@(x) SigmoidDerivative(x),obj.Output);
            D_Weights2 = mtimes(arg1, (arg2 .* arg3));

            arg1 = transpose(obj.Input);
            arg2 = 2*(obj.Y - obj.Output);
            arg3 = arrayfun(@(x) SigmoidDerivative(x), obj.Output);
            arg4 = arg2 .* arg3;
            arg5 = transpose(obj.Weights2);
            arg6 = arrayfun(@(x) SigmoidDerivative(x),obj.Layer1);
            arg7 = mtimes(arg4, arg5) .* arg6;
            D_Weights1 = mtimes(arg1, arg7);

            obj.Weights1 = obj.Weights1 + D_Weights1;
            obj.Weights2 = obj.Weights2 + D_Weights2;
        end
    end
end
