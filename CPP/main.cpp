#include <iostream>
#include "NeuralNet.h"

int main() {
  double x[ROWS][IN_COLS] = { {0.0,0.0,1.0},
              {0.0,1.0,1.0},
              {1.0,0.0,1.0},
              {1.0,1.0,1.0} };
  double y[ROWS][OUT_COLS] = { {0.0},
              {1.0},
              {1.0},
              {0.0} };

  NeuralNet nn(x,y);
  double error[ITERATIONS];

  for (int i = 0; i < ITERATIONS; i++)
  {
    nn.feedForward();
    nn.backProp();
    error[i] = 0;
    for (int j = 0; j < ROWS; j++)
    {
      error[i] += (nn.y[j][0] - nn.output[j][0]) * (nn.y[j][0] - nn.output[j][0]);
    }
  }

  for(int i = 0; i < ROWS; i++)
  {
    for(int j = 0; j < OUT_COLS; j++)
    {
      std::cout << "output["<<i<<"]["<<j<<"]: " << nn.output[i][j] << std::endl;
    }
  }

  double test1[IN_COLS] = {0.0, 0.0, 1.0};
  double test2[IN_COLS] = {0.0, 1.0, 1.0};
  double test3[IN_COLS] = {1.0, 0.0, 1.0};
  double test4[IN_COLS] = {1.0, 1.0, 1.0};
  nn.predict(test1);
  nn.predict(test2);
  nn.predict(test3);
  nn.predict(test4);
}
