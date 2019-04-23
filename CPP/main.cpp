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
  const int loops = 15000;
  double error[loops];

  for (int i = 0; i < loops; i++)
  {
    nn.feedForward();
    nn.backProp();
    error[i] = 0;
    for (int j = 0; j < ROWS; j++)
    {
      error[i] += (nn.y[j][0] - nn.output[j][0]) * (nn.y[j][0] - nn.output[j][0]);
    }
    std::cout << "Error: " << error[i] << std::endl;
  }

  std::cout << "Weights1: " << std::endl;
  for(int i = 0; i < IN_COLS; i++)
  {
    std::cout << "weights1["<<i<<"]: ";
    for(int j = 0; j < NODES; j++)
    {
      std::cout << nn.weights1[i][j] << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << "Weights2: " << std::endl;
  for(int i = 0; i < NODES; i++)
  {
    for(int j = 0; j < OUT_COLS; j++)
    {
      std::cout << "weights2["<<i<<"]["<<j<<"]: " << nn.weights2[i][j] << std::endl;
    }
  }
  std::cout << "Output: " << std::endl;
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
