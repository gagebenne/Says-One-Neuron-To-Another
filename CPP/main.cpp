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

  for (int i = 0; i < 1500; i++)
  {
    nn.feedForward();
    nn.backProp();
  }

  std::cout << "Output: " << std::endl;
  for(int i = 0; i < ROWS; i++)
  {
    for(int j = 0; j < OUT_COLS; j++)
    {
      std::cout << "output["<<i<<"]["<<j<<"]: " << nn.output[i][j] << std::endl;
    }
  }
}
