#ifndef NN_H 
#define NN_H 

#define ROWS 4
#define IN_COLS 3
#define OUT_COLS 1
#define NODES 4
#define ITERATIONS 1500


#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

class NeuralNet 
{
  public:
    double input[ROWS][IN_COLS];
    double layer1[ROWS][NODES];
    double weights1[IN_COLS][NODES];
    double weights2[NODES][OUT_COLS];
    double y[ROWS][OUT_COLS];
    double output[ROWS][OUT_COLS];

    NeuralNet(double x[ROWS][IN_COLS], double y[ROWS][OUT_COLS]);
    void feedForward();
    void backProp();
    void predict(double x[IN_COLS]);
    double Sig(double x);
    double DSig(double x);

  private:

};

#endif

