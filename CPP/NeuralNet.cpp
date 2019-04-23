#include "NeuralNet.h"
#define RAND_MAX 1

NeuralNet::NeuralNet(double x[ROWS][IN_COLS], double y[ROWS][OUT_COLS])
{
  srand((unsigned)time(0));

  for(int i=0; i < ROWS; i++)
    for(int j = 0; j < IN_COLS; j++)
      this->input[i][j] = x[i][j];

  for(int i=0; i < ROWS; i++)
    for(int j = 0; j < IN_COLS; j++)
      this->weights1[i][j] = ((double) rand() / (RAND_MAX));

  for(int i=0; i < ROWS; i++)
    for(int j = 0; j < IN_COLS; j++)
      this->weights2[i][j] = ((double) rand() / (RAND_MAX));

  for(int i=0; i < ROWS; i++)
    for(int j = 0; j < IN_COLS; j++)
      this->y[i][j] = y[i][j];

  for(int i=0; i < ROWS; i++)
    for(int j = 0; j < OUT_COLS; j++)
      this->output[i][j] = 0.0;

}

void NeuralNet::feedForward()
{
  for(int i = 0; i < ROWS; i++)
  {
    for(int j = 0; j < NODES; j++)
    {
      double temp = 0.0;
      for(int k = 0; k < IN_COLS; k++)
      {
        temp += input[i][k] * weights1[k][j];
      }
      //std::cout << "temp: " << temp << std::endl;
      std::cout << "Sig(temp): " << Sig(temp) << std::endl;
      this->layer1[i][j] = Sig(temp);
    }
  }

  for(int i = 0; i < ROWS; i++)
  {
    for(int j = 0; j < OUT_COLS; j++)
    {
      double temp = 0.0;
      for(int k = 0; k < NODES; k++)
      {
        temp += layer1[i][k] * weights2[k][j];
      }
      this->output[i][j] = Sig(temp);
    }
  }
}

void NeuralNet::backProp()
{
  for(int i = 0; i < NODES; i++)
  {
    for(int j = 0; j < OUT_COLS; j++)
    {
      double temp = 0.0;
      for(int k = 0; k < ROWS; k++)
      {
        temp += layer1[k][i] * (2.0 * (y[k][j]-output[k][j]) * DSig(output[k][j]));
      }
      this->weights2[i][j] += temp;
    }
  }


  double tempArr[ROWS][NODES];
  for(int i = 0; i < ROWS; i++)
  {
    for(int j = 0; j < NODES; j++)
    {
      double temp = 0.0;
      for(int k = 0; k < OUT_COLS; k++)
      {
        temp += (2.0 * (y[i][k]-output[i][k]) * DSig(output[i][k])) * weights2[j][k];
      }
      tempArr[i][j] = temp;
    }
  }

  for(int i = 0; i < IN_COLS; i++)
  {
    for(int j = 0; j < NODES; j++)
    {
      double temp = 0.0;
      for(int k = 0; k < ROWS; k++)
      {
        temp += input[k][i] * tempArr[k][j];
      }
      this->weights1[i][j] += temp * DSig(layer1[i][j]);
    }
  }
}

double NeuralNet::Sig(double x)
{
  return (1.0 / (1.0 + exp(-x)));
}

double NeuralNet::DSig(double x)
{
  return x * (1.0 - x);
}
