#include "NeuralNet.h"

NeuralNet::NeuralNet(double x[ROWS][IN_COLS], double y2[ROWS][OUT_COLS])
{
  srand((unsigned)time(0)); 

  for(int i=0; i < ROWS; i++) 
    for(int j = 0; j < IN_COLS; j++)
      this->input[i][j] = x[i][j];

  for(int i=0; i < IN_COLS; i++) 
    for(int j = 0; j < NODES; j++)
      this->weights1[i][j] = (rand()%1000000)/1000000.0; 

  for(int i=0; i < NODES; i++) 
    for(int j = 0; j < OUT_COLS; j++)
      this->weights2[i][j] = (rand()%1000000)/1000000.0;

  for(int i=0; i < ROWS; i++) 
    for(int j = 0; j < OUT_COLS; j++)
      this->y[i][j] = y2[i][j];

  for(int i=0; i < ROWS; i++) 
    for(int j = 0; j < OUT_COLS; j++)
      this->output[i][j] = 0.0;
  
  for(int i=0; i < IN_COLS; i++) 
    for(int j = 0; j < NODES; j++)
      this->layer1[i][j] = 0.0;
  
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
      tempArr[i][j] = temp * DSig(layer1[i][j]);
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
      this->weights1[i][j] += temp;
    }
  }
}

void NeuralNet::predict(double x[IN_COLS])
{
  double nodes[NODES];
  for(int j = 0; j < NODES; j++)
  {
    double temp = 0.0;
    for(int k = 0; k < IN_COLS; k++)
    {
      temp += x[k] * weights1[k][j];
    }
    nodes[j] = Sig(temp);
  }

  double out[OUT_COLS];
  for(int j = 0; j < OUT_COLS; j++)
  {
    double temp = 0.0;
    for(int k = 0; k < NODES; k++)
    {
      temp += nodes[k] * weights2[k][j];
    }
    out[j] = Sig(temp);
  }

  std::cout << "== Prediction == " << std::endl;
  std::cout << "Input: " << std::endl;
  for(int j = 0; j < IN_COLS; j++)
  {
    std::cout << "input["<<j<<"]: " << x[j] << std::endl;
  }
  std::cout << "Output: " << std::endl;
  for(int j = 0; j < OUT_COLS; j++)
  {
    std::cout << "output["<<j<<"]: " << out[j] << std::endl;
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

