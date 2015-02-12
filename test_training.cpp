#include <iostream>

#include "math.h"
#include "training.h"

using namespace gradient_descent;

int main() {
  Example e1 = { { {0, 1}, {1, -1} }, 1 };
  Example e2 = { { {1, 1} }, -1 };
  // L2LogLoss func(0.1);
  LogLoss func;

  std::cout << "batch st" << std::endl;
  DenseVector weights = trainBatchSt(func, {e1, e2}, 2, 0.2, 0.01, 10000);
  std::cout << "weights: " << weights[0] << ", " << weights[1] << std::endl;

  std::cout << "batch mt" << std::endl;
  weights = trainBatchMt(func, {e1, e2}, 2, 0.2, 0.01, 10000, 2);
  std::cout << "weights: " << weights[0] << ", " << weights[1] << std::endl;

  std::cout << "stochastic st" << std::endl;
  weights = trainStochasticSt(func, {e1, e2}, 2, 0.2, 0.01, 10000);
  std::cout << "weights: " << weights[0] << ", " << weights[1] << std::endl;

  std::cout << "stochastic mt" << std::endl;
  weights = trainStochasticMt(func, {e1, e2}, 2, 0.2, 0.01, 10000, 2);
  std::cout << "weights: " << weights[0] << ", " << weights[1] << std::endl;
}
