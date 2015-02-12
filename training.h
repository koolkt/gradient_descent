#pragma once

#include <vector>

#include "math.h"

namespace gradient_descent {

DenseVector trainBatchSt(const LossFunction& lossFunction,
                         const std::vector<Example> examples,
                         size_t numFeatures,
                         double learningRate,
                         double epsilon,
                         size_t maxIterations);

DenseVector trainBatchMt(const LossFunction& lossFunction,
                         const std::vector<Example> examples,
                         size_t numFeatures,
                         double learningRate,
                         double epsilon,
                         size_t maxIterations,
                         size_t numThreads);

DenseVector trainStochasticSt(const LossFunction& lossFunction,
                              const std::vector<Example> examples,
                              size_t numFeatures,
                              double learningRate,
                              double epsilon,
                              size_t maxIterations);

DenseVector trainStochasticMt(const LossFunction& lossFunction,
                              const std::vector<Example> examples,
                              size_t numFeatures,
                              double learningRate,
                              double epsilon,
                              size_t maxIterations,
                              size_t numThreads);

} // namespace
