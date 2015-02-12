#include <cmath>
#include <limits>
#include <numeric>
#include <thread>

#include "io.h"
#include "math.h"
#include "training.h"

namespace gradient_descent {

DenseVector initWeights(size_t numFeatures) {
  return DenseVector(numFeatures);
}

bool checkConvergence(double prevLoss, double curLoss, double epsilon) {
  return std::abs(prevLoss - curLoss) / prevLoss < epsilon;
}

void logCurrentLoss(double curLoss, size_t iteration) {
  LOG << "Loss after " << iteration + 1 << " iteration(s): " << curLoss;
}

void logConvergence(size_t iteration) {
  LOG << "Reached convergence after " << iteration + 1 << " iteration(s).";
}

void logTermination() {
  LOG << "Finished training.";
}

DenseVector trainBatchSt(const LossFunction& lossFunction,
                         const std::vector<Example> examples,
                         size_t numFeatures,
                         double learningRate,
                         double epsilon,
                         size_t maxIterations) {
  DenseVector modelWeights = initWeights(numFeatures);
  DenseVector weightOffsets(numFeatures);
  double prevLoss = std::numeric_limits<double>::max();

  for (size_t i = 0; i < maxIterations; ++i) {
    double curLoss = 0;
    std::fill(weightOffsets.begin(), weightOffsets.end(), 0);
    for (const Example& e : examples) {
      curLoss += update(lossFunction, modelWeights, e, learningRate, weightOffsets);
    }
    addInPlace(modelWeights, weightOffsets);
    curLoss += lossFunction.getRegularizationTerm(modelWeights);
    // logCurrentLoss(curLoss, i);
    if (checkConvergence(prevLoss, curLoss, epsilon)) {
      logConvergence(i);
      break;
    }
    prevLoss = curLoss;
  }

  logTermination();
  return modelWeights;
}

DenseVector trainBatchMt(const LossFunction& lossFunction,
                         const std::vector<Example> examples,
                         size_t numFeatures,
                         double learningRate,
                         double epsilon,
                         size_t maxIterations,
                         size_t numThreads) {
  DenseVector modelWeights = initWeights(numFeatures);
  double prevLoss = std::numeric_limits<double>::max();

  std::vector<std::thread> threads(numThreads);
  std::vector<DenseVector> weightOffsets(numThreads, DenseVector(numFeatures));
  std::vector<double> partialLoss(numThreads);
  size_t splitSize = std::max(examples.size() / numThreads, 1UL);

  for (size_t i = 0; i < maxIterations; ++i) {
    for (size_t t = 0; t < numThreads; ++t) {
      threads[t] = std::thread([&, t] () {
        partialLoss[t] = 0;
        std::fill(weightOffsets[t].begin(), weightOffsets[t].end(), 0);
        size_t examplesBegin = t * splitSize;
        size_t examplesEnd = std::min(examplesBegin + splitSize, examples.size());
        for (size_t j = examplesBegin; j < examplesEnd; ++j) {
          partialLoss[t] += update(lossFunction,
                                   modelWeights,
                                   examples[j],
                                   learningRate,
                                   weightOffsets[t]);
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }    
    double curLoss = std::accumulate(partialLoss.begin(), partialLoss.end(), 0);
    for (auto& wo : weightOffsets) {
      addInPlace(modelWeights, wo);
    }
    curLoss += lossFunction.getRegularizationTerm(modelWeights);
    // logCurrentLoss(curLoss, i);
    if (checkConvergence(prevLoss, curLoss, epsilon)) {
      logConvergence(i);
      break;
    }
    prevLoss = curLoss;
  }

  logTermination();
  return modelWeights;
}

DenseVector trainStochasticSt(const LossFunction& lossFunction,
                              const std::vector<Example> examples,
                              size_t numFeatures,
                              double learningRate,
                              double epsilon,
                              size_t maxIterations) {
  DenseVector modelWeights = initWeights(numFeatures);
  double prevLoss = std::numeric_limits<double>::max();

  for (size_t i = 0; i < maxIterations; ++i) {
    double curLoss = 0;
    for (const Example& e : examples) {
      curLoss += stochasticUpdate(lossFunction, modelWeights, e, learningRate);
    }
    curLoss += lossFunction.getRegularizationTerm(modelWeights);
    // logCurrentLoss(curLoss, i);
    if (checkConvergence(prevLoss, curLoss, epsilon)) {
      logConvergence(i);
      break;
    }
    prevLoss = curLoss;
  }

  logTermination();
  return modelWeights;
}

DenseVector trainStochasticMt(const LossFunction& lossFunction,
                              const std::vector<Example> examples,
                              size_t numFeatures,
                              double learningRate,
                              double epsilon,
                              size_t maxIterations,
                              size_t numThreads) {
  DenseVector modelWeights = initWeights(numFeatures);
  double prevLoss = std::numeric_limits<double>::max();

  std::vector<std::thread> threads(numThreads);
  std::vector<DenseVector> partialWeights(numThreads, DenseVector(numFeatures));
  std::vector<double> partialLoss(numThreads);
  size_t splitSize = std::max(examples.size() / numThreads, 1UL);

  for (size_t i = 0; i < maxIterations; ++i) {
    for (size_t t = 0; t < numThreads; ++t) {
      threads[t] = std::thread([&, t] () {
        partialLoss[t] = 0;
        partialWeights[t] = modelWeights;
        size_t examplesBegin = t * splitSize;
        size_t examplesEnd = std::min(examplesBegin + splitSize, examples.size());
        for (size_t j = examplesBegin; j < examplesEnd; ++j) {
          partialLoss[t] += stochasticUpdate(lossFunction,
                                             partialWeights[t],
                                             examples[j],
                                             learningRate);
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }    
    double curLoss = std::accumulate(partialLoss.begin(), partialLoss.end(), 0);
    std::fill(modelWeights.begin(), modelWeights.end(), 0);
    for (auto& w : partialWeights) {
      addInPlace(modelWeights, w);
    }
    scaleInPlace(modelWeights, 1.0 / numThreads);
    curLoss += lossFunction.getRegularizationTerm(modelWeights);
    // logCurrentLoss(curLoss, i);
    if (checkConvergence(prevLoss, curLoss, epsilon)) {
      logConvergence(i);
      break;
    }
    prevLoss = curLoss;
  }

  logTermination();
  return modelWeights;
}

} // namespace
