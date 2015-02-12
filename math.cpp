#include <cmath>

#include "math.h"

namespace gradient_descent {

double dotProduct(const SparseVector& sparseVector, const DenseVector& denseVector) {
  double product = 0;
  for (const FeatureValue& fv : sparseVector) {
    product += fv.value * denseVector[fv.feature];
  }
  return product;
}

double squaredNorm(const DenseVector& denseVector) {
  double norm = 0;
  for (double x : denseVector) {
    norm += x * x;
  }
  return norm;
}

void addInPlace(DenseVector& lhs, const DenseVector& rhs) {
  if (rhs.size() > lhs.size()) {
    lhs.resize(rhs.size());
  }
  for (size_t i = 0; i < rhs.size(); ++i) {
    lhs[i] += rhs[i];
  }
}

void scaleInPlace(DenseVector& vector, double scale) {
  for (double& x : vector) {
    x *= scale;
  }
}

double logistic(double x) {
  return 1 / (1 + std::exp(-x));
}

double LossFunction::getRegularizationTerm(const DenseVector& modelWeights) const {
  return 0;
}

double LogLoss::predict(double dotProd) const {
  return logistic(dotProd);
}

double LogLoss::getLoss(double targetValue, double dotProd) const {  
  return std::log(1 + std::exp(-targetValue * dotProd));
}
    
double LogLoss::getGradient(double modelWeight,
                            double featureValue,
                            double targetValue,
                            double dotProd) const {
  return -targetValue * featureValue * logistic(targetValue * dotProd);
}

L2LogLoss::L2LogLoss(double regularization)
  : regularization_(regularization) {
}

double L2LogLoss::getGradient(double modelWeight,
                              double featureValue,
                              double targetValue,
                              double dotProd) const {
  return 2 * regularization_ * modelWeight -
    targetValue * featureValue * logistic(targetValue * dotProd);
}

double L2LogLoss::getRegularizationTerm(const DenseVector& modelWeights) const {
  return regularization_ * squaredNorm(modelWeights);
}

double update(const LossFunction& lossFunction,
              const DenseVector& modelWeights,
              const Example& example,
              double learningRate,
              DenseVector& weightOffsets) {
  double dotProd = dotProduct(example.featureVector, modelWeights);
  for (const FeatureValue& fv : example.featureVector) {
    weightOffsets[fv.feature] -=
      learningRate * lossFunction.getGradient(modelWeights[fv.feature],
                                              fv.value,                                              
                                              example.targetValue,
                                              dotProd);
  }
  return lossFunction.getLoss(example.targetValue, dotProd);
}

double stochasticUpdate(const LossFunction& lossFunction,
                        DenseVector& modelWeights,
                        const Example& example,
                        double learningRate) {
  return update(lossFunction,
                modelWeights,
                example,
                learningRate,
                modelWeights);
}

double predict(const LossFunction& lossFunction,
               const DenseVector& modelWeights,
               const SparseVector& featureVector) {
  return lossFunction.predict(dotProduct(featureVector, modelWeights));
}


} // namespace
