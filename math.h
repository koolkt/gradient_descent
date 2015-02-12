#pragma once

#include <vector>

namespace gradient_descent {

using DenseVector = std::vector<double>;

struct FeatureValue {
  size_t feature;
  double value;
};

using SparseVector = std::vector<FeatureValue>;

struct Example {
  SparseVector featureVector;
  double targetValue;
};

double dotProduct(const SparseVector& sparseVector, const DenseVector& denseVector);

double squaredNorm(const DenseVector& denseVector);

void addInPlace(DenseVector& lhs, const DenseVector& rhs);

void scaleInPlace(DenseVector& vector, double scale);

double logistic(double x);

class LossFunction {
public:
  virtual double predict(double dotProd) const = 0;

  // Addend of (unregularized) loss corresponding to a single example.
  virtual double getLoss(double targetValue, double dotProd) const = 0;

  // Component of gradient of (regularized) loss corresponding to a single
  // example.
  virtual double getGradient(double modelWeight,
                             double featureValue,
                             double targetValue,
                             double dotProd) const = 0;

  // Regularization term in loss function.
  virtual double getRegularizationTerm(const DenseVector& modelWeights) const;
};           

class LogLoss : public LossFunction {
public:
  double predict(double dotProd) const;

  double getLoss(double targetValue, double dotProd) const;

  virtual double getGradient(double modelWeight,
                             double featureValue,
                             double targetValue,
                             double dotProd) const;
};           
  
class L2LogLoss : public LogLoss {
public:
  L2LogLoss(double regularization);

  double getGradient(double modelWeight,
                     double featureValue,
                     double targetValue,
                     double dotProd) const;

  double getRegularizationTerm(const DenseVector& modelWeights) const;

private:
  double regularization_;
};

double update(const LossFunction& lossFunction,
              const DenseVector& modelWeights,
              const Example& example,            
              double learningRate,
              DenseVector& weightOffsets);

double stochasticUpdate(const LossFunction& lossFunction,
                        DenseVector& modelWeights,
                        const Example& example,
                        double learningRate);

double predict(const LossFunction& lossFunction,
               const DenseVector& modelWeights,
               const SparseVector& featureVector);

} // namespace
