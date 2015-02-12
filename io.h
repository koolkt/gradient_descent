#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "math.h"

namespace gradient_descent {

struct EndlHelper {
  ~EndlHelper() { std::cout << std::endl; }
};

#define LOG (EndlHelper(), std::cout)

SparseVector readFeatureVector(std::istream& in);

Example readExample(std::istream& in);

void writeModel(const DenseVector& modelWeights, std::ostream& out);

DenseVector readModel(std::istream& in);

} // namespace
