#include <sstream>

#include "io.h"

namespace gradient_descent {

SparseVector readFeatureVector(std::istream& in) {
  SparseVector featureVector;
  std::string entry;
  while (in >> entry) {
    FeatureValue featureValue;
    std::istringstream entryStream(entry);
    std::string featureStr;
    std::getline(entryStream, featureStr, ':');
    std::istringstream(featureStr) >> featureValue.feature;
    entryStream >> featureValue.value;
    featureVector.push_back(featureValue);
  }
  return featureVector;
}

Example readExample(std::istream& in) {
  Example example;
  in >> example.targetValue;
  example.featureVector = readFeatureVector(in);
  return example;
}

void writeModel(const DenseVector& modelWeights, std::ostream& out) {
  out << modelWeights.size() << std::endl;
  for (double w : modelWeights) {
    out << w << " ";
  }
  out << std::endl;
}

DenseVector readModel(std::istream& in) {
  size_t size;
  in >> size;
  DenseVector modelWeights(size);
  for (int i = 0; i < size; ++i) {
    in >> modelWeights[i];
  }
  return modelWeights;
}

} // namespace




