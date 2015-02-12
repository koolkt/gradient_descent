#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "io.h"
#include "math.h"
#include "training.h"

using namespace gradient_descent;

bool validateLine(const std::string& s) {
  for (char c : s) {
    if (c != ' ' && c != '\n') {
      return true;
    }
  }
  return false;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    LOG << "Need to specify training file.";
    return 1;
  }
  std::vector<Example> examples;
  std::ifstream trainingFile(argv[1]);
  std::string line;
  size_t numFeatures = 0;
  while (std::getline(trainingFile, line)) {
    if (validateLine(line)) {
      std::istringstream lineStream(line);
      examples.push_back(readExample(lineStream));
      for (const FeatureValue& fv : examples.back().featureVector) {
        numFeatures = std::max(numFeatures, fv.feature + 1);
      }
    }
  }
  trainingFile.close();

  LOG << "Read " << examples.size() << " examples.";
  LOG << "Total features: " << numFeatures << ".";

  // DenseVector model = trainBatchSt(L2LogLoss(0.001), examples, numFeatures, 0.1, 1e-6, 1e6);
  // DenseVector model = trainBatchMt(L2LogLoss(0.001), examples, numFeatures, 0.1, 1e-6, 1e6, 4);
  // DenseVector model = trainStochasticSt(L2LogLoss(0.001), examples, numFeatures, 0.1, 1e-6, 1e6);
  DenseVector model = trainStochasticMt(L2LogLoss(0.001), examples, numFeatures, 0.1, 1e-6, 1e6, 4);

  if (argc > 2) {
    std::ofstream modelFile(argv[2]);
    writeModel(model, modelFile);
    modelFile.close();
    LOG << "Written model to file.";
  }
}
