#include <iostream>
#include <string>

#include "io.h"

using namespace gradient_descent;

int main() {
  std::vector<Example> e = readExamples(std::cin);
  std::cout << e.size() << std::endl;
}
