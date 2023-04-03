#include "QuEST.h"
#include <eigen3/Eigen/Dense>
#include <vector>
#include <iostream>
#include "hhl.hpp"

double amplitudeEncode(const Eigen::VectorXd& VEC, Qureg qureg) {
  // assumes VEC is real
  const double NORM = VEC.norm();
  Eigen::VectorXd normVec = VEC / NORM;
  std::vector<double> im(VEC.size(), 0.0);

  setAmps(qureg, 0, normVec.data(), im.data(), VEC.size());

  return NORM;
}