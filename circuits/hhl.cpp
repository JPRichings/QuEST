#include "QuEST.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "hhl.hpp"

int main(void) 
{
  QuESTEnv quenv = createQuESTEnv();

  Eigen::Matrix<double, 6, 6> A {
    {0, 0, 0, 3, 2, 6},
    {0, 0, 0, -3, 1, 10},
    {0, 0, 0, 3, 8, 18},
    {18, 8, 3, 0, 0, 0},
    {10, 1, -3, 0, 0, 0},
    {6, 2, 3, 0, 0, 0}
  };
  Eigen::Vector<double,6> b {32, -4, 68, 68, -4, 32};

  Eigen::VectorXd x = solveClassically(A, b);

  std::cout << "Classical solution is:\n" << x << std::endl;

  destroyQuESTEnv(quenv);

  return 0;
}