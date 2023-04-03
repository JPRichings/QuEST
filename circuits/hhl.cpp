#include "QuEST.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "hhl.hpp"

int main(void) 
{
  // Implementation of Harrow-Hassidim-Lloyd algorithm to solve Ax=b.  
  // Solve problem classically first.
  
  Eigen::Matrix<double, 6, 6, Eigen::RowMajor> cA {
    {0, 0, 0, 3, 2, 6},
    {0, 0, 0, -3, 1, 10},
    {0, 0, 0, 3, 8, 18},
    {3, -3, 3, 0, 0, 0},
    {2, 1, 8, 0, 0, 0},
    {6, 10, 18, 0, 0, 0}
  };
  Eigen::Vector<double,6> cb {32, -4, 68, 0, 0, 0};

  Eigen::VectorXd x = solveClassically(cA, cb);

  // Solve problem with HHL
  std::cout << "Classical solution is:\n" << x << std::endl;

  QuESTEnv quenv = createQuESTEnv();
  
  // number of qubits required
  const std::size_t NQUBITS_B = std::ceil(std::log2(cb.size()));
  const std::size_t NQUBITS_SCRATCH = 8;

  Qureg qureg_b = createQureg(NQUBITS_B, quenv);
  Qureg qureg_scratch = createQureg(NQUBITS_SCRATCH, quenv);

  const double NORM = amplitudeEncode(cb, qureg_b);
  initZeroState(qureg_scratch);
  

  destroyQureg(qureg_b, quenv);
  destroyQureg(qureg_scratch, quenv);
  destroyQuESTEnv(quenv);

  return 0;
}