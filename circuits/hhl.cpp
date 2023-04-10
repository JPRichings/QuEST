#include "QuEST.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
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
  const std::size_t NQUBITS_CLOCK = 6;
  const std::size_t NQUBITS_ANCILLA = 1;
  const std::size_t NQUBITS = NQUBITS_B + NQUBITS_CLOCK + NQUBITS_ANCILLA;

  // location of qubit registers
  const std::size_t B_START = 0;
  const std::size_t CLOCK_START = NQUBITS_B;
  const std::size_t ANCILLA_START = NQUBITS_B + NQUBITS_CLOCK;

  Qureg qureg = createQureg(NQUBITS, quenv);

  initZeroState(qureg);
  const double NORM = amplitudeEncode(cb, B_START, qureg);

  // In order to simulate, we have to Eigensolve A
  // Note that solving this problem is equivalent to inverting A, so we hope 
  // it is not necessary on real quantum hardware.
  const EigResult EIGA = eigensolve(cA);

  std::cout << "Eigenvalues of A:\n" << EIGA.eigenvalues().transpose() << std::endl;
  std::cout << "Eigenvectors of A:\n" << EIGA.eigenvectors() << std::endl;

  // Set up for QPE
  const double T = 1.0 /  NQUBITS_CLOCK;
  unsigned int k = 1;
  std::vector<ComplexMatrixN> U(NQUBITS_CLOCK);
  for (std::size_t i = 0; i < NQUBITS_CLOCK; ++i) {
    U.at(i) = createComplexMatrixN(NQUBITS_B);
    constructEvolutionOperator(EIGA, std::complex<double>(0,k*T), U.at(i));
    k *= 2;
  }

  // QPE
  quantumPhaseEstimation(U, B_START, NQUBITS_B, CLOCK_START, NQUBITS_CLOCK, qureg);
  // invert eigenvalues
  //conditionalRotationY();
  // uncompute

  reportState(qureg);

  destroyQureg(qureg, quenv);
  destroyComplexMatrixN(U[0]);
  destroyComplexMatrixN(U[1]);
  destroyComplexMatrixN(U[2]);
  destroyQuESTEnv(quenv);

  return 0;
}