#include "QuEST.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <vector>
#include <numeric>
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

  // In order to simulate, we have to Eigensolve A
  // Note that solving this problem is equivalent to inverting A, so we hope 
  // it is not necessary on real quantum hardware.
  const EigResult EIGA = eigensolve(cA);

  std::cout << "Eigenvalues of A:\n" << EIGA.eigenvalues().transpose() << std::endl;
  std::cout << "Eigenvectors of A:\n" << EIGA.eigenvectors() << std::endl;

  // Set up for QPE
  const double T = (2 * M_PI) / (std::pow(2,NQUBITS_CLOCK) * std::abs(EIGA.eigenvalues()(0)));
  unsigned int k = 1;
  std::vector<ComplexMatrixN> U(NQUBITS_CLOCK);
  for (std::size_t i = 0; i < NQUBITS_CLOCK; ++i) {
    U.at(i) = createComplexMatrixN(NQUBITS_B);
    constructEvolutionOperator(EIGA, std::complex<double>(0,k*T), U.at(i));
    k *= 2;
  }
  
  // Repeat QPE, C-ROTY until ancilla is measured at 1
  const std::size_t MAX_ITER = 100;
  std::size_t iter = 0;
  int m_ancilla = 0;
  double p_ancilla; 
  while(!m_ancilla && iter < MAX_ITER) {
    initZeroState(qureg);
    const double NORM = amplitudeEncode(cb, B_START, qureg);
    // QPE
    quantumPhaseEstimation(U, B_START, NQUBITS_B, CLOCK_START, NQUBITS_CLOCK, qureg);
    // invert eigenvalues
    conditionalRotationY(EIGA, NQUBITS_B, NQUBITS_CLOCK, NQUBITS_ANCILLA, qureg);

    m_ancilla = measureWithStats(qureg, ANCILLA_START, &p_ancilla);
    std::printf("Ancilla is %d with probability %g.\n", m_ancilla, p_ancilla);
    std::printf("Norm = %g, iter = %d.\n", calcTotalProb(qureg), iter++);
  }

  if (!m_ancilla) {
    std::printf("Reached MAX_ITER, aborting.\n");
    for (auto& umat : U) destroyComplexMatrixN(umat);
    reportState(qureg);
    destroyQureg(qureg, quenv);
    destroyQuESTEnv(quenv);
    return 0;
  }

  // set up for inverse QPE
  for (auto& umat : U) destroyComplexMatrixN(umat);
  U.clear();
  U.resize(NQUBITS_CLOCK);

  k = 1;
  for (std::size_t i = 0; i < NQUBITS_CLOCK; ++i) {
    U.at(i) = createComplexMatrixN(NQUBITS_B);
    constructEvolutionOperator(EIGA, std::complex<double>(0,-k*T), U.at(i));
    k *= 2;
  }

  // uncompute
  inverseQuantumPhaseEstimation(U, B_START, NQUBITS_B, CLOCK_START, NQUBITS_CLOCK, qureg);

  reportState(qureg);

  std::vector<double> p_b(std::pow(2, NQUBITS_B));
  std::vector<int> qubits(NQUBITS_B);
  std::iota(qubits.begin(), qubits.end(), B_START); 
  calcProbOfAllOutcomes(p_b.data(), qureg, qubits.data(), NQUBITS_B);
  std::printf("|b> register probabilities\n");
  for (std::size_t idx = 0; idx < p_b.size(); ++idx) {
    std::printf("  |%d> : %g\n", idx, p_b.at(idx));
  }

  for (auto& umat : U) destroyComplexMatrixN(umat);
  destroyQureg(qureg, quenv);
  destroyQuESTEnv(quenv);

  return 0;
}