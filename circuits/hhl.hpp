#include "QuEST.h"
#include <complex>
#include <vector>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>

using EigResult = Eigen::ComplexEigenSolver<Eigen::MatrixXcd>;

template <typename DerivedA, typename Derivedb> 
Eigen::VectorXd solveClassically(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<Derivedb>& b) {
  Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
  return x;
}

template <typename DerivedA>
EigResult eigensolve(const Eigen::MatrixBase<DerivedA>& A) { 
  return EigResult(A, true);
}

double amplitudeEncode(const Eigen::VectorXd& VEC, const std::size_t START_QUBIT, Qureg qureg);

Eigen::MatrixXcd expMat(const EigResult& EIGA, const std::complex<double> KIT);

void constructEvolutionOperator(const EigResult& EIGA, const std::complex<double> KIT, ComplexMatrixN& U);

void quantumPhaseEstimation(const std::vector<ComplexMatrixN>& U, const std::size_t B_START, 
  const std::size_t NQUBITS_B, const std::size_t CLOCK_START, const std::size_t NQUBITS_CLOCK, Qureg qureg);

void inverseQuantumPhaseEstimation(const std::vector<ComplexMatrixN>& U, const std::size_t B_START, 
  const std::size_t NQUBITS_B, const std::size_t CLOCK_START, const std::size_t NQUBITS_CLOCK, Qureg qureg);

void inverseQFT(const std::size_t START_QUBIT, const std::size_t NQUBITS, Qureg qureg);

void conditionalRotationY(const EigResult& EIGA, const std::size_t NQUBITS_B, const std::size_t NQUBITS_CLOCK,
  const std::size_t NQUBITS_ANCILLA, Qureg qureg);

Eigen::Matrix2cd updateRy(const std::complex<double> ANGLE);

void eigenToComplexMatrix2(const Eigen::Matrix2cd& E, ComplexMatrix2& U);

