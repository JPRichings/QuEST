#include "QuEST.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <vector>
#include <iostream>
#include <cmath>
#include "hhl.hpp"

double amplitudeEncode(const Eigen::VectorXd& VEC, const std::size_t START, Qureg qureg) {
  // assumes VEC is real
  const double NORM = VEC.norm();
  const std::size_t START_IDX = std::pow(2, START) - 1;
  Eigen::VectorXd normVec = VEC / NORM;
  std::vector<double> im(VEC.size(), 0.0);

  setAmps(qureg, START_IDX, normVec.data(), im.data(), VEC.size());

  return NORM;
}

Eigen::MatrixXcd expMat(const EigResult& EIGA, const std::complex<double> KIT) {
  // to calculate exp(iAt) we find the eigendecompostion of A, then calculate
  // exp(kiAt) = sum(exp(k*i*lamba*t)|lambda><lambda|), where lambda is the 
  // eigenvalues, and |lambda> the eigenvectors
  const std::size_t DIM = EIGA.eigenvalues().rows();
  Eigen::MatrixXcd expmat = Eigen::MatrixXcd::Zero(DIM, DIM);

  for (std::size_t i = 0; i < DIM; ++i) {
    expmat += std::exp(KIT * EIGA.eigenvalues()(i)) * EIGA.eigenvectors().col(i) * EIGA.eigenvectors().col(i).transpose();
  }

  return expmat;
}

void constructEvolutionOperator(const EigResult& EIGA, const std::complex<double> KIT, ComplexMatrixN& U) {
  const std::size_t DIM = std::pow(2, U.numQubits);
  const std::size_t ADIM = EIGA.eigenvalues().rows();

  Eigen::MatrixXcd expkiAt = expMat(EIGA, KIT);

  // Evolution operator is exp(iAt)
  for (std::size_t row = 0; row < ADIM; ++row) {
    for (std::size_t col = 0; col < ADIM; ++col) {
      U.real[row][col] = expkiAt(row,col).real();
      U.imag[row][col] = expkiAt(row,col).imag();
    }
  }

  // if our A matrix is not a power of 2 in size, we pad the rest with 0
  for (std::size_t row = ADIM; row < DIM; ++row) {
    for (std::size_t col = ADIM; col < DIM; ++col) {
      U.real[row][col] = 0.0;
      U.imag[row][col] = 0.0;
    }
  }

  return;
}

void quantumPhaseEstimation(const std::vector<ComplexMatrixN>& U, const std::size_t B_START, const std::size_t NQUBITS_B, 
const std::size_t CLOCK_START, const std::size_t NQUBITS_CLOCK, Qureg qureg) {
  const std::size_t NCTRLS = 1;
  std::vector<int> targs(NQUBITS_B);

  // Apply Hadamard gate to clock qubits
  for (std::size_t qid = CLOCK_START; qid < CLOCK_START + NQUBITS_CLOCK; ++qid) {
    hadamard(qureg, qid);
  }

  // apply controlled time evolution gates
  int tid = B_START;
  for (auto& targ : targs) {
    targ = tid;
    ++tid;
  }
  std::size_t uid = 0;
  for (int ctrl = CLOCK_START + NQUBITS_CLOCK - 1; ctrl >= CLOCK_START; --ctrl) {
    applyMultiControlledMatrixN(qureg, &ctrl, NCTRLS, targs.data(), NQUBITS_B, U[uid]);
    ++uid;
  }

  // inverse QFT on clock qubits
  inverseQFT(CLOCK_START, NQUBITS_CLOCK, qureg);
  
  return;
}

void inverseQFT(const std::size_t START_QUBIT, const std::size_t NQUBITS, Qureg qureg) {
  const std::size_t END_QUBIT = START_QUBIT + NQUBITS - 1;
  double angle = 0.0;

  for (std::size_t qid = END_QUBIT; qid <= START_QUBIT; --qid) {
    for (std::size_t ctrl = END_QUBIT; ctrl > qid; --ctrl) {
      std::size_t m = END_QUBIT - ctrl;
      angle = M_PI / std::pow(2, m);
      controlledPhaseShift(qureg, ctrl, qid, angle);
    }
    hadamard(qureg, qid);
  }

  return;
}

void conditionalRotationY(const EigResult& A, const std::size_t NQUBITS_B, const std::size_t NQUBITS_CLOCK, 
const std::size_t NQUBITS_ANCILLA, Qureg qureg) {
  const std::size_t NQUBITS = NQUBITS_B + NQUBITS_CLOCK + NQUBITS_ANCILLA;
  double angle;
  Eigen::VectorXi measurement(NQUBITS_CLOCK);
  Eigen::Matrix2cd Ry;
  Eigen::MatrixXi IB = Eigen::MatrixXi::Identity(NQUBITS_B, NQUBITS_B);
  Eigen::SparseMatrix<std::complex<double>> op(std::pow(2,NQUBITS), std::pow(2,NQUBITS));
  op.reserve(std::pow(2, NQUBITS) + 2 * std::pow(2, NQUBITS_B));

  // need to figure out qubit ordering in QuEST to build this operator correctly...

  return;
}