#include "QuEST.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <vector>
#include <iostream>
#include <cmath>
#include <complex>
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
  std::iota(targs.begin(), targs.end(), B_START);
  std::size_t uid = U.size() - 1;
  for (int ctrl = CLOCK_START + NQUBITS_CLOCK - 1; ctrl >= CLOCK_START; --ctrl) {
    applyMultiControlledMatrixN(qureg, &ctrl, NCTRLS, targs.data(), NQUBITS_B, U.at(uid));
    --uid;
  }

  // inverse QFT on clock qubits
  inverseQFT(CLOCK_START, NQUBITS_CLOCK, qureg);
  
  return;
}

void inverseQuantumPhaseEstimation(const std::vector<ComplexMatrixN>& U, const std::size_t B_START, const std::size_t NQUBITS_B, 
const std::size_t CLOCK_START, const std::size_t NQUBITS_CLOCK, Qureg qureg) {
  // QFT on clock qubits
  std::vector<int> targs(NQUBITS_CLOCK);
  std::iota(targs.begin(), targs.end(), CLOCK_START);
  applyQFT(qureg, targs.data(), NQUBITS_CLOCK);

  // apply controlled time evolution gates
  const std::size_t NCTRLS = 1;
  targs.resize(NQUBITS_B);
  std::iota(targs.begin(), targs.end(), B_START);
  std::size_t uid = 0;
  for (int ctrl = CLOCK_START; ctrl < CLOCK_START + NQUBITS_CLOCK; ++ctrl) {
    applyMultiControlledMatrixN(qureg, &ctrl, NCTRLS, targs.data(), NQUBITS_B, U.at(uid));
    ++uid;
  }

  // apply Hadamard gate to clock qubits
  for (std::size_t qid = CLOCK_START; qid < CLOCK_START + NQUBITS_CLOCK; ++qid) {
    hadamard(qureg, qid);
  }

  return;
}
 
void inverseQFT(const std::size_t START_QUBIT, const std::size_t NQUBITS, Qureg qureg) {
  const std::size_t END_QUBIT = START_QUBIT + NQUBITS - 1;
  double angle = 0.0;

  // swaps
  std::size_t qid1; 
  std::size_t qid2;
  for (qid1 = START_QUBIT + (NQUBITS / 2) - 1, qid2 = START_QUBIT + (NQUBITS / 2); qid1 >= START_QUBIT; --qid1, ++qid2) {
    swapGate(qureg, qid1, qid2);
  }

  // main iQFT
  for (std::size_t qid = START_QUBIT; qid <= END_QUBIT; ++qid) {
    for (std::size_t ctrl = START_QUBIT; ctrl < qid; ++ctrl) {
      std::size_t m = qid - ctrl;
      angle = M_PI / std::pow(2, m-1);
      controlledPhaseShift(qureg, ctrl, qid, angle);
    }
    hadamard(qureg, qid);
  }

  return;
}

void conditionalRotationY(const EigResult& EIGA, const std::size_t NQUBITS_B, const std::size_t NQUBITS_CLOCK, 
const std::size_t NQUBITS_ANCILLA, Qureg qureg) {
  const std::size_t NQUBITS = NQUBITS_B + NQUBITS_CLOCK + NQUBITS_ANCILLA;
  const std::size_t DIM = std::pow(2, NQUBITS);
  const std::size_t ANCILLA_ID = NQUBITS - 1;
 
  std::complex<double> angle;
  Eigen::Matrix2cd Ry;

  ComplexMatrix2 CM_Ry;

  // set up control vectors
  std::vector<int> ctrl_qubits(NQUBITS_CLOCK);
  std::iota(ctrl_qubits.begin(), ctrl_qubits.end(), NQUBITS_B);
  std::vector<std::vector<int>> ctrl_state(EIGA.eigenvalues().rows());
  ctrl_state.at(0) = {0, 1, 0, 0, 0, 0}; // 1
  ctrl_state.at(1) = {0, 1, 0, 0, 0, 1}; // -1
  ctrl_state.at(2) = {1, 1, 1, 0, 0, 0}; // 3.5
  ctrl_state.at(3) = {1, 1, 1, 0, 0, 1}; // -3.5
  ctrl_state.at(4) = {1, 1, 1, 1, 1, 0}; // 15.5
  ctrl_state.at(5) = {1, 1, 1, 1, 1, 1}; // -15.5

  for (std::size_t i = 0; i < EIGA.eigenvalues().rows(); ++i) {
    angle = 2.0 * std::asin(
      EIGA.eigenvalues()(0) / EIGA.eigenvalues()(i)
    );
    Ry = updateRy(angle);

    eigenToComplexMatrix2(Ry, CM_Ry);
    multiStateControlledUnitary(qureg, ctrl_qubits.data(), ctrl_state.at(i).data(), NQUBITS_CLOCK, ANCILLA_ID, CM_Ry);
  }

  return;
}

Eigen::Matrix2cd updateRy(const std::complex<double> ANGLE) {
  Eigen::Matrix2cd Ry;
  Ry(0,0) = std::cos(ANGLE/2.0);
  Ry(0,1) = -std::sin(ANGLE/2.0);
  Ry(1,0) = std::sin(ANGLE/2.0);
  Ry(1,1) = std::cos(ANGLE/2.0);

  return Ry;
}

void eigenToComplexMatrix2(const Eigen::Matrix2cd& E, ComplexMatrix2& U) {
  for (std::size_t row = 0; row < 2; ++row) {
    for (std::size_t col = 0; col < 2; ++col) {
      U.real[row][col] = E(row,col).real();
      U.imag[row][col] = E(row,col).imag();
    } 
  }

  return;
}