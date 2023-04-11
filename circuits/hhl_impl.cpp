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
  std::size_t uid = 0;
  for (int ctrl = CLOCK_START + NQUBITS_CLOCK - 1; ctrl >= CLOCK_START; --ctrl) {
    applyMultiControlledMatrixN(qureg, &ctrl, NCTRLS, targs.data(), NQUBITS_B, U.at(uid));
    ++uid;
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
  std::size_t uid = NQUBITS_CLOCK - 1;
  for (int ctrl = CLOCK_START; ctrl < CLOCK_START + NQUBITS_CLOCK; ++ctrl) {
    applyMultiControlledMatrixN(qureg, &ctrl, NCTRLS, targs.data(), NQUBITS_B, U.at(uid));
    --uid;
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
  for (qid1 = START_QUBIT + (NQUBITS / 2) - 1, qid2 = START_QUBIT + (NQUBITS / 2) + 1; qid1 >= START_QUBIT; --qid1, ++qid2) {
    swapGate(qureg, qid1, qid2);
  }

  // main iQFT
  for (std::size_t qid = START_QUBIT; qid <= END_QUBIT; ++qid) {
    for (std::size_t ctrl = START_QUBIT; ctrl < qid; ++ctrl) {
      std::size_t m = ctrl - qid;
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
  const Eigen::MatrixXcd IB = Eigen::MatrixXcd::Identity(std::pow(2, NQUBITS_B), std::pow(2, NQUBITS_B));
  const Eigen::MatrixXcd IC = Eigen::MatrixXcd::Identity(std::pow(2, NQUBITS_CLOCK), std::pow(2, NQUBITS_CLOCK));
  const Eigen::Matrix2cd IA = Eigen::Matrix2cd::Identity();
  Eigen::MatrixXcd op = Eigen::MatrixXcd::Zero(DIM, DIM); 
  Eigen::MatrixXcd m = Eigen::MatrixXcd::Zero(std::pow(2, NQUBITS_CLOCK), std::pow(2, NQUBITS_CLOCK)); 

  ComplexMatrixN U = createComplexMatrixN(NQUBITS);
  std::vector<int> targs(NQUBITS); 
  std::iota(targs.begin(), targs.end(), 0);

  // set up measurement vectors
  std::vector<Eigen::VectorXcd> measurements(EIGA.eigenvalues().rows());
  for (auto& vec : measurements) {
    vec = Eigen::VectorXcd::Zero(std::pow(2,NQUBITS_CLOCK));
  }
  measurements.at(0)(0b010000) = 1; // 1    
  measurements.at(1)(0b010001) = 1; // -1
  measurements.at(2)(0b111000) = 1; // 3.5
  measurements.at(3)(0b111001) = 1; // -3.5
  measurements.at(4)(0b111110) = 1; // 15.5
  measurements.at(5)(0b111111) = 1; // -15.5

  for (std::size_t i = 0; i < EIGA.eigenvalues().rows(); ++i) {
    angle = 2.0 * std::asin(
      EIGA.eigenvalues()(0) / EIGA.eigenvalues()(i)
    );
    Ry = updateRy(angle);
    m = measurements.at(i) * measurements.at(i).transpose();
    op = Eigen::KroneckerProduct<Eigen::MatrixX2cd, Eigen::MatrixXcd>(
        Ry, Eigen::KroneckerProduct<Eigen::MatrixXcd, Eigen::MatrixXcd>(m, IB)
      );
    op += Eigen::KroneckerProduct<Eigen::MatrixX2cd, Eigen::MatrixXcd>(
        IA, Eigen::KroneckerProduct<Eigen::MatrixXcd, Eigen::MatrixXcd>(IC-m, IB)
    );

    eigenToComplexMatrixN(op, U);
    applyMatrixN(qureg, targs.data(), NQUBITS, U);
  }

  destroyComplexMatrixN(U);
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

void eigenToComplexMatrixN(const Eigen::MatrixXcd& E, ComplexMatrixN& U) {
  const std::size_t DIM = std::pow(2, U.numQubits);

  if (DIM != E.rows() || DIM != E.cols()) {
    std::cerr << "ComplexMatrixN and Eigen Matrix are not the same size!" << std::endl;
    return;
  }

  for (std::size_t row = 0; row < DIM; ++row) {
    for (std::size_t col = 0; col < DIM; ++col) {
      U.real[row][col] = E(row,col).real();
      U.imag[row][col] = E(row,col).imag();
    } 
  }

  return;
}