#include "QuEST.h"
#include <eigen3/Eigen/Dense>

template <typename DerivedA, typename Derivedb> 
Eigen::VectorXd solveClassically(const Eigen::MatrixBase<DerivedA>& A, const Eigen::MatrixBase<Derivedb>& b) {
  Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
  return x;
}