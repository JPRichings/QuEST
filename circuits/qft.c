#include <stdio.h>
#include <math.h>
#include "QuEST.h"

double calcPhaseShift(const int M) {
  return  ( M_PI / pow(2, (M-1)) );
}

void qft_qubit(Qureg qureg, const int NUM_QUBITS, const int QUBIT_ID) {
  int control_id = 0;
  double angle = 0.0;
  
  hadamard(qureg, QUBIT_ID);
  int m = 2;
  for (int control = QUBIT_ID+1; control < NUM_QUBITS; ++control) {
    angle = calcPhaseShift(m++);
    controlledPhaseShift(qureg, control, QUBIT_ID, angle);
  }

  return;
}

void qft(Qureg qureg, const int NUM_QUBITS) {
  for (int qid = 0; qid < NUM_QUBITS; ++qid) 
    qft_qubit(qureg, NUM_QUBITS, qid);
  return;
}

int main (void) {
  const unsigned int NUM_QUBITS = 4;
  
  // Initialise QuEST
  QuESTEnv quenv = createQuESTEnv();

  // create quantum register
  Qureg qureg = createQureg(NUM_QUBITS, quenv);

  // initialise input register to |0..0>
  initZeroState(qureg);

  // report model
  reportQuregParams(qureg);
  reportQuESTEnv(quenv);

  // apply QFT to input register
  qft(qureg, NUM_QUBITS);

  printf("Total number of gates: %d\n", (NUM_QUBITS * (NUM_QUBITS+1))/2 );

  // results
  qreal prob_0 = getProbAmp(qureg, 0);
  printf("Measured probability amplitude of |0..0> state: %g\n", prob_0);
  printf("Calculated probability amplitude of |0..0>, C0 = 1 / 2^%d: %g\n",
    NUM_QUBITS, 1.0 / pow(2,NUM_QUBITS));

  // Finalise QuEST
  destroyQureg(qureg, quenv);
  destroyQuESTEnv(quenv);
  
  return 0;
}