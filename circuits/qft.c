#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "QuEST.h"

double calcPhaseShift(const int M) {
  return  ( M_PI / pow(2, (M-1)) );
}

void qftQubit(Qureg qureg, const int NUM_QUBITS, const int QUBIT_ID) {
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
    qftQubit(qureg, NUM_QUBITS, qid);
  return;
}

void writeState(const int * const STATE, const size_t NUM_QUBITS) {
  printf("|");
  for (size_t n = 0; n < NUM_QUBITS; ++n) printf("%d", STATE[n]);
  printf(">\n");
  return;
}

int main (void) {
  const unsigned int NUM_QUBITS = 4;
  
  printf("Simulating %d-Qubit QFT\n\n", NUM_QUBITS);

  // Initialise QuEST
  QuESTEnv quenv = createQuESTEnv();

  // create quantum register
  Qureg qureg = createQureg(NUM_QUBITS, quenv);

  // initialise input register to |0..0>
  initZeroState(qureg);

  // report model
  reportQuregParams(qureg);
  printf("\n");
  reportQuESTEnv(quenv);
  printf("\n");

  // apply QFT to input register
  qft(qureg, NUM_QUBITS);

  printf("Total number of gates: %d\n", (NUM_QUBITS * (NUM_QUBITS+1))/2 );

  // results
  qreal prob;
  prob = getProbAmp(qureg, 0);
  printf("Measured probability amplitude of |0..0> state: %g\n", prob);
  printf("Calculated probability amplitude of |0..0>, C0 = 1 / 2^%d: %g\n",
    NUM_QUBITS, 1.0 / pow(2,NUM_QUBITS));

  printf("Measuring final state: (all probabilities should be 0.5)\n");
  int outcome;
  int * state = (int *) malloc(NUM_QUBITS * sizeof(int));
  for (int n = 0; n < NUM_QUBITS; ++n) {
    outcome = measureWithStats(qureg, n, &prob);
    state[n] = outcome;
    printf("Qubit %d measured in state %d with probability %g\n",
      n, outcome, prob);
  }
  printf("\n");
  printf("Final state:\n");
  writeState(state, NUM_QUBITS);

  // Finalise QuEST
  free(state);
  destroyQureg(qureg, quenv);
  destroyQuESTEnv(quenv);
  
  return 0;
}