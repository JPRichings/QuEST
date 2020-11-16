#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "QuEST.h"

double calcPhaseShif(const int);
void qftQubit(Qureg, const int, const int);
void qft(Qureg, const int);
void writeState(const int * const, const size_t);

int main (int argc, char *argv[]) 
{
  int num_qubits;

  if (argc < 2) {
    num_qubits = 4;
  } else if (argc > 2) {
    printf("Error: Too many arguments! Usage: ./qft $NUMBER_OF_QUBITS\n");
    return -1;
  } else {
    // arcg == 2
    num_qubits = atoi(argv[1]);
    if (num_qubits < 1) {
      printf("Error: num_qubits < 1, you requested %d qubits\n", num_qubits);
      return -1;
    }
  }

  // initialise QuEST
  QuESTEnv quenv = createQuESTEnv();

  // create quantum register
  Qureg qureg = createQureg(num_qubits, quenv);

  if (!quenv.rank)
    printf("Simulating %d-Qubit QFT\n\n", num_qubits);

  // initialise input register to |0..0>
  initZeroState(qureg);

  // report model
  if (!quenv.rank) {
    reportQuregParams(qureg);
    printf("\n");
    reportQuESTEnv(quenv);
    printf("\n");
  }

  // apply QFT to input register
  qft(qureg, num_qubits);

  if (!quenv.rank)
    printf("Total number of gates: %d\n", (num_qubits * (num_qubits+1))/2 );

  // results
  qreal prob_0 = getProbAmp(qureg, 0);
  if (!quenv.rank) {
    printf("Measured probability amplitude of |0..0> state: %g\n", prob_0);
    printf("Calculated probability amplitude of |0..0>, C0 = 1 / 2^%d: %g\n",
      num_qubits, 1.0 / pow(2,num_qubits));

    printf("Measuring final state: (all probabilities should be 0.5)\n");
  }

  int * state = (int *) malloc(num_qubits * sizeof(int));
  qreal * probs = (qreal *) malloc(num_qubits * sizeof(qreal));
  for (int n = 0; n < num_qubits; ++n) {
    state[n] = measureWithStats(qureg, n, probs+n);
  }
  if (!quenv.rank) {
    for (int n = 0; n < num_qubits; ++n)
      printf("Qubit %d measured in state %d with probability %g\n",
        n, state[n], probs[n]);
    printf("\n");
    printf("Final state:\n");
    writeState(state, num_qubits);
  }

  // free resources
  free(state);
  free(probs);
  destroyQureg(qureg, quenv);
  destroyQuESTEnv(quenv);
  
  return 0;
}

double calcPhaseShift(const int M) {
  return  ( M_PI / pow(2, (M-1)) );
}

void qftQubit(Qureg qureg, const int num_qubits, const int QUBIT_ID) {
  int control_id = 0;
  double angle = 0.0;
  
  hadamard(qureg, QUBIT_ID);
  int m = 2;
  for (int control = QUBIT_ID+1; control < num_qubits; ++control) {
    angle = calcPhaseShift(m++);
    controlledPhaseShift(qureg, control, QUBIT_ID, angle);
  }

  return;
}

void qft(Qureg qureg, const int num_qubits) {
  for (int qid = 0; qid < num_qubits; ++qid) 
    qftQubit(qureg, num_qubits, qid);
  return;
}

void writeState(const int * const STATE, const size_t num_qubits) {
  printf("|");
  for (size_t n = 0; n < num_qubits; ++n) printf("%d", STATE[n]);
  printf(">\n");
  return;
}