#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "QuEST.h"

void hadamardBench(Qureg, const unsigned int, const unsigned int);
void getMonoTime(struct timespec *);
double getElapsedSeconds(const struct timespec * const,
  const struct timespec * const);

int main (int argc, char* argv[])
{
  int tmp;
  unsigned int num_qubits, reps;

  if (argc < 3) {
    printf("Error: Not enough arguments! Usage: ./db $NUMBER_OF_QUBITS $NUMBER_OF_REPETITIONS\n");
    return -1;
  } else if (argc > 3) {
    printf("Error: Too many arguments! Usage: ./db $NUMBER_OF_QUBITS $NUMBER_OF_REPETITIONS\n");
    return -1;
  } else {
    // arcg == 3
    tmp = atoi(argv[1]);
    if (tmp < 1) {
      printf("Error: num_qubits < 1, you requested %d qubits\n", tmp);
      return -1;
    }
    if (tmp > 48) {
      printf("Error: num_qubits > 48, you requested %d qubits. If you have 5 petabytes of RAM, you may be able to remove this constraint.\n",
      tmp);
      return -1;
    }
    num_qubits = (unsigned int) tmp;

    tmp = atoi(argv[2]);
    if (tmp < 1) {
      printf("Error: reps < 1, must be a completely positive number\n");
      return -1;
    }
    reps = (unsigned int) tmp;
  }

  const unsigned int N_GATES = reps * num_qubits;

  struct timespec run_start, run_stop;
  struct timespec db_start, db_stop;
  double run_time, db_time, gate_time;

  getMonoTime(&run_start);

  QuESTEnv quenv = createQuESTEnv();

  Qureg qureg = createQureg(num_qubits, quenv);

  if (!quenv.rank)
    printf("Simulating %d-Qubit depth benchmark\n\n", num_qubits);

  initZeroState(qureg);

  if (!quenv.rank) {
    reportQuregParams(qureg);
    printf("\n");
    reportQuESTEnv(quenv);
    printf("\n");
    printf("Running hadamard benchmark:\n");
    printf("  %d qubits\n", num_qubits);
    printf("  1 Hadamard gate per qubit\n");
    printf("  %d repetitions\n", reps);
    printf("TOTAL: %d gates\n", N_GATES);
  }

  getMonoTime(&db_start);
  hadamardBench(qureg, num_qubits, reps);
  getMonoTime(&db_stop);

  destroyQureg(qureg, quenv);

  getMonoTime(&run_stop);

  // results
  run_time = getElapsedSeconds(&run_start, &run_stop);
  db_time = getElapsedSeconds(&db_start, &db_stop);
  gate_time = db_time / N_GATES;

  if (!quenv.rank) {
    printf("Results:\n");
    printf("  Run Time = %g s\n", run_time);
    printf("  Circuit Time = %g s\n", db_time);
    printf("  Time per gate = %g s\n", gate_time);
  }

  destroyQuESTEnv(quenv);

  return 0;
}

void hadamardBench(Qureg qureg, const unsigned int N_QUBITS,
const unsigned int N_REPS) {
  for (unsigned int rep = 0; rep < N_REPS; ++rep) {
    for (unsigned int qubit_id = 0; qubit_id < N_QUBITS; ++qubit_id) {
      hadamard(qureg, qubit_id);
    }
  }
  return;
}

void getMonoTime(struct timespec * time) {
  clock_gettime(CLOCK_MONOTONIC, time);
  return;
}

double getElapsedSeconds(const struct timespec * const start,
const struct timespec * const stop) {
  const unsigned long BILLION = 1000000000UL;
  const unsigned long long TOTAL_NS = 
    BILLION * (stop->tv_sec - start->tv_sec)
    + (stop->tv_nsec - start->tv_nsec);

  return (double) TOTAL_NS / BILLION;
}