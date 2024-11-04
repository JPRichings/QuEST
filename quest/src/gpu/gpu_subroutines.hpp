/** @file
 * CUDA GPU-accelerated signatures of the subroutines called by accelerator.cpp.
 */

#ifndef GPU_SUBROUTINES_HPP
#define GPU_SUBROUTINES_HPP

#include "quest/include/types.h"
#include "quest/include/qureg.h"
#include "quest/include/paulis.h"
#include "quest/include/matrices.h"

#include <vector>

using std::vector;


/*
 * GETTERS
 */

qcomp gpu_statevec_getAmp_sub(Qureg qureg, qindex ind);


/*
 * COMMUNICATION BUFFER PACKING
 */

template <int NumQubits> qindex gpu_statevec_packAmpsIntoBuffer(Qureg qureg, vector<int> qubits, vector<int> qubitStates);

qindex gpu_statevec_packPairSummedAmpsIntoBuffer(Qureg qureg, int qubit1, int qubit2, int qubit3, int bit2);


/*
 * SWAPS
 */

template <int NumCtrls> void gpu_statevec_anyCtrlSwap_subA(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, int targ1, int targ2);
template <int NumCtrls> void gpu_statevec_anyCtrlSwap_subB(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates);
template <int NumCtrls> void gpu_statevec_anyCtrlSwap_subC(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, int targ, int targState);


/*
 * DENSE MATRIX
 */

template <int NumCtrls> void gpu_statevec_anyCtrlOneTargDenseMatr_subA(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, int targ, CompMatr1 matr);
template <int NumCtrls> void gpu_statevec_anyCtrlOneTargDenseMatr_subB(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, qcomp fac0, qcomp fac1);

template <int NumCtrls> void gpu_statevec_anyCtrlTwoTargDenseMatr_sub(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, int targ1, int targ2, CompMatr2 matr);

template <int NumCtrls, int NumTargs, bool ApplyConj> void gpu_statevec_anyCtrlAnyTargDenseMatr_sub(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, vector<int> targs, CompMatr matr);


/*
 * DIAGONAL MATRIX
 */

template <int NumCtrls> void gpu_statevec_anyCtrlOneTargDiagMatr_sub(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, int targ, DiagMatr1 matr);

template <int NumCtrls> void gpu_statevec_anyCtrlTwoTargDiagMatr_sub(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, int targ1, int targ2, DiagMatr2 matr);

template <int NumCtrls, int NumTargs, bool ApplyConj, bool HasPower> void gpu_statevec_anyCtrlAnyTargDiagMatr_sub(Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, vector<int> targs, DiagMatr matr, qcomp exponent);

template <bool HasPower> void gpu_statevec_allTargDiagMatr_sub(Qureg qureg, FullStateDiagMatr matr, qcomp exponent);

template <bool HasPower, bool MultiplyOnly> void gpu_densmatr_allTargDiagMatr_sub(Qureg qureg, FullStateDiagMatr matr, qcomp exponent);



/*
 * PAULI TENSOR AND GADGET
 */

template <int NumCtrls, int NumTargs> void gpu_statevector_anyCtrlPauliTensorOrGadget_subA(
    Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, vector<int> suffixTargsXY, 
    qindex suffixMaskXY, qindex allMaskYZ, qcomp powI, qcomp fac0, qcomp fac1
);

template <int NumCtrls> void gpu_statevector_anyCtrlPauliTensorOrGadget_subB(
    Qureg qureg, vector<int> ctrls, vector<int> ctrlStates,
    qindex suffixMaskXY, qindex bufferMaskXY, qindex allMaskYZ, qcomp powI, qcomp fac0, qcomp fac1
);

template <int NumCtrls> void gpu_statevector_anyCtrlAnyTargZOrPhaseGadget_sub(
    Qureg qureg, vector<int> ctrls, vector<int> ctrlStates, vector<int> targs, 
    qcomp fac0, qcomp fac1
);



/*
 * QUREG COMBINATION
 */

void gpu_statevec_setQuregToSuperposition_sub(qcomp facOut, Qureg outQureg, qcomp fac1, Qureg inQureg1, qcomp fac2, Qureg inQureg2);

void gpu_densmatr_mixQureg_subA(qreal outProb, Qureg outQureg, qreal inProb, Qureg inDensMatr);
void gpu_densmatr_mixQureg_subB(qreal outProb, Qureg outQureg, qreal inProb, Qureg inStateVec);
void gpu_densmatr_mixQureg_subC(qreal outProb, Qureg outQureg, qreal inProb);


/*
 * DECOHERENCE
 */

void gpu_densmatr_oneQubitDephasing_subA(Qureg qureg, int qubit, qreal prob);
void gpu_densmatr_oneQubitDephasing_subB(Qureg qureg, int qubit, qreal prob);

void gpu_densmatr_twoQubitDephasing_subA(Qureg qureg, int qubitA, int qubitB, qreal prob);
void gpu_densmatr_twoQubitDephasing_subB(Qureg qureg, int qubitA, int qubitB, qreal prob);

void gpu_densmatr_oneQubitDepolarising_subA(Qureg qureg, int qubit, qreal prob);
void gpu_densmatr_oneQubitDepolarising_subB(Qureg qureg, int qubit, qreal prob);

void gpu_densmatr_twoQubitDepolarising_subA(Qureg qureg, int qubit1, int qubit2, qreal prob);
void gpu_densmatr_twoQubitDepolarising_subB(Qureg qureg, int qubit1, int qubit2, qreal prob);
void gpu_densmatr_twoQubitDepolarising_subC(Qureg qureg, int qubit1, int qubit2, qreal prob);
void gpu_densmatr_twoQubitDepolarising_subD(Qureg qureg, int qubit1, int qubit2, qreal prob);
void gpu_densmatr_twoQubitDepolarising_subE(Qureg qureg, int qubit1, int qubit2, qreal prob);
void gpu_densmatr_twoQubitDepolarising_subF(Qureg qureg, int qubit1, int qubit2, qreal prob);
void gpu_densmatr_twoQubitDepolarising_subG(Qureg qureg, int qubit1, int qubit2, qreal prob);

void gpu_densmatr_oneQubitPauliChannel_subA(Qureg qureg, int ketQubit, qreal pI, qreal pX, qreal pY, qreal pZ);
void gpu_densmatr_oneQubitPauliChannel_subB(Qureg qureg, int ketQubit, qreal pI, qreal pX, qreal pY, qreal pZ);

void gpu_densmatr_oneQubitDamping_subA(Qureg qureg, int qubit, qreal prob);
void gpu_densmatr_oneQubitDamping_subB(Qureg qureg, int qubit, qreal prob);
void gpu_densmatr_oneQubitDamping_subC(Qureg qureg, int qubit, qreal prob);
void gpu_densmatr_oneQubitDamping_subD(Qureg qureg, int qubit, qreal prob);


/*
 * PARTIAL TRACE
 */

template <int NumTargs> void gpu_densmatr_partialTrace_sub(Qureg inQureg, Qureg outQureg, vector<int> targs, vector<int> pairTargs);


/*
 * PROBABILITIES
 */

qreal gpu_statevec_calcTotalProb_sub(Qureg qureg);
qreal gpu_densmatr_calcTotalProb_sub(Qureg qureg);

template <int NumQubits> qreal gpu_statevec_calcProbOfMultiQubitOutcome_sub(Qureg qureg, vector<int> qubits, vector<int> outcomes);
template <int NumQubits> qreal gpu_densmatr_calcProbOfMultiQubitOutcome_sub(Qureg qureg, vector<int> qubits, vector<int> outcomes);

template <int NumQubits> void gpu_statevec_calcProbsOfAllMultiQubitOutcomes_sub(qreal* outProbs, Qureg qureg, vector<int> qubits);
template <int NumQubits> void gpu_densmatr_calcProbsOfAllMultiQubitOutcomes_sub(qreal* outProbs, Qureg qureg, vector<int> qubits);


/*
 * PROJECTORS
 */

template <int NumQubits> void gpu_statevec_multiQubitProjector_sub(Qureg qureg, vector<int> qubits, vector<int> outcomes, qreal prob);
template <int NumQubits> void gpu_densmatr_multiQubitProjector_sub(Qureg qureg, vector<int> qubits, vector<int> outcomes, qreal prob);



#endif // GPU_SUBROUTINES_HPP