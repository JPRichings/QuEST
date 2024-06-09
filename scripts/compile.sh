# A Unix bash script which compiles QuEST alongside user
# source code into a single executable. This script should
# be run in the same directory containing the quest/ direc.
# 
# @author Tyson Jones



# USER SETTINGS

# numerical precision (1, 2, 4)
FLOAT_PRECISION=2

# deployment mode (0, 1)
ENABLE_DISTRIBUTION=0
ENABLE_MULTITHREADING=0
ENABLE_GPU_ACCELERATION=0

# deployment params
GPU_CC=90

# backend compilers
BASE_COMPILER=g++
OMP_COMPILER=g++
MPI_COMPILER=mpic++
GPU_COMPILER=nvcc

# linker
LINKER=g++

# the type of OMP_COMPILER (CLANG or other)
OMP_COMPILER_TYPE=CLANG

# name of the compiled executable
EXEC_FILE="main"

# user files (.c or .cpp, intermixed)
USER_FILES=(
    "main.c"
)

# location of user files
USER_DIR="."

# user code language-specific compilers
USER_C_COMPILER=gcc
USER_CXX_COMPILER=g++

# user code language-specific compiler flags
USER_C_COMP_FLAGS='-std=c99'
USER_CXX_COMP_FLAGS='-std=c++14'

# user linker flags
USER_LINK_FLAGS='-lstdc++' 



# CONSTANTS

USER_OBJ_PREF="user_"
QUEST_OBJ_PREF="quest_"

INDENT='  '



# QUEST FILE LAYOUT

# directories
ROOT_DIR='quest'
INCLUDE_DIR="${ROOT_DIR}/include"
SRC_DIR="${ROOT_DIR}/src"
API_DIR="${SRC_DIR}/api"
OMP_DIR="${SRC_DIR}/cpu"
GPU_DIR="${SRC_DIR}/gpu"
MPI_DIR="${SRC_DIR}/comm"
CORE_DIR="${SRC_DIR}/core"

# files in CORE_DIR
CORE_FILES=(
    "accelerator"
    "errors"
    "localiser"
    "memory"
    "utilities"
    "validation"
)

# files in API_DIR
API_FILES=(
    "calculations"
    "debug"
    "decoherence"
    "environment"
    "initialisations"
    "maths"
    "modes"
    "operations"
    "qasm"
    "qureg"
    "structures"
)

# files in GPU_DIR
GPU_FILES=(
    "config"
    "cuda_subroutines"
)

# files in OMP_DIR
OMP_FILES=(
    "omp_subroutines"
)

# files in MPI_DIR
MPI_FILES=(
    "communication"
)



# COMPILER AND LINKER FLAG OPTIONS

echo "deployment modes:"

# compiler flags given to all backend files (but not user files)
BACKEND_COMP_FLAGS='-std=c++14 -O3'

# GPU-specific flags
GPU_COMP_FLAGS="-x cu -arch=sm_${GPU_CC}"
GPU_LINK_FLAGS="-lcudart -lcuda"

# MPI-specific flags
MPI_COMP_FLAGS=''
MPI_LINK_FLAGS='-lmpi'

# OMP-specific flags (form depends on OMP compiler type)
if [ "$OMP_COMPILER_TYPE" == "CLANG" ]
then
    OMP_COMP_FLAGS='-Xclang -fopenmp'
    OMP_LINK_FLAGS='-lomp'
else
    OMP_COMP_FLAGS='-fopenmp'
    OMP_LINK_FLAGS='-fopenmp'
fi

# define pre-processor macros to indicate deployment mode
MODE_FLAGS="-DENABLE_DISTRIBUTION=${ENABLE_DISTRIBUTION} "
MODE_FLAGS+="-DENABLE_MULTITHREADING=${ENABLE_MULTITHREADING} "
MODE_FLAGS+="-DENABLE_GPU_ACCELERATION=${ENABLE_GPU_ACCELERATION}"

# define pre-processor macros to set qcomp precision
PREC_FLAG="-DFLOAT_PRECISION=${FLOAT_PRECISION}"

# point compilers to QuEST src
HEADER_FLAGS="-I. -I${INCLUDE_DIR}"



# ASSEMBLE FLAGS

# flags given to every compilation unit
GLOBAL_COMP_FLAGS="${HEADER_FLAGS} ${MODE_FLAGS} ${PREC_FLAG}"

# choose linker flags (extended below)
ALL_LINK_FLAGS="${USER_LINK_FLAGS}"

# choose compiler and flags for CPU/OMP files
if [ $ENABLE_MULTITHREADING == 1 ]
then
    echo "${INDENT}(multithreading enabled)"
    CPU_FILES_COMPILER=$OMP_COMPILER
    CPU_FILES_FLAGS=$OMP_COMP_FLAGS
    ALL_LINK_FLAGS+=" ${OMP_LINK_FLAGS}"
else
    echo "${INDENT}(multithreading disabled)"
    CPU_FILES_COMPILER=$BASE_COMPILER
    CPU_FILES_FLAGS=''
fi

# choose compiler and flags for GPU files
if [ $ENABLE_GPU_ACCELERATION == 1 ]
then
    echo "${INDENT}(GPU-acceleration enabled)"
    GPU_FILES_COMPILER=$GPU_COMPILER
    GPU_FILES_FLAGS=$GPU_COMP_FLAGS
    ALL_LINK_FLAGS+=" ${GPU_LINK_FLAGS}"
else
    echo "${INDENT}(GPU-acceleration disabled)"
    GPU_FILES_COMPILER=$BASE_COMPILER
    GPU_FILES_FLAGS=''
fi

# choose compiler and flags for communication files
if [ $ENABLE_DISTRIBUTION == 1 ]
then
    echo "${INDENT}(distribution enabled)"
    MPI_FILES_COMPILER=$MPI_COMPILER
    MPI_FILES_FLAGS=$MPI_COMP_FLAGS
    ALL_LINK_FLAGS+=" ${MPI_LINK_FLAGS}"
else
    echo "${INDENT}(distribution disabled)"
    MPI_FILES_COMPILER=$BASE_COMPILER
    MPI_FILES_FLAGS=''
fi

# display precision
if [ $FLOAT_PRECISION == 1 ]; then
    echo "${INDENT}(single precision)"
elif [ $FLOAT_PRECISION == 2 ]; then
    echo "${INDENT}(double precision)"
elif [ $FLOAT_PRECISION == 4 ]; then
    echo "${INDENT}(quad precision)"
else
    echo ""
    echo "INVALID FLOAT_PRECISION (${FLOAT_PRECISION})"
    echo "Exiting..."
    exit
fi

echo ""



# REPORTING COMILERS FLAGS


echo "chosen compilers and flags..."

# user compilers
echo "${INDENT}user compilers and flags (for .c and .cpp files respectively):"
echo "${INDENT}${INDENT}${USER_C_COMPILER} ${USER_C_COMP_FLAGS}"
echo "${INDENT}${INDENT}${USER_CXX_COMPILER} ${USER_CXX_COMP_FLAGS}"

# base compiler
echo "${INDENT}base files compiler and backend flags:"
echo "${INDENT}${INDENT}${BASE_COMPILER} ${BACKEND_COMP_FLAGS}"

# OMP/CPU
echo "${INDENT}CPU files compiler and flags:"
echo "${INDENT}${INDENT}${CPU_FILES_COMPILER} ${CPU_FILES_FLAGS}"

# GPU
echo "${INDENT}GPU files compiler and flags:"
echo "${INDENT}${INDENT}${GPU_FILES_COMPILER} ${GPU_FILES_FLAGS}"

# MPI
echo "${INDENT}distributed files compiler and flags:"
echo "${INDENT}${INDENT}${MPI_FILES_COMPILER} ${MPI_FILES_FLAGS}"

# linker 
echo "${INDENT}linker and all linker flags:"
echo "${INDENT}${INDENT}${LINKER} ${ALL_LINK_FLAGS}"

# globals
echo "${INDENT}header flags:"
echo "${INDENT}${INDENT}${HEADER_FLAGS}"

echo ""



# COMPILING USER CODE

# abort script if any compilation fails
set -e

echo "compiling user files:"

for fn in ${USER_FILES[@]}
do
    # choose C or C++ compiler for each user file
    if [[ $fn == *.cpp ]]
    then
        echo "${INDENT}${fn} (C++) ..."
        COMP=$USER_CXX_COMPILER
        FLAG=$USER_CXX_COMP_FLAGS
    else
        echo "${INDENT}${fn} (C) ..."
        COMP=$USER_C_COMPILER
        FLAG=$USER_C_COMP_FLAGS
    fi

    # compile
    $COMP -c $USER_DIR/$fn -o ${USER_OBJ_PREF}${fn}.o $FLAG $GLOBAL_COMP_FLAGS
done

echo ""



# COMPILING CORE

echo "compiling core files in C++"

for fn in ${CORE_FILES[@]}
do
    echo "${INDENT}${fn}.cpp ..."
    $BASE_COMPILER -c $CORE_DIR/$fn.cpp -o ${QUEST_OBJ_PREF}${fn}.o $BACKEND_COMP_FLAGS $GLOBAL_COMP_FLAGS
done

echo ""



# COMPILING API

echo "compiling API files in C++:"

for fn in ${API_FILES[@]}
do
    echo "${INDENT}${fn}.cpp ..."
    $BASE_COMPILER -c $API_DIR/$fn.cpp -o ${QUEST_OBJ_PREF}${fn}.o $BACKEND_COMP_FLAGS $GLOBAL_COMP_FLAGS
done

echo ""



# COMPILING OMP

echo "compiling CPU/OMP files..."

for fn in ${OMP_FILES[@]}; do
    echo "${INDENT}${fn}.cpp ..."
    $CPU_FILES_COMPILER -c $OMP_DIR/$fn.cpp -o ${QUEST_OBJ_PREF}${fn}.o $CPU_FILES_FLAGS $BACKEND_COMP_FLAGS $GLOBAL_COMP_FLAGS
done

echo ""



# COMPILING GPU

echo "compiling GPU files..."

for fn in ${GPU_FILES[@]}; do
    echo "${INDENT}${fn}.cpp ..."
    $GPU_FILES_COMPILER -c $GPU_DIR/$fn.cpp -o ${QUEST_OBJ_PREF}${fn}.o $GPU_FILES_FLAGS $BACKEND_COMP_FLAGS $GLOBAL_COMP_FLAGS
done

echo ""



# COMPILING MPI

echo "compiling communication/MPI files..."

for fn in ${MPI_FILES[@]}; do
    echo "${INDENT}${fn}.cpp ..."
    $MPI_FILES_COMPILER -c $MPI_DIR/$fn.cpp -o ${QUEST_OBJ_PREF}${fn}.o $MPI_FILES_FLAGS $BACKEND_COMP_FLAGS $GLOBAL_COMP_FLAGS
done

echo ""



# LINKING

echo "linking all files..."

# collect list of all objects
OBJECTS=""
OBJECTS+=" $(printf " ${USER_OBJ_PREF}%s.o" "${USER_FILES[@]}")"
OBJECTS+=" $(printf " ${QUEST_OBJ_PREF}%s.o" "${CORE_FILES[@]}")"
OBJECTS+=" $(printf " ${QUEST_OBJ_PREF}%s.o" "${API_FILES[@]}")"
OBJECTS+=" $(printf " ${QUEST_OBJ_PREF}%s.o" "${GPU_FILES[@]}")"
OBJECTS+=" $(printf " ${QUEST_OBJ_PREF}%s.o" "${OMP_FILES[@]}")"
OBJECTS+=" $(printf " ${QUEST_OBJ_PREF}%s.o" "${MPI_FILES[@]}")"

# link all objects
$LINKER $OBJECTS $ALL_LINK_FLAGS -o $EXEC_FILE

echo "${INDENT}compiled executable ${EXEC_FILE}"
echo ""



# CLEAN UP

echo "deleting temporary object files..."

rm *.o

echo "${INDENT}done"
echo ""