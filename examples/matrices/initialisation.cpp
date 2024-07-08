#include "quest.h"
#include <stdlib.h>
#include <vector>


void demo_getInlineCompMatr() {

    // inline literal; identical to getCompMatr1() for consistencty with C API
    CompMatr1 a = getInlineCompMatr1( {{1,2i},{3i+.1,-4}} );
    reportCompMatr1(a);

    // we must specify all elements (only necessary in C++)
    CompMatr2 b = getInlineCompMatr2({
        {1, 2, 3, 4},
        {5, 6, 8, 8},
        {5, 7, 6, 1},
        {4, 4, 3, 3}
    });
    reportCompMatr2(b);
}


void demo_getCompMatr() {

    // inline literal (C++ only)
    CompMatr1 a = getCompMatr1( {{1,2i},{3i+.1,-4}} );
    reportCompMatr1(a);

    // 2D vector (C++ only)
    std::vector<std::vector<qcomp>> vec {{-9,-8},{-7,-6}};
    CompMatr1 b = getCompMatr1(vec);
    reportCompMatr1(b);

    // 2D compile-time array
    qcomp arr[2][2] = {{5, 4},{3, 2}};
    CompMatr1 c = getCompMatr1(arr);
    reportCompMatr1(c);

    // nested pointers
    int dim = 4;
    qcomp** ptrs = (qcomp**) malloc(dim * sizeof(qcomp));
    for (int i=0; i<dim; i++) {
        ptrs[i] =(qcomp*) malloc(dim * sizeof(qcomp));
        for (int j=0; j<dim; j++)
            ptrs[i][j] = i + j*1i;
    }
    CompMatr2 d = getCompMatr2(ptrs);
    reportCompMatr2(d);

    // array of pointers
    qcomp* ptrArr[dim];
    for (int i=0; i<dim; i++)
        ptrArr[i] = ptrs[i];
    CompMatr2 e = getCompMatr2(ptrArr);
    reportCompMatr2(e);

    // cleanup
    for (int i=0; i<dim; i++)
        free(ptrs[i]);
    free(ptrs);
}


void demo_setInlineCompMatr() {

    // inline literal; identical to setCompMatrN() for consistencty with C API
    CompMatrN a = createCompMatrN(1);
    setInlineCompMatrN(a, 1, {{.3,.4},{.6,.7}});
    reportCompMatrN(a);
    destroyCompMatrN(a);

    // we must specify all elements (only necessary in C++)
    CompMatrN b = createCompMatrN(3);
    setInlineCompMatrN(b, 3, {
        {1,2,3,4,5,6,7,8},
        {8i,7i,6i,5i,0,0,0,0},
        {0,0,0,0,9,9,9,9},
        {8,7,6,5,4,3,2,1},
        {3,3,3,3,3,3,3,3},
        {0,0,0,0,1i,1i,1i,1i},
        {9,8,7,6,5,4,3,2},
        {-1,-2,-3,-4,4,3,2,1}    
    });
    reportCompMatrN(b);
    destroyCompMatrN(b);
}


void demo_setCompMatr() {

    // inline literal (C++ only)
    CompMatrN a = createCompMatrN(1);
    setCompMatrN(a, {{1,2i},{3i+.1,-4}});
    reportCompMatrN(a);

    // 2D vector (C++ only)
    std::vector<std::vector<qcomp>> vec {
        {-9,-8, -8, -9},
        {7, 7, 6, 6},
        {0, -1, -2, -3},
        {-4i, -5i, 0, 0}
    };
    CompMatrN b = createCompMatrN(2);
    setCompMatrN(b, vec);
    reportCompMatrN(b);

    // nested pointers
    int dim = 8;
    qcomp** ptrs = (qcomp**) malloc(dim * sizeof(qcomp));
    for (int i=0; i<dim; i++) {
        ptrs[i] = (qcomp*) malloc(dim * sizeof(qcomp));
        for (int j=0; j<dim; j++)
            ptrs[i][j] = i + j*1i;
    }
    CompMatrN c = createCompMatrN(3);
    setCompMatrN(c, ptrs);
    reportCompMatrN(c);
    destroyCompMatrN(c);

    // array of pointers
    qcomp* ptrArr[dim];
    for (int i=0; i<dim; i++)
        ptrArr[i] = ptrs[i];
    CompMatrN d = createCompMatrN(3);
    setCompMatrN(d, ptrArr);
    reportCompMatrN(d);
    destroyCompMatrN(d);

    // cleanup
    for (int i=0; i<dim; i++)
        free(ptrs[i]);
    free(ptrs);
}


int main() {
    
    initQuESTEnv();

    demo_getInlineCompMatr();
    demo_getCompMatr();
    demo_setInlineCompMatr();
    demo_setCompMatr();

    finalizeQuESTEnv();
    return 0;
}