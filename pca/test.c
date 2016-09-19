/* Doesn't always work: 
    cc -std=c99 -o testlapack test.c ; ./testlapack
 sometimes fails with a segmentation fault
 to compile with gcc, use
   LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH gcc -std=c99 -o testlapack test.c -I$CRAY_LIBSCI_PREFIX_DIR/include -L$CRAY_LIBSCI_PREFIX_DIR/lib -l:libsci_intel.so
 not sure if this always works or not
 */

#include <stdio.h>
#include <stdlib.h>
#include "lapacke.h"

void main(int argc, char ** argv) {
    int numrows = 7000000;
    int numcols = 6;
    int numeigs = 100;

    double * AV = (double *) malloc( numrows * numeigs * sizeof(double));
    if (AV == NULL) {
        fprintf(stderr, "Error allocating AV\n");
        exit(-1);
    }

    for(int r = 0; r < numrows; r = r + 1) {
        for(int c = 0; c < numeigs; c = c + 1) {
            AV[numcols*r + c] = (r+1)/((double)(c+1));
        }
    }

    double * U = (double *) malloc( numrows * numeigs * sizeof(double));
    double * VT = (double *) malloc( numeigs * numeigs * sizeof(double));
    double * V = (double *) malloc( numeigs * numeigs * sizeof(double));
    double * finalV = (double *) malloc( numcols * numeigs * sizeof(double));
    double * singvals = (double *) malloc( numeigs * sizeof(double));
    if (U == NULL || VT == NULL || V == NULL || finalV == NULL || singvals == NULL) {
        fprintf(stderr, "Could not allocate the memory needed to compute the SVD of AV\n");
        exit(-1);
    }
    LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', numrows, numeigs, AV, numeigs, singvals, U, numeigs, VT, numeigs);

    printf("Succeeded\n");
}
