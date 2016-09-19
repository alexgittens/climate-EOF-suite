#include "pca.h"
#include "cblas.h"

/*******************************************************************************************/
/*  MPI-based functions for doing various shared memory matrix computations  */
/*******************************************************************************************/

// computes means of the rows of A, subtracts them from A, and returns them in meanVec on the root process
// assumes memory has already been allocated for meanVec
void computeAndSubtractRowMeans(double *localRowChunk, double *meanVec, distMatrixInfo *matInfo) {
    int mpi_rank = matInfo->mpi_rank;
    int numcols = matInfo->numcols;
    int localrows = matInfo->localrows;
    int * rowcounts = matInfo->rowcounts;
    int * rowoffsets = matInfo->rowoffsets;
    MPI_Comm *comm = matInfo->comm;

    double *onesVec = (double *) malloc( numcols * sizeof(double));
    double *localMeanVec = (double *) malloc( localrows * sizeof(double));

    for(int idx = 0; idx < numcols; idx = idx + 1) {
        onesVec[idx]=1;
    }
    cblas_dgemv(CblasRowMajor, CblasNoTrans, localrows, numcols, 1.0/((double)numcols), localRowChunk, numcols, onesVec, 1, 0, localMeanVec, 1);
    cblas_dger(CblasRowMajor, localrows, numcols, -1.0, localMeanVec, 1, onesVec, 1, localRowChunk, numcols);
    if (mpi_rank != 0) {
        MPI_Gatherv(localMeanVec, localrows, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, *comm);
    } else {
        MPI_Gatherv(localMeanVec, localrows, MPI_DOUBLE, meanVec, rowcounts, rowoffsets, MPI_DOUBLE, 0, *comm);
    }
    free(onesVec);
    free(localMeanVec);
}

// rescales the rows of A by the given weights
// weights only needs to be defined on the root process
void rescaleRows(double *localRowChunk, double *weights, distMatrixInfo *matInfo) {
    int mpi_rank = matInfo->mpi_rank;
    int numcols = matInfo->numcols;
    int localrows = matInfo->localrows;
    int * rowcounts = matInfo->rowcounts;
    int * rowoffsets = matInfo->rowoffsets;
    MPI_Comm *comm = matInfo->comm;

    double *localweights = (double *) malloc(localrows * sizeof(double));

    if(mpi_rank != 0) {
        MPI_Scatterv(NULL, rowcounts, rowoffsets, MPI_DOUBLE, localweights, localrows, MPI_DOUBLE, 0, *comm);
    } else {
        MPI_Scatterv(weights, rowcounts, rowoffsets, MPI_DOUBLE, localweights, localrows, MPI_DOUBLE, 0, *comm);
    }
    for(int rowIdx = 0; rowIdx < localrows; rowIdx = rowIdx + 1)
        cblas_dscal(numcols, localweights[rowIdx], localRowChunk + (rowIdx*numcols), 1);
    free(localweights);
}

// computes A^T*A*v and stores back in v
void distributedGramianVecProd(const double *localRowChunk, double *v, const distMatrixInfo *matInfo, scratchMatrices * scratchSpace) {
    multiplyGramianChunk(localRowChunk, v, v, scratchSpace->Scratch, matInfo->localrows, matInfo->numcols, 1); // TODO: write an appropriate mat-vec function instead of using the mat-mat function
    MPI_Allreduce(v, scratchSpace->Scratch2, matInfo->numcols, MPI_DOUBLE, MPI_SUM, *(matInfo->comm));
    cblas_dcopy(matInfo->numcols, scratchSpace->Scratch2, 1, v, 1);
}

// computes A*mat and stores result on the rank 0 process in matProd (assumes the memory has already been allocated)
void distributedMatMatProd(const double *localRowChunk, const double *mat,
    double *matProd, const distMatrixInfo *matInfo, const distGatherInfo
    *eigInfo, scratchMatrices * scratchSpace) {
    multiplyAChunk(localRowChunk, mat, scratchSpace->Scratch3,
        matInfo->localrows, matInfo->numcols, eigInfo->numeigs);
    if (matInfo->mpi_rank != 0) {
        MPI_Gatherv(scratchSpace->Scratch3,
            matInfo->localrows*eigInfo->numeigs, MPI_DOUBLE, NULL, NULL, NULL,
            MPI_DOUBLE, 0, *(matInfo->comm));
    } else {
        MPI_Gatherv(scratchSpace->Scratch3,
            matInfo->localrows*eigInfo->numeigs, MPI_DOUBLE, matProd,
            eigInfo->elementcounts, eigInfo->elementoffsets, MPI_DOUBLE, 0,
            *(matInfo->comm));
    }
}
/*******************************************************************************************/
/* Non-MPI matrix manipulation helper functions */
/*******************************************************************************************/

// computes C = A*Omega 
void multiplyAChunk(const double A[], const double Omega[], double C[], const
    int rowsA, const int colsA, const int colsOmega) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, colsOmega, colsA, 1.0, A, colsA, Omega, colsOmega, 0.0, C, colsOmega);
}

/* computes C = A'*(A*Omega) = A*Scratch , so Scratch must have size rowsA*colsOmega */
void multiplyGramianChunk(const double A[], const double Omega[], double C[],
    double Scratch[], const int rowsA, const int colsA, const int colsOmega) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, colsOmega, colsA, 1.0, A, colsA, Omega, colsOmega, 0.0, Scratch, colsOmega);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, colsA, colsOmega, rowsA, 1.0, A, colsA, Scratch, colsOmega, 0.0, C, colsOmega);
}

// copies matrix A
void dgecopy(const double * A, long m, long n, long incRowA, long incColA, double * B, long incRowB, long incColB)
{
    for (long j=0; j<n; ++j) {
        for (long i=0; i<m; ++i) {
            B[i*incRowB+j*incColB] = A[i*incRowA+j*incColA];
        }
    }
}

// stores the transpose of matrix A in matrix B
void mattrans(const double * A, long m, long n, double * B) {
    dgecopy(A, m, n, n, 1, B, 1, m); 
}

// A <- hadadmardProduct(A, B)
void dhad(double * A, const double * B, long numelems) {
    for(long i=0; i < numelems; i++) 
        A[i] = A[i]*B[i];
}

// flips the left-right ordering of the columns of a matrix stored in rowmajor format
void flipcolslr(double * A, long m, long n) {
    for(long idx = 0; idx < n/2; idx = idx + 1) 
        cblas_dswap(m, A + idx, n, A + (n - 1 - idx), n);
}

void freeScratchMatrices(scratchMatrices * scratchSpace) {
    free(scratchSpace->Scratch);
    free(scratchSpace->Scratch2);
    free(scratchSpace->Scratch3);
}
