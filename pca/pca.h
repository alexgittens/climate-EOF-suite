#ifndef PCA_HEADER_SEEN
#define PCA_HEADER_SEEN
#include "mpi.h"
#include "lapacke.h"

/*********************************************************
 * Auxiliary structures
 * *******************************************************/

// M_PI is not always defined in math.h
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679L
#endif

enum climatedatasource {CESM, CFSRO};

struct distMatrixInfo {
    int mpi_size;
    int mpi_rank;
    MPI_Comm * comm;

    int numrows;
    int numcols;
    int localrows;
    int startingrow;

    int littlePartitionSize;
    int bigPartitionSize;
    int numLittlePartitions;
    int numBigPartitions;

    int * rowcounts;
    int * rowoffsets;
};
typedef struct distMatrixInfo distMatrixInfo;

// used for MPI gatherv-type operations
// contains offsets into the eigenvector matrix for each mpi process
struct distGatherInfo {
    int numrows;
    int numcols;
    int numeigs;
    int * elementcounts;
    int * elementoffsets;
    int * rowcounts;
    int * rowoffsets;
};
typedef struct distGatherInfo distGatherInfo;

struct scratchMatrices {
    double *Scratch, *Scratch2, *Scratch3; //Scratch and Scratch2 are used in multiplyGramianChunk, Scratch3 in distributedMatMatProd
};
typedef struct scratchMatrices scratchMatrices;

/*********************************************************
 * ARPACK routines
 * *******************************************************/

extern void dsaupd_(int * ido, char * bmat, int * n, char * which,
                    int * nev, double * tol, double * resid, 
                    int * ncv, double * v, int * ldv,
                    int * iparam, int * ipntr, double * workd,
                    double * workl, int * lworkl, int * info);

extern void dseupd_(int *rvec, char *HowMny, int *select,
                    double *d, double *Z, int *ldz,
                    double *sigma, char *bmat, int *n,
                    char *which, int *nev, double *tol,
                    double *resid, int *ncv, double *V,
                    int *ldv, int *iparam, int *ipntr,
                    double *workd, double *workl,
                    int *lworkl, int *info);

/*********************************************************
 * LAPACK routines
 * *******************************************************/

lapack_int LAPACKE_dgesdd(int matrix_layout, char jobz, lapack_int m,
    lapack_int n, double * a, lapack_int lda, double * s, double * u,
    lapack_int ldu, double * vt, lapack_int ldvt);

/*********************************************************
 * MPI-based matrix computation routines
 * *******************************************************/

/* computes C = A'*(A*Omega) = A*Scratch , so Scratch must have size rowsA*colsOmega */
void multiplyGramianChunk(const double A[], const double Omega[], double C[],
    double Scratch[], const int rowsA, const int colsA, const int colsOmega);

// computes C = A*Omega 
void multiplyAChunk(const double A[], const double Omega[], double C[], const
    int rowsA, const int colsA, const int colsOmega);
 
// computes A^T*A*v and stores back in v
void distributedGramianVecProd(const double *localRowChunk, double *v, const distMatrixInfo *matInfo, scratchMatrices * scratchSpace);

// computes C = A*Omega 
void multiplyAChunk(const double A[], const double Omega[], double C[], const
    int rowsA, const int colsA, const int colsOmega);

// computes A*mat and stores result on the rank 0 process in matProd (assumes the memory has already been allocated)
void distributedMatMatProd(const double *localRowChunk, const double *mat,
    double *matProd, const distMatrixInfo *matInfo, const distGatherInfo
    *eigInfo, scratchMatrices * scratchSpace);

// computes means of the rows of A, subtracts them from A, and returns them in meanVec on the root process
// assumes memory has already been allocated for meanVec
void computeAndSubtractRowMeans(double *localRowChunk, double *meanVec, distMatrixInfo *matInfo);

// rescales the rows of A by the given weights
// weights only needs to be defined on the root process
void rescaleRows(double *localRowChunk, double *weights, distMatrixInfo *matInfo);

/*********************************************************
 * non-MPI helper matrix routines
 * *******************************************************/

// copies matrix A
void dgecopy(const double * A, long m, long n, long incRowA, long incColA, double * B, long incRowB, long incColB);

// stores the transpose of matrix A in matrix B
void mattrans(const double * A, long m, long n, double * B);

// A <- hadadmardProduct(A, B)
void dhad(double * A, const double * B, long numelems);

// flips the left-right ordering of the columns of a matrix stored in rowmajor format
void flipcolslr(double * A, long m, long n);

/*********************************************************
 * IO routines
 * *******************************************************/

// assumes matInfo was preallocated
double getMatrixInfo(char * infilename, char * datasetname, MPI_Comm *comm, MPI_Info *info, distMatrixInfo *matInfo);

void freeMatrixInfo(distMatrixInfo * matInfo);

// note eigInfo is only populated for rank 0, so can send NULL in otherwise
double genGatherInfo(distMatrixInfo *matInfo, int numeigs, distGatherInfo *eigInfo);

void freeGatherInfo(distGatherInfo *eigInfo);

void freeScratchMatrices(scratchMatrices * scratchSpace);

// localRowChunk will be allocated inside the function
double loadMatrix(char * infilename, char * datasetname, distMatrixInfo
    *matInfo, double ** localRowChunk, MPI_Comm *comm, MPI_Info *info);

// loads the latitudes of each row from the given file, and returns the corresponding vector of row weights in rowWeights
// expects each line of the file to be in the form ^idx,latitude$
// returns sqrt(cos(lat)) as the weights vector
double loadRowWeights(char * weightsFname, double * rowWeights);

void writeSVD(char * outfname, distGatherInfo *eigInfo, double * U, double * V, double * singvals, double * meanVec, double * rowWeights);

//converts string to uppercase
void upperCasify(char * sPtr);

#endif

