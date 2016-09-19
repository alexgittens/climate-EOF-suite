#include "hdf5.h"
#include "mpi.h"
#include "pca.h"
#include <stdio.h>
#include <math.h>
#include <ctype.h>

// assumes matInfo was preallocated
double getMatrixInfo(char * infilename, char * datasetname, MPI_Comm *comm, MPI_Info *info, distMatrixInfo *matInfo) {
    double startTime = MPI_Wtime();

    int mpi_size, mpi_rank;
    MPI_Comm_size(*comm, &mpi_size);
    MPI_Comm_rank(*comm, &mpi_rank);

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, *comm, *info);

    hid_t file_id = H5Fopen(infilename, H5F_ACC_RDONLY, plist_id);
    if (file_id < 0) {
        fprintf(stderr, "Error opening %s\n", infilename);
        exit(-1);
    }

    hid_t dataset_id = H5Dopen(file_id, datasetname, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening %s in %s : are you sure this dataset exists?\n", datasetname, infilename);
        exit(-1);
    }

    hid_t dataset_space = H5Dget_space(dataset_id);
    hsize_t dims[2];
    herr_t status = H5Sget_simple_extent_dims(dataset_space, dims, NULL);
    if (status < 0 || status != 2) {
       fprintf(stderr, "Error reading %s from %s : remember it should be a 2d matrix\n", datasetname, infilename); 
       exit(-1);
    }

    H5Sclose(dataset_space);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    H5Pclose(plist_id);

    int numrows = dims[0];
    int numcols = dims[1];
    
	int littlePartitionSize = numrows/mpi_size;
    int bigPartitionSize = littlePartitionSize + 1;
    int numLittlePartitions = mpi_size - numrows % mpi_size;
    int numBigPartitions = numrows % mpi_size;
    int localrows, startingrow;

    if (mpi_rank < numBigPartitions) {
        localrows = bigPartitionSize;
        startingrow = bigPartitionSize*mpi_rank;
    } else {
        localrows = littlePartitionSize;
        startingrow = bigPartitionSize*numBigPartitions + 
                      littlePartitionSize*(mpi_rank - numBigPartitions);
    }

    matInfo->mpi_size = mpi_size;
    matInfo->mpi_rank = mpi_rank;
    matInfo->numrows = numrows;
    matInfo->numcols = numcols;
    matInfo->localrows = localrows;
    matInfo->startingrow = startingrow;
    matInfo->littlePartitionSize = littlePartitionSize;
    matInfo->bigPartitionSize = bigPartitionSize;
    matInfo->numLittlePartitions = numLittlePartitions;
    matInfo->numBigPartitions = numBigPartitions;

    matInfo->comm = comm;
    
    matInfo->rowcounts = (int *) malloc ( mpi_size * sizeof(int) );
    matInfo->rowoffsets = (int *) malloc ( mpi_size * sizeof(int) );
    if (matInfo->rowcounts == NULL || matInfo->rowoffsets == NULL) {
        fprintf(stderr, "Could not allocate memory for the matrix chunk offset information\n");
        exit(-1);
    }
    for(int idx = 0; idx < numBigPartitions; idx = idx + 1) {
        matInfo->rowcounts[idx] = bigPartitionSize;
        matInfo->rowoffsets[idx] = bigPartitionSize * idx;
    }
    for(int idx = numBigPartitions; idx < mpi_size; idx = idx + 1) {
        matInfo->rowcounts[idx] = littlePartitionSize;
        matInfo->rowoffsets[idx] = bigPartitionSize * numBigPartitions + littlePartitionSize * (idx - numBigPartitions);
    }

    return MPI_Wtime() - startTime;
}

void freeMatrixInfo(distMatrixInfo * matInfo) {
    free(matInfo->rowcounts);
    free(matInfo->rowoffsets);
}

double genGatherInfo(distMatrixInfo *matInfo, int numeigs, distGatherInfo *eigInfo) {
    double startTime = MPI_Wtime();

    int mpi_size = matInfo->mpi_size;
    int bigPartitionSize = matInfo->bigPartitionSize;
    int numBigPartitions = matInfo->numBigPartitions;
    int littlePartitionSize = matInfo->littlePartitionSize;
    int numLittlePartitions = matInfo->numLittlePartitions;

    eigInfo->elementcounts = (int *) malloc( mpi_size * sizeof(int));
    eigInfo->elementoffsets = (int *) malloc( mpi_size * sizeof(int));
    eigInfo->rowcounts = (int *) malloc( mpi_size * sizeof(int));
    eigInfo->rowoffsets = (int *) malloc( mpi_size * sizeof(int));

    if (eigInfo->elementcounts == NULL || eigInfo->elementoffsets == NULL
        || eigInfo->rowcounts == NULL || eigInfo->rowoffsets == NULL) {
        fprintf(stderr, "Couldn't allocate memory for the eigenvector matrix information\n");
        exit(-1);
    }

    for(int idx = 0; idx < numBigPartitions; idx = idx + 1) {
        eigInfo->elementcounts[idx] = bigPartitionSize * numeigs;
        eigInfo->elementoffsets[idx] = bigPartitionSize * numeigs * idx;

        eigInfo->rowcounts[idx] = bigPartitionSize;
        eigInfo->rowoffsets[idx] = bigPartitionSize * idx;
    }
    for(int idx = numBigPartitions; idx < mpi_size; idx = idx + 1) {
        eigInfo->elementcounts[idx] = littlePartitionSize * numeigs;
        eigInfo->elementoffsets[idx] = bigPartitionSize * numeigs * numBigPartitions + 
                          littlePartitionSize * numeigs * (idx - numBigPartitions);

        eigInfo->rowcounts[idx] = littlePartitionSize;
        eigInfo->rowoffsets[idx] = bigPartitionSize * numBigPartitions + littlePartitionSize * (idx - numBigPartitions);
    }
    eigInfo->numrows = matInfo->numrows;
    eigInfo->numcols = matInfo->numcols;
    eigInfo->numeigs = numeigs;

    return MPI_Wtime() - startTime;
}

void freeGatherInfo(distGatherInfo *eigInfo) {
    free(eigInfo->elementcounts);
    free(eigInfo->elementoffsets);
    free(eigInfo->rowcounts);
    free(eigInfo->rowoffsets);
}

// localRowChunk will be allocated inside the function
double loadMatrix(char * infilename, char * datasetname, distMatrixInfo
    *matInfo, double ** localRowChunk, MPI_Comm *comm, MPI_Info *info) {
    double startTime = MPI_Wtime();

    // assuming double inputs, check that the chunks aren't too big to be read
    // by HDF5 in parallel mode
    if (matInfo->bigPartitionSize*matInfo->numcols >= 268435456 &&
        matInfo->mpi_rank == 0) {
        fprintf(stderr, "MPIIO-based HDF5 is limited to reading 2GiB at most in "
            "each call to read; try increasing the number of processors\n");
        exit(-1); }

    *localRowChunk = (double *) malloc( matInfo->localrows * matInfo->numcols *sizeof(double) );
    if (*localRowChunk == NULL) {
        fprintf(stderr, "Could not allocate enough memory for the local chunk "
            "of rows for %s in process %d\n", datasetname, matInfo->mpi_rank);
        exit(-1);
    }

    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, *comm, *info);
    hid_t file_id = H5Fopen(infilename, H5F_ACC_RDONLY, plist_id);
    if (file_id < 0) {
        fprintf(stderr, "Error opening %s\n", infilename);
        exit(-1);
    }

    hid_t dataset_id = H5Dopen(file_id, datasetname, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error opening %s in %s : are you sure this dataset "
            "exists?\n", datasetname, infilename);
        exit(-1);
    }

    hsize_t offset[2], count[2], offset_out[2];
    count[0] = matInfo->localrows;
    count[1] = matInfo->numcols;
    offset[0] = matInfo->mpi_rank < matInfo->numBigPartitions ? 
                (matInfo->mpi_rank * matInfo->bigPartitionSize) : 
                (matInfo->numBigPartitions * matInfo->bigPartitionSize + 
                    (matInfo->mpi_rank - matInfo->numBigPartitions) * 
                    matInfo->littlePartitionSize );
    offset[1] = 0;

    hid_t filespace = H5Dget_space(dataset_id);
    if ( H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 ) {
        fprintf(stderr, "Error selecting input file hyperslab in process %d\n",
            matInfo->mpi_rank);
        exit(-1);
    }

    hid_t memspace = H5Screate_simple(2, count, NULL);
    offset_out[0] = 0;
    offset_out[1] = 0;
    if ( H5Sselect_hyperslab(memspace, H5S_SELECT_SET, offset_out, NULL, count, NULL) < 0 ) {
        fprintf(stderr, "Error selecting memory hyperslab in process %d\n",
            matInfo->mpi_rank);
        exit(-1);
    }

    hid_t daccess_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(daccess_id, H5FD_MPIO_INDEPENDENT); // collective io seems slow for this

    if (matInfo->mpi_rank == 0) {
        printf("Loading matrix from dataset %s in file %s\n", datasetname,
            infilename);
    }
    if( H5Dread(dataset_id, H5T_NATIVE_DOUBLE, memspace, filespace, daccess_id, *localRowChunk) < 0) {
        fprintf(stderr, "Error reading dataset in process %d\n", matInfo->mpi_rank);
        exit(-1);
    }

    H5Pclose(daccess_id);
    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    H5Pclose(plist_id);

    return MPI_Wtime() - startTime;
}

// loads the latitudes of each row from the given file, and returns the corresponding vector of row weights in rowWeights
// expects each line of the file to be in the form ^idx,latitude$
// returns sqrt(cos(lat)) as the weights vector; should be called only by one process
double loadRowWeights(char * weightsFname, double * rowWeights) {
    int rowIdx;
    double latVal;
    double loadTime = MPI_Wtime();

    printf("Loading latitudes for each observation from file %s\n", weightsFname);

    FILE * fin = fopen(weightsFname, "r");
    if( fin == NULL ) {
        fprintf(stderr, "Can't open latitude csv file %s!\n", weightsFname);
        exit(1);
    }
    while(fscanf(fin, "%d,%lf", &rowIdx, &latVal) != EOF) {
        rowWeights[rowIdx] = sqrt(cos(latVal*M_PI/180.0));
    }
    fclose(fin);
    return MPI_Wtime() - loadTime;
}

void writeSVD(char * outfname, distGatherInfo *eigInfo, double * U, double * V, double * singvals, double * meanVec, double * rowWeights) {

    hid_t file_id = H5Fcreate(outfname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Could not create output file %s\n", outfname);
        exit(-1);
    }


    hsize_t dims[2];
    dims[0] = eigInfo->numrows;
    dims[1] = eigInfo->numeigs;
    hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
    hid_t dataset_id = H5Dcreate2(file_id, "/U", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
    if (dataset_id < 0) {
        fprintf(stderr, "Error creating dataset U in %s\n", outfname);
        exit(-1);
    }
    if( H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace_id, plist_id, U) < 0) {
        fprintf(stderr, "Error writing U to %s\n", outfname);
        exit(-1);
    }
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    dims[0] = eigInfo->numcols;
    dims[1] = eigInfo->numeigs;
    dataspace_id = H5Screate_simple(2, dims, NULL);
    dataset_id = H5Dcreate2(file_id, "/V", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error creating dataset V in %s\n", outfname);
        exit(-1);
    }
    if ( H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace_id, plist_id, V) < 0 ) {
        fprintf(stderr, "Error writing V to %s\n", outfname);
        exit(-1);
    }
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    dims[0] = eigInfo->numeigs;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate2(file_id, "/S", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error creating dataset S in %s\n", outfname);
        exit(-1);
    }
    if ( H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace_id, plist_id, singvals) < 0) {
        fprintf(stderr, "Error writing S to %s\n", outfname);
        exit(-1);
    }
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    dims[0] = eigInfo->numrows;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate2(file_id, "/rowMeans", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error creating dataset rowMeans in %s\n", outfname);
        exit(-1);
    }
    if ( H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace_id, plist_id, meanVec) < 0) {
        fprintf(stderr, "Error writing rowMeans to %s\n", outfname);
        exit(-1);
    }
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    dims[0] = eigInfo->numrows;
    dataspace_id = H5Screate_simple(1, dims, NULL);
    dataset_id = H5Dcreate2(file_id, "/rowWeights", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_id < 0) {
        fprintf(stderr, "Error creating dataset rowWeights in %s\n", outfname);
        exit(-1);
    }
    if ( H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, dataspace_id, plist_id, rowWeights) < 0) {
        fprintf(stderr, "Error writing rowWeights to %s\n", outfname);
        exit(-1);
    }
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);

    H5Pclose(plist_id);
    H5Fclose(file_id);
}

void upperCasify(char * sPtr) {
    while(*sPtr != '\0') {
        *sPtr = toupper((unsigned char)*sPtr);
        sPtr = sPtr + 1;
    }
}
