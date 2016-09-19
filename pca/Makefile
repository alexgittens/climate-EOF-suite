#SHELL="/bin/bash"
.SHELLARGS="-l -c"

# use 40 slurm nodes : takes less than 12 minutes
# NB: the determining factor in the number of nodes is that Parallel-HDF5 can only load less than 2GB in each call to h5fread
CFSROFNAMEIN=../rawdata/cfsro.h5
CFSROTEMPDATASETNAME=temp
CFSROLATFNAME=../rawdata/CFSROObservedLatitudes.csv
CFSROMETADATADIR=../rawdata
CFSRORANK=100
CFSROOUTFNAME=../eofs/cfsropcas.h5
CFSROEOF3DFNAME=../eofs/cfsroeofs.nc

# use 30 slurm nodes
# NB: this has to hold both temp and rho in memory
CESMFNAMEIN=../rawdata/cesm.h5
CESMTEMPDATASETNAME=temp
CESMRHODATASETNAME=rho
CESMLATFNAME=../rawdata/CESMObservedLatitudes.csv
CESMMETADATADIR=../rawdata
CESMRANK=100
CESMOUTFNAME=../eofs/cesmpcas.h5
CESMEOF3DFNAME=../eofs/cesmeofs.nc

all: pca 

# deprecated, but keep around in case need to compile on Edison at some point
#	cc -o pca pca.c -I$$HDF5_INCLUDE_OPTS -static -lhdf5 -larpack -L. -L$$CRAY_LD_LIBRARY_PATH

# the -Xlinker -Map=a.map creates a file showing what libraries are linked against
# -l:libsci_intel.so tells it to link against shared library
pca: computations.c io.c pca.h pca.c 
	module load cray-hdf5-parallel && \
	cc -std=c99 -g -o pca pca.c computations.c io.c -larpack -I. -L.

cesm: pca
	module load python cray-hdf5-parallel && \
	srun -u -c 1 -n 960 ./pca CESM ${CESMFNAMEIN} ${CESMTEMPDATASETNAME} ${CESMRHODATASETNAME} ${CESMLATFNAME} ${CESMRANK} ${CESMOUTFNAME} && \
	python cesmreshape.py ${CESMOUTFNAME} "latweighting+centering" ${CESMMETADATADIR} ${CESMEOF3DFNAME}

cfsro: pca
	module load python cray-hdf5-parallel && \
	srun -u -c 1 -n 1252 ./pca CFSRO ${CFSROFNAMEIN} ${CFSROTEMPDATASETNAME} ${CFSROLATFNAME} ${CFSRORANK} ${CFSROOUTFNAME} && \
	python cfsroreshape.py ${CFSROOUTFNAME} "latweighting+centering" ${CFSROMETADATADIR} ${CFSROEOF3DFNAME}

