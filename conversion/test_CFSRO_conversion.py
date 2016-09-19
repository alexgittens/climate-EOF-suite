# Code for testing that the CFSRO climate data has been converted correctly from the 
# multiple netcdf files into one large hdf5 file, and the latitudes and level depths 
# recorded for each observation are correct
#
# To run: first do a 
#  module load python h5py
# then start ipython and run this code interactively

import numpy as np
from netCDF4 import Dataset
import h5py

# FOR TESTING AFTER FULL CONVERSION, RUN THIS BLOCK OF CODE
# randomly sample columns and check for equality with the data from the corresponding
# original files and check that the latitudes and level depths for each observation were recorded accurately

numColSamples = 20

outputFname = "../rawdata/cfsro.h5"
outputMat = h5py.File(outputFname, "r")["rows"]
numCols = outputMat.shape[1]
colIndices = np.sort(np.random.randint(numCols, size=numColSamples))
sampledCols = outputMat[:, colIndices]

baseDir = "/global/cscratch1/sd/nrcavana/CFSR_OCEAN/"
varname = "POT_L160_Avg_1"
metadataFname = "../rawdata/CFRSOMetadata.npz"
metadata = np.load(metadataFname)
timeOffsets = np.concatenate([np.array(item) for item in metadata["timeSliceOffsets"]])
fileNames = np.concatenate([np.array(item) for item in metadata["fileNames"]])
fNames = fileNames[colIndices]
timeSliceOffsets = timeOffsets[colIndices]
recordedLats = np.tile(metadata["observedLatCoords"], (numColSamples, 1)).transpose()
recordedLevelDepths =  np.tile(metadata["observedLevelDepths"], (numColSamples, 1)).transpose()

rawCols = np.empty_like(sampledCols)
rawLats = np.empty_like(sampledCols)
rawDepths = np.empty_like(sampledCols)
for (idx, fName) in enumerate(fNames):
    timeOffset = timeSliceOffsets[idx]
    rawFin = Dataset(baseDir + fName, "r")
    rawData = rawFin[varname][:].data
    rawMask = np.logical_not(rawFin[varname][:].mask)
    lonGrid, latGrid = np.meshgrid(rawFin["lon"][:], rawFin["lat"][:])
    latMesh = np.tile(latGrid, (rawData.shape[1],1,1))
    rawCols[:, idx] = rawData[timeOffset, rawMask[timeOffset, ...]]
    rawLats[:, idx] = latMesh[rawMask[timeOffset, ...]]

    levelDepths = rawFin["level0"]
    curLevelDepths = []
    for levNum in xrange(rawData.shape[1]):
        curLevelDepths = np.concatenate([curLevelDepths, [levelDepths[levNum]] * np.count_nonzero(rawMask[timeOffset, levNum, ...])])
    rawDepths[:, idx] = curLevelDepths
    rawFin.close()
    print "Loaded column %d/%d" % (idx+1, numColSamples)

# these should be zero on success
print np.linalg.norm(rawCols - sampledCols)
print np.linalg.norm(rawLats - recordedLats)
print np.linalg.norm(rawDepths - recordedLevelDepths)
