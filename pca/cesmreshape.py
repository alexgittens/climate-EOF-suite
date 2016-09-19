import numpy as np
from numba import jit
import csv, sys, h5py, netCDF4

FILLVALUE = -999

def main(inSource, preprocessMethod, metadataDir, outDest):

    fin = h5py.File(inSource, "r")
    U = fin["U"][:]
    V = fin["V"][:]
    S = fin["S"][:]
    rowMeans = fin["rowMeans"][:]

    latGrid = np.array(map(float, open(metadataDir + "/CESMLatList.lst").readlines()))
    lonGrid = np.array(map(float, open(metadataDir + "/CESMLonList.lst").readlines()))
    dates = np.array(map(lambda x: list(x.rstrip()), open(metadataDir + "/CESMColumnDates.lst").readlines()))
    mapToLocations = np.array(map(int, open(metadataDir + "/CESMObservedLocations.lst").readlines()))

    writeEOFs(outDest, latGrid, lonGrid, numdepths, dates, mapToLocations,
            preprocessMethod, rowMeans, U, S, V)

def writeEOFs(outDest, latGrid, lonGrid, numdepths, dates, mapToLocations, preprocessMethod, rowMeans, U, S, V):


    @jit
    def toEOF(vector):
        eof = -999*np.ones((numdepths, len(latGrid), len(lonGrid)), vector.dtype)
        curIndex = 0
        curWriteIndexOffset = 0

        for depth in range(numdepths):
            for lat in range(len(latGrid)):
                for lon in range(len(lonGrid)):
                    if (curWriteIndexOffset == len(vector)):
                        return eof
                    elif (curIndex == mapToLocations[curWriteIndexOffset]):
                        eof[depth, lat, lon] = vector[curWriteIndexOffset]
                        curWriteIndexOffset += 1
                    curIndex += 1

    rootgrp = netCDF4.Dataset(outDest, "w", format="NETCDF4")

    latDim = rootgrp.createDimension("lat", len(latGrid))
    lonDim = rootgrp.createDimension("lon", len(lonGrid))
    depthDim = rootgrp.createDimension("depth", numdepths)
    eofsDim = rootgrp.createDimension("eofs", S.shape[0])
    datesDim = rootgrp.createDimension("dates", dates.shape[0])
    datelengthDim = rootgrp.createDimension("lengthofdates", dates.shape[1])

    rootgrp.createVariable("lat", latGrid.dtype, ("lat",))[:] = latGrid
    rootgrp.createVariable("lon", lonGrid.dtype, ("lon",))[:] = lonGrid
    rootgrp.createVariable("coldates", 'S1', ("dates", "lengthofdates"))[:] = dates
    rootgrp.createVariable("temporalEOFs", V.dtype, ("eofs", "dates"))[:] = V.T
    rootgrp.createVariable("singvals", S.dtype, ("eofs",))[:] = S
    rootgrp.createVariable("meanTemps", rowMeans.dtype, ("depth", "lat", "lon"), fill_value = FILLVALUE)[:] = toEOF(rowMeans)
    
    for eofNum in range(U.shape[1]):
        print "Converting EOF {0}".format(eofNum)
        curEOF = rootgrp.createVariable("EOF" + str(eofNum), U.dtype, ("depth", "lat", "lon"), fill_value = FILLVALUE)
        curEOF[:] = toEOF(U[:, eofNum])
        curEOF.type_of_statistical_processing = preprocessMethod
        curEOF.level_type = "Depth below sea level (m)"
        curEOF.grid_type = "Latitude/longitude"
    
    rootgrp.close()

numdepths = 60 # TODO: this should be supplied in the metadatadir
inSource = sys.argv[1]
preprocessMethod = sys.argv[2]
metadataDir = sys.argv[3]
outDest = sys.argv[4]
main(inSource, preprocessMethod, metadataDir, outDest)
