# Run to convert the dumped metadata for the CESM data into csv files

import numpy as np
import csv

outDir = '../rawdata'
metadataFname = outDir + "/tempCESMMetadata.npz"
observedLatFname = outDir + "/CESMObservedLatitudes.csv"
observedDepthFname = outDir + "/CESMObservedDepths.csv"
latListFname = outDir + "/CESMLatList.lst"
lonListFname = outDir + "/CESMLonList.lst"
depthListFname = outDir + "/CESMDepthList.lst"
observedLocationsFname = outDir + "/CESMObservedLocations.lst"
observedTareaFname = outDir + "/CESMObservedTareas.lst"
dateListFname = outDir + "/CESMColumnDates.lst"

metadata = np.load(metadataFname)
recordedLats = metadata["observedLatCoords"]
recordedTareas = metadata["observedTareas"]
recordedLevelIndices =  metadata["observedLevelDepths"]

def strLine(number):
    return str(number) + "\n"

with open(dateListFname, 'w') as fout:
    timeStamps = [item for sublist in metadata["timeStamps"] for item in sublist]
    fout.writelines( map( lambda str: str + "\n", map(lambda stamp: str(int(stamp)), timeStamps)))

with open(latListFname, 'w') as fout:
    fout.writelines( map(strLine, metadata["latList"]) )

with open(lonListFname, 'w') as fout:
    fout.writelines( map(strLine, metadata["lonList"]) )

with open(observedLocationsFname, 'w') as fout:
    fout.writelines( map(strLine, metadata["observedLocations"]) )

with open(observedLatFname, 'w') as csvfile:
    fieldnames = ['rowidx', 'latitude']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for (idx, val) in enumerate(recordedLats):
        writer.writerow({'rowidx' : idx, 'latitude' : val})

with open(observedTareaFname, 'w') as csvfile:
    fieldnames = ['rowidx', 'tarea']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    for (idx, val) in enumerate(recordedTareas):
        writer.writerow({'rowidx' : idx, 'tarea' : val})

