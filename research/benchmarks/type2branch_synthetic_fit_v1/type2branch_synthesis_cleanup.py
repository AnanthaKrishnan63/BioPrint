"""Translation of pinned KSDSLD CLI cleanup, not .NET runtime parity.

Sources (GPLv3) are retained under references/type2branch_synthesis/cleanup.
Program.LoadDataset applies ThresholdPartitioner then CleanFTs before fitting;
KeystrokeDynamicsSynthesizer applies the same stages to generated outputs.
"""
import numpy as np
from type2branch_csv_bridge import integer_rows

INVALID_TIMING = -(2**31)


def cleanup(rows):
    original=integer_rows(rows)
    result=original.copy()
    # Partitions are determined before clamping negative and >1500ms values.
    partitions=np.flatnonzero(original[:,2]>1500)
    timings=result[:,1:]
    timings[timings<0]=2**31-1
    np.minimum(timings,1500,out=timings)
    result[0,2]=INVALID_TIMING
    result[partitions,2]=INVALID_TIMING
    return result,partitions
