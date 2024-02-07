# IMe library
Inhibition Method header-only C library

If you use this work, please cite:
> D. Loreti, M. Artioli and A. Ciampolini, "Solving Linear Systems on High Performance Hardware with Resilience to Multiple Hard Faults," 2020 International Symposium on Reliable Distributed Systems (SRDS), Shanghai, China, 2020, pp. 266-275, [doi: 10.1109/SRDS51746.2020.00034](https://dx.doi.org/10.1109/SRDS51746.2020.00034)

## Inhibition Method
It's a linear systems solver for dense unstructured matrices.
It has a closed iterative, exact, naturally parallel algorithm, now enhanced with fault tolerance to mulitple hard-faults, particularly suitable for HPC environments.

## The library
It's written in C with MPI in "include-only" style.
Its routines follow a naming scheme similar to ScaLAPACK.
 
