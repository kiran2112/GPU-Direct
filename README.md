# MPI Alltoall using GPU-Direct
Kiran Ravikumar - kiran.r@gatech.edu

Code to study the performance of MPI-Alltoall using GPU-Direct.

The code can be compiled on IBM XL compiler using

module load cuda
mpif90 -O3 -qcuda GPU_DIRECT.F90 -o GPU_DIRECT.x

To run the code remember to use the following line in the batch script

source $MPI_ROOT/jsm_pmix/bin/export_smpi_env -gpu

The code performs the alltoall using two methods,
1. A dummy array is copied from the device to the host. A blocking MPI_Alltoall is performed on the Host. A dummy array is then copied back from the Host to the device. The dummy copies mimic the practical use case where the send and receive buffers need to be copied from and to the device.
2. The blocking MPI_ALLTOALL is performed using gpu arrays directly.

The performance of the Second method is ecpected to be similar to the first method if not better.

The code outputs the time it takes perform the alltoall using gpu arrays in method 2 and the time it takes to perform the alltoall+copy in+copy out in the method 1. The effective BW reported in the output is calculated using this time and the message size that is exchanged. Therefore it is important to not that the effective BW is not related to the network BW directly.
The correctness of the two methods can be verified by ensuring that the max global error reported in the output is 0.
