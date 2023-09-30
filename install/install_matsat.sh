
gcc -O3 src/abdsat_sp.c -o matsat -fopenmp -m64 -march=native -lm -lcblas -lblas -DUSE_BLAS

nvcc -rdc=true -arch sm_35 -O3 -o matsat_gpu src_gpu/abdsat_sp.cu

