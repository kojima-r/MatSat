
gcc -O3 src/abdsat_sp.c -o abd_sat -fopenmp -m64 -march=native -lm -lcblas -lblas -DUSE_BLAS

#gcc -O3 src/test.c -o test -fopenmp -lm -Wall -pg

kinst-ompp gcc -O3 src/abdsat_sp.c -o abd_sat_omp -fopenmp -lm 

nvcc -rdc=true -arch sm_35 -O3 -o abdsat_gpu src_gpu/abdsat_sp.cu

