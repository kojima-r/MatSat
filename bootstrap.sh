
gcc -O3 src/abdsat_sp.c -o test -fopenmp -m64 -march=native -lm -lcblas -lblas -DUSE_BLAS

#gcc -O3 src/test.c -o test -fopenmp -lm -Wall -pg

kinst-ompp gcc -O3 src/abdsat_sp.c -o test -fopenmp -lm 
