# MatSat
This is an implementation of MatSat (https://arxiv.org/abs/2108.06481)

## Basic usage

### Source code
This repository contains four implementations.
- C++ (GPU version): `src_gpu/abdsat_sp.cu`
- python: `python/tensorized_sat_solver_sp.py`
- Naive C code: `src/abdsat_sp.c`
- Naive C with dense matrix format (deprecated): `src/abdsat_sp.c`

### Compile & Example

#### (1) Naive C code
Complie command using gcc with openmp and blas libraries
```
gcc -O3 src/matsat_sp.c -o matsat -fopenmp -m64 -march=native -lm -lblas -DUSE_BLAS
```
Solving the example problem: 3SAT_inst500
```
time ./matsat 3SAT_inst500 [<seed> <max_itration> <sample_size(retry)>]
```

#### (2) C++ (GPU version)
Complie command using nvcc (nvidia CUDA compile tool)
```
nvcc -rdc=true -arch sm_35 -O3 -o matsat_gpu src_gpu/matsat_sp.cu
```
Solving the example problem: 3SAT_inst500
```
time ./matsat 3SAT_inst500 [<seed> <max_itration> <sample_size(retry)>]
```

#### (3) Python implementation
Solving the example problem: 3SAT_inst500
```
python python/matsat_sp.py 3SAT_inst500
```

### SAT problem
Input File = 3SAT instance in DIMACS format

Please see the exmaple problem using a text editor: `3SAT_inst500`

[Experiments](./experiment/README.md)

### Citation

```
@inproceedings{sato2021matsat,
  title={MatSat: A matrix-based differentiable SAT solver},
  author = {Taisuke Sato and Ryosuke Kojima},
  howpublished={The 11th International Workshop on Pragmatics of SAT (PoS 2021), CoRR},
  url={https://arxiv.org/abs/2108.06481}
  year={2021}
}
```

