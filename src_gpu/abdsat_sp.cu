
#define MAX 2048

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <float.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#if defined(_WIN32) || (_MSC_VER)
#define VC_MODE
#endif

#ifdef VC_MODE
#include <sys/types.h>
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

#define nsat 7 	// =3(always),3-SAT
#define MAX_NSAT_CHECK 10
int split=256;
float perturb=0.5;

#define block 1024
#define block1 1
#define block32 32
#define grid1 1
#define grid32 32

int n = 0;		// |variables|
int m = 0;		// |clauses|
int max_try = 3;
int *base_M;
static uint32_t seed = 0;

__device__ int d_m;
__device__ int d_n;
__device__ int d_split;
__device__ float d_perturb;
__device__ int d_max_itr;
__device__ int d_nblockA;
__device__ int d_nblockB;
__device__ float d_J;
__device__ float d_J1;
__device__ float d_J2;
__device__ float d_alpha;
__device__ float d_sumJaJa;
__device__ float du_min;
__device__ float du_max;
__device__ float dd;
__device__ int d_ini_error;
__device__ int d_final_error;
__device__ int d_final_error_index;
__device__ int d_itr_count;
__device__ float* d_u;
__device__ float* d_u0;
__device__ float* d_wu;
__device__ float* d_wM;
__device__ float* d_wu_cnt;
__device__ float* d_wM_cnt;
__device__ float d_wu_sum;
__device__ float d_wM_sum;
__device__ float* d_E;
__device__ float* d_FF;
__device__ float* d_Ja;
__device__ float* d_JaJa;
__device__ float* d_floatA;
__device__ float* d_floatA2;
__device__ int* d_M;
__device__ int* d_a;
__device__ int* d_Cp0;
__device__ int* d_intA;

// for anealing
//__device__ int d_max_try;

#define INDEX(ROW, COL, rows) ((COL) * (rows) + (ROW))

#define CHECK(call)                                                            \
{                                                                              \
	const cudaError_t error = call;                                            \
	if (error != cudaSuccess)                                                  \
	{                                                                          \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
		fprintf(stderr, "code: %d, reason: %s\n", error,                       \
				cudaGetErrorString(error));                                    \
		exit(1);                                                               \
	}                                                                          \
}

#ifdef VC_MODE
inline double seconds()
{
    _timeb tp;
    _ftime(&tp);
    return ((double)tp.time + (double)tp.millitm / 1000.0);
}
#else
inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
#endif


/*
 * Display a variety of information on the first CUDA device in this system,
 * including driver version, runtime version, compute capability, bytes of
 * global memory, etc.
 */

int printDeviceInfo(int dev, cudaDeviceProp deviceProp)
{
	printf("Device %d: \"%s\"\n", dev, deviceProp.name);

	int driverVersion = 0, runtimeVersion = 0;

	cudaDriverGetVersion(&driverVersion);
	cudaRuntimeGetVersion(&runtimeVersion);
	printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10);
	printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
			deviceProp.major, deviceProp.minor);
	printf("  Total amount of global memory:                 %.2f GBytes (%llu "
			"bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
			(unsigned long long)deviceProp.totalGlobalMem);
	printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
			"GHz)\n", deviceProp.clockRate * 1e-3f,
			deviceProp.clockRate * 1e-6f);
	printf("  Memory Clock rate:                             %.0f Mhz\n",
			deviceProp.memoryClockRate * 1e-3f);
	printf("  Memory Bus Width:                              %d-bit\n",
			deviceProp.memoryBusWidth);

	if (deviceProp.l2CacheSize)
	{
		printf("  L2 Cache Size:                                 %d bytes\n",
				deviceProp.l2CacheSize);
	}

	printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
			"2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
			deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
			deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
			deviceProp.maxTexture3D[2]);
	printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
			"2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
			deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
			deviceProp.maxTexture2DLayered[1],
			deviceProp.maxTexture2DLayered[2]);
	printf("  Total amount of constant memory:               %zu bytes\n",
			deviceProp.totalConstMem);
	printf("  Total amount of shared memory per block:       %zu bytes\n",
			deviceProp.sharedMemPerBlock);
	printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
	printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
	printf("  Maximum number of threads per multiprocessor:  %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
	printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
	printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
	printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
	printf("  Maximum memory pitch:                          %zu bytes\n",
			deviceProp.memPitch);

	return 1;
}


int randint(uint32_t *xp, int nums)
{
	if(seed == 0) {
		clock_t clk = clock();
		time_t sec;
		time(&sec);
		seed = clk * sec;
	}
	uint32_t x=0;
	int i;
	for(i=0;i<nums;i++){
		x = (i > 0) ? xp[i-1] : seed;
		x = x ^ (x << 13);
		x = x ^ (x >> 17);
		x = x ^ (x << 15);
		xp[i] = x;
	}
	seed = x;

	return i;
}

void rand_u(float *xp, int nums) 
{
	uint32_t *yp = (uint32_t *)calloc(nums,sizeof(uint32_t));
	uint32_t mask = 0x7fffffff;
	uint32_t maxint = 0x80000000;
	randint(yp, nums);
	for (int i = 0; i<nums; i++) {
		uint32_t my;
		my = yp[i] & mask;
		xp[i] = (float)my / maxint;
	}
	free(yp);
}



// PosOcc[q][p] == 1 <=> var q(0<=q<n) occurs positively in p(0<=p<m)-th clause in M
// NegOcc[q][p] == 1 <=> var q(0<=q<n) occurs negatively in p(0<=p<m)-th clause in M


void read_cnf(const char* File){
	// Construct matrix M from DIMACS format file "File"
	// where vars are numberd from 1 to n whereas
	// they are numbered from 0 to n-1 in M,PosOcc,NegOcc
	FILE *fp ;
	int lit[MAX_NSAT_CHECK];

	static char separator[] = " \t\n";
	char *token;

	int n_lit;
	char c1[3];
	char c2[4];
	char buf[MAX];
	int p = 0;
	fp = fopen(File,"r");
	if(fp == NULL) {
		printf( "file open failed\n" );
		exit(1);
	};
		
	while((fgets(buf,MAX,fp) != NULL)){
		if(buf[0] == 'c'){
			continue;
		}
		if(buf[0] == 'p'){
			sscanf(buf,"%s %s %d %d",c1,c2, &n, &m);
			printf("n=%d\n",n);
			printf("m=%d\n",m);
			base_M = (int *)calloc(m*nsat,sizeof(int));
//			for (int p=0;p<m;p++){ M[p] = base_M + p * nsat; }
			continue;
		}
		n_lit = 0;
		if ((token = strtok(buf, separator)) != NULL) {
			do {
				lit[n_lit] = atoi(token);
			}while (++n_lit < MAX_NSAT_CHECK && (token = strtok(NULL, separator)) != NULL);
			if (n_lit > nsat) {
				printf( "over nsat=%d literals, ignored\n",nsat);
			}
		}
		for (int j=n_lit;j<nsat;j++){ lit[j]=0; }
		for (int l=0;l<nsat;l++){
			if (p<m){
				base_M[INDEX(p,l,m)] = lit[l];
			} else {
				printf("****\n");
			}  // M[p][l]=0 if |p-th clause|<l
		}
		//for (int j=0;j<k;j++){ printf(" %d",M[p][j]); }; printf("\n");
		p++;
	}
	fclose(fp);
	// printf("k= %d n=%d m=%d max_try=%d max_itr=%d\n",nsat,n,m,max_try,max_itr);
	// for (p=0;p<m;p++){ printf("%d %d %d\n",M[p][0],M[p][1],M[p][2]); }

}




// BlockDim.x should be less than 1025, more than 63 and 2^n where n is a positive integer.
__global__ void reduceAddInt(int* d_iInt, int *d_result, unsigned int n, int s)
{
	unsigned int tid = threadIdx.x;
	unsigned int nth = gridDim.x * blockDim.x;

	__shared__ int smem[1024];

	smem[tid] = 0;
	int idx = tid + blockIdx.x * blockDim.x;
	while(idx < n){
		smem[tid] += d_iInt[idx];
		idx += nth;
	}
	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile int *vsmem = smem;
		if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
		if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
		if (blockDim.x >= 16) vsmem[tid] += vsmem[tid +  8];
		if (blockDim.x >= 8)  vsmem[tid] += vsmem[tid +  4];
		if (blockDim.x >= 4)  vsmem[tid] += vsmem[tid +  2];
		if (blockDim.x >= 2)  vsmem[tid] += vsmem[tid +  1];
	}

	// write result for this block to global mem
	if (tid == 0)
		d_result[s] = smem[0];

}



// BlockDim.x should be less than 1025 and 2^n where n is a positive integer.
__global__ void reduceAddFloat(float* d_iFloat, float *d_result, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int nth = gridDim.x * blockDim.x;

	__shared__ float smem[1024];

	smem[tid] = 0.0;
	int idx = tid + blockIdx.x * blockDim.x;
	while(idx < n){
		smem[tid] += d_iFloat[idx];
		idx += nth;
	}

	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile float *vsmem = smem;
		if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
		if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
		if (blockDim.x >= 16) vsmem[tid] += vsmem[tid +  8];
		if (blockDim.x >= 8)  vsmem[tid] += vsmem[tid +  4];
		if (blockDim.x >= 4)  vsmem[tid] += vsmem[tid +  2];
		if (blockDim.x >= 2)  vsmem[tid] += vsmem[tid +  1];
	}

	// write result for this block to global mem
	if (tid == 0) d_result[blockIdx.x] = smem[0];


}


// BlockDim.x should be less than 1025, more than 63 and 2^n where n is a positive integer.
__global__ void reduceMinMaxFloat(float* d_iFloat1, float* d_iFloat2, float *d_minresult, float *d_maxresult, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int nth = gridDim.x * blockDim.x;

	__shared__ float minsmem[1024];
	__shared__ float maxsmem[1024];
	minsmem[tid] = FLT_MAX;
	maxsmem[tid] = FLT_MIN;

	int idx = tid + blockIdx.x * blockDim.x;
	while(idx < n){
		float tmpf = d_iFloat1[idx];
		if (minsmem[tid] > tmpf) minsmem[tid] = tmpf;
		tmpf = d_iFloat2[idx];
		if (maxsmem[tid] < tmpf) maxsmem[tid] = tmpf;
		idx += nth;
	}
	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512){
		if (minsmem[tid] > minsmem[tid + 512]) minsmem[tid] = minsmem[tid + 512];
		if (maxsmem[tid] < maxsmem[tid + 512]) maxsmem[tid] = maxsmem[tid + 512];
	}
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256){
		if (minsmem[tid] > minsmem[tid + 256]) minsmem[tid] = minsmem[tid + 256];
		if (maxsmem[tid] < maxsmem[tid + 256]) maxsmem[tid] = maxsmem[tid + 256];
	}
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128){
		if (minsmem[tid] > minsmem[tid + 128]) minsmem[tid] = minsmem[tid + 128];
		if (maxsmem[tid] < maxsmem[tid + 128]) maxsmem[tid] = maxsmem[tid + 128];
	}
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64){
		if (minsmem[tid] > minsmem[tid + 64]) minsmem[tid] = minsmem[tid + 64];
		if (maxsmem[tid] < maxsmem[tid + 64]) maxsmem[tid] = maxsmem[tid + 64];
	}
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile float *minvsmem = minsmem;
		if (blockDim.x >= 64 && minvsmem[tid] > minvsmem[tid + 32]) minvsmem[tid] = minvsmem[tid + 32];
		if (blockDim.x >= 32 && minvsmem[tid] > minvsmem[tid + 16]) minvsmem[tid] = minvsmem[tid + 16];
		if (blockDim.x >= 16 && minvsmem[tid] > minvsmem[tid + 8]) minvsmem[tid] = minvsmem[tid + 8];
		if (blockDim.x >= 8  && minvsmem[tid] > minvsmem[tid + 4]) minvsmem[tid] = minvsmem[tid + 4];
		if (blockDim.x >= 4  && minvsmem[tid] > minvsmem[tid + 2]) minvsmem[tid] = minvsmem[tid + 2];
		if (blockDim.x >= 2  && minvsmem[tid] > minvsmem[tid + 1]) minvsmem[tid] = minvsmem[tid + 1];
		volatile float *maxvsmem = maxsmem;
		if (blockDim.x >= 64 && maxvsmem[tid] < maxvsmem[tid + 32]) maxvsmem[tid] = maxvsmem[tid + 32];
		if (blockDim.x >= 32 && maxvsmem[tid] < maxvsmem[tid + 16]) maxvsmem[tid] = maxvsmem[tid + 16];
		if (blockDim.x >= 16 && maxvsmem[tid] < maxvsmem[tid + 8]) maxvsmem[tid] = maxvsmem[tid + 8];
		if (blockDim.x >= 8  && maxvsmem[tid] < maxvsmem[tid + 4]) maxvsmem[tid] = maxvsmem[tid + 4];
		if (blockDim.x >= 4  && maxvsmem[tid] < maxvsmem[tid + 2]) maxvsmem[tid] = maxvsmem[tid + 2];
		if (blockDim.x >= 2  && maxvsmem[tid] < maxvsmem[tid + 1]) maxvsmem[tid] = maxvsmem[tid + 1];
	}

	// write result for this block to global mem
	if (tid == 0) d_minresult[blockIdx.x] = minsmem[0];
	if (tid == 0) d_maxresult[blockIdx.x] = maxsmem[0];

}


// BlockDim.x should be less than 1025, more than 63 and 2^n where n is a positive integer.
__global__ void reduceMinInt(int* d_iInt, int *d_result, int *d_result_index,unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int nth = gridDim.x * blockDim.x;

	__shared__ int smem[1024];
	__shared__ int smem_arg[1024];
	smem[tid] = INT_MAX;
	smem_arg[tid] = INT_MAX;

	int idx = tid + blockIdx.x * blockDim.x;
	while(idx < n){
		int tmpf = d_iInt[idx];
		if (smem[tid] > tmpf){
			smem[tid] = tmpf;
			smem_arg[tid] = idx;
		}
		idx += nth;
	}
	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512)
		if (smem[tid] > smem[tid + 512]){
			smem[tid] = smem[tid + 512];
			smem_arg[tid] = smem_arg[tid + 512];
		}
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256)
		if (smem[tid] > smem[tid + 256]){
			smem[tid] = smem[tid + 256];
			smem_arg[tid] = smem_arg[tid + 256];
		}
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128)
		if (smem[tid] > smem[tid + 128]){
			smem[tid] = smem[tid + 128];
			smem_arg[tid] = smem_arg[tid + 128];
		}
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)
		if (smem[tid] > smem[tid + 64]){
			smem[tid] = smem[tid + 64];
			smem_arg[tid] = smem_arg[tid + 64];
		}
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile int *vsmem = smem;
		volatile int *vsmem_arg = smem_arg;
		if (blockDim.x >= 64 && vsmem[tid] > vsmem[tid + 32]){
			vsmem[tid] = vsmem[tid + 32];
			vsmem_arg[tid] = vsmem_arg[tid + 32];
		}
		if (blockDim.x >= 32 && vsmem[tid] > vsmem[tid + 16]){
			vsmem[tid] = vsmem[tid + 16];
			vsmem_arg[tid] = vsmem_arg[tid + 16];
		}
		if (blockDim.x >= 16 && vsmem[tid] > vsmem[tid + 8]){
			vsmem[tid] = vsmem[tid + 8];
			vsmem_arg[tid] = vsmem_arg[tid + 8];
		}
		if (blockDim.x >= 8  && vsmem[tid] > vsmem[tid + 4]){
			vsmem[tid] = vsmem[tid + 4];
			vsmem_arg[tid] = vsmem_arg[tid + 4];
		}
		if (blockDim.x >= 4  && vsmem[tid] > vsmem[tid + 2]){
			vsmem[tid] = vsmem[tid + 2];
			vsmem_arg[tid] = vsmem_arg[tid + 2];
		}
		if (blockDim.x >= 2  && vsmem[tid] > vsmem[tid + 1]){
			vsmem[tid] = vsmem[tid + 1];
			vsmem_arg[tid] = vsmem_arg[tid + 1];
		}
	}

	// write result for this block to global mem
	if (tid == 0){
		d_result[blockIdx.x] = smem[0];
		d_result_index[blockIdx.x] = smem_arg[0];
	}

}


__global__ void d_compute_error_1(int s){
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float lins = du_min + s * dd;
	if(idx < d_n)
		d_a[idx] = ( d_u[idx] >= lins ? 1 : 0);
}

__global__ void d_compute_error_2(){
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;

	__shared__ int smem[1024];
	smem[tid] = 0;
	if(idx < d_m){
		int M;
		int Cp = 0;
		for(int i=0;i<nsat;i++){
			M = d_M[INDEX(idx, i, d_m)];
			//Cp += (M < 0 ? 1 - d_a[-M-1] : d_a[M-1]);
			if(M < 0){
				Cp += 1 - d_a[-M-1];
			}else if (M>0){
				Cp += d_a[M-1];
			}
		}
		smem[tid] = (Cp == 0);
		d_wM_cnt[idx]=(Cp == 0);
	}

	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile int *vsmem = smem;
		if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
		if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
		if (blockDim.x >= 16) vsmem[tid] += vsmem[tid +  8];
		if (blockDim.x >= 8)  vsmem[tid] += vsmem[tid +  4];
		if (blockDim.x >= 4)  vsmem[tid] += vsmem[tid +  2];
		if (blockDim.x >= 2)  vsmem[tid] += vsmem[tid +  1];
	}

	// write result for this block to global mem
	if (tid == 0) d_Cp0[blockIdx.x] = smem[0];

}

__global__ void d_init_wM(){
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;
	if(idx < d_m){
		d_wM[idx]=1;
	}
}

__global__ void d_compute_error_wM(){
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;

	__shared__ int smem[1024];
	smem[tid] = 0;
	if(idx < d_m){
		int M;
		int Cp = 0;
		for(int i=0;i<nsat;i++){
			M = d_M[INDEX(idx, i, d_m)];
			//Cp += (M < 0 ? 1 - d_a[-M-1] : d_a[M-1]);
			if(M < 0){
				Cp += 1 - d_a[-M-1];
			}else if (M>0){
				Cp += d_a[M-1];
			}
		}
		d_wM[idx] += (Cp == 0);
		smem[tid] = d_wM[idx] ;
	}

	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile int *vsmem = smem;
		if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
		if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
		if (blockDim.x >= 16) vsmem[tid] += vsmem[tid +  8];
		if (blockDim.x >= 8)  vsmem[tid] += vsmem[tid +  4];
		if (blockDim.x >= 4)  vsmem[tid] += vsmem[tid +  2];
		if (blockDim.x >= 2)  vsmem[tid] += vsmem[tid +  1];
	}

	// write result for this block to global mem
	if (tid == 0) d_wM_cnt[blockIdx.x] = smem[0];

}

__global__ void d_compute_normalize_wM(){
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;
	float d_wM_mean =d_wM_sum / (float) d_m;
	if(idx < d_m){
		d_wM[idx] = d_wM[idx]/d_wM_mean;
	}
}

__device__ int d_compute_error ()
{
	// Compute thresholds lins[]
	reduceMinMaxFloat<<<grid32, block>>>(d_u, d_u, d_floatA, d_floatA2, d_n);
	reduceMinMaxFloat<<<grid1, block32>>>(d_floatA, d_floatA2, &du_min, &du_max, 32);

	dd = (du_max - du_min)/d_split;

	// Compute num_false[s] = sum((Q*[a;1-a])==0) for each threshold s
	for (int s=0;s<d_split;s++){
		d_compute_error_1<<<d_nblockA, block>>>(s);
		d_compute_error_2<<<d_nblockB, block>>>();
		//d_intA[s]<=sum_0^n(d_Cp0)
		reduceAddInt<<<grid1, block>>>(d_Cp0, d_intA, d_nblockB,s);
	}

	// compute final_error = the least error
	reduceMinInt<<<grid1, block32>>>(d_intA, &d_final_error, &d_final_error_index, d_split);

	//d_final_error_index
	return d_final_error;
}

__device__ int d_compute_error_and_update_weight()
{
	// Compute thresholds lins[]
	reduceMinMaxFloat<<<grid32, block>>>(d_u, d_u, d_floatA, d_floatA2, d_n);
	reduceMinMaxFloat<<<grid1, block32>>>(d_floatA, d_floatA2, &du_min, &du_max, 32);

	dd = (du_max - du_min)/d_split;

	// Compute num_false[s] = sum((Q*[a;1-a])==0) for each threshold s
	for (int s=0;s<d_split;s++){
		d_compute_error_1<<<d_nblockA, block>>>(s);
		d_compute_error_2<<<d_nblockB, block>>>();
		//d_intA[s]<=sum_0^n(d_Cp0)
		reduceAddInt<<<grid1, block>>>(d_Cp0, d_intA, d_nblockB,s);
	}

	// compute final_error = the least error
	reduceMinInt<<<grid1, block32>>>(d_intA, &d_final_error, &d_final_error_index, d_split);

	d_compute_error_1<<<d_nblockA, block>>>(d_final_error_index);
	d_compute_error_wM<<<d_nblockB, block>>>();
	reduceAddFloat<<<grid1, block>>>(d_wM_cnt, &d_wM_sum, d_nblockB);
	d_compute_normalize_wM<<<d_nblockB, block>>>();

	//d_final_error_index
	return d_final_error;
}

__global__ void d_compute_weight_u_0()
{
	// set index
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;

	//__shared__ float smem[1024];
	//smem[tid] = 0.0;
	if(idx < d_n){
		d_wu[idx]=0;
		//float w = d_uw[idx];
		//smem[tid] = w;
	}
}

__global__ void d_compute_weight_u_1()
{
	// set index
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;
	__shared__ float smem[1024];
	smem[tid] = 0.0;
	if(idx < d_m){
		int M;
		for(int i=0;i<nsat;i++){
			M = d_M[INDEX(idx, i, d_m)];
			if(M < 0){
				M = -M-1;
				atomicAdd(&d_wu[M], 1);
				smem[tid]+=1;
			}else if (M>0){
				M = M-1;
				atomicAdd(&d_wu[M], 1);
				smem[tid]+=1;
			}
		}
	}
	__syncthreads();
	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile float *vsmem = smem;
		if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
		if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
		if (blockDim.x >= 16) vsmem[tid] += vsmem[tid +  8];
		if (blockDim.x >= 8)  vsmem[tid] += vsmem[tid +  4];
		if (blockDim.x >= 4)  vsmem[tid] += vsmem[tid +  2];
		if (blockDim.x >= 2)  vsmem[tid] += vsmem[tid +  1];
	}
	// write result for this block to global mem
	if (tid == 0) d_wu_cnt[blockIdx.x] = smem[0];
}

__global__ void d_compute_weight_u_2()
{
	// set index
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;

	//__shared__ float smem[1024];
	//smem[tid] = 0.0;
	float d_wu_mean =d_wu_sum / (float) d_n;
	if(idx < d_n){
		d_wu[idx] = d_wu[idx]/d_wu_mean;
	}
}

__global__ void d_compute_weight_u_3()
{
	// set index
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;
	if(idx < d_m){
		d_wM[idx]=0;
		int M;
		for(int i=0;i<nsat;i++){
			M = d_M[INDEX(idx, i, d_m)];
			if(M < 0){
				M = -M-1;
				d_wM[idx]+=d_wu[M];
			}else if (M>0){
				M = M-1;
				d_wM[idx]+=d_wu[M];
			}
		}
	}
}

__global__ void d_compute_J_Ja_1 ()
{
	// set index
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;

	__shared__ float smem[1024];
	smem[tid] = 0.0;
	if(idx < d_n){
		float du = d_u[idx];
		float df = du * (1 - du);
		smem[tid] = df * df;
		d_Ja[idx] = df * (1 - 2 * du);
	}

	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile float *vsmem = smem;
		if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
		if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
		if (blockDim.x >= 16) vsmem[tid] += vsmem[tid +  8];
		if (blockDim.x >= 8)  vsmem[tid] += vsmem[tid +  4];
		if (blockDim.x >= 4)  vsmem[tid] += vsmem[tid +  2];
		if (blockDim.x >= 2)  vsmem[tid] += vsmem[tid +  1];
	}

	// write result for this block to global mem
	if (tid == 0) d_FF[blockIdx.x] = smem[0];


}


__global__ void d_compute_J_Ja_2 ()
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int tid = threadIdx.x;
	__shared__ float smem[1024];
	smem[tid] = 0.0;
	

	if(idx < d_m){
		// Compute C = Q*[u;1-u]
		int M[nsat];
		float dC = 0.0;
		for(int i=0;i<nsat;i++){
			M[i] = d_M[INDEX(idx, i, d_m)];
			if(M[i]!=0){
				dC += (M[i] < 0 ? 1 - d_u[-M[i]-1] : d_u[M[i]-1]);
			}
		}
		// Compute E = (C<=1).*(1-C) = 1-min(C,1)
		smem[tid] = (dC <= 1.0 ? 1.0 - dC : 0.0);
		int q;
		int pn;
		float Jaq;
		float w = 0.0;
		for(int i=0;i<nsat;i++){
			if(M[i]!=0){
				q = (M[i] > 0 ? M[i]-1 : -M[i]-1);
				w = (M[i] > 0 ? d_wu[M[i]-1] : d_wu[-M[i]-1]);
				//w+=d_wM[idx];
				pn= (M[i] > 0 ? w : -w);
				Jaq= (dC < 1.0 ? -1.0*pn : 0.0);
				atomicAdd(&d_Ja[q], Jaq);
			}
		}
	}


	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile float *vsmem = smem;
		if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
		if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
		if (blockDim.x >= 16) vsmem[tid] += vsmem[tid +  8];
		if (blockDim.x >= 8)  vsmem[tid] += vsmem[tid +  4];
		if (blockDim.x >= 4)  vsmem[tid] += vsmem[tid +  2];
		if (blockDim.x >= 2)  vsmem[tid] += vsmem[tid +  1];
	}

	// write result for this block to global mem
	if (tid == 0) d_E[blockIdx.x] = smem[0];

}


__global__ void d_update_u_1 ()
{
	// set index
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;

	__shared__ float smem[1024];
	smem[tid] = 0.0;
	if(idx <d_n)
		smem[tid] = d_Ja[idx]*d_Ja[idx];

	__syncthreads();

	// in-place reduction in shared memory
	if (blockDim.x >= 1024 && tid < 512) smem[tid] += smem[tid + 512];
	__syncthreads();
	if (blockDim.x >= 512 && tid < 256) smem[tid] += smem[tid + 256];
	__syncthreads();
	if (blockDim.x >= 256 && tid < 128) smem[tid] += smem[tid + 128];
	__syncthreads();
	if (blockDim.x >= 128 && tid < 64)  smem[tid] += smem[tid + 64];
	__syncthreads();
	// unrolling warp
	if (tid < 32)
	{
		volatile float *vsmem = smem;
		if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
		if (blockDim.x >= 32) vsmem[tid] += vsmem[tid + 16];
		if (blockDim.x >= 16) vsmem[tid] += vsmem[tid +  8];
		if (blockDim.x >= 8)  vsmem[tid] += vsmem[tid +  4];
		if (blockDim.x >= 4)  vsmem[tid] += vsmem[tid +  2];
		if (blockDim.x >= 2)  vsmem[tid] += vsmem[tid +  1];
	}

	// write result for this block to global mem
	if (tid == 0) d_JaJa[blockIdx.x] = smem[0];
	
}

__global__ void d_update_u_2 ()
{
	// set thread ID
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < d_n)
		d_u[idx] = d_u[idx] - d_alpha * d_Ja[idx];
}


__global__ void d_set_device_ptr (
	float* ptr_d_wu,float* ptr_d_wM,float* ptr_d_wu_cnt,float* ptr_d_wM_cnt,
	float* ptr_d_u, float* ptr_d_u0, float* ptr_d_E, float* ptr_d_FF, float* ptr_d_Ja, float* ptr_d_JaJa,
	float* ptr_d_floatA, float* ptr_d_floatA2, int* ptr_d_M, int* ptr_d_a, int* ptr_d_Cp0, int* ptr_d_intA)
{
	d_wu=ptr_d_wu;
	d_wM=ptr_d_wM;
	d_wu_cnt=ptr_d_wu_cnt;
	d_wM_cnt=ptr_d_wM_cnt;
	d_u = ptr_d_u;
	d_u0 = ptr_d_u0;
	d_E = ptr_d_E;
	d_FF = ptr_d_FF;
	d_Ja = ptr_d_Ja;
	d_JaJa = ptr_d_JaJa;
	d_floatA = ptr_d_floatA;
	d_floatA2 = ptr_d_floatA2;
	d_M = ptr_d_M;
	d_a = ptr_d_a;
	d_Cp0 = ptr_d_Cp0;
	d_intA = ptr_d_intA;
}


__global__ void d_calc ()
{
	d_compute_error();
	cudaDeviceSynchronize();
	d_ini_error = d_final_error;

	for (int j=0;j<d_max_itr;j++){

		d_compute_weight_u_0<<<d_nblockA, block>>>();
		d_compute_weight_u_1<<<d_nblockB, block>>>();
		reduceAddFloat<<<grid1, block>>>(d_wu_cnt, &d_wu_sum, d_nblockA);
		d_compute_weight_u_2<<<d_nblockA, block>>>();
		//d_compute_weight_u_3<<<d_nblockB, block>>>();

		d_itr_count++;
		d_compute_J_Ja_1<<<d_nblockA, block>>>();
		reduceAddFloat<<<grid1, block>>>(d_FF, &d_J1, d_nblockA);
		d_compute_J_Ja_2<<<d_nblockB, block>>>();
		reduceAddFloat<<<grid1, block>>>(d_E, &d_J2, d_nblockA);

		d_update_u_1<<<d_nblockA, block>>>();
		reduceAddFloat<<<grid1, block>>>(d_JaJa, &d_sumJaJa, d_nblockA);
		cudaDeviceSynchronize();
		d_J = d_J1+d_J2;
		d_alpha = (d_J)/d_sumJaJa;

		// anealing
		//float r = j/d_max_itr;
		//d_alpha = (0.1+ 0.9*(1-r)) * d_alpha;

		d_update_u_2<<<d_nblockA, block>>>();

		cudaDeviceSynchronize();
//		printf("	j=%d	J=%10.6f\n",j,d_J);

		if (d_J<2.0 || j==d_max_itr-1){
			// Compute the least error = final_error
			d_compute_error_and_update_weight();
		} // if (J<1.0){

		cudaDeviceSynchronize();

		if (d_final_error == 0) { break; }

	}

}


__global__ void d_update_initu (int i)
{
	unsigned int nth = gridDim.x * blockDim.x;
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//float d = 0.1*i/d_max_try;
	while(idx < d_n){
		//perturbation
		d_u[idx] = (1.0-d_perturb)*d_u[idx] +d_perturb*d_u0[idx];
		//
		//d_u[idx] = (0.6+d)*d_u[idx] + (0.4-d)*d_u0[idx];
		idx += nth;
	}
}


int main(int argc, char** argv)
{
	// Argument parser
	char* File = argv[1];
	if(argc>2){
		seed = strtol(argv[2],NULL,0);
		printf("seed=%d\n",seed);
	}
	int max_itr = 300;
	if(argc>3){
		max_itr = strtol(argv[3],NULL,0);
	}

	printf("max_itr=%d\n",max_itr);
	if(argc>4){
		max_try = strtol(argv[4],NULL,0);
	}
	printf("max_try=%d\n",max_try);
	if(argc>5){
		split = strtol(argv[5],NULL,0);
	}
	printf("split=%d\n",split);
	if(argc>6){
		perturb = strtof(argv[6],NULL);
	}
	printf("perturb=%f\n",perturb);
	read_cnf(File);


	// (m x n) matrix for a SAT instantce read from "File"
	// M[p][q] = r <=> var |r|(1<=|r|<=n) occurs
	// in p(0<=p<m)-th clause as q(= 0,1,2)-th literal
	// [-2 3 -5 0] = 1st row in DIMACS file where n = 5
	//  => M[0][0] = -2, M[0][1] = 3, M[0][2] = -5
	//     Q(0,:) = [0 0 1 0 0  0 1 0 0 1] = [Q1 Q2]
	//     Q1 = [0 0 1 0 0;...], Q2 = [0 1 0 0 1;...]

	float* u  = (float *)calloc(n,sizeof(float));	// continuous assignment(model)	 (n x 1)
	float* u0 = (float *)calloc(n,sizeof(float));   // continuous assignment(model)	 (n x 1)
	// float* Ja = (float *)calloc(n,sizeof(float));   // Ja = (Q2-Q1)'*(C<1) + F.*(1-2*u)	J's Jacobian
	// {(Q2-Q1)'*(C<1)}[p]
	//   = |neg. occ. of var p in clauses falsified by u|
	//    - |pos. occ. of var p in clauses falsified by u|
	float J=0;     // = sum(E) + ||u.*(1-u)||^2  cost function J


	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0){
		printf("There are no available device(s) that support CUDA\n");
		return 1;
	}

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printDeviceInfo(dev, deviceProp);

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	int nblockA = (n + block - 1) / block;
	int nblockB = (m + block - 1) / block;

	CHECK(cudaMemcpyToSymbol( d_m, &m, sizeof(int)));	
	CHECK(cudaMemcpyToSymbol( d_n, &n, sizeof(int)));	
	CHECK(cudaMemcpyToSymbol( d_max_itr, &max_itr, sizeof(int)));	
	CHECK(cudaMemcpyToSymbol( d_split, &split, sizeof(int)));	
	CHECK(cudaMemcpyToSymbol( d_perturb, &perturb, sizeof(float)));	
	CHECK(cudaMemcpyToSymbol( d_nblockA, &nblockA, sizeof(int)));	
	CHECK(cudaMemcpyToSymbol( d_nblockB, &nblockB, sizeof(int)));	
	float* ptr_d_u;
	float* ptr_d_u0;
	float* ptr_d_wu;
	float* ptr_d_wu_cnt;
	float* ptr_d_wM;
	float* ptr_d_wM_cnt;
	float* ptr_d_E;
	float* ptr_d_FF;
	float* ptr_d_Ja;
	float* ptr_d_JaJa;
	float* ptr_d_floatA;
	float* ptr_d_floatA2;
	int* ptr_d_M;
	int* ptr_d_a;
	int* ptr_d_Cp0;
	int* ptr_d_intA;
	CHECK(cudaMalloc((void**)&ptr_d_u,n*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_u0,n*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_wu,n*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_wu_cnt,n*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_wM,m*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_wM_cnt,m*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_E,m*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_FF,n*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_Ja,n*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_JaJa,n*sizeof(float)));
	CHECK(cudaMalloc((void**)&ptr_d_M,m*nsat*sizeof(int)));
	CHECK(cudaMalloc((void **)&ptr_d_floatA,n*sizeof(float)));
	CHECK(cudaMalloc((void **)&ptr_d_floatA2,n*sizeof(float)));
	CHECK(cudaMalloc((void **)&ptr_d_intA,n*sizeof(int)));
	CHECK(cudaMalloc((void**)&ptr_d_a,n*sizeof(int)));
	CHECK(cudaMalloc((void**)&ptr_d_Cp0,m*sizeof(int)));
	CHECK(cudaMemcpy(ptr_d_M, base_M, m* nsat * sizeof(int), cudaMemcpyHostToDevice));

	// Initialize u by uniform dist. over [0 1]
	rand_u(u,n);
	CHECK(cudaMemcpy(ptr_d_u, u, n*sizeof(float), cudaMemcpyHostToDevice));

	d_set_device_ptr<<<grid1, block1>>>(
		ptr_d_wu,ptr_d_wM,ptr_d_wu_cnt,ptr_d_wM_cnt,
		ptr_d_u, ptr_d_u0, ptr_d_E, ptr_d_FF, ptr_d_Ja, ptr_d_JaJa,
		ptr_d_floatA, ptr_d_floatA2, ptr_d_M, ptr_d_a, ptr_d_Cp0, ptr_d_intA);


	int itr_count=0;
	CHECK(cudaMemcpyToSymbol(d_itr_count, &itr_count, sizeof(int)));
	// for anealing
	//CHECK(cudaMemcpyToSymbol(d_max_try, &max_try, sizeof(int)));	


	double iStart = seconds();
	//	struct timespec startTime, endTime;
	//	clock_gettime(CLOCK_REALTIME, &startTime);


	int ini_error,final_error;
	d_init_wM<<<nblockB, block>>>();
	for (int i=0;i<max_try;i++){

		/*	Iterate gradiate descent by Newton's method
			for j=1:max_itr               % enter j-loop to mimize J
				C = Q2t1 + Q12*u;     % = Q*[u;1-u]	 (m x 1)
				E = (C<=1).*(1-C);    % = 1-min(C,1)	(m x 1)
				F = u.*(1-u);         % (n x 1)
				%% update u by Newton's method
				J = sum(E) + sumsq(F);% cost function J
				Ja = (Q2-Q1)'*(C<1) + F.*(1-2*u); % J's Jacobian  (n x 1)
				alpha = J/sumsq(Ja);
				u = u - alpha*Ja;
			endfor
		*/

		d_calc<<<grid1, block1>>>();

		rand_u(u0,n);
		CHECK(cudaMemcpyAsync(ptr_d_u0, u0, n*sizeof(float), cudaMemcpyHostToDevice,stream));

		CHECK(cudaMemcpyFromSymbol(&ini_error, d_ini_error, sizeof(int)));
		CHECK(cudaMemcpyFromSymbol(&final_error, d_final_error, sizeof(int)));
		CHECK(cudaMemcpyFromSymbol(&J, d_J, sizeof(float)));
		printf("i=%d	error(%d->%d): J=%f \n",i,ini_error,final_error,J);

		if (final_error == 0)
			break;

		CHECK(cudaStreamSynchronize(stream));
		d_update_initu<<<nblockA, block>>>(i);

		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError());
	} // for (i=0;i<max_try;i++){

	double iElaps = seconds() - iStart;
	CHECK(cudaMemcpyFromSymbol(&itr_count, d_itr_count, sizeof(int)));
	printf("final error = %d\n",final_error);
	printf("all time = %f\n",iElaps);
	printf("itration time = %f\n",iElaps/itr_count);

//	clock_gettime(CLOCK_REALTIME, &endTime);
//	if (endTime.tv_nsec < startTime.tv_nsec) {
//		long sec=endTime.tv_sec - startTime.tv_sec - 1;
//		long nsec=endTime.tv_nsec + 1000000000 - startTime.tv_nsec;
//		float f=sec+nsec/1000000000.0;
//		printf("all time = %f\n",f);
//		printf("itration time = %f\n",f/itr_count);
//	} else {
//		long sec=endTime.tv_sec - startTime.tv_sec;
//		long nsec=endTime.tv_nsec - startTime.tv_nsec;
//		float f=sec+nsec/1000000000.0;
//		printf("all time = %f\n",f);
//		printf("itration time = %f\n",f/itr_count);
//	}

	free(base_M);
	free(u);
	free(u0);

	CHECK(cudaFree(ptr_d_u));
	CHECK(cudaFree(ptr_d_u0));
	CHECK(cudaFree(ptr_d_wu));
	CHECK(cudaFree(ptr_d_wM));
	CHECK(cudaFree(ptr_d_wu_cnt));
	CHECK(cudaFree(ptr_d_wM_cnt));
	CHECK(cudaFree(ptr_d_E));
	CHECK(cudaFree(ptr_d_FF));
	CHECK(cudaFree(ptr_d_Ja));
	CHECK(cudaFree(ptr_d_JaJa));
	CHECK(cudaFree(ptr_d_M));
	CHECK(cudaFree(ptr_d_floatA));
	CHECK(cudaFree(ptr_d_floatA2));
	CHECK(cudaFree(ptr_d_intA));
	CHECK(cudaFree(ptr_d_a));
	CHECK(cudaFree(ptr_d_Cp0));

	CHECK(cudaStreamDestroy(stream));

	// reset device
	CHECK(cudaDeviceReset());

	return EXIT_SUCCESS;

}




