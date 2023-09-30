/* file = test.c

How to use:
> ./test <File> <k> <n> <m> <sample_size> <max_itr>
File = 3SAT instance in DIMACS format
k = 3, n = |variable|, m = |clause|, sample_size = |retry|

(1) compile
> gcc -O3 test.c -o test -fopenmp -lm

(2) prepare 3SAT instances by Octave
> octave&
>> n=100; m=426; Q = gen_kSAT(3,n,m); write_SAT(Q,"3SAT_inst100");
>> n=500; m=2130; Q = gen_kSAT(3,n,m); write_SAT(Q,"3SAT_inst500");
>> n=1000; m=4260; Q = gen_kSAT(3,n,m); write_SAT(Q,"3SAT_inst1000");
>> n=10000; m=42600; Q = gen_kSAT(3,n,m); write_SAT(Q,"3SAT_inst10000");

(3) execute and measure time 
> time ./test 3SAT_inst100 3 100	426	10	50
> time ./test 3SAT_inst500 3 500	2130	10	300
> time ./test 3SAT_inst1000 3 1000	4260	100	1000
> time ./test 3SAT_inst10000 3 10000	42600	100	5000

*/

#define MAX 2048		// MAX chars in DIMACS format per row

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <omp.h>
static uint32_t seed = 0;

int randint(uint32_t *xp, int nums)
{
	int i;
	uint32_t x;
	clock_t clk;
	time_t sec;

	if(seed == 0) {
		clk = clock();
		time(&sec);
		seed = clk * sec;
	}

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
	int i;
	uint32_t mask, maxint;
	uint32_t yp[nums];

	mask = 0x7fffffff;
	maxint = 0x80000000;
	randint(yp, nums);
	for (i = 0; i<nums; i++) {
		uint32_t my;
		my = yp[i] & mask;
		xp[i] = (float)my / maxint;
	}
}

int k = 3;				// =3(always),3-SAT
int n = 0;		// |variables|
int m = 0;		// |clauses|
int **M,*base_M;

int **PosOcc,*base_PosOcc;
// PosOcc[q][p] == 1 <=> var q(0<=q<n) occurs positively in p(0<=p<m)-th clause in M
int **NegOcc,*base_NegOcc;
// NegOcc[q][p] == 1 <=> var q(0<=q<n) occurs negatively in p(0<=p<m)-th clause in M

int **PosNegOcc,*base_PosNegOcc;

void read_cnf(char* File){
	// Construct matrix M from DIMACS format file "File"
	//	 where vars are numberd from 1 to n whereas
	//	 they are numbered from 0 to n-1 in M,PosOcc,NegOcc
	FILE *fp ;
	int lit[k];
	int n_lit;
	char c1[3];
	char c2[3];
	char buf[MAX] = {0};
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
			M = (int **)calloc(m,sizeof(int *));        // m x k
			base_M = (int *)calloc(m*k,sizeof(int));
			for (int p=0;p<m;p++){ M[p] = base_M + p * k; }
			continue;
		}
		n_lit = sscanf(buf,"%d  %d  %d",&lit[0],&lit[1],&lit[2]);
		if (n_lit != k) {
			printf( "not k=3 literals, ignored\n" );
			continue;
		}
		for (int l=0;l<k;l++){
			M[p][l] = lit[l];
		}
		p++;
	}
	fclose(fp);
	// printf("k= %d	n=%d	m=%d sample_size=%d	max_itr=%d\n",k,n,m,sample_size,max_itr);
	// for (p=0;p<m;p++){ printf("%d	%d	%d\n",M[p][0],M[p][1],M[p][2]); }
}

/*
void constructPosNegOcc(){
	PosOcc = (int **)calloc(n,sizeof(int *));   // n x m
	base_PosOcc = (int *)calloc(n*m,sizeof(int));
	for (int q=0;q<n;q++) { PosOcc[q] = base_PosOcc + q * m; }
	NegOcc = (int **)calloc(n,sizeof(int *));   // n x m
	base_NegOcc = (int *)calloc(n*m,sizeof(int));
	for (int q=0;q<n;q++) { NegOcc[q] = base_NegOcc + q * m; }

	// Construct PosOcc and NegOcc

	for (int q=0;q<n;q++) {
		 for (int p=0;p<m;p++){
			 if ( M[p][0] == q+1  || M[p][1] == q+1  || M[p][2] == q+1  ) { PosOcc[q][p] = 1; }
			 if ( M[p][0] == -q-1 || M[p][1] == -q-1 || M[p][2] == -q-1 ) { NegOcc[q][p] = 1; }
		 }
	}
}
*/

void constructPosNegOcc(){
	PosNegOcc = (int **)calloc(n,sizeof(int *));   // n x m
	base_PosNegOcc = (int *)calloc(n*m,sizeof(int));
	for (int q=0;q<n;q++) { PosNegOcc[q] = base_PosNegOcc + q * m; }

	// Construct PosNegOcc
	for (int q=0;q<n;q++) {
		 for (int p=0;p<m;p++){
			 if ( M[p][0] == q+1  || M[p][1] == q+1  || M[p][2] == q+1  ) { PosNegOcc[q][p] = 1; }
			 if ( M[p][0] == -q-1 || M[p][1] == -q-1 || M[p][2] == -q-1 ) { PosNegOcc[q][p] = -1; }
		 }
	}
}

int compute_error(float* u,float* C){
	float d,u_min,u_max;
	int split = 10;
	float lins[split];
	int a[n];
	int n1,n2,n3;  // for 3SAT
	int num_false[split];
	int final_error;

	// Compute thresholds lins[]
	u_min = u_max = u[0];
	for (int q=1;q<n;q++){
		if (u_min > u[q]){ u_min = u[q]; }
		if (u_max < u[q]){ u_max = u[q]; }
	}
	d = (u_max - u_min)/split;
	for (int s=0;s<split;s++){ lins[s] = u_min + s * d; }
	//printf("u_min=%lf, u_max=%lf\n",u_min,u_max);
	//for (s=0;s<split;s++){ printf("lins[%d]=%lf\n",s,lins[s]); }

	// Compute num_false[s] = sum((Q*[a;1-a])==0) for each threshold s
#pragma omp parallel
{
#pragma omp for
	for (int s=0;s<split;s++){
		for (int q=0;q<n;q++){ a[q] = ( u[q] >= lins[s] ? 1 : 0); }
		for (int p=0;p<m;p++){				// compute C = Q*[a;1-a]
			C[p] = ((n1 = M[p][0]) < 0 ? 1 - a[-n1-1] : a[n1-1])
				+ ((n2 = M[p][1]) < 0 ? 1 - a[-n2-1] : a[n2-1])
				+ ((n3 = M[p][2]) < 0 ? 1 - a[-n3-1] : a[n3-1]);
		}
		int x = 0;
		for (int p=0;p<m;p++){ x += (C[p] == 0); }
		num_false[s] = x;
	}
}
	//for (s=0;s<split;s++){ printf("num_false[%d]=%d\n",s,num_false[s]); }

	// compute final_error = the least error
	final_error = 1000000;
	for (int s=0;s<split;s++){
		if (final_error > num_false[s]) { final_error = num_false[s]; }
	}
	return final_error;
}


float compute_J(float* C,float* E,float* F,float* u){
	float J = 0;
	int	n1,n2,n3;		// for 3SAT
#pragma omp parallel reduction(-:J)
{
#pragma omp for
	// Compute C = Q*[u;1-u]
	for (int p=0;p<m;p++){
		C[p] = ((n1 = M[p][0]) < 0 ? 1 - u[-n1-1] : u[n1-1])
			 + ((n2 = M[p][1]) < 0 ? 1 - u[-n2-1] : u[n2-1])
			 + ((n3 = M[p][2]) < 0 ? 1 - u[-n3-1] : u[n3-1]);
		// Compute E = (C<=1).*(1-C) = 1-min(C,1)
		E[p] = (C[p] <= 1 ? 1 - C[p] : 0);
		// for (p=0;p<m;p++){ printf("E[%d]=%lf\n",p,E[p]); }
		J += E[p];
	}
	//for (p=0;p<m;p++){ printf("C[%d]=%lf\n",p,C[p]); }
}

#pragma omp parallel reduction(-:J)
{
#pragma omp for
	// Compute F = u.*(1-u)
	for (int q=0;q<n;q++){
		F[q] = u[q] * (1 - u[q]);
		J += F[q] * F[q];
	}
	// for (q=0;q<n;q++){ printf("F[%d]=%lf\n",q,F[q]); }
}

	return J;
}

void compute_Ja(float* Ja,float* C,float* F,float* u){
#pragma omp parallel
{
#pragma omp for
	for (int q=0;q<n;q++){
		float x = 0;
		for (int p=0;p<m;p++){
			//x += (NegOcc[q][p] - PosOcc[q][p]) * (C[p] < 1);
			x += -PosNegOcc[q][p] * (C[p] < 1);
		}
		Ja[q] = x + F[q] * (1 - 2 * u[q]);
	}
}
}


void update_u(float* Ja,float J,float* u){
	float alpha;
	// Update u
	float x = 0;
	for (int q=0;q<n;q++){ x += Ja[q] * Ja[q]; };
	alpha = J/x;
	for (int q=0;q<n;q++){ u[q] = u[q] - alpha * Ja[q]; }
	//printf("x=%lf, alpha=%lf\n",x,alpha);
	//for (int p=0;p<n;p++){ printf("u[%d]=%lf\n",p,u[p]); }
}

int main(int argc, char** argv)
{
	char* File = argv[1];
	if(argc>2){
		seed = strtol(argv[2],NULL,0);
	}
	//int	sample_size = strtol(argv[5],NULL,0);	// |retry|
	//int	max_itr = strtol(argv[6],NULL,0);
	int sample_size = 10;//strtol(argv[5],NULL,0);	// |retry|
	int max_itr = 300;//strtol(argv[6],NULL,0);
	read_cnf(File);


	// (m x n) matrix for a SAT instantce read from "File"
	// M[p][q] = r <=> var |r|(1<=|r|<=n) occurs
	//		in p(0<=p<m)-th clause as q(= 0,1,2)-th literal
	// [-2 3 -5 0] = 1st row in DIMACS file where n = 5
	//	=> M[0][0] = -2, M[0][1] = 3, M[0][2] = -5
	//		 Q(0,:) = [0 0 1 0 0	0 1 0 0 1] = [Q1 Q2]
	//		 Q1 = [0 0 1 0 0;...], Q2 = [0 1 0 0 1;...]


	float u[n];    // continuous assignment(model)	 (n x 1)
	float u0[n];   // continuous assignment(model)	 (n x 1)
	float C[m];    // = Q*[u;1-u]                   (m x 1)
	float E[m];    // = 1-min(C,1) = (C<=1).*(1-C)	 (m x 1)
	float F[n];    // = u.*(1-u)                    (n x 1)
	float J;       // = sum(E) + ||u.*(1-u)||^2  cost function J
	float Ja[n];   // = (Q2-Q1)'*(C<1) + F.*(1-2*u)	J's Jacobian
	// {(Q2-Q1)'*(C<1)}[p]
	//   = |neg. occ. of var p in clauses falsified by u|
	//    - |pos. occ. of var p in clauses falsified by u|

	int ini_error,final_error;

	// PosOcc[q][p] = 1 <=> var q(0<=q<n) positively in p(0<=p<m)-th clause in M
	// NegOcc[p][q] = 1 <=> var q(0<=q<n) negatively in p(0<=p<m)-th clause in M
	constructPosNegOcc();

	// Initialize u by uniform dist. over [0 1]
	rand_u(u,n);

	struct timespec startTime, endTime;
	clock_gettime(CLOCK_REALTIME, &startTime);

	int itr_count=0;
	// Sample a SAT solution for M retrying at most "sample_size" times
	for (int i=0;i<sample_size;i++){
		final_error=ini_error = compute_error(u,C);
/*	Iterate gradiate descent by Newton's method

		for j=1:max_itr				% enter j-loop to mimize J
			C = Q2t1 + Q12*u;		% = Q*[u;1-u]	 (m x 1)
			E = (C<=1).*(1-C);	 % = 1-min(C,1)	(m x 1)
			F = u.*(1-u);				% (n x 1) 

			%% update u by Newton's method
			J = sum(E) + sumsq(F);						 % cost function J
			Ja = (Q2-Q1)'*(C<1) + F.*(1-2*u);	% J's Jacobian	(n x 1)
			alpha = J/sumsq(Ja);
			u = u - alpha*Ja;
		endfor
*/
		for (int j=0;j<max_itr;j++){
			itr_count++;
			// Compute J = sum(E) + sumsq(F)
			//printf("	j=%d	J=%lf\n",j,J);
			J=compute_J(C,E,F,u);

			// Compute Ja = (Q2-Q1)'*(C<1) + F.*(1-2*u)
			compute_Ja(Ja,C,F,u);
			
			update_u(Ja,J,u);
			if (J<2.0){
				// Compute the least error = final_error
				final_error = compute_error(u,C);
				// printf(" final_error = %d\n",final_error);
			} // if (J<1.0){
			if (final_error == 0) { break; }
		} // for (j=1;j<max_itr;j++){

		printf("i=%d	error(%d->%d): J=%f \n",i,ini_error,final_error,J);
		if (final_error == 0) { break; }

		rand_u(u0,n);
		for (int q=0;q<n;q++){ u[q] = 0.5*u[q] + 0.5*u0[q]; }

	} // for (i=0;i<sample_size;i++){	 

	clock_gettime(CLOCK_REALTIME, &endTime);
	if (endTime.tv_nsec < startTime.tv_nsec) {
		long sec=endTime.tv_sec - startTime.tv_sec - 1;
		long nsec=endTime.tv_nsec + 1000000000 - startTime.tv_nsec;
		double f=sec+nsec/1000000000.0;
		printf("all time = %f\n",f);
		printf("itration time = %f\n",f/itr_count);
	} else {
		long sec=endTime.tv_sec - startTime.tv_sec;
		long nsec=endTime.tv_nsec - startTime.tv_nsec;
		double f=sec+nsec/1000000000.0;
		printf("all time = %f\n",f);
		printf("itration time = %f\n",f/itr_count);
	}

	free(base_M);
	free(M);
	free(base_PosOcc);
	free(PosOcc);
	free(base_NegOcc);
	free(NegOcc);

}
