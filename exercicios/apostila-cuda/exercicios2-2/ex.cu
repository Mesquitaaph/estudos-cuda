/** 
  * Programa ... 
  * Executar com:
  * nvcc -o ex.exe ex.cu --run
  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 1024*4*2

// wrapper para checar erros nas chamadas de funções de CUDA
#define CUDA_SAFE_CALL(call) { \
  cudaError_t err = call; \
  if(err != cudaSuccess) { \
    fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  }\
}

#define SQR(x) ((x)*(x))

clock_t start, end;

__global__ void prod_int_p(double* a, double* b, double* c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockDim.x;
  
  if(i > N-1) return;

  extern __shared__ double comp_sum[];
  double *sum = comp_sum;

  sum[threadIdx.x] = (a[i]) * (b[i]);
  // printf("sum[%d] = %lf\n", i, sum[threadIdx.x]);
  __syncthreads();

  for(int passo = 1; passo < n; passo *= 2) {
    double aux;
    if(threadIdx.x >= passo) aux = sum[threadIdx.x-passo];

    __syncthreads();

    if(threadIdx.x >= passo) sum[threadIdx.x] = sum[threadIdx.x] + aux;

    __syncthreads();
  }

  if(threadIdx.x == (n-1)) {
    c[blockIdx.x] = sum[threadIdx.x];
    // printf("c[%d] = %lf\n", blockIdx.x, c[blockIdx.x]);
  }
}

// __global__ void prod_int_p(double* a, double* b, double* c, double* sum) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   int n = blockDim.x;

//   if(i > N-1) return;

//   // extern __shared__ double sum[];
//   sum[i] = (a[i]) * (b[i]);
//   __syncthreads();

//   for(int passo = 1; passo < n; passo *= 2) {
//     double aux;
//     if(threadIdx.x >= passo) aux = sum[i-passo];

//     __syncthreads();

//     if(threadIdx.x >= passo) sum[i] = sum[i] + aux;

//     __syncthreads();
//   }

//   if(threadIdx.x == (n-1)) {
//     c[blockIdx.x] = sum[i];
//   }
// }

void prod_int(double* a, double* b, double* sum) {
  for(int i = 0; i < N; i++) {
    *sum += (a[i]) * (b[i]);
  }
}

int main() {
  double *v_a, *v_b, *v_sum, *s_sum, *d_a, *d_b, *d_sum;
  double *v_1, *d_1;
  double soma_p = 0;

  int n_bytes = N*sizeof(double);
  int n_threads = 32;
  int n_blocos1 = (N + n_threads-1)/n_threads, n_blocos2;

  size_t quant_mem = n_blocos1 * sizeof(double);

  v_a = (double*) malloc(n_bytes);
  v_b = (double*) malloc(n_bytes);
  v_sum = (double*) malloc(quant_mem);
  v_1 = (double*) malloc(quant_mem);
  s_sum = (double*) malloc(sizeof(double));

  for(int i = 0; i < N; i++) {
    v_a[i] = (double)(i%525);
    v_b[i] = (double)(i%332);

    if(i < n_blocos1) v_1[i] = 1.0;
  }
  
  *s_sum = 0.0;
  
  cudaMalloc((void**) &d_a, n_bytes);
  cudaMalloc((void**) &d_b, n_bytes);
  cudaMalloc((void**) &d_sum, quant_mem);
  cudaMalloc((void**) &d_1, quant_mem);

  cudaMemcpy(d_a, v_a, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, v_b, n_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_1, v_1, quant_mem, cudaMemcpyHostToDevice);


  start = clock();
  prod_int_p<<<n_blocos1, n_threads, n_threads*sizeof(double)>>>(d_a, d_b, d_sum);

  printf("n_blocos1 = %d\n", n_blocos1);
  n_blocos2 = n_blocos1/n_threads;

  if(n_blocos2 > 0) {
    prod_int_p<<<n_blocos2, n_threads, n_blocos2*sizeof(double)>>>(d_sum, d_1, d_a);
    cudaMemcpy(v_sum, d_a, n_blocos2*sizeof(double), cudaMemcpyDeviceToHost);
    for(int k = 0; k < n_blocos2; k++) soma_p += v_sum[k];
  } else {
    cudaMemcpy(v_sum, d_sum, quant_mem, cudaMemcpyDeviceToHost);
    for(int k = 0; k < n_blocos1; k++) soma_p += v_sum[k];
  }

  end = clock();
  printf("Tempo Paralelo = %ld\n", end - start);

  cudaFree(d_sum); cudaFree(d_a); cudaFree(d_b); cudaFree(d_1);

  start = clock();
  prod_int(v_a, v_b, s_sum);
  end = clock();
  printf("Tempo Sequencial = %ld\n", end - start);


  printf("Sequencial = %lf | Paralelo = %lf\n", *s_sum, soma_p);

  int correct = *s_sum == soma_p;
  if(correct) printf("Correto\n");
  else printf("Incorreto\n");


  free(v_a); free(v_b); free(v_sum); free(s_sum); free(v_1);
  return 0;
}