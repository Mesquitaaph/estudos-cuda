/** 
  * Programa ... 
  * Executar com:
  * nvcc -o ex.exe ex.cu --run
  */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TAM_BLOCO 1024
#define N 1024*4*2
#define M 1024
#define L 512

// wrapper para checar erros nas chamadas de funções de CUDA
#define CUDA_SAFE_CALL(call) { \
  cudaError_t err = call; \
  if(err != cudaSuccess) { \
    fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  }\
}

#define SQR(x) ((x)*(x))
#define MIN(a, b) a < b ? a : b

clock_t start, end;

// kernel para execucao paralela na GPU
__global__ void multMatriz(double *A, double *B, double *C) {
  // coordenadas globais das threads
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if(i > N-1 || j > L-1) return;

  // calcula o elemento C(i,j)
  double valor = 0;
  for(int k = 0; k < M; k++) {
    valor += A[i*M + k] * B[k*L + j];
  }

  // escreve o valor calculado da matriz de saida
  C[i*L + j] = valor;
}

void multMatrizSeq(double *A, double *B, double *C) {
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < L; j++) {
      double valor = 0;
      for(int k = 0; k < M; k++) {
        valor += A[M*i + k] * B[k*L + j];
      }
      C[i*L + j] = valor;
    }
  }
}

int main() {
  double *A, *B, *C, *s_C;
  double *d_A, *d_B, *d_C;

  long mtx1_bytes = N*M*sizeof(double);
  long mtx2_bytes = M*L*sizeof(double);
  long mtx_res_bytes = N*L*sizeof(double);

  A = (double*) malloc(mtx1_bytes);
  B = (double*) malloc(mtx2_bytes);
  C = (double*) malloc(mtx_res_bytes);
  s_C = (double*) malloc(mtx_res_bytes);

  CUDA_SAFE_CALL(cudaMalloc((void**) &d_A, mtx1_bytes));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_B, mtx2_bytes));
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_C, mtx_res_bytes));

  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      A[i*M + j] = (double)((i*M + j) %525);
      // printf("%lf ", A[i*M + j]);
    }
    // printf("\n");
  }
  // printf("\n");

  for(int i = 0; i < M; i++) {
    for(int j = 0; j < L; j++) {
      B[i*L + j] = (double)((i*L + j) %332);
      // printf("%lf ", B[i*L + j]);
    }
    // printf("\n");
  }
  // printf("\n");
  
  start = clock();

  // A equacao threadsBloco.x * threadsBloco.y * threadsBloco.z = 1024 deve ser verdade para variaveis do tipo dim3 para que o kernel execute.
  dim3 threadsBloco(sqrt(TAM_BLOCO), sqrt(TAM_BLOCO));
  dim3 blocosGrade((N + threadsBloco.x - 1)/threadsBloco.x, (L + threadsBloco.y - 1)/threadsBloco.y);

  CUDA_SAFE_CALL(cudaMemcpy(d_A, A, mtx1_bytes, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_B, B, mtx2_bytes, cudaMemcpyHostToDevice));

  // printf("threadsBloco.x = %u, threadsBloco.y = %u\n", threadsBloco.x, threadsBloco.y);
  // printf("blocosGrade.x = %u, blocosGrade.y = %u\n", blocosGrade.x, blocosGrade.y);

  multMatriz<<<blocosGrade, threadsBloco>>>(d_A, d_B, d_C);
  cudaMemcpy(C, d_C, mtx_res_bytes, cudaMemcpyDeviceToHost);

  end = clock();
  printf("Tempo Paralelo = %ld\n", end - start);

  cudaFree(d_A); cudaFree(d_B); cudaFree(d_B);
  
  start = clock();

  multMatrizSeq(A, B, s_C);

  end = clock();
  printf("Tempo Sequencial = %ld\n", end - start);

  int correct = 1;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < L; j++) {
      if(s_C[i*L + j] != C[i*L + j]) {
        correct = 0;
        printf("C[%d][%d] = %lf\n", i, j, C[i*L + j]);
        break;
      }
      // printf("%lf", s_C[i*L + j]);
    }
    // printf("\n");
    if(correct == 0) break;
  }
  // printf("\n");

  free(A); free(B); free(C); 

  if(correct) printf("Correto\n");
  else printf("Incorreto\n");

  return 1;
}