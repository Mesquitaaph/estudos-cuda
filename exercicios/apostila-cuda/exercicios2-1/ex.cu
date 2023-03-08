/** 
  * Programa ... 
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

typedef struct {
  double x;
  double y;
  double z;
  double dist;
} POINT;

time_t start, end;

double dist_s(POINT p1, POINT p2) {
  return sqrt(SQR(p1.x - p2.x) + SQR(p1.y - p2.y) + SQR(p1.z - p2.z));
}

__device__ double dist(POINT p1, POINT p2) {
  return sqrt(SQR(p1.x - p2.x) + SQR(p1.y - p2.y) + SQR(p1.z - p2.z));
}

__global__ void calcula_dist_p(POINT* p, POINT* pts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("%d\n", i);
  pts[i].dist = dist(p[0], pts[i]);
}

void calcula_dist(POINT p, POINT* pts) {
  for(int i = 0; i < N; i++) {
    pts[i].dist = dist_s(p, pts[i]);
  }
  // printf("dist = %lf\n", pts[4].dist);
}

int main() {
  POINT p[1] = {{0.0, 0.0, 0.0, 0.0}};
  POINT pontos[N], pontos_s[N], *d_pontos, *d_p;
  int n_bytes = N*sizeof(POINT);

  for(int i = 0; i < N; i++) {
    pontos[i] = {(double)(i%1024), (double)(i%525), (double)(i%332), 0.0};
    pontos_s[i] = {(double)(i%1024), (double)(i%525), (double)(i%332), 0.0};
  }

  cudaMalloc((void**) &d_p, sizeof(POINT));
  cudaMalloc((void**) &d_pontos, n_bytes);

  cudaMemcpy(d_p, p, sizeof(POINT), cudaMemcpyHostToDevice);
  cudaMemcpy(d_pontos, pontos, n_bytes, cudaMemcpyHostToDevice);

  int n_threads = 1024;
  int n_blocos = (N + n_threads-1)/n_threads;

  start = time(NULL);
  calcula_dist_p<<<n_blocos, n_threads>>>(d_p, d_pontos);
  end = time(NULL);
  printf("Tempo Paralelo = %I64d\n", end - start);

  cudaMemcpy(pontos, d_pontos, n_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_p); cudaFree(d_pontos);

  start = time(NULL);
  calcula_dist(p[0], pontos_s);
  end = time(NULL);
  printf("Tempo Sequencial = %I64d\n", end - start);

  int correct = 1;
  for(int i = 0; i < N; i++) {
    if(pontos[i].dist != pontos_s[i].dist) {
      correct = 0;
      // printf("%lf, %lf\n", pontos[i].dist, pontos_s[i].dist);
    }
  }

  if(correct) printf("Correto\n");
  else printf("Incorreto\n");

  return 0;
}