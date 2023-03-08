/** 
  * Programa inicial apenas para testar a compilação de aplicações CUDA (.cu)
  */

#include <stdio.h>
#include <stdlib.h>

// wrapper para checar erros nas chamadas de funções de CUDA
#define CUDA_SAFE_CALL(call) { \
  cudaError_t err = call; \
  if(err != cudaSuccess) { \
    fprintf(stderr,"Erro no arquivo '%s', linha %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
  }\
}

int main() {
  printf("Hello world\n");

  return 0;
}