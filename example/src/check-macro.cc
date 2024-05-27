#include <stdio.h>

int main(int argc, char* argv[])
{
#ifdef __clang__
  printf("__clang__ is defined\n");
#endif

#ifdef __HIPCC__
  printf("__HIPCC__ is defined\n");
#endif

#ifdef __CUDACC__
  printf("__CUDACC__ is defined\n");
#endif
#ifdef __NVCC__
  printf("__NVCC__ is defined\n");
#endif

  return 0;
}
