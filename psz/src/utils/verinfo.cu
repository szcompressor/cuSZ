#include <nvml.h>

#include <cstdio>

#include "utils/verinfo.h"

// REF: https://stackoverflow.com/a/70302416
int print_NVIDIA_driver()
{
  auto retval = nvmlInit();
  if (NVML_SUCCESS != retval) {
    printf("(failed to initialize NVML: %s)\n", nvmlErrorString(retval));
    return 1;
  }

  char version_str[NVML_DEVICE_PART_NUMBER_BUFFER_SIZE + 1];
  retval = nvmlSystemGetDriverVersion(
      version_str, NVML_DEVICE_PART_NUMBER_BUFFER_SIZE);
  if (retval != NVML_SUCCESS) {
    fprintf(stderr, "%s\n", nvmlErrorString(retval));
    return 1;
  }

  nvmlShutdown();
  printf("- NVIDIA driver: %s\n", version_str);

  return 0;
}