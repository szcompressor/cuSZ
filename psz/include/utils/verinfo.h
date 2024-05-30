#ifndef C8F01682_B23E_44FD_A419_10254525A26F
#define C8F01682_B23E_44FD_A419_10254525A26F

#ifdef __cplusplus
extern "C" {
#endif

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

void print_CXX_ver();
void print_NVCC_ver();
int print_CUDA_driver();
int print_NVIDIA_driver();
void CUDA_devices();

#define membw_base1000 DDR_memory_bandwidth_GBps_base1000
#define membw_base1024 DDR_memory_bandwidth_GiBps_base1024
float DDR_memory_bandwidth_GBps_base1000(
    float mem_bus_bitwidth, float clock_rate);
float DDR_memory_bandwidth_GiBps_base1024(
    float mem_bus_bitwidth, float clock_rate_Hz);

#ifdef __cplusplus
}
#endif

#endif /* C8F01682_B23E_44FD_A419_10254525A26F */
