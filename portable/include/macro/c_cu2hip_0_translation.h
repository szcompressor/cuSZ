#define cudaStream_t hipStream_t
#define cudaStreamCreate(...) hipStreamCreate(__VA_ARGS__)
#define cudaStreamDestroy(...) hipStreamDestroy(__VA_ARGS__)

#define cudaDeviceSynchronize(...) hipDeviceSynchronize(__VA_ARGS__)
#define cudaStreamSynchronize(STREAM) hipStreamSynchronize((hipStream_t)STREAM)

#define cudaMemcpy(...) hipMemcpy(__VA_ARGS__)
#define cudaMemcpyAsync(...) hipMemcpyAsync(__VA_ARGS__)
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define cudaMalloc(...) hipMalloc(__VA_ARGS__)
#define cudaMallocHost(...) hipHostMalloc(__VA_ARGS__)
#define cudaMallocManaged(...) hipMallocManaged(__VA_ARGS__)
#define cudaMallocShared(...) hipMallocManaged(__VA_ARGS__)
#define cudaFree(...) hipFree(__VA_ARGS__)
#define cudaFreeHost(...) hipHostFree(__VA_ARGS__)
#define cudaMemset(...) hipMemset(__VA_ARGS__)

#define cudaGetDevice(...) hipGetDevice(__VA_ARGS__)
#define cudaSetDevice(...) hipSetDevice(__VA_ARGS__)
#define cudaDeviceProp hipDeviceProp_t
#define cudaGetDeviceProperties(...) hipGetDeviceProperties(__VA_ARGS__)
#define cudaDeviceGetAttribute(...) hipDeviceGetAttribute(__VA_ARGS__)
#define CUdeviceptr hipDeviceptr_t
#define cuMemGetAddressRange(...) hipMemGetAddressRange(__VA_ARGS__)
#define cudaFuncSetAttribute(...) hipFuncSetAttribute(__VA_ARGS__)
#define cudaFuncAttribute hipFuncAttribute
#define cudaDriverGetVersion(...) hipDriverGetVersion(__VA_ARGS__)
#define cudaRuntimeGetVersion(...) hipRuntimeGetVersion(__VA_ARGS__)
#define cudaGetDeviceCount(...) hipGetDeviceCount(__VA_ARGS__)

#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString(...) hipGetErrorString(__VA_ARGS__)
#define cudaGetLastError(...) hipGetLastError(__VA_ARGS__)

#define cudaDeviceReset(...) hipDeviceReset(__VA_ARGS__)