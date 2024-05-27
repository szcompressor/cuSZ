#define GpuStreamT cudaStream_t
#define GpuStreamCreate(...) cudaStreamCreate(__VA_ARGS__)
#define GpuStreamDestroy(...) cudaStreamDestroy(__VA_ARGS__)

#define GpuDeviceSync(...) cudaDeviceSynchronize(__VA_ARGS__)
#define GpuStreamSync(STREAM) cudaStreamSynchronize((cudaStream_t)STREAM)

#define GpuMemcpy(...) cudaMemcpy(__VA_ARGS__)
#define GpuMemcpyAsync(...) cudaMemcpyAsync(__VA_ARGS__)
#define GpuMemcpyKind cudaMemcpyKind
#define GpuMemcpyH2D cudaMemcpyHostToDevice
#define GpuMemcpyH2H cudaMemcpyHostToHost
#define GpuMemcpyD2H cudaMemcpyDeviceToHost
#define GpuMemcpyD2D cudaMemcpyDeviceToDevice

#define GpuMalloc(...) cudaMalloc(__VA_ARGS__)
#define GpuMallocHost(...) cudaMallocHost(__VA_ARGS__)
#define GpuMallocManaged(...) cudaMallocManaged(__VA_ARGS__)
#define GpuMallocShared(...) cudaMallocManaged(__VA_ARGS__)
#define GpuFree(...) cudaFree(__VA_ARGS__)
#define GpuFreeHost(...) cudaFreeHost(__VA_ARGS__)
#define GpuMemset(...) cudaMemset(__VA_ARGS__)

#define GpuGetDevice(...) cudaGetDevice(__VA_ARGS__)
#define GpuSetDevice(...) cudaSetDevice(__VA_ARGS__)
#define GpuDeviceProp cudaDeviceProp
#define GpuGetDeviceProperties(...) cudaGetDeviceProperties(__VA_ARGS__)
#define GpuDeviceGetAttribute(...) cudaDeviceGetAttribute(__VA_ARGS__)
#define GpuDevicePtr CUdeviceptr
#define GpuMemGetAddressRange(...) cuMemGetAddressRange(__VA_ARGS__)
#define GpuFuncSetAttribute(...) cudaFuncSetAttribute(__VA_ARGS__)
#define GpuFuncAttribute cudaFuncAttribute
#define GpuDriverGetVersion(...) cudaDriverGetVersion(__VA_ARGS__)
#define GpuRuntimeGetVersion(...) cudaRuntimeGetVersion(__VA_ARGS__)
#define GpuGetDeviceCount(...) cudaGetDeviceCount(__VA_ARGS__)

#define GpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#define GpuDevAttrMaxSharedMemoryPerBlock cudaDevAttrMaxSharedMemoryPerBlock
#define GpuDevAttrMaxSharedMemoryPerBlockOptin cudaDevAttrMaxSharedMemoryPerBlockOptin
#define GpuFuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize

#define GpuErrorT cudaError_t
#define GpuSuccess cudaSuccess
#define GpuGetErrorString(...) cudaGetErrorString(__VA_ARGS__)
#define GpuGetLastError(...) cudaGetLastError(__VA_ARGS__)

#define GpuDeviceReset(...) cudaDeviceReset(__VA_ARGS__)
