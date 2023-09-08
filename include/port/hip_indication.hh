#define GpuStreamT hipStream_t
#define GpuStreamCreate(...) hipStreamCreate(__VA_ARGS__)
#define GpuStreamDestroy(...) hipStreamDestroy(__VA_ARGS__)

#define GpuDeviceSync(...) hipDeviceSynchronize(__VA_ARGS__)
#define GpuStreamSync(STREAM) hipStreamSynchronize((hipStream_t)STREAM)

#define GpuMemcpy(...) hipMemcpy(__VA_ARGS__)
#define GpuMemcpyAsync(...) hipMemcpyAsync(__VA_ARGS__)
#define GpuMemcpyKind hipMemcpyKind
#define GpuMemcpyH2D hipMemcpyHostToDevice
#define GpuMemcpyH2H hipMemcpyHostToHost
#define GpuMemcpyD2H hipMemcpyDeviceToHost
#define GpuMemcpyD2D hipMemcpyDeviceToDevice

#define GpuMalloc(...) hipMalloc(__VA_ARGS__)
#define GpuMallocHost(...) hipHostMalloc(__VA_ARGS__)
#define GpuMallocManaged(...) hipMallocManaged(__VA_ARGS__)
#define GpuMallocShared(...) hipMallocManaged(__VA_ARGS__)
#define GpuFree(...) hipFree(__VA_ARGS__)
#define GpuFreeHost(...) hipHostFree(__VA_ARGS__)
#define GpuMemset(...) hipMemset(__VA_ARGS__)

#define GpuGetDevice(...) hipGetDevice(__VA_ARGS__)
#define GpuSetDevice(...) hipSetDevice(__VA_ARGS__)
#define GpuDeviceProp hipDeviceProp_t
#define GpuGetDeviceProperties(...) hipGetDeviceProperties(__VA_ARGS__)
#define GpuDeviceGetAttribute(...) hipDeviceGetAttribute(__VA_ARGS__)
#define GpuDevicePtr hipDeviceptr_t
#define GpuMemGetAddressRange(...) hipMemGetAddressRange(__VA_ARGS__)
#define GpuFuncSetAttribute(...) hipFuncSetAttribute(__VA_ARGS__)
#define GpuFuncAttribute hipFuncAttribute
#define GpuDriverGetVersion(...) hipDriverGetVersion(__VA_ARGS__)
#define GpuRuntimeGetVersion(...) hipRuntimeGetVersion(__VA_ARGS__)
#define GpuGetDeviceCount(...) hipGetDeviceCount(__VA_ARGS__)

#define GpuDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define GpuDevAttrMaxSharedMemoryPerBlock hipDeviceAttributeMaxSharedMemoryPerBlock
#define GpuDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define GpuFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

#define GpuErrorT hipError_t
#define GpuSuccess hipSuccess
#define GpuGetErrorString(...) hipGetErrorString(__VA_ARGS__)
#define GpuGetLastError(...) hipGetLastError(__VA_ARGS__)

#define GpuDeviceReset(...) hipDeviceReset(__VA_ARGS__)