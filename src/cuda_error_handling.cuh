#ifndef CUDA_ERROR_HANDLING
#define CUDA_ERROR_HANDLING

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdio>

// back compatibility start
static void HandleError(cudaError_t err, const char* file, int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
// back compatibility end

static void check_cuda_error(cudaError_t status, const char* file, int line)
{
    if (cudaSuccess != status) {
        printf("\nCUDA error/status reference (as of CUDA 11):\n");
        printf("cudaSuccess                         -> %d\n", cudaSuccess);
        printf("cudaErrorInvalidValue               -> %d\n", cudaErrorInvalidValue);
        printf("cudaErrorMemoryAllocation           -> %d\n", cudaErrorMemoryAllocation);
        printf("cudaErrorInitializationError        -> %d\n", cudaErrorInitializationError);
        printf("cudaErrorCudartUnloading            -> %d\n", cudaErrorCudartUnloading);
        printf("cudaErrorProfilerDisabled           -> %d\n", cudaErrorProfilerDisabled);
        printf("cudaErrorProfilerNotInitialized (Deprecated)-> %d\n", cudaErrorProfilerNotInitialized);
        printf("cudaErrorProfilerAlreadyStarted (Deprecated)-> %d\n", cudaErrorProfilerAlreadyStarted);
        printf("cudaErrorProfilerAlreadyStopped (Deprecated)-> %d\n", cudaErrorProfilerAlreadyStopped);
        printf("cudaErrorInvalidConfiguration       -> %d\n", cudaErrorInvalidConfiguration);
        printf("cudaErrorInvalidPitchValue          -> %d\n", cudaErrorInvalidPitchValue);
        printf("cudaErrorInvalidSymbol              -> %d\n", cudaErrorInvalidSymbol);
        printf("cudaErrorInvalidHostPointer     (Deprecated)-> %d\n", cudaErrorInvalidHostPointer);
        printf("cudaErrorInvalidDevicePointer   (Deprecated)-> %d\n", cudaErrorInvalidDevicePointer);
        printf("cudaErrorInvalidTexture             -> %d\n", cudaErrorInvalidTexture);
        printf("cudaErrorInvalidTextureBinding      -> %d\n", cudaErrorInvalidTextureBinding);
        printf("cudaErrorInvalidChannelDescriptor   -> %d\n", cudaErrorInvalidChannelDescriptor);
        printf("cudaErrorInvalidMemcpyDirection     -> %d\n", cudaErrorInvalidMemcpyDirection);
        printf("cudaErrorAddressOfConstant      (Deprecated)-> %d\n", cudaErrorAddressOfConstant);
        printf("cudaErrorTextureFetchFailed     (Deprecated)-> %d\n", cudaErrorTextureFetchFailed);
        printf("cudaErrorTextureNotBound        (Deprecated)-> %d\n", cudaErrorTextureNotBound);
        printf("cudaErrorSynchronizationError   (Deprecated)-> %d\n", cudaErrorSynchronizationError);
        printf("cudaErrorInvalidFilterSetting       -> %d\n", cudaErrorInvalidFilterSetting);
        printf("cudaErrorInvalidNormSetting         -> %d\n", cudaErrorInvalidNormSetting);
        printf("cudaErrorMixedDeviceExecution   (Deprecated)-> %d\n", cudaErrorMixedDeviceExecution);
        printf("cudaErrorNotYetImplemented      (Deprecated)-> %d\n", cudaErrorNotYetImplemented);
        printf("cudaErrorMemoryValueTooLarge    (Deprecated)-> %d\n", cudaErrorMemoryValueTooLarge);
        printf("cudaErrorInsufficientDriver         -> %d\n", cudaErrorInsufficientDriver);
        printf("cudaErrorInvalidSurface             -> %d\n", cudaErrorInvalidSurface);
        printf("cudaErrorDuplicateVariableName      -> %d\n", cudaErrorDuplicateVariableName);
        printf("cudaErrorDuplicateTextureName       -> %d\n", cudaErrorDuplicateTextureName);
        printf("cudaErrorDuplicateSurfaceName       -> %d\n", cudaErrorDuplicateSurfaceName);
        printf("cudaErrorDevicesUnavailable         -> %d\n", cudaErrorDevicesUnavailable);
        printf("cudaErrorIncompatibleDriverContext  -> %d\n", cudaErrorIncompatibleDriverContext);
        printf("cudaErrorMissingConfiguration       -> %d\n", cudaErrorMissingConfiguration);
        printf("cudaErrorPriorLaunchFailure     (Deprecated)-> %d\n", cudaErrorPriorLaunchFailure);
        printf("cudaErrorLaunchMaxDepthExceeded     -> %d\n", cudaErrorLaunchMaxDepthExceeded);
        printf("cudaErrorLaunchFileScopedTex        -> %d\n", cudaErrorLaunchFileScopedTex);
        printf("cudaErrorLaunchFileScopedSurf       -> %d\n", cudaErrorLaunchFileScopedSurf);
        printf("cudaErrorSyncDepthExceeded          -> %d\n", cudaErrorSyncDepthExceeded);
        printf("cudaErrorLaunchPendingCountExceeded -> %d\n", cudaErrorLaunchPendingCountExceeded);
        printf("cudaErrorInvalidDeviceFunction      -> %d\n", cudaErrorInvalidDeviceFunction);
        printf("cudaErrorNoDevice                   -> %d\n", cudaErrorNoDevice);
        printf("cudaErrorInvalidDevice              -> %d\n", cudaErrorInvalidDevice);
        printf("cudaErrorStartupFailure             -> %d\n", cudaErrorStartupFailure);
        printf("cudaErrorInvalidKernelImage         -> %d\n", cudaErrorInvalidKernelImage);
        printf("cudaErrorDeviceUninitialized        -> %d\n", cudaErrorDeviceUninitialized);
        printf("cudaErrorMapBufferObjectFailed      -> %d\n", cudaErrorMapBufferObjectFailed);
        printf("cudaErrorUnmapBufferObjectFailed    -> %d\n", cudaErrorUnmapBufferObjectFailed);
        printf("cudaErrorArrayIsMapped              -> %d\n", cudaErrorArrayIsMapped);
        printf("cudaErrorAlreadyMapped              -> %d\n", cudaErrorAlreadyMapped);
        printf("cudaErrorNoKernelImageForDevice     -> %d\n", cudaErrorNoKernelImageForDevice);
        printf("cudaErrorAlreadyAcquired            -> %d\n", cudaErrorAlreadyAcquired);
        printf("cudaErrorNotMapped                  -> %d\n", cudaErrorNotMapped);
        printf("cudaErrorNotMappedAsArray           -> %d\n", cudaErrorNotMappedAsArray);
        printf("cudaErrorNotMappedAsPointer         -> %d\n", cudaErrorNotMappedAsPointer);
        printf("cudaErrorECCUncorrectable           -> %d\n", cudaErrorECCUncorrectable);
        printf("cudaErrorUnsupportedLimit           -> %d\n", cudaErrorUnsupportedLimit);
        printf("cudaErrorDeviceAlreadyInUse         -> %d\n", cudaErrorDeviceAlreadyInUse);
        printf("cudaErrorPeerAccessUnsupported      -> %d\n", cudaErrorPeerAccessUnsupported);
        printf("cudaErrorInvalidPtx                 -> %d\n", cudaErrorInvalidPtx);
        printf("cudaErrorInvalidGraphicsContext     -> %d\n", cudaErrorInvalidGraphicsContext);
        printf("cudaErrorNvlinkUncorrectable        -> %d\n", cudaErrorNvlinkUncorrectable);
        printf("cudaErrorJitCompilerNotFound        -> %d\n", cudaErrorJitCompilerNotFound);
        printf("cudaErrorInvalidSource              -> %d\n", cudaErrorInvalidSource);
        printf("cudaErrorFileNotFound               -> %d\n", cudaErrorFileNotFound);
        printf("cudaErrorSharedObjectSymbolNotFound -> %d\n", cudaErrorSharedObjectSymbolNotFound);
        printf("cudaErrorSharedObjectInitFailed     -> %d\n", cudaErrorSharedObjectInitFailed);
        printf("cudaErrorOperatingSystem            -> %d\n", cudaErrorOperatingSystem);
        printf("cudaErrorInvalidResourceHandle      -> %d\n", cudaErrorInvalidResourceHandle);
        printf("cudaErrorIllegalState               -> %d\n", cudaErrorIllegalState);
        printf("cudaErrorSymbolNotFound             -> %d\n", cudaErrorSymbolNotFound);
        printf("cudaErrorNotReady                   -> %d\n", cudaErrorNotReady);
        printf("cudaErrorIllegalAddress             -> %d\n", cudaErrorIllegalAddress);
        printf("cudaErrorLaunchOutOfResources       -> %d\n", cudaErrorLaunchOutOfResources);
        printf("cudaErrorLaunchTimeout              -> %d\n", cudaErrorLaunchTimeout);
        printf("cudaErrorLaunchIncompatibleTexturing-> %d\n", cudaErrorLaunchIncompatibleTexturing);
        printf("cudaErrorPeerAccessAlreadyEnabled   -> %d\n", cudaErrorPeerAccessAlreadyEnabled);
        printf("cudaErrorPeerAccessNotEnabled       -> %d\n", cudaErrorPeerAccessNotEnabled);
        printf("cudaErrorSetOnActiveProcess         -> %d\n", cudaErrorSetOnActiveProcess);
        printf("cudaErrorContextIsDestroyed         -> %d\n", cudaErrorContextIsDestroyed);
        printf("cudaErrorAssert                     -> %d\n", cudaErrorAssert);
        printf("cudaErrorTooManyPeers               -> %d\n", cudaErrorTooManyPeers);
        printf("cudaErrorHostMemoryAlreadyRegistered-> %d\n", cudaErrorHostMemoryAlreadyRegistered);
        printf("cudaErrorHostMemoryNotRegistered    -> %d\n", cudaErrorHostMemoryNotRegistered);
        printf("cudaErrorHardwareStackError         -> %d\n", cudaErrorHardwareStackError);
        printf("cudaErrorIllegalInstruction         -> %d\n", cudaErrorIllegalInstruction);
        printf("cudaErrorMisalignedAddress          -> %d\n", cudaErrorMisalignedAddress);
        printf("cudaErrorInvalidAddressSpace        -> %d\n", cudaErrorInvalidAddressSpace);
        printf("cudaErrorInvalidPc                  -> %d\n", cudaErrorInvalidPc);
        printf("cudaErrorLaunchFailure              -> %d\n", cudaErrorLaunchFailure);
        printf("cudaErrorCooperativeLaunchTooLarge  -> %d\n", cudaErrorCooperativeLaunchTooLarge);
        printf("cudaErrorNotPermitted               -> %d\n", cudaErrorNotPermitted);
        printf("cudaErrorNotSupported               -> %d\n", cudaErrorNotSupported);
        printf("cudaErrorSystemNotReady             -> %d\n", cudaErrorSystemNotReady);
        printf("cudaErrorSystemDriverMismatch       -> %d\n", cudaErrorSystemDriverMismatch);
        printf("cudaErrorCompatNotSupportedOnDevice -> %d\n", cudaErrorCompatNotSupportedOnDevice);
        printf("cudaErrorStreamCaptureUnsupported   -> %d\n", cudaErrorStreamCaptureUnsupported);
        printf("cudaErrorStreamCaptureInvalidated   -> %d\n", cudaErrorStreamCaptureInvalidated);
        printf("cudaErrorStreamCaptureMerge         -> %d\n", cudaErrorStreamCaptureMerge);
        printf("cudaErrorStreamCaptureUnmatched     -> %d\n", cudaErrorStreamCaptureUnmatched);
        printf("cudaErrorStreamCaptureUnjoined      -> %d\n", cudaErrorStreamCaptureUnjoined);
        printf("cudaErrorStreamCaptureIsolation     -> %d\n", cudaErrorStreamCaptureIsolation);
        printf("cudaErrorStreamCaptureImplicit      -> %d\n", cudaErrorStreamCaptureImplicit);
        printf("cudaErrorCapturedEvent              -> %d\n", cudaErrorCapturedEvent);
        printf("cudaErrorStreamCaptureWrongThread   -> %d\n", cudaErrorStreamCaptureWrongThread);
        printf("cudaErrorTimeout                    -> %d\n", cudaErrorTimeout);
        printf("cudaErrorGraphExecUpdateFailure     -> %d\n", cudaErrorGraphExecUpdateFailure);
        printf("cudaErrorUnknown                    -> %d\n", cudaErrorUnknown);
        printf("cudaErrorApiFailureBase (Deprecated)-> %d\n", cudaErrorApiFailureBase);
        printf("\n");
        printf("CUDA API failed at \e[31m\e[1m%s:%d\e[0m with error: %s (%d)\n", file, line, cudaGetErrorString(status), status);
        exit(EXIT_FAILURE);
    }
}

static void check_cusparse_error(cusparseStatus_t status, const char* file, int line)
{
    if (CUSPARSE_STATUS_SUCCESS != status) {
        printf("\nCUSPARSE status reference (as of CUDA 11):\n");
        printf("CUSPARSE_STATUS_SUCCESS                   -> %d\n", CUSPARSE_STATUS_SUCCESS);
        printf("CUSPARSE_STATUS_NOT_INITIALIZED           -> %d\n", CUSPARSE_STATUS_NOT_INITIALIZED);
        printf("CUSPARSE_STATUS_ALLOC_FAILED              -> %d\n", CUSPARSE_STATUS_ALLOC_FAILED);
        printf("CUSPARSE_STATUS_INVALID_VALUE             -> %d\n", CUSPARSE_STATUS_INVALID_VALUE);
        printf("CUSPARSE_STATUS_ARCH_MISMATCH             -> %d\n", CUSPARSE_STATUS_ARCH_MISMATCH);
        printf("CUSPARSE_STATUS_EXECUTION_FAILED          -> %d\n", CUSPARSE_STATUS_EXECUTION_FAILED);
        printf("CUSPARSE_STATUS_INTERNAL_ERROR            -> %d\n", CUSPARSE_STATUS_INTERNAL_ERROR);
        printf("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED -> %d\n", CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
        printf("CUSPARSE_STATUS_NOT_SUPPORTED             -> %d\n", CUSPARSE_STATUS_NOT_SUPPORTED);
        // printf("CUSPARSE_STATUS_INSUFFICIENT_RESOURCES    -> %d\n", CUSPARSE_STATUS_INSUFFICIENT_RESOURCES); // not working before CUDA 11
        printf("\n");
        printf("CUSPARSE API failed at \e[31m\e[1m%s:%d\e[0m with error: %s (%d)\n", file, line, cusparseGetErrorString(status), status);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(err) (check_cuda_error(err, __FILE__, __LINE__))
#define CHECK_CUSPARSE(err) (check_cusparse_error(err, __FILE__, __LINE__))

#endif
