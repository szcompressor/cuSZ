#ifndef AE6DCA2E_F19B_41DB_80CB_11230E548F92
#define AE6DCA2E_F19B_41DB_80CB_11230E548F92

#include "busyheader.hh"
#include "port.hh"

#if defined(PSZ_USE_CUDA)

#include <cuda_runtime.h>

#elif defined(PSZ_USE_HIP)

#include <hip/hip_runtime.h>

#elif defined(PSZ_USE_1API)

#warning "compile-time warning: need to put general entry to oneapi"

#endif

struct psz_gpu_exception : public std::exception {
  psz_gpu_exception(
      const char* err, int err_code, const char* file, int line)
  {
    std::stringstream ss;
    ss << "GPU API failed at \e[31m\e[1m" << file << ':' << line
       << "\e[0m with error: " << err << '(' << err_code << ')';
    err_msg = ss.str();
  }
  const char* what() const noexcept { return err_msg.c_str(); }
  std::string err_msg;
};

static void psz_check_gpu_error_impl(GpuErrorT status, const char* file, int line)
{
  if (GpuSuccess != status) {
    throw psz_gpu_exception(GpuGetErrorString(status), status, file, line);
  }
}

#define CHECK_GPU(err) (psz_check_gpu_error_impl(err, __FILE__, __LINE__))


#endif /* AE6DCA2E_F19B_41DB_80CB_11230E548F92 */
