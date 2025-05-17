#ifndef AE6DCA2E_F19B_41DB_80CB_11230E548F92
#define AE6DCA2E_F19B_41DB_80CB_11230E548F92

#include "detail/busyheader.hh"

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)

#include <exception>
#include <sstream>

namespace psz {

struct exception_gpu_general : public std::exception {
  exception_gpu_general(cudaError_t gpu_error_status, const char* _file_, const int _line_)
  {
    const char* err = cudaGetErrorString(gpu_error_status);
    std::stringstream ss;
    ss << "GPU API failed at \e[31m\e[1m" << _file_ << ':' << _line_;
    ss << "\e[0m with error: " << err << '(' << (int)gpu_error_status << ')';
    err_msg = ss.str();
  }
  const char* what() const noexcept { return err_msg.c_str(); }
  std::string err_msg;
};

}  // namespace psz

// proxy: not safe to put throw inside a macro expansion
static void throw_exception_gpu_general(
    cudaError_t GPU_ERROR_CODE, const char* _file_, const int _line_)
{
  if (cudaSuccess != GPU_ERROR_CODE) {
    throw psz::exception_gpu_general(GPU_ERROR_CODE, _file_, _line_);
  }
}

#define CHECK_GPU(GPU_ERROR_CODE) (throw_exception_gpu_general(GPU_ERROR_CODE, __FILE__, __LINE__))

#define AD_HOC_CHECK_GPU_WITH_LINE(GPU_ERROR_CODE, FILE, LINE) \
  (throw_exception_gpu_general(GPU_ERROR_CODE, FILE, LINE))

#elif defined(PSZ_USE_1API)

/*
DPCT1009:2: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
#define cudaGetErrorString(...) \
  "cudaGetErrorString is not supported" /*cudaGetErrorString(__VA_ARGS__)*/

struct psz_gpu_exception : public std::exception {
  psz_gpu_exception(const char* err, int err_code, const char* file, int line)
  {
    std::stringstream ss;
    ss << "GPU API failed at \e[31m\e[1m" << file << ':' << line << "\e[0m with error: " << err
       << '(' << err_code << ')';
    err_msg = ss.str();
  }
  const char* what() const noexcept { return err_msg.c_str(); }
  std::string err_msg;
};

static void psz_check_gpu_error_impl(cudaError_t status, const char* file, int line)
{
  /*
  DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
  */
  if (cudaSuccess != status) {
    /*
    DPCT1001:0: The statement could not be removed.
    */
    throw psz_gpu_exception(cudaGetErrorString(status), status, file, line);
  }
}

#endif

#endif /* AE6DCA2E_F19B_41DB_80CB_11230E548F92 */
