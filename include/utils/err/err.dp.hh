#ifndef F7DB8F4D_0B6F_4137_A719_D30533955896
#define F7DB8F4D_0B6F_4137_A719_D30533955896

/*
DPCT1009:2: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
#define GpuGetErrorString(...) \
  "cudaGetErrorString is not supported" /*cudaGetErrorString(__VA_ARGS__)*/

struct psz_gpu_exception : public std::exception {
  psz_gpu_exception(const char* err, int err_code, const char* file, int line)
  {
    std::stringstream ss;
    ss << "GPU API failed at \e[31m\e[1m" << file << ':' << line
       << "\e[0m with error: " << err << '(' << err_code << ')';
    err_msg = ss.str();
  }
  const char* what() const noexcept { return err_msg.c_str(); }
  std::string err_msg;
};

static void psz_check_gpu_error_impl(
    GpuErrorT status, const char* file, int line)
{
  /*
  DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
  */
  if (GpuSuccess != status) {
    /*
    DPCT1001:0: The statement could not be removed.
    */
    throw psz_gpu_exception(GpuGetErrorString(status), status, file, line);
  }
}

#endif /* F7DB8F4D_0B6F_4137_A719_D30533955896 */
