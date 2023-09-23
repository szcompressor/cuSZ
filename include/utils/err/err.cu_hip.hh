#ifndef F4EEFA21_1B38_47EA_A59B_5C6A1A6D02DB
#define F4EEFA21_1B38_47EA_A59B_5C6A1A6D02DB

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
  if (GpuSuccess != status) {
    throw psz_gpu_exception(GpuGetErrorString(status), status, file, line);
  }
}

#define CHECK_GPU(err) (psz_check_gpu_error_impl(err, __FILE__, __LINE__))

#endif /* F4EEFA21_1B38_47EA_A59B_5C6A1A6D02DB */
