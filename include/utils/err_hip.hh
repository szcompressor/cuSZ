/**
 * @file err_hip.hh
 * @author Jiannan Tian
 * @brief CUDA runtime error handling macros.
 * @version 0.2
 * @date 2020-09-20
 * Created on: 2019-10-08
 *
 * @copyright (C) 2020 by Washington State University, The University of
 * Alabama, Argonne National Laboratory See LICENSE in top-level directory
 *
 */

#ifndef DA0F0534_B680_453A_9234_EAEEE540333F
#define DA0F0534_B680_453A_9234_EAEEE540333F

#include <hip/hip_runtime.h>

#include "busyheader.hh"

struct psz_gpu_exception : public std::exception {
  psz_gpu_exception(const char* err, int err_code, const char* file, int line)
  {
    std::stringstream ss;
    ss << "CUDA API failed at \e[31m\e[1m" << file << ':' << line
       << "\e[0m with error: " << err << '(' << err_code << ')';
    err_msg = ss.str();
  }
  const char* what() const noexcept { return err_msg.c_str(); }
  std::string err_msg;
};

static void check_hip_error(hipError_t status, const char* file, int line)
{
  if (hipSuccess != status) {
    // TODO print exact error
    throw psz_gpu_exception(hipGetErrorString(status), status, file, line);
  }
}

#define CHECK_GPU(err) (check_hip_error(err, __FILE__, __LINE__))

#endif /* DA0F0534_B680_453A_9234_EAEEE540333F */
