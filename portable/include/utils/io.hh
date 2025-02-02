#ifndef _PORTABLE_UTILS_IO_HH
#define _PORTABLE_UTILS_IO_HH

// Jiannan Tian
// (created) 2019-08-27 (update) 2020-09-20...2024-12-22

#include <fstream>
#include <iostream>

#define PORTABLE_IO_SUCCESS 0
#define PORTABLE_FAIL_NULLPTR 1
#define PORTABLE_IFS_FAIL_TO_OPEN -1
#define PORTABLE_OFS_FAIL_TO_OPEN -2

namespace _portable::utils {

template <typename T>
int fromfile(const std::string& fname, T* _a, size_t const dtype_len)
{
  std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
  if (not ifs.is_open()) return PORTABLE_IFS_FAIL_TO_OPEN;
  if (not _a) return PORTABLE_FAIL_NULLPTR;

  ifs.read((char*)(_a), std::streamsize(dtype_len * sizeof(T)));
  ifs.close();
  return PORTABLE_IO_SUCCESS;
}

template <typename T>
int tofile(const std::string& fname, T* const _a, size_t const dtype_len)
{
  std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::out);
  if (not ofs.is_open()) return PORTABLE_OFS_FAIL_TO_OPEN;
  ofs.write((const char*)(_a), std::streamsize(dtype_len * sizeof(T)));
  ofs.close();
  return PORTABLE_IO_SUCCESS;
}

}  // namespace _portable::utils

#endif /* _PORTABLE_UTILS_IO_HH */
