#ifndef _PORTABLE_UTILS_IO_HH
#define _PORTABLE_UTILS_IO_HH

// Jiannan Tian
// (created) 2019-08-27 (update) 2020-09-20...2024-12-22

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <type_traits>

#define PORTABLE_IO_SUCCESS 0
#define PORTABLE_FAIL_NULLPTR 1
#define PORTABLE_IFS_FAIL_TO_OPEN -1
#define PORTABLE_OFS_FAIL_TO_OPEN -2
#define PORTABLE_IFS_SHORT_READ -3
#define PORTABLE_OFS_WRITE_ERR -4

namespace _ptb::utils {

template <typename T>
int fromfile(const std::string& fname, T* _a, size_t const dtype_len)
{
  static_assert(std::is_trivially_copyable_v<T>, "fromfile requires trivially copyable T");
  if (not _a) return PORTABLE_FAIL_NULLPTR;

  std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
  if (not ifs.is_open()) return PORTABLE_IFS_FAIL_TO_OPEN;

  auto expected = std::streamsize(dtype_len * sizeof(T));
  ifs.read(reinterpret_cast<char*>(_a), expected);
  if (ifs.gcount() != expected) return PORTABLE_IFS_SHORT_READ;

  return PORTABLE_IO_SUCCESS;
}

// Convenience wrapper for binaries: exit on failure with a clear message.
// Use in CLIs/example bins where there is no graceful recovery path; the
// silent-zero-buffer behavior of the bare `fromfile` masks missing/typo'd
// file paths and produces nonsensical downstream metrics.
template <typename T>
void fromfile_or_die(const std::string& fname, T* _a, size_t const dtype_len)
{
  auto rc = fromfile(fname, _a, dtype_len);
  if (rc == PORTABLE_IO_SUCCESS) return;
  std::cerr << "[_ptb::utils::fromfile] failed to read \"" << fname << "\" (rc=" << rc << ")";
  if (rc == PORTABLE_IFS_FAIL_TO_OPEN)
    std::cerr << "  — file does not exist or is not readable";
  else if (rc == PORTABLE_FAIL_NULLPTR)
    std::cerr << "  — destination buffer is null";
  std::cerr << std::endl;
  std::exit(2);
}

template <typename T>
int tofile(const std::string& fname, T* const _a, size_t const dtype_len)
{
  static_assert(std::is_trivially_copyable_v<T>, "tofile requires trivially copyable T");
  if (not _a) return PORTABLE_FAIL_NULLPTR;

  std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::out);
  if (not ofs.is_open()) return PORTABLE_OFS_FAIL_TO_OPEN;

  auto expected = std::streamsize(dtype_len * sizeof(T));
  ofs.write(reinterpret_cast<const char*>(_a), expected);
  if (ofs.fail()) return PORTABLE_OFS_WRITE_ERR;

  return PORTABLE_IO_SUCCESS;
}

// Returns the byte size of fname, or 0 if the file cannot be opened.
// Callers must treat 0 as an error — an empty valid file is indistinguishable,
// but opening should be checked separately before reading.
inline size_t filesize(const std::string& fname)
{
  std::ifstream in(fname.c_str(), std::ifstream::ate | std::ifstream::binary);
  if (not in.is_open()) return 0;
  auto pos = in.tellg();
  return pos < 0 ? 0 : static_cast<size_t>(pos);
}

}  // namespace _ptb::utils

#endif /* _PORTABLE_UTILS_IO_HH */
