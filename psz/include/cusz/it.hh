/**
 * @file it.hh
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-03-13
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "../busyheader.hh"

template <typename T, int DIM, int BLOCK>
struct psz_buf {
 private:
  T* _buf;
  size_t _len{1};
  static const int stridey{BLOCK};
  static const int stridez{BLOCK * BLOCK};

 public:
  psz_buf(bool do_memset = true)
  {
    if (DIM == 1) _len = BLOCK;
    if (DIM == 2) _len = BLOCK * BLOCK;
    if (DIM == 3) _len = BLOCK * BLOCK * BLOCK;
    _buf = new T[_len];
    if (do_memset) memset(_buf, 0x0, sizeof(T) * _len);
  }

  ~psz_buf() { delete[] _buf; }

  T*& buf() { return _buf; }

  T& operator()(int x) { return _buf[x]; }
  T& operator()(int x, int y) { return _buf[x + y * stridey]; }
  T& operator()(int x, int y, int z)
  {
    return _buf[x + y * stridey + z * stridez];
  }
};

// template <typename T, typename IDX = uint32_t>
// struct psz_outlier_seq {
//  private:
//   T* _data;
//   IDX* _idx;
//   uint32_t _count{0};
//   uint32_t _cap;

//  public:
//   psz_outlier_seq(size_t cap) : _cap(cap)
//   {
//     _data = new T[cap + 1];
//     _idx = new IDX[cap + 1];
//     memset(_data, 0x0, sizeof(T) * cap);
//   }

//   ~psz_outlier_seq()
//   {
//     delete[] _data;
//     delete[] _idx;
//   }

//   T*& val() { return _data; }
//   IDX*& idx() { return _idx; }
//   uint32_t const count() { return _count; }

//   void record(T data, IDX idx)
//   {
//     if (_count > _cap) throw std::runtime_error("Outlier overflows.");
//     _data[_count] = data;
//     _idx[_count] = idx;
//     ++_count;
//   }
// };