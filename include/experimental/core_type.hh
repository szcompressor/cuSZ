#ifndef AC9D7D11_7C3E_42BD_B96A_7C0F1E02A2C5
#define AC9D7D11_7C3E_42BD_B96A_7C0F1E02A2C5

struct core_type {
  //
  static size_t linearize(size_t x, size_t y, size_t z) { return x * y * z; }

#if defined(PSZ_USE_CUDA) || defined(PSZ_USE_HIP)
  static dim3 range3(size_t x, size_t y, size_t z) { return dim3(x, y, z); }
  static size_t linearize(dim3 l3) { return l3.x * l3.y * l3.z; }
#elif defined(PSZ_USE_1API)
  static sycl::range<3> range3(size_t x, size_t y, size_t z)
  {
    return sycl::range<3>(z, y, x);
  }
  static size_t linearize(sycl::range<3> l3) { return l3[0] * l3[1] * l3[2]; }
#endif
};

#endif /* AC9D7D11_7C3E_42BD_B96A_7C0F1E02A2C5 */
