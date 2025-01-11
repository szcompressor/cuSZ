#include <array>

namespace {
dim3 div3(dim3 l, dim3 subl)
{
  return dim3((l.x - 1) / subl.x + 1, (l.y - 1) / subl.y + 1, (l.z - 1) / subl.z + 1);
};
}  // namespace

namespace psz::kernelconfig {

struct lorenzo_utils {
  static int ndim(dim3 len3)
  {
    if (len3.z == 1 and len3.y == 1)
      return 1;
    else if (len3.z == 1 and len3.y != 1)
      return 2;
    else
      return 3;
  };

  static int ndim(std::array<size_t, 3> len3)
  {
    if (len3[2] == 1 and len3[1] == 1)
      return 1;
    else if (len3[2] == 1 and len3[1] != 1)
      return 2;
    else
      return 3;
  };
};

template <int dim>
struct c_lorenzo;

template <int dim>
struct x_lorenzo;

template <>
struct c_lorenzo<1> {
  static constexpr dim3 tile = dim3(256, 1, 1);
  static constexpr dim3 sequentiality = dim3(4, 1, 1);  // x-sequentiality == 4
  static constexpr dim3 thread_block = dim3(256 / 4, 1, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };
};

template <>
struct c_lorenzo<2> {
  static constexpr dim3 tile = dim3(16, 16, 1);
  static constexpr dim3 sequentiality = dim3(1, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 thread_block = dim3(16, 2, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };
};

template <>
struct c_lorenzo<3> {
  static constexpr dim3 tile = dim3(32, 8, 8);
  static constexpr dim3 sequentiality = dim3(1, 1, 8);  // z-sequentiality == 8
  static constexpr dim3 thread_block = dim3(32, 8, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };
};

template <>
struct x_lorenzo<1> {
  static constexpr dim3 tile = dim3(256, 1, 1);
  static constexpr dim3 sequentiality = dim3(8, 1, 1);  // x-sequentiality == 8
  static constexpr dim3 thread_block = dim3(256 / 8, 1, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };
};

template <>
struct x_lorenzo<2> {
  static constexpr dim3 tile = dim3(16, 16, 1);
  static constexpr dim3 sequentiality = dim3(1, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 thread_block = dim3(16, 2, 1);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };
};

template <>
struct x_lorenzo<3> {
  static constexpr dim3 tile = dim3(32, 8, 8);
  static constexpr dim3 sequentiality = dim3(1, 8, 1);  // y-sequentiality == 8
  static constexpr dim3 thread_block = dim3(32, 1, 8);
  static dim3 thread_grid(dim3 len3) { return div3(len3, tile); };
};

};  // namespace psz::kernelconfig