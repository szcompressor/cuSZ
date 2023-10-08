#ifndef F3B260F6_21FF_47E6_8E0D_1A02944EA8C9
#define F3B260F6_21FF_47E6_8E0D_1A02944EA8C9

#include <cmath>

namespace psz {
namespace criterion {
namespace seq {

template <typename T>
struct eq {
  bool operator()(T x, T n) const { return x == n; }
};

template <typename T>
struct in_ball_shifted {
  bool operator()(T x, T radius) const { return x >= 0 and x < 2 * radius; }
};

template <typename T>
struct in_ball {
  bool operator()(T x, T radius) const { return std::fabs(x) < radius; }
};

}  // namespace seq
}  // namespace criterion
}  // namespace psz

#endif /* F3B260F6_21FF_47E6_8E0D_1A02944EA8C9 */
