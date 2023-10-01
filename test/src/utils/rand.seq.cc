/**
 * @file rand.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-08-25
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "../rand.hh"

#include <iostream>
#include <random>

namespace psz {
namespace testutils {
namespace cpp {

int randint(size_t upper_limit)
{
  std::random_device dev;
  std::mt19937 gen(dev());

  std::uniform_int_distribution<> distribute(0, upper_limit);

  return distribute(gen);
}

template <typename T = float>
T randfp(T upper_limit, T lower_limit)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<T> dis(lower_limit, upper_limit);

  return dis(gen);
}

template <>
void rand_array<int>(int* array, size_t len)
{
  for (auto i = 0; i < len; i++) { array[i] = randint(1000); }
}

template <>
void rand_array<float>(float* array, size_t len)
{
  for (auto i = 0; i < len / 2; i++) {
    auto idx = randint(len);
    array[idx] = randfp<float>();
  }
}

template <>
void rand_array<double>(double* array, size_t len)
{
  for (auto i = 0; i < len / 5; i++) {
    auto idx = randint(len);
    array[idx] = randfp<double>();
  }
}

}  // namespace cpp
}  // namespace testutils
}  // namespace psz

template float psz::testutils::cpp::randfp<float>(float, float);
template double psz::testutils::cpp::randfp<double>(double, double);