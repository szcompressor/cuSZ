/**
 * @file test_pncodec_func.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-01-05
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include <bitset>
#include <iostream>

#include "cusz/suint.hh"
#include "cusz/type.h"

using std::bitset;
using std::cout;
using std::endl;

template <typename T>
void demo(T input)
{
  auto encoded = psz::ZigZag<T>::encode(input);
  auto decoded = psz::ZigZag<T>::decode(encoded);

  cout << "original: " << bitset<32>(input) << '\t' << input << '\n'      //
       << "encoded:  " << bitset<32>(encoded) << '\t' << encoded << '\n'  //
       << "decoded:  " << bitset<32>(decoded) << '\t' << decoded << "\n\n";
}

template <typename T>
bool test(T input)
{
  auto encoded = psz::ZigZag<T>::encode(input);
  auto decoded = psz::ZigZag<T>::decode(encoded);

  auto pass1 =
      encoded == (input >= 0 ? (2 * input) : (2 * std::abs(input) - 1));
  auto pass2 = decoded == input;
  return pass1 and pass2;
}

int main()
{
  demo(0), demo(1), demo(-1), demo(2), demo(-2);

  auto test1 = test((int8_t)0) and test((int8_t)1) and test((int8_t)-1) and
               test((int8_t)2) and test((int8_t)-2);
  auto test2 = test((int16_t)0) and test((int16_t)1) and test((int16_t)-1) and
               test((int16_t)2) and test((int16_t)-2);
  auto test4 = test((int32_t)0) and test((int32_t)1) and test((int32_t)-1) and
               test((int32_t)2) and test((int32_t)-2);
  auto test8 = test((int64_t)0) and test((int64_t)1) and test((int64_t)-1) and
               test((int64_t)2) and test((int64_t)-2);

  cout << "test1: " << (test1 ? "okay" : "failed") << endl;
  cout << "test2: " << (test2 ? "okay" : "failed") << endl;
  cout << "test4: " << (test4 ? "okay" : "failed") << endl;
  cout << "test8: " << (test8 ? "okay" : "failed") << endl;

  if (test1 and test2 and test4 and test8)
    return 0;
  else
    return -1;
}
