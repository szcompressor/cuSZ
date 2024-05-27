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

using std::bitset;
using std::cout;
using std::endl;

template <int BYTEWIDTH>
void f(typename psz::typing::Int<BYTEWIDTH>::T input)
{
    auto encoded = PN<4>::encode(input);
    auto decoded = PN<4>::decode(encoded);

    cout << "original: " << bitset<32>(input) << '\t' << input << '\n'      //
         << "encoded:  " << bitset<32>(encoded) << '\t' << encoded << '\n'  //
         << "decoded:  " << bitset<32>(decoded) << '\t' << decoded << "\n\n";
}

template <int BYTEWIDTH>
bool test(typename psz::typing::Int<BYTEWIDTH>::T input)
{
    auto encoded = PN<4>::encode(input);
    auto decoded = PN<4>::decode(encoded);

    auto pass1 = encoded == (input >= 0 ? (2 * input) : (2 * abs(input) - 1));
    auto pass2 = decoded == input;
    return pass1 and pass2;
}

int main()
{
    f<4>(0), f<4>(1), f<4>(-1), f<4>(2), f<4>(-2);

    auto test1 = test<1>(0) and test<1>(1) and test<1>(-1) and test<1>(2) and test<1>(-2);
    auto test2 = test<2>(0) and test<2>(1) and test<2>(-1) and test<2>(2) and test<2>(-2);
    auto test4 = test<4>(0) and test<4>(1) and test<4>(-1) and test<4>(2) and test<4>(-2);
    auto test8 = test<8>(0) and test<8>(1) and test<8>(-1) and test<8>(2) and test<8>(-2);

    cout << "test1: " << (test1 ? "okay" : "failed") << endl;
    cout << "test2: " << (test2 ? "okay" : "failed") << endl;
    cout << "test4: " << (test4 ? "okay" : "failed") << endl;
    cout << "test8: " << (test8 ? "okay" : "failed") << endl;

    if (test1 and test2 and test4 and test8)
        return 0;
    else
        return -1;
}
