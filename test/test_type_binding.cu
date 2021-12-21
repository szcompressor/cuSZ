/**
 * @file test_type_binding.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-10-06
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <cstdint>
#include <iostream>
#include <typeinfo>
#include "../src/binding.hh"
#include "../src/wrapper.hh"

using namespace std;

int main()
{
    //
    using Predictor = cusz::PredictorLorenzo<float, uint32_t, float>;
    cout << typeid(Predictor::Origin).name() << endl;
    cout << typeid(Predictor::Precision).name() << endl;

    using binding1 = struct PredictorReducerEncoderBinding<
        cusz::PredictorLorenzo<float, uint32_t, float>, cusz::CSR10<float>, cusz::HuffmanCoarse<uint32_t, uint32_t>>;

    binding1::type_matching();

    cout << typeid(binding1::T1).name() << endl;
    cout << typeid(binding1::PREDICTOR::Origin).name() << endl;

    using binding2 = struct PredictorReducerBinding<  //
        cusz::PredictorLorenzo<float, float, float>, cusz::CSR10<float>>;

    binding2::type_matching();

    cout << typeid(binding2::T1).name() << endl;

    return 0;
}
