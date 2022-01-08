/**
 * @file test_rle.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-04-01
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "common/capsule.hh"
#include "kernel/rle.cuh"
#include "ood/codec_runlength.hh"

void test_encoding(size_t& N, size_t& num_runs)
{
    // input data on the host
    char _data[] = "aaabbbbbcddeeeeeeeeeffaaaaaaa";
    N            = (sizeof(_data) / sizeof(char)) - 1;
    num_runs     = N;

    DataPack<char> fullfmt_data("char array");
    fullfmt_data
        .SetLen(N)  //
        .SetHostSpace(_data)
        .AllocDeviceSpace()
        .template Move<transfer::h2d>();

    DataPack<char> compact_data("compact bundle: data");
    compact_data.SetLen(N).AllocHostSpace().AllocDeviceSpace();
    DataPack<int> lengths("compact bundle: index");
    lengths.SetLen(N).AllocHostSpace().AllocDeviceSpace();

    cout << "Before encoding, num_runs is set to " << num_runs << '\n' << endl;

    // form 1, call kernel directly
    // kernel::RunLengthEncoding(fullfmt_data.dptr(), N, sparse_data.dptr(), lengths.dptr(), num_runs);

    // form 2, ood
    auto encoder = RunLengthCodec<char>(&fullfmt_data, &compact_data, &lengths);
    encoder.SetFullLen(N).SetRunLen(N).Encode();
    num_runs = encoder.RunLen();

    compact_data.template Move<transfer::d2h>();
    lengths.template Move<transfer::d2h>();

    ////////////////////////////////////////////////////////////////////////////////
    cout << "After encoding, num_runs is modified to " << num_runs << '\n' << endl;
    //    std::cout << "run-length encoded output:" << std::endl;
    //    for (size_t i = 0; i < num_runs; i++) std::cout << "(" << compact_data.hptr()[i] << "," << lengths.hptr()[i]
    //    << ")"; std::cout << std::endl;
    ////////////////////////////////////////////////////////////////////////////////
}

void test_decoding(size_t& N, size_t& num_runs)
{
    ////////////////////////////////////////////////////////////////////////////////
    char h_input[7]   = {'a', 'b', 'c', 'd', 'e', 'f', 'a'};
    int  h_lengths[7] = {3, 5, 1, 2, 9, 2, 7};
    std::cout << "run-length encoded input:" << std::endl;
    for (size_t i = 0; i < num_runs; i++) std::cout << "(" << h_input[i] << "," << h_lengths[i] << ")";
    std::cout << std::endl << std::endl;
    ////////////////////////////////////////////////////////////////////////////////

    DataPack<char> compact_data("compact data");
    DataPack<int>  lengths("lengths");
    DataPack<char> fullfmt_data("fullfmt data");

    compact_data.SetLen(num_runs).SetHostSpace(h_input).AllocDeviceSpace().template Move<transfer::h2d>();
    lengths.SetLen(num_runs).SetHostSpace(h_lengths).AllocDeviceSpace().template Move<transfer::h2d>();

    fullfmt_data.SetLen(N).AllocDeviceSpace().AllocHostSpace();

    // form 1, call kernel directly
    // kernel::RunLengthDecoding(fullfmt_data.dptr(), N, compact_data.dptr(), lengths.dptr(), num_runs);

    // form 2, ood
    auto decoder = RunLengthCodec<char>(&fullfmt_data, &compact_data, &lengths);
    decoder.SetFullLen(N).SetRunLen(num_runs).Decode();

    fullfmt_data.template Move<transfer::d2h>();
    for (auto i = 0; i < N; i++) cout << fullfmt_data.hptr()[i];
    cout << '\n';
}

void test_encoding_fpkey(size_t& N, size_t& num_runs)
{
    // input data on the host
    float _data[] = {
        // sum: 21
        1.1, 1.1,                      // 2
        2.1, 2.1, 2.1, 2.1, 2.1, 2.1,  // 6
        3.1, 3.1, 3.1, 3.1,            // 4
        1.1, 1.1, 1.1,                 // 3
        4.1, 4.1, 4.1, 4.1, 4.1,       // 5
        2.1, 2.1                       // 2

    };
    N        = (sizeof(_data) / sizeof(float));
    num_runs = N;

    DataPack<float> fullfmt_data("char array");
    fullfmt_data
        .SetLen(N)  //
        .SetHostSpace(_data)
        .AllocDeviceSpace()
        .template Move<transfer::h2d>();

    DataPack<float> compact_data("compact bundle: data");
    compact_data.SetLen(N).AllocHostSpace().AllocDeviceSpace();
    DataPack<int> lengths("compact bundle: index");
    lengths.SetLen(N).AllocHostSpace().AllocDeviceSpace();

    cout << "Before encoding, num_runs is set to " << num_runs << '\n' << endl;

    // form 1, call kernel directly
    // kernel::RunLengthEncoding(fullfmt_data.dptr(), N, sparse_data.dptr(), lengths.dptr(), num_runs);

    // form 2, ood
    auto encoder = RunLengthCodec<float>(&fullfmt_data, &compact_data, &lengths);
    encoder.SetFullLen(N).SetRunLen(N).Encode();
    num_runs = encoder.RunLen();

    compact_data.template Move<transfer::d2h>();
    lengths.template Move<transfer::d2h>();

    ////////////////////////////////////////////////////////////////////////////////
    cout << "After encoding, num_runs is modified to " << num_runs << '\n' << endl;
    std::cout << "run-length encoded output:" << std::endl;
    for (size_t i = 0; i < num_runs; i++) std::cout << "(" << compact_data.hptr()[i] << "," << lengths.hptr()[i] << ")";
    std::cout << std::endl;
    ////////////////////////////////////////////////////////////////////////////////
}

void test_decoding_fpkey(size_t& N, size_t& num_runs)
{
    ////////////////////////////////////////////////////////////////////////////////
    float h_input[6]   = {1.1, 2.1, 3.1, 1.1, 4.1, 2.1};
    int   h_lengths[6] = {2, 6, 4, 3, 5, 2};
    std::cout << "run-length encoded input:" << std::endl;
    for (size_t i = 0; i < num_runs; i++) std::cout << "(" << h_input[i] << "," << h_lengths[i] << ")";
    std::cout << std::endl << std::endl;
    ////////////////////////////////////////////////////////////////////////////////

    DataPack<float> compact_data("compact data");
    DataPack<int>   lengths("lengths");
    DataPack<float> fullfmt_data("fullfmt data");

    compact_data.SetLen(num_runs).SetHostSpace(h_input).AllocDeviceSpace().template Move<transfer::h2d>();
    lengths.SetLen(num_runs).SetHostSpace(h_lengths).AllocDeviceSpace().template Move<transfer::h2d>();

    fullfmt_data.SetLen(N).AllocDeviceSpace().AllocHostSpace();

    // form 1, call kernel directly
    // kernel::RunLengthDecoding(fullfmt_data.dptr(), N, compact_data.dptr(), lengths.dptr(), num_runs);

    // form 2, ood
    auto decoder = RunLengthCodec<float>(&fullfmt_data, &compact_data, &lengths);
    decoder.SetFullLen(N).SetRunLen(num_runs).Decode();

    fullfmt_data.template Move<transfer::d2h>();
    for (auto i = 0; i < N; i++) cout << fullfmt_data.hptr()[i] << ' ';
    cout << '\n';
}

int main()
{
    size_t N        = 0;  // known (set inside encoding)
    size_t num_runs = 0;  // unknown, set to N
    cout << "test encoding" << endl;
    test_encoding_fpkey(N, num_runs);
    cout << "test decoding" << endl;
    test_decoding_fpkey(N, num_runs);

    return 0;
}
