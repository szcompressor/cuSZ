/**
 * @file run_rle.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.2
 * @date 2021-04-02
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <iostream>
#include <string>

#include "../analysis/analyzer.hh"
#include "../ood/codec_runlength.hh"

using std::cerr;
using std::cout;
using std::endl;
using std::string;

int main(int argc, char** argv)
{
    if (argc != 3) {
        cout << "help:\n  rle <path/to/input> <input-size>\n\n";
        exit(0);
    }
    auto  fname = string(argv[1]);
    char* end;
    auto  len = std::strtod(argv[2], &end);

    using Quant = uint16_t;

    DataPack<Quant> quant("quant", len);
    DataPack<Quant> compact_data("compact-data", len);
    DataPack<int>   compact_idx("compact-index", len);
    quant.AllocHostSpace().AllocDeviceSpace().Move<transfer::fs2h>(fname).Move<transfer::h2d>();
    compact_data.AllocDeviceSpace().AllocHostSpace();
    compact_idx.AllocDeviceSpace().AllocHostSpace();

    auto codec    = RunLengthCodec<Quant>(&quant, &compact_data, &compact_idx);
    auto num_runs = codec.SetFullLen(len).Encode().RunLen();

    cout << "num runs:\t" << num_runs << endl;

    auto ori_len  = sizeof(float) * len;  // original data is float32
    auto compact1 = (sizeof(Quant) + sizeof(int)) * num_runs;
    auto compact2 = (sizeof(Quant) + sizeof(uint16_t)) * num_runs;
    auto compact3 = (sizeof(Quant) + sizeof(uint8_t)) * num_runs;

    Analyzer analyzer;
    auto     res = analyzer.GetMaxMinRng<int, AnalyzerExecutionPolicy::cuda_device, AnalyzerMethod::thrust>(
        compact_idx.dptr(), len);

    cout << "file:\t" << fname << '\n';
    cout << "max run-length:\t" << res.max_val << '\n';

    auto h_freq = new unsigned int[256]();
    compact_idx.Move<transfer::d2h>();
    compact_data.Move<transfer::d2h>();
    auto h_c1 = reinterpret_cast<uint8_t*>(compact_idx.safe_hptr());
    for (auto i = 0; i < num_runs * sizeof(int); i++) { h_freq[h_c1[i]] += 1; }

    auto h_c2 = reinterpret_cast<uint8_t*>(compact_data.safe_hptr());
    for (auto i = 0; i < num_runs * sizeof(Quant); i++) { h_freq[h_c2[i]] += 1; }

    analyzer.EstimateFromHistogram(h_freq, 256).PrintCompressibilityInfo(false, false, 8);

    cout << "reduction rate (int):\t" << (1.0 * ori_len) / compact1 << '\n';
//    cout << "reduction rate (uint16):\t" << (1.0 * ori_len) / compact2 << '\n';
//    cout << "reduction rate (uint8):\t" << (1.0 * ori_len) / compact3 << '\n';
    cout << endl;
}