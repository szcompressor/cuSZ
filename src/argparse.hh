#ifndef ARGPARSE_HH
#define ARGPARSE_HH

/**
 * @file argparse.hh
 * @author Jiannan Tian
 * @brief Argument parser (header).
 * @version 0.1
 * @date 2020-09-20
 * Created on: 20-04-24
 *
 * @copyright (C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <cstdlib>
#include <iostream>
#include <regex>
#include <string>

#include "types.hh"
#include "utils/format.hh"

using std::string;

extern const char* version_text;
extern const int   version;
extern const int   compatibility;

struct alignas(8) SZWorkFlow {
    bool input_use_demo{false};

    bool    pre_binning{false};
    /**/ bool pre_log_trans{false};

    bool autotune_huffman_chunk{true};

    bool lossy_construct{false};
    bool lossy_reconstruct{false};
    bool lossy_dryrun{false};

    bool exp_export_codebook{false};
    bool exp_partitioning_imbalance{false};

    bool    verify_huffman{false};
    /**/ bool verify_eb_quality{false};

    bool skip_write_output{false};
    bool skip_huffman_enc{false};

    // bool    lossless_huffman{true};  // ????
    bool lossless_nvcomp_cascade{false};
    bool lossless_gzip{false};

    bool gtest{false};
};

struct alignas(8) SZDataRepresent {
    int quant_byte;
    int huff_byte;
};

class ArgPack {
   public:
    // TODO [ ] metadata
    // TODO [x] metric/stat

    // mega variable
    struct {
        bool input_use_demo{false};

        bool    pre_binning{false};
        /**/ bool pre_log_trans{false};

        bool autotune_huffman_chunk{true};

        bool lossy_construct{false};
        bool lossy_reconstruct{false};
        bool lossy_dryrun{false};

        bool exp_export_codebook{false};
        bool exp_partitioning_imbalance{false};

        bool    verify_huffman{false};
        /**/ bool verify_eb_quality{false};

        bool skip_write_output{false};
        bool skip_huffman_enc{false};

        // bool    lossless_huffman{true};  // ????
        bool lossless_nvcomp_cascade{false};
        bool lossless_gzip{false};

        bool gtest{false};
    } sz_workflow;

    struct HuffmanWorkFlow {
        bool dryrun;
        bool encode;
        bool decode;
    } hwf;

    struct {
        string path2file;
        struct {
            string huff_base, out_quant, out_outlier, out_yamp;
        } compress;
        struct {
            string in_quant, in_outlier, in_yamp, in_origin, out_xdata;
        } decompress;
    } subfiles;

    stat_t stat;

    int read_args_status{0};

    string mode;  // abs (absolute), r2r (relative to value range)
    string demo_dataset;
    string opath;
    string dtype;

    int quant_byte{2}, huff_byte{4};
    int huffman_chunk{512};
    int ndim{-1};

    double eb{0.0};

    size_t    len{1};
    UInteger4 dim4{1, 1, 1, 1};
    UInteger4 nblk4{1, 1, 1, 1};
    UInteger4 stride4{1, 1, 1, 1};
    int       GPU_block_size{1};
    int       dict_size{1024};
    int       radius{512};

    // experiment
    UInteger4 part4{1, 1, 1, 1};

    bool verbose{false};
    bool to_verify{false};

    bool get_huff_entropy{false};
    bool get_huff_avg_bitcount{false};

    // for standalone Huffman
    int input_rep{2};
    int huffman_datalen{-1};

    static string format(const string& s);

    int trap(int _status);

    void CheckArgs();

    void HuffmanCheckArgs();

    static void cuszDoc();

    static void HuffmanDoc();

    static void cuszFullDoc();

    ArgPack() = default;

    static int          SelfMultiple4(Integer4 i) { return i._0 * i._1 * i._2 * i._3; }
    static unsigned int SelfMultiple4(UInteger4 i) { return i._0 * i._1 * i._2 * i._3; }

    void ParseCuszArgs(int argc, char** argv);

    void ParseHuffmanArgs(int argc, char** argv);

    void SortOutFilenames();
};

typedef ArgPack argpack;

#endif  // ARGPARSE_HH
