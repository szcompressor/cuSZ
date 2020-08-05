#include <string>
#include <vector>

//#include "SDRB.hh"
#include "../argparse.hh"
#include "../constants.hh"
#include "../cuda_mem.cuh"
//#include "../cusz_workflow.cuh"
#include "../huffman_workflow.cuh"
//#include "../filter.cuh"
#include "../io.hh"
#include "../types.hh"

using std::string;

int main(int argc, char** argv)
{
    auto ap = new argpack(argc, argv, /* use standalone Huffman */ true);

    size_t len        = ap->huffman_datalen;
    auto   huff_chunk = ap->huffman_chunk;
    auto   basename   = ap->fname;
    auto   dict_size  = ap->dict_size;
    auto   input_rep  = ap->input_rep;
    auto   huff_rep   = ap->huffman_rep;

    cout << log_info << "datum file:\t" << basename << endl;
    cout << log_info << "datum size:\t" << len * (input_rep / 8) << endl;
    // TODO add dict size print to cusz
    cout << log_info << "dict size:\t" << dict_size << endl;

    // metadata
    typedef std::tuple<size_t, size_t, size_t> tuple3ul;
    // size_t num_huff_outlier = 0;
    size_t total_bits, total_uInt, huffman_metadata_size;

    cout << log_dbg << "data type:\t" << ap->dtype << endl;
    cout << log_dbg << "using uint" << ap->input_rep << "_t as quant. rep." << endl;
    cout << log_dbg << "using uint" << ap->huffman_rep << "_t as Huffman codeword rep." << endl;

    // program status
    auto encode_processed = false;
    auto decode_processed = false;
    auto good_codec       = true;

    // TODO what is for dryrun?

    cout << log_dbg << "encoding...using uint" << input_rep << " input and uint" << huff_rep << " huff-rep" << endl;
    tuple3ul t;

    if (ap->to_encode) {
        if (ap->input_rep == 8 and ap->huffman_rep == 32) {
            auto datum       = io::ReadBinaryFile<uint8_t>(basename, len);
            auto d_datum     = mem::CreateDeviceSpaceAndMemcpyFromHost(datum, len);
            t                = HuffmanEncode<uint8_t, uint32_t>(basename, d_datum, len, huff_chunk, dict_size);
            encode_processed = true;
        }
        else if (ap->input_rep == 16 and ap->huffman_rep == 32) {
            auto datum       = io::ReadBinaryFile<uint16_t>(basename, len);
            auto d_datum     = mem::CreateDeviceSpaceAndMemcpyFromHost(datum, len);
            t                = HuffmanEncode<uint16_t, uint32_t>(basename, d_datum, len, huff_chunk, dict_size);
            encode_processed = true;
        }
        else if (ap->input_rep == 32 and ap->huffman_rep == 32) {
            auto datum       = io::ReadBinaryFile<uint32_t>(basename, len);
            auto d_datum     = mem::CreateDeviceSpaceAndMemcpyFromHost(datum, len);
            t                = HuffmanEncode<uint32_t, uint32_t>(basename, d_datum, len, huff_chunk, dict_size);
            encode_processed = true;
        }
    }

    std::tie(total_bits, total_uInt, huffman_metadata_size) = t;

    if (ap->to_decode) {
        cout << log_dbg << "decoding...using uint" << input_rep << " input and uint" << huff_rep << " huff-rep" << endl;

        if (ap->input_rep == 8 and ap->huffman_rep == 32) {
            auto xdatum = HuffmanDecode<uint8_t, uint32_t>(basename, len, huff_chunk, total_uInt, dict_size);
            if (ap->verify_huffman) {
                auto datum = io::ReadBinaryFile<uint8_t>(basename, len);
                for (auto i = 0; i < len; i++) {
                    if (datum[i] != xdatum[i]) {
                        good_codec = false;
                        break;
                    }
                }
            }
            decode_processed = true;
        }

        else if (ap->input_rep == 16 and ap->huffman_rep == 32) {
            auto xdatum = HuffmanDecode<uint16_t, uint32_t>(basename, len, huff_chunk, total_uInt, dict_size);
            if (ap->verify_huffman) {
                auto datum = io::ReadBinaryFile<uint16_t>(basename, len);
                for (auto i = 0; i < len; i++) {
                    if (datum[i] != xdatum[i]) {
                        good_codec = false;
                        break;
                    }
                }
            }
            decode_processed = true;
        }

        else if (ap->input_rep == 32 and ap->huffman_rep == 32) {
            auto xdatum = HuffmanDecode<uint32_t, uint32_t>(basename, len, huff_chunk, total_uInt, dict_size);
            if (ap->verify_huffman) {
                auto datum = io::ReadBinaryFile<uint32_t>(basename, len);
                for (auto i = 0; i < len; i++) {
                    if (datum[i] != xdatum[i]) {
                        good_codec = false;
                        break;
                    }
                }
            }
            decode_processed = true;
        }
    }

    if (ap->to_encode and not encode_processed) cout << log_err << "somehow the asked encoding is not processed." << endl;
    if (ap->to_decode and not decode_processed) cout << log_err << "somehow the asked decoding is not processed." << endl;

    if (decode_processed and good_codec)
        cout << log_dbg << "Good codec!" << endl;
    else if (decode_processed and not good_codec)
        cout << log_err << "Not a good codec." << endl;
}