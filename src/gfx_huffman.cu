
#include "huffman_workflow.cuh"

int main(int argc, char** argv) {
    if (argc != 7) {
        cout << "<program> <datum name>.b<8,16,32> <len> <chunk size> <b8|b16|b32> <h32|h64> <bin number>" << endl;
        cout << "example:" << endl;
        cout << "\t./huff CLDHGH_sample.b16 6480000 1024 b16 h32 1024" << endl;
        exit(0);
    }
    auto fname      = std::string(argv[1]);
    auto len        = atoi(argv[2]);
    auto chunk_size = atoi(argv[3]);
    auto bin_rep    = std::string(argv[4]);
    auto huff_rep   = std::string(argv[5]);
    auto dict_size  = atoi(argv[6]);

    cout << "[INFO] filename " << fname << endl;
    cout << "[INFO] chunk_size\t" << chunk_size << endl;
    cout << "[INFO] quantization code using " << bin_rep << " as internal representation." << endl;
    cout << "[INFO] Huffman code using " << huff_rep << " as internal representation." << endl;
    cout << "[INFO] DICTIONARY SIZE\t" << dict_size << endl;

    std::tuple<size_t, size_t> total_bits__total_uInt;

    //................................................................................
    // encoding
    //................................................................................
    if (huff_rep == "h32") {
        if (bin_rep == "b8") {
            assert(dict_size <= 256);
            total_bits__total_uInt = HuffmanEncode<uint32_t, uint8_t>(fname, len, chunk_size, dict_size);
        } else if (bin_rep == "b16") {
            assert(dict_size <= 65536);
            total_bits__total_uInt = HuffmanEncode<uint32_t, uint16_t>(fname, len, chunk_size, dict_size);
        } else if (bin_rep == "b32") {
            total_bits__total_uInt = HuffmanEncode<uint32_t, uint32_t>(fname, len, chunk_size, dict_size);
        } else {
            cerr << "[ERR] should use <b8|b16|b32>" << endl;
            // throw std::invalid_argument("b8|b16|b32");
        }
    } else if (huff_rep == "h64") {
        if (bin_rep == "b8") {
            assert(dict_size <= 256);
            total_bits__total_uInt = HuffmanEncode<uint64_t, uint8_t>(fname, len, chunk_size, dict_size);
        } else if (bin_rep == "b16") {
            assert(dict_size <= 65536);
            total_bits__total_uInt = HuffmanEncode<uint64_t, uint16_t>(fname, len, chunk_size, dict_size);
        } else if (bin_rep == "b32") {
            total_bits__total_uInt = HuffmanEncode<uint64_t, uint32_t>(fname, len, chunk_size, dict_size);
        } else
            cerr << "[ERR] should use <b8|b16|b32>" << endl;
    } else {
        cerr << "[ERR] Must use 32 or 64 bit as internal representation for Huffman code!" << endl;
    }

    //................................................................................
    // decoding
    //................................................................................
    size_t total_bits, total_uInt;
    std::tie(total_bits, total_uInt) = total_bits__total_uInt;

    if (huff_rep == "h32") {
        if (bin_rep == "b8") {
            assert(dict_size <= 256);
            HuffmanDecode<uint32_t, uint8_t>(fname, len, chunk_size, total_uInt, dict_size);
        } else if (bin_rep == "b16") {
            assert(dict_size <= 65536);
            HuffmanDecode<uint32_t, uint16_t>(fname, len, chunk_size, total_uInt, dict_size);
        } else if (bin_rep == "b32") {
            HuffmanDecode<uint32_t, uint32_t>(fname, len, chunk_size, total_uInt, dict_size);
        } else {
            cerr << "[ERR] should use <b8|b16|b32>" << endl;
            // throw std::invalid_argument("b8|b16|b32");
        }
    } else if (huff_rep == "h64") {
        if (bin_rep == "b8") {
            assert(dict_size <= 256);
            HuffmanDecode<uint64_t, uint8_t>(fname, len, chunk_size, total_uInt, dict_size);
        } else if (bin_rep == "b16") {
            assert(dict_size <= 65536);
            //            cout << "Decoding H64" << endl;
            HuffmanDecode<uint64_t, uint16_t>(fname, len, chunk_size, total_uInt, dict_size);
        } else if (bin_rep == "b32") {
            HuffmanDecode<uint64_t, uint32_t>(fname, len, chunk_size, total_uInt, dict_size);
        } else
            cerr << "[ERR] should use <b8|b16|b32>" << endl;
    } else {
        cerr << "[ERR] Must use 32 or 64 bit as internal representation for Huffman code!" << endl;
    }
    cout << endl;

    return 0;
}
