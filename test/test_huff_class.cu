
#include <string>
using std::string;

//#include "../src/huff_interface.cuh"
#include "../src/kernel/hist.h"
#include "../src/ood/codec_huffman.hh"
#include "../src/par_huffman.cuh"

string filename = "/home/jtian/sdrb-cesm/FLDS.dat.quant";

using ExampleTypeData = uint16_t;
using ExampleTypeHuff = uint32_t;

// static const int huff_type_bytes = sizeof(ExampleTypeHuff);
// static const int huff_type_bits  = sizeof(ExampleTypeHuff) * 8;
size_t data_len   = 6480000;
size_t chunk_size = 8192;
size_t num_syms   = 1024;

template <typename Data, typename Huff>
int TestEncoding(string filename)
{
    size_t num_chunks = (data_len - 1 + chunk_size) / chunk_size;

    // input, external to Huffman codec
    // --------------------------------
    DataPack<Data>     data("uint16 _quant code");
    DataPack<uint32_t> freq("hist-_freq");
    data.SetLen(data_len)
        .AllocHostSpace()
        .AllocDeviceSpace()
        .template Move<transfer::fs2h>(filename)
        .template Move<transfer::h2d>();
    freq.SetLen(num_syms).AllocDeviceSpace();

    float dummy;
    kernel_wrapper::get_frequency<Data, false>(data.dptr(), data.Len(), freq.dptr(), freq.Len(), dummy);

    ////////////////////////////////////////////////////////////////////////////////
    //  ^                                  | Huffman
    //  |  external to Huffman class       v
    ////////////////////////////////////////////////////////////////////////////////

    // internal to Huffman codec, calculating memory footprint
    // -------------------------------------------------------
    static const auto num_gpu_arrays = 4;
    struct {
        size_t sizes[num_gpu_arrays]{};
        size_t types[num_gpu_arrays]{sizeof(Huff), sizeof(uint8_t), sizeof(size_t), sizeof(Huff)};
        size_t bytes[num_gpu_arrays]{};
        size_t entries[num_gpu_arrays]{};
    } gpu_arrays;

    gpu_arrays.sizes[0] = num_syms;                                                         // book
    gpu_arrays.sizes[1] = sizeof(Huff) * (2 * sizeof(Huff) * 8) + sizeof(Data) * num_syms;  // reverse_book
    gpu_arrays.sizes[2] = num_chunks;                                                       // bits_of_chunks
    gpu_arrays.sizes[3] = data_len;                                                         // enc_space

    for (auto& i : gpu_arrays.sizes) { cout << "gpu_array_sizes " << i << '\n'; }

    DataPack<Huff>    book("canonical codebook", gpu_arrays.sizes[0]);
    DataPack<uint8_t> reverse_book("reverse codebook", gpu_arrays.sizes[1]);
    DataPack<size_t>  bits_of_chunks("chunkwise bits of chunks", gpu_arrays.sizes[2]);
    DataPack<Huff>    enc_space("workspace for encoding", gpu_arrays.sizes[3]);

    book.AllocDeviceSpace();
    reverse_book.AllocDeviceSpace();
    bits_of_chunks.AllocDeviceSpace().AllocHostSpace();
    enc_space.AllocDeviceSpace();

    //    cout << "bits of chunks dptr allocated?\t" << bits_of_chunks.DptrAllocated() << endl;

    // to export to host
    // -----------------
    DataPack<size_t> units_of_chunks("chunkwise units of chunks", num_chunks);
    DataPack<size_t> entries_of_chunks("chunkwise entries of chunks", num_chunks);
    DataPack<Huff>   compact_format("sparse format");

    units_of_chunks.AllocHostSpace();
    entries_of_chunks.AllocHostSpace();
    compact_format.Note(placeholder::length_unknown).Note(placeholder::alloc_with_caution);

    // make big wrapper
    // ----------------
    struct HuffmanEncodingDataBundle<Data, Huff> hedp {
    };

    hedp.external  = {&data, &freq};
    hedp.internal  = {&book, &reverse_book, &bits_of_chunks, &enc_space, &compact_format};
    hedp.chunkwise = {&units_of_chunks, &entries_of_chunks};

    HuffmanCodec<uint16_t, uint32_t> codec(&hedp, num_syms, chunk_size);

    auto metadata = codec
                        .SetOptionalNames(filename)  //
                        .ConstructCodebook()
                        .Encode()
                        .GatherEncMetadata()
                        .GetEncMetadata();

    compact_format.SetLen(metadata.num_units).AllocDeviceSpace();
    codec.GatherEncChunks();
    compact_format.template Move<transfer::d2h>();

    return 0;
}

template <typename Data, typename Huff>
int TestDecoding()
{
    // can be known
    size_t num_chunks = (data_len - 1 + chunk_size) / chunk_size;
    // ----

    DataPack<Data>    data("decoded data", data_len);
    DataPack<size_t>  num_bits_units("num bits units", 2 * num_chunks);
    size_t            rb_size = sizeof(Huff) * (2 * sizeof(Huff) * 8) + sizeof(Data) * num_syms;
    DataPack<uint8_t> reverse_book("reverse book", rb_size);

    return 0;
}

int main()
{
    TestEncoding<uint16_t, uint32_t>(filename);

    // TestDecoding<uint16_t, uint32_t>();

    return 0;
}

// snippet for memory pool
/*
    for (auto i = 0; i < num_gpu_arrays; i++) gpu_arrays.bytes[i] = gpu_arrays.sizes[i] * gpu_arrays.types[i];
    std::copy(gpu_arrays.bytes, gpu_arrays.bytes + num_gpu_arrays - 1, gpu_arrays.entries + 1);
    for (auto i = 1; i < num_gpu_arrays; i++) gpu_arrays.entries[i] += gpu_arrays.entries[i - 1];
    size_t _nbyte_gpu_internal = std::accumulate(gpu_arrays.bytes, gpu_arrays.bytes + num_gpu_arrays, (unsigned
   int)0);

    // internal to Huffman codec, calculating
    // --------------------------------------
    uint8_t* _gpu_internal = nullptr;
    cudaMalloc((void**)&_gpu_internal, _nbyte_gpu_internal);
    cudaMemset(_gpu_internal, 0x00, _nbyte_gpu_internal);
    DataPack<Huff>    book("canonical codebook", gpu_arrays.sizes[0]);
    DataPack<uint8_t> reverse_book("reverse codebook", gpu_arrays.sizes[1]);
    DataPack<size_t>  bits_of_chunks("chunkwise bits of chunks", gpu_arrays.sizes[2]);
    DataPack<Huff>    enc_space("workspace for encoding", gpu_arrays.sizes[3]);

    book.SetDeviceSpace(reinterpret_cast<Huff*>(_gpu_internal + gpu_arrays.entries[0]))
        .template Memset<space::device>(0xff);
    reverse_book.SetDeviceSpace(reinterpret_cast<uint8_t*>(_gpu_internal + gpu_arrays.entries[1]));
    bits_of_chunks.SetDeviceSpace(reinterpret_cast<size_t*>(_gpu_internal + gpu_arrays.entries[2]));
    enc_space.SetDeviceSpace(reinterpret_cast<Huff*>(_gpu_internal + gpu_arrays.entries[3]));
    */