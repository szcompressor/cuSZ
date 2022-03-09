/**
 * @file ex_api_app.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2022-03-07
 *
 * (C) 2022 by Washington State University, Argonne National Laboratory
 *
 */

#include "../../src/app.cuh"

void demo_encapsulate_io()
{
    char* cesm  = getenv(const_cast<char*>("CESM"));
    auto  fname = std::string(cesm);
    cout << fname << "\n";

    cusz::app<float> cusz;

    cout << "\ncompression demo:\n";
    {
        std::string config("do=compress,dtype=f32,eb=3.4e-4,mode=r2r,size=3600x1800,radius=512");
        config   = config + ",input=" + fname;
        auto ctx = new cuszCTX(config.c_str(), false);

        cusz.cusz_dispatch(ctx);
    }

    cout << "\ndecompression demo:\n";
    {
        std::string config("do=decompress");
        config   = config + ",input=" + fname + ".cusza";
        config   = config + ",compare=" + fname;
        auto ctx = new cuszCTX(config.c_str(), false);

        cusz.cusz_dispatch(ctx);
    }
}

void demo_expose_io()
{
    using T = float;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    Capsule<T>    uncompressed("uncompressed");
    Capsule<BYTE> compressed("compressed");
    Capsule<T>    decompressed("decompressed"), cmp("cmp");

    char* cesm  = getenv(const_cast<char*>("CESM"));
    auto  fname = std::string(cesm);
    cout << fname << "\n";

    cusz::app<T> cusz_instance;

    BYTE*  d_compressed;
    size_t compressed_len;

    // extended compression
    {
        std::string config("do=compress,dtype=f32,eb=3.3e-4,mode=r2r,size=3600x1800,radius=512");
        config   = config + ",input=" + fname;
        auto ctx = new cuszCTX(config.c_str(), false);

        auto len = (*ctx).x * (*ctx).y * (*ctx).z;

        cusz::app<T>::input_uncompressed<T>(uncompressed, len, fname);
        if ((*ctx).mode == "r2r") (*ctx).eb *= uncompressed.prescan().get_rng();

        // core compression
        auto core_compressed = [&]() {
            cusz_instance.init_compressor(ctx);
            cusz_instance.cusz_compress(uncompressed.dptr, ctx, stream, (*ctx).report.time);
            cusz_instance.get_compressed(d_compressed, compressed_len);
        };

        core_compressed();
    }

    // extended decompression
    {
        auto header = new cuszHEADER;
        cudaMemcpy(header, d_compressed, sizeof(cuszHEADER), cudaMemcpyDeviceToHost);

        auto len = (*header).get_uncompressed_len();
        decompressed.set_len(len).template alloc<cusz::LOC::HOST_DEVICE, cusz::ALIGNDATA::SQUARE_MATRIX>();
        cmp.set_len(len);

        std::string config("do=decompress");
        config   = config + ",input=" + fname + ".cusza";
        config   = config + ",compare=" + fname;
        auto ctx = new cuszCTX(config.c_str(), false);

        // core decompression
        {
            cusz_instance.init_compressor(header);
            cusz_instance.cusz_decompress(d_compressed, header, decompressed.dptr, stream, (*ctx).report.time);

            cusz_instance.try_compare(header, decompressed, cmp, (*ctx).fname.origin_cmp);
            cusz_instance.try_write(decompressed, fname, (*ctx).to_skip.write2disk);
        }
    }

    cusz_instance.destroy_compressor();
}

int main()
{
    demo_encapsulate_io();
    cout << "--------------------------------------------------------------------------------\n";
    demo_expose_io();

    return 0;
}