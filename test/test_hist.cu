#include "../src/cuda_mem.cuh"
#include "../src/io.hh"
#include "../src/thrust_hist.cuh"

int main()
{
    auto len  = (unsigned int)512 * 512 * 512;
    auto data = io::read_binary_to_new_array<float>("/home/jtian/sdrb-nyx/baryon_density.dat", len);

    auto nbin = 128u;
    auto keys = new unsigned int[nbin]();
    auto hist = new unsigned int[nbin]();

    auto d_data = mem::create_devspace_memcpy_h2d(data, nbin);
    auto d_keys = mem::create_devspace_memcpy_h2d(keys, nbin);
    auto d_hist = mem::create_devspace_memcpy_h2d(hist, nbin);

    float* min_val = nullptr;
    float* max_val = nullptr;

    par_ops::use_thrust::Histogram<float>(d_data, len, d_keys, d_hist, nbin, min_val, max_val);
}