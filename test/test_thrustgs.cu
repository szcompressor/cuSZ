#include <iostream>
#include "../src/cuda_mem.cuh"
#include "../src/gather_scatter.cuh"

using std::cout;
using std::endl;

int main()
{
    using Data             = float;
    using Index            = unsigned int;
    const unsigned int len = 512 * 512 * 512;

    // Data h_dense[len]    = {0};
    auto h_dense         = new Data[len]();
    h_dense[1]           = 1.1;
    h_dense[10]          = 2.2;
    auto         d_dense = mem::CreateDeviceSpaceAndMemcpyFromHost(h_dense, len);
    unsigned int nnz;

    // gather
    Data*  h_sparse;
    Index* h_idxmap;
    std::tie(h_sparse, h_idxmap) = ThrustGatherDualQuantOutlier<Data, Index>(d_dense, len, nnz);
    cout << "\ngather" << endl;
    // for (auto i = 0; i < nnz; i++) { cout << h_sparse[i] << "\t" << h_idxmap[i] << endl; }

    // scatter
    cudaMemset(d_dense, 0x00, len * sizeof(Data));
    auto d_idxmap = mem::CreateDeviceSpaceAndMemcpyFromHost(h_idxmap, nnz);
    auto d_sparse = mem::CreateDeviceSpaceAndMemcpyFromHost(h_sparse, nnz);
    ThrustScatterDualQuantOutlier<Data, Index>(d_dense, len, nnz, d_sparse, d_idxmap);
    cudaMemcpy(h_dense, d_dense, len * sizeof(Data), cudaMemcpyDeviceToHost);
    cout << "\nscatter" << endl;
    // for (auto i = 0; i < len; i++) { cout << h_dense[i] << endl; }
}