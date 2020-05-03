// 200211
#ifndef SDRB_HH
#define SDRB_HH

#include <string>
#include <unordered_map>

#include "types.hh"

size_t dims_HACC[]        = {280953867, 1, 1, 1, 1};
size_t dims_CESM[]        = {3600, 1800, 1, 1, 2};
size_t dims_Hurricane[]   = {500, 500, 100, 1, 3};
size_t dims_NYX[]         = {512, 512, 512, 1, 3};
size_t dims_QMCPACK1[]    = {288, 69, 7935, 1, 3};
size_t dims_QMCPACK2[]    = {69, 69, 33120, 1, 3};
size_t dims_EXAFEL_demo[] = {388, 59200, 1, 1, 2};
size_t dims_ARAMCO[]      = {235, 849, 849, 1, 3};

static std::unordered_map<std::string, size_t*> dataset_entries = {  //
    {std::string("hacc"), dims_HACC},
    {std::string("cesm"), dims_CESM},
    {std::string("hurricane"), dims_Hurricane},
    {std::string("nyx"), dims_NYX},
    {std::string("qmc"), dims_QMCPACK1},
    {std::string("qmcpre"), dims_QMCPACK2},
    {std::string("exafeldemo"), dims_EXAFEL_demo},
    {std::string("aramco"), dims_ARAMCO}};

size_t* InitializeDemoDims(std::string const& datum,  //
                           size_t             cap,
                           bool               override = false,
                           size_t             new_d0   = 1,
                           size_t             new_d1   = 1,
                           size_t             new_d2   = 1,
                           size_t             new_d3   = 1) {
    auto dims_L16 = new size_t[16]();
    int  BLK;

    if (not override) {
        auto dims_datum = dataset_entries.at(datum);
        std::copy(dims_datum, dims_datum + 4, dims_L16);
        dims_L16[nDIM] = dims_datum[4];
    } else {
        size_t dims_override[] = {new_d0, new_d1, new_d2, new_d3};
        std::copy(dims_override, dims_override + 4, dims_L16);
        dims_L16[nDIM] = ((size_t)new_d0 != 1) + ((size_t)new_d1 != 1) + ((size_t)new_d2 != 1) + ((size_t)new_d3 != 1);
    }

    if (dims_L16[nDIM] == 1)
        BLK = B_1d;
    else if (dims_L16[nDIM] == 2)
        BLK = B_2d;
    else if (dims_L16[nDIM] == 3)
        BLK = B_3d;

    dims_L16[nBLK0]  = (dims_L16[DIM0] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK1]  = (dims_L16[DIM1] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK2]  = (dims_L16[DIM2] - 1) / (size_t)BLK + 1;
    dims_L16[nBLK3]  = (dims_L16[DIM3] - 1) / (size_t)BLK + 1;
    dims_L16[LEN]    = dims_L16[DIM0] * dims_L16[DIM1] * dims_L16[DIM2] * dims_L16[DIM3];
    dims_L16[CAP]    = cap;
    dims_L16[RADIUS] = cap / 2;

    return dims_L16;
}

/*
// TODO move somewhere else
double* setupErrorBoundFamily(config_t* eb_config) {
    auto ebs_L4 = new double[4]();
    ebs_L4[0]   = eb_config->eb_final;            // eb
    ebs_L4[1]   = 1 / eb_config->eb_final;        // 1/eb
    ebs_L4[2]   = 2 * eb_config->eb_final;        // 2* eb
    ebs_L4[3]   = 1 / (2 * eb_config->eb_final);  // 1/(2*eb)
    return ebs_L4;
}
*/

#endif
