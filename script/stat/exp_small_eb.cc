/**
 * @file exp_small_eb.cc
 * @author Jiannan InputTypeian
 * @brief
 * @version 0.3
 * @date 2021-08-20
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <omp.h>
#include "busyheader.hh"

#include "../src/common.hh"
#include "../src/utils.hh"

std::string fname;
size_t      len;
double      eb;

template <typename InputType = float, typename ComputeType = double>
void f(bool r2r = false)
{
    float*  input32;
    double* input64;

    ComputeType* compute;
    ComputeType* input_dup = new ComputeType[len];

    if constexpr (std::is_same_v<InputType, float>) {
        input32 = new float[len];
        io::read_binary_to_array<float>(fname, input32, len);
    }
    else if constexpr (std::is_same_v<InputType, double>) {
        input64 = new double[len];
        io::read_binary_to_array<double>(fname, input64, len);
    }
    else {
        static_assert("wrong type: must be `float` or `double`");
    }

    if constexpr (std::is_same_v<InputType, ComputeType>) {
        if constexpr (std::is_same_v<InputType, float>)
            compute = input32;
        else if constexpr (std::is_same_v<InputType, double>)
            compute = input64;
    }
    else {
        compute = new ComputeType[len];

        if constexpr (std::is_same_v<InputType, float>)
            for (auto i = 0; i < len; i++) compute[i] = input32[i];
        else if constexpr (std::is_same_v<InputType, double>)
            for (auto i = 0; i < len; i++) compute[i] = input64[i];
    }

    // make a duplicate for verification
    memcpy(input_dup, compute, len * sizeof(ComputeType));

    if (r2r) {
        auto minval = std::min_element(compute, compute + len);
        auto maxval = std::max_element(compute, compute + len);
        auto rng    = maxval - minval;
        eb *= rng;
    }

    double ebx2   = eb * 2;
    double ebx2_r = 1 / (ebx2);

    {
        auto a = hires::now();
        // omp_set_num_threads(1);
#pragma omp parallel
        {
            printf("thread number: %d\n", omp_get_num_threads());
            auto p_ebx2_r = static_cast<ComputeType>(ebx2_r);
            auto p_ebx2   = static_cast<ComputeType>(ebx2);
            auto i        = 0;
#pragma omp parallel for shared(len, compute) private(i, p_ebx2_r, p_ebx2)
            for (i = 0; i < len; i++) compute[i] = std::round(compute[i] * ebx2_r) * ebx2;
        }
        auto z = hires::now();
        cout << "dryrun time: " << static_cast<duration_t>(z - a).count() << " sec\n";
    }

    Stat stat;

    cusz::verify_data(&stat, compute, input_dup, len);
    psz::print_metrics<double>(&stat, 0, false);
}

int main(int argc, char** argv)
{
    if (argc < 4) throw std::runtime_error("./prog fname len eb <input type> <compute type> <if r2r>");
    fname = std::string(argv[1]);

    char* _end;
    len = std::strtol(argv[2], &_end, 10);
    eb  = std::strtod(argv[3], &_end);

    if (argc < 5) f<float, float>();

    bool if_r2r = false;

    if (argc == 7) if_r2r = std::string(argv[5]) == "r2r" or std::string(argv[5]) == "true";

    if (argc >= 6) {
        if (std::string(argv[4]) == "float" and std::string(argv[5]) == "float")
            f<float, float>(if_r2r);
        else if (std::string(argv[4]) == "float" and std::string(argv[5]) == "double")
            f<float, double>(if_r2r);
        else if (std::string(argv[4]) == "double" and std::string(argv[5]) == "double")
            f<double, double>(if_r2r);
        else if (std::string(argv[4]) == "double" and std::string(argv[5]) == "float")
            f<double, float>(if_r2r);
    }
}