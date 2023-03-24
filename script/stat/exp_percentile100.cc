/**
 * @file exp_precentile10.cc
 * @author Jiannan Tian
 * @brief
 * @version 0.3
 * @date 2021-12-11
 *
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include "../src/analysis/analyzer.hh"
#include "../src/common.hh"

int main(int argc, char** argv)
{
    if (argc < 4) {
        cout << "./<prog> <filename> <len> <host|device>\n";
        return 0;
    }

    int len = atoi(argv[2]);

    Capsule<float> in(atoi(argv[2]));
    in  //
        .template malloc()
        .template mallochost()
        .template fromfile(argv[1])
        .host2device();

    auto where = std::string(argv[3]);
    auto a     = hires::now();

    std::vector<float> res;

    if (where == "host")
        res = Analyzer::percentile100<float, ExecutionPolicy::host>(in.hptr, in.len);
    else if (where == "device") {
        res = Analyzer::percentile100<float, ExecutionPolicy::cuda_device>(in.dptr, in.len);
    }

    auto b = hires::now();
    cout << static_cast<duration_t>(b - a).count() << " sec\n";

    for (auto i = 0; i < res.size(); i++) {
        if (i != 0 and i % 5 == 0) printf("\n");
        printf("%f\t", res[i]);
    }
    cout << '\n';

    in.template free().template freehost();

    return 0;
}
