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

int disposable_non_performance_critical()
{
    char* cesm  = getenv(const_cast<char*>("CESM"));
    auto  fname = std::string(cesm);
    cout << fname << "\n";

    {
        std::string config("do=compress,dtype=f32,eb=3.3e-4,mode=r2r,size=3600x1800,radius=512");
        config = config + ",input=" + fname;

        auto ctx = new cuszCTX(config.c_str(), true);
        cusz::app<float>::defaultpath(ctx);
    }

    {
        std::string config("do=decompress");
        config = config + ",input=" + fname + ".cusza";
        config = config + ",compare=" + fname;

        auto ctx = new cuszCTX(config.c_str(), true);
        cusz::app<float>::defaultpath(ctx);
    }
}

int g() {}

int main()
{
    f();
    cout << "--------------------------------------------------------------------------------\n";
    g();
}