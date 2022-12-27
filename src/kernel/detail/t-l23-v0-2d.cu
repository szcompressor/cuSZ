/**
 * @file t-l23-v0-2d.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "t-l23-utils.hh"

int main()
{
    struct RefactorTestFramework<float, uint16_t> test {};
    test.set_eb(1e-4).init_data_2d().test2d_v0_against_origin().destroy();

    return 0;
}