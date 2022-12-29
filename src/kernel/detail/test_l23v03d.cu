/**
 * @file test_l23v02d.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-29
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "test_utils.hh"

int main()
{
    struct RefactorTestFramework<float, uint16_t> test {};
    test.set_eb(1e-4).init_data_3d().test3d_v0_against_origin().destroy();

    return 0;
}