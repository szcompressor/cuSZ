/**
 * @file t-l23-v0r1-3d.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2022-12-26
 *
 * (C) 2022 by Indiana University, Argonne National Laboratory
 *
 */

#include "t-l23-utils.hh"

int main()
{
    struct RefactorTestFramework<float, uint16_t> test {};
    test.set_eb(1e-4).init_data_3d().test3d_v0r1_against_origin().destroy();

    return 0;
}
