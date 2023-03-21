/**
 * @file t-l23-v0c-2d.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.4
 * @date 2023-01-04
 *
 * (C) 2023 by Indiana University, Argonne National Laboratory
 *
 */

#include "test_l23_utils.hh"

int main()
{
    struct RefactorTestFramework<float, uint16_t> test {};
    test.set_eb(1e-4).init_data_2d().test2d_v0compaction_against_origin().destroy();

    return 0;
}