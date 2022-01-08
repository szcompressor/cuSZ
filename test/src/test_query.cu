/**
 * @file query.cu
 * @author Jiannan Tian
 * @brief
 * @version 0.1.3
 * @date 2020-10-05
 *
 * @copyright (C) 2020 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include "query.hh"

#ifdef MAIN
int main(int argc, char** argv)
{
    GetMachineProperties();
    GetDeviceProperty();
    return 0;
}
#endif