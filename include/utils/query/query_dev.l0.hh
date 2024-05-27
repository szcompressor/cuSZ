//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef BC913DD9_0EFD_46CB_BDF7_59EF95B36D7D
#define BC913DD9_0EFD_46CB_BDF7_59EF95B36D7D

namespace info = sycl::info;

struct l0_diagnostics {
  static void show_device()
  {
    sycl::queue q(sycl::gpu_selector_v);

    // Output platform and device information.
    auto device = q.get_device();
    auto p_name = device.get_platform().get_info<info::platform::name>();
    cout << std::setw(20) << "Platform Name: " << p_name << "\n";
    auto p_version = device.get_platform().get_info<info::platform::version>();
    cout << std::setw(20) << "Platform Version: " << p_version << "\n";
    auto d_name = device.get_info<info::device::name>();
    cout << std::setw(20) << "Device Name: " << d_name << "\n";
    auto max_work_group = device.get_info<info::device::max_work_group_size>();
    cout << std::setw(20) << "Max Work Group: " << max_work_group << "\n";
    auto max_compute_units =
        device.get_info<info::device::max_compute_units>();
    cout << std::setw(20) << "Max Compute Units: " << max_compute_units
         << "\n\n";
  }
};

#endif /* BC913DD9_0EFD_46CB_BDF7_59EF95B36D7D */