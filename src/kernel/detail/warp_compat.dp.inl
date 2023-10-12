namespace psz::dpcpp::compat {

template <typename T, bool EXPERIMENTAL_MASKED = true>
T shift_sub_group_right(
    unsigned int member_mask, sycl::sub_group g, T x, unsigned int delta,
    int logical_sub_group_size = 32)
{
  if constexpr (EXPERIMENTAL_MASKED) {
    /*
    DPCT1108: '__shfl_up_sync' was migrated with the experimental feature
    masked sub_group function which may not be supported by all compilers or
    runtimes. You may need to adjust the code.
    */
    return dpct::experimental::shift_sub_group_right(
        member_mask, g, x, delta, logical_sub_group_size);
  }
  else {
    /*
    DPCT1023: The SYCL sub-group does not support mask options for
    dpct::shift_sub_group_right. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_up_sync.
    */
    return dpct::shift_sub_group_right(g, x, delta, logical_sub_group_size);
  }
}

}  // namespace psz::dpcpp::compat