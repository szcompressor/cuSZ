# ------------------------------------------------------------------------------
# Pure-ctest unit tests for _portable primitives.
#
# Each test is a standalone binary that uses assert() and exits 0 on pass.
# No gtest. Add a new test by adding a portable_add_test(<name>) line plus
# the corresponding test_<name>.cc file in this directory.
# ------------------------------------------------------------------------------

function(portable_add_test name)
  add_executable(test_${name} test_${name}.cc)
  target_link_libraries(test_${name} PRIVATE PORTABLE)
  # assert() must remain active even in Release builds for tests to be useful.
  target_compile_options(test_${name} PRIVATE -UNDEBUG)
  add_test(NAME portable_${name} COMMAND test_${name})
endfunction()

portable_add_test(str2num)
portable_add_test(kv_parse)
portable_add_test(check_in)
portable_add_test(val_eq)
portable_add_test(arg_builder)
portable_add_test(kv_binder)
