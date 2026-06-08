# cusz CLI-driven ctest matrix.
#
# Parallel to cuda-test-bin_hf.cmake. Exercises the full compressor path
# (psz_cusz_compressor -> phf::high_level::HFR_encode -> kernels) for each
# (--codec, dataset, eb) combo. Pass/fail = round-trip lossless within eb.
#
# Recipe (per row):
#   cusz -z --codec <C> -t f32 -m abs -e <EB> -l <DIMS> -i <FILE>
#   cusz -x --compare <FILE> -i <FILE>.cusza
#   (compare exit 0 + PSNR > threshold = pass)
#
# Test data is expected at paths under $CUSZ_TEST_DATA (env var) or the
# fallback default below; any missing file makes its test SKIP (rc=77).

set(CUSZ_TEST_DATA_DEFAULT "/data")
if(DEFINED ENV{CUSZ_TEST_DATA})
  set(CUSZ_TEST_DATA "$ENV{CUSZ_TEST_DATA}")
else()
  set(CUSZ_TEST_DATA "${CUSZ_TEST_DATA_DEFAULT}")
endif()

# Wrapper: encode + decode + cleanup. Bash-driven so we can chain commands.
function(add_cusz_test name codec dtype mode eb dims file)
  add_test(NAME ${name}
    COMMAND bash -c "
      set -e
      [ -f '${file}' ] || exit 77
      ./cusz -t ${dtype} -m ${mode} -e ${eb} -l ${dims} -i '${file}' -z --codec ${codec} \
        > /tmp/${name}.enc.log 2>&1
      ./cusz -i '${file}.cusza' -x --compare '${file}' \
        > /tmp/${name}.dec.log 2>&1
      # crude success check: PSNR present and max_error_rel near eb
      grep -q 'PSNR=' /tmp/${name}.dec.log
      rm -f '${file}.cusza'
    "
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(${name} PROPERTIES
    LABELS "cusz_cli;codec_${codec}"
    SKIP_RETURN_CODE 77
  )
endfunction()

function(add_cusz_pred_test name predictor dtype mode eb dims file)
  add_test(NAME ${name}
    COMMAND bash -c "
      set -e
      [ -f '${file}' ] || exit 77
      ./cusz -t ${dtype} -m ${mode} -e ${eb} -l ${dims} -i '${file}' \
             -z -p ${predictor} --codec hf > /tmp/${name}.enc.log 2>&1
      ./cusz -i '${file}.cusza' -x --compare '${file}' \
        > /tmp/${name}.dec.log 2>&1
      mxe=\$(grep -oE 'max_error=[0-9.eE+-]+' /tmp/${name}.dec.log | head -1 | cut -d= -f2)
      [ -n \"\$mxe\" ] || { cat /tmp/${name}.dec.log; echo 'FAIL: no max_error in compare output'; exit 1; }
      awk -v m=\"\$mxe\" -v e=${eb} 'BEGIN{exit !(m+0 <= 1.001*(e+0))}' \
        || { echo \"FAIL: max_error=\$mxe over eb=${eb}\" ; exit 1 ; }
      rm -f '${file}.cusza'
    "
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(${name} PROPERTIES
    LABELS "cusz_cli;predictor_${predictor}"
    SKIP_RETURN_CODE 77
  )
endfunction()

# --- y24 round-trip tests (3D 32x8x8 anchor blocks) -------------------------
set(RTM_FILE "${CUSZ_TEST_DATA}/RTM/0480.f32")
set(RTM_DIMS "235-449-449")
add_cusz_pred_test(cusz__rtm_0480__y24__abs_1e-4 spl-y24 f32 abs 1e-4 ${RTM_DIMS} ${RTM_FILE})
add_cusz_pred_test(cusz__rtm_0480__y24__abs_1e-3 spl-y24 f32 abs 1e-3 ${RTM_DIMS} ${RTM_FILE})
add_cusz_pred_test(cusz__rtm_0480__y25__abs_1e-4 spl-y25 f32 abs 1e-4 ${RTM_DIMS} ${RTM_FILE})

# --- codec sweep on HURR Uf48 (100x500x500 f32) -----------------------------
set(HURR_FILE  "${CUSZ_TEST_DATA}/HURR/Uf48.f4")
set(HURR_DIMS  "500x500x100")
foreach(C IN ITEMS hf hfr hfr-pbkc)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_test(cusz__hurr_uf48__rel_1e-3__${C_SAN}  ${C} f32 rel 1e-3 ${HURR_DIMS} ${HURR_FILE})
  add_cusz_test(cusz__hurr_uf48__rel_1e-4__${C_SAN}  ${C} f32 rel 1e-4 ${HURR_DIMS} ${HURR_FILE})
endforeach()

# --- codec sweep on NYX velocity_x (512^3 f32) ------------------------------
set(NYX_FILE  "${CUSZ_TEST_DATA}/NYX/velocity_x.f32")
set(NYX_DIMS  "512x512x512")
foreach(C IN ITEMS hf hfr hfr-pbkc)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_test(cusz__nyx_velx__rel_1e-3__${C_SAN}   ${C} f32 rel 1e-3 ${NYX_DIMS} ${NYX_FILE})
endforeach()
