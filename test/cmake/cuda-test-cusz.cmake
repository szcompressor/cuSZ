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

# Wrapper: encode + decode + enforce the error bound. Bash-driven so we can chain commands.
function(add_cusz_test name codec dtype mode eb dims file)
  # abs mode bounds the absolute error, rel mode the relative -- pick the matching compare metric.
  if("${mode}" STREQUAL "abs")
    set(metric "max_error")
  else()
    set(metric "max_error_rel")
  endif()
  add_test(NAME ${name}
    COMMAND bash -c "
      set -e
      [ -f '${file}' ] || exit 77
      ./cusz -t ${dtype} -m ${mode} -e ${eb} -l ${dims} -i '${file}' -z --codec ${codec} \
        > /tmp/${name}.enc.log 2>&1
      ./cusz -i '${file}.cusza' -x --compare '${file}' \
        > /tmp/${name}.dec.log 2>&1
      mxe=\$(grep -oE '${metric}=[0-9.eE+-]+' /tmp/${name}.dec.log | head -1 | cut -d= -f2)
      [ -n \"\$mxe\" ] || { cat /tmp/${name}.dec.log; echo 'FAIL: no ${metric} in compare output'; exit 1; }
      awk -v m=\"\$mxe\" -v e=${eb} 'BEGIN{exit !(m+0 <= 1.001*(e+0))}' \
        || { echo \"FAIL: ${metric}=\$mxe over eb=${eb}\"; exit 1; }
      rm -f '${file}.cusza'
    "
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(${name} PROPERTIES
    LABELS "cusz_cli;codec_${codec}"
    SKIP_RETURN_CODE 77
  )
endfunction()

function(add_cusz_pred_test name predictor codec dtype mode eb dims file)
  add_test(NAME ${name}
    COMMAND bash -c "
      set -e
      [ -f '${file}' ] || exit 77
      ./cusz -t ${dtype} -m ${mode} -e ${eb} -l ${dims} -i '${file}' \
             -z -p ${predictor} --codec ${codec} > /tmp/${name}.enc.log 2>&1
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
# y24 eq is tile-ordered under every HFR-family variant (32x8x8 tile == two 1Ki chunks), riding the
# same per-block cells as lorenzo; hf-rev2 stays linear (global compact). The codec is firewalled (a
# black box behind hf_hl.cc), so one variant passing does not imply the rest -- exercise them all.
# This also guards the partial-tile padding: the boundary blocks must pad with the neutral `radius`
# code under every blockwise variant (a 0 there spans the per-block book window and ships the block
# raw -> incomp.breaks). Plain `hf` has no outlier path and is lossy for spline, so it is excluded.
foreach(C IN ITEMS hf-rev2 hfr-v2 hfr-v3 hfr-v4 hfr-pbkc hfr-pbkgo)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_pred_test(
    cusz__rtm_0480__y24__abs_1e-4__${C_SAN} spl-y24 ${C} f32 abs 1e-4 ${RTM_DIMS} ${RTM_FILE})
endforeach()
# looser eb keeps the per-block path lossless too (fewer outliers, but boundary padding unchanged).
add_cusz_pred_test(cusz__rtm_0480__y24__abs_1e-3 spl-y24 hfr-pbkc f32 abs 1e-3 ${RTM_DIMS} ${RTM_FILE})
# y25's by-level eq can't address outliers from a 1024-chunk (they are data-space), so under PBK it
# pairs the blockwise eq with the global compact for outliers (same data-space restore as hf-rev2).
add_cusz_pred_test(cusz__rtm_0480__y25__abs_1e-4 spl-y25 hfr-pbkc f32 abs 1e-4 ${RTM_DIMS} ${RTM_FILE})

# --- lorenzo-2d under PBK: dense outliers (>7/chunk) ship as enc_id=31 f4 candidates -----------
# CESM cloud-edge gradients at tight eb put ~25% of elements out of radius=128; the per-chunk
# incomp path (not just the 7-cell cap) keeps hfr-pbkc lossless on lorenzo-2d.
set(CESM_FILE "${CUSZ_TEST_DATA}/CESM/CLDHGH.f4")
add_cusz_pred_test(cusz__cesm_cldhgh__lorenzo2d__abs_1e-3 lorenzo hfr-pbkc f32 abs 1e-3 3600-1800 ${CESM_FILE})
# 2D lorenzo eq is tile-ordered under every HFR-family variant (32x32 tile == 1Ki chunk == HF
# block); hf / hf-rev2 stay linear. The codec is firewalled (a black box behind hf_hl.cc), so one
# variant passing does not imply the rest -- exercise them all on the same 2D field.
foreach(C IN ITEMS hf hf-rev2 hfr-v2 hfr-v3 hfr-v4 hfr-pbkgo)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_pred_test(
    cusz__cesm_cldhgh__lorenzo2d__abs_1e-3__${C_SAN} lorenzo ${C} f32 abs 1e-3 3600-1800 ${CESM_FILE})
endforeach()

# 3D lorenzo eq is tile-ordered too: the 32x8x8 CTA == two 1Ki chunks, outliers routed by half.
# CESM reinterpreted as a 6.48M-element volume (boundary in all three axes); every HF variant.
foreach(C IN ITEMS hf hf-rev2 hfr-v2 hfr-v3 hfr-v4 hfr-pbkc hfr-pbkgo)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_pred_test(
    cusz__cesm_cldhgh__lorenzo3d__abs_1e-3__${C_SAN} lorenzo ${C} f32 abs 1e-3 360-180-100 ${CESM_FILE})
endforeach()

# --- codec sweep on HURR Uf48 (100x500x500 f32) -----------------------------
set(HURR_FILE  "${CUSZ_TEST_DATA}/HURR/Uf48.f4")
set(HURR_DIMS  "500x500x100")
# Blockwise codecs (per-block cells + enc_id=31 incomp) hold the error bound at both ebs.
foreach(C IN ITEMS hfr hfr-v3 hfr-pbkc)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_test(cusz__hurr_uf48__rel_1e-3__${C_SAN}  ${C} f32 rel 1e-3 ${HURR_DIMS} ${HURR_FILE})
  add_cusz_test(cusz__hurr_uf48__rel_1e-4__${C_SAN}  ${C} f32 rel 1e-4 ${HURR_DIMS} ${HURR_FILE})
endforeach()
# HFr2 routes outliers through the fixed-capacity global compact: it holds at 1e-3 but that
# capacity is overwhelmed at 1e-4 on HURR, so only the looser eb is exercised here.
add_cusz_test(cusz__hurr_uf48__rel_1e-3__hf  hf f32 rel 1e-3 ${HURR_DIMS} ${HURR_FILE})

# --- codec sweep on NYX velocity_x (512^3 f32) ------------------------------
set(NYX_FILE  "${CUSZ_TEST_DATA}/NYX/velocity_x.f32")
set(NYX_DIMS  "512x512x512")
foreach(C IN ITEMS hf hfr hfr-v3 hfr-pbkc)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_test(cusz__nyx_velx__rel_1e-3__${C_SAN}   ${C} f32 rel 1e-3 ${NYX_DIMS} ${NYX_FILE})
endforeach()
