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

# Same as add_cusz_pred_test, but sets both --codec1 and --codec2 (HiCR / HiTP archive shapes).
function(add_cusz_dualcodec_pred_test name predictor codec1 codec2 dtype mode eb dims file)
  add_test(NAME ${name}
    COMMAND bash -c "
      set -e
      [ -f '${file}' ] || exit 77
      ./cusz -t ${dtype} -m ${mode} -e ${eb} -l ${dims} -i '${file}' \
             -z -p ${predictor} --codec1 ${codec1} --codec2 ${codec2} > /tmp/${name}.enc.log 2>&1
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
# same per-block cells as lorenzo; hf and hf-rev2 stay linear (global compact). The codec is
# firewalled (a black box behind hf_hl.cc), so one variant passing does not imply the rest --
# exercise them all. This also guards the partial-tile padding: the boundary blocks must pad with
# the neutral `radius` code under every blockwise variant (a 0 there spans the per-block book
# window and ships the block raw -> incomp.breaks).
foreach(C IN ITEMS hf hf-rev2 hfr-v2 hfr-v4 hfr-pbkc hfr-pbkgo)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_pred_test(
    cusz__rtm_0480__y24__abs_1e-4__${C_SAN} spl-y24 ${C} f32 abs 1e-4 ${RTM_DIMS} ${RTM_FILE})
endforeach()
# looser eb keeps the per-block path lossless too (fewer outliers, but boundary padding unchanged).
add_cusz_pred_test(cusz__rtm_0480__y24__abs_1e-3 spl-y24 hfr-pbkc f32 abs 1e-3 ${RTM_DIMS} ${RTM_FILE})
# y25's by-level eq can't address outliers from a 1024-chunk (they are data-space), so under PBK it
# pairs the blockwise eq with the global compact for outliers (same data-space restore as hf-rev2).
add_cusz_pred_test(cusz__rtm_0480__y25__abs_1e-4 spl-y25 hfr-pbkc f32 abs 1e-4 ${RTM_DIMS} ${RTM_FILE})
# spline's only outlier sink is the global compact (no per-block incomp), so plain `hf` (not
# HFR-family, not hf-rev2) must also enable it -- regression test for the dropped-outlier bug.
add_cusz_pred_test(cusz__rtm_0480__y25__abs_1e-4__hf spl-y25 hf f32 abs 1e-4 ${RTM_DIMS} ${RTM_FILE})
# TCMS (codec1=lc, HiTP-eq shape) keeps the global compact too, at this same eb/dataset where the
# HFR-family rows above are already known to hold (i.e. within the compact's fixed capacity).
add_cusz_pred_test(cusz__rtm_0480__y24__abs_1e-4__lc spl-y24 lc-tcms f32 abs 1e-4 ${RTM_DIMS} ${RTM_FILE})
add_cusz_pred_test(cusz__rtm_0480__y25__abs_1e-4__lc spl-y25 lc-tcms f32 abs 1e-4 ${RTM_DIMS} ${RTM_FILE})

# --- lorenzo-2d under PBK: dense outliers (>7/chunk) ship as enc_id=31 f4 candidates -----------
# CESM cloud-edge gradients at tight eb put ~25% of elements out of radius=128; the per-chunk
# incomp path (not just the 7-cell cap) keeps hfr-pbkc lossless on lorenzo-2d.
set(CESM_FILE "${CUSZ_TEST_DATA}/CESM/CLDHGH.f4")
add_cusz_pred_test(cusz__cesm_cldhgh__lorenzo2d__abs_1e-3 lorenzo hfr-pbkc f32 abs 1e-3 3600-1800 ${CESM_FILE})
# 2D lorenzo eq is tile-ordered under every HFR-family variant (32x32 tile == 1Ki chunk == HF
# block); hf / hf-rev2 stay linear. The codec is firewalled (a black box behind hf_hl.cc), so one
# variant passing does not imply the rest -- exercise them all on the same 2D field.
# lc: TCMS eq-only (no HF/HFR at all); tile-ordered here too, so it shares the same padded-len_eq
# byte-count requirement as the HF/HFR variants above.
foreach(C IN ITEMS hf hf-rev2 hfr-v2 hfr-v4 hfr-pbkgo lc-tcms)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_pred_test(
    cusz__cesm_cldhgh__lorenzo2d__abs_1e-3__${C_SAN} lorenzo ${C} f32 abs 1e-3 3600-1800 ${CESM_FILE})
endforeach()

# 1D lorenzo has no tile order at all -- the simplest reproduction of the LC decode-side
# eq destination bug (mem->eq_d() vs d_space), independent of any tile-order sizing question.
add_cusz_pred_test(cusz__cesm_cldhgh__lorenzo1d__abs_1e-3__lc lorenzo lc-tcms f32 abs 1e-3 6480000 ${CESM_FILE})

# HiTP (codec1=lc, codec2=lc) additionally BITR-compresses [anchor][spfmt]; at this eb that
# region is zero-length (no anchor for lorenzo, no outliers), which used to crash BITR's decode
# (the chunked kernel never writes *outsize for a zero-byte input) -- guards that empty-input path.
add_cusz_dualcodec_pred_test(
  cusz__cesm_cldhgh__lorenzo2d__abs_1e-3__hitp lorenzo lc-tcms lc-bitr f32 abs 1e-3 3600-1800 ${CESM_FILE})
# HiCR (codec1=hf, codec2=lc): plain HF for eq, RTR wraps [HF][anchor][spfmt] together.
add_cusz_dualcodec_pred_test(
  cusz__cesm_cldhgh__lorenzo2d__abs_1e-3__hicr lorenzo hf lc-rtr f32 abs 1e-3 3600-1800 ${CESM_FILE})
# HiCR via hf-rev2 (codec1=hf-rev2, codec2=lc): same archive shape as codec1=hf (both alias into
# the same Huffman_rev2 encode + RTR pass2 on compress), but decode used to check codec1==HF only
# and miss HFr2, landing in the HiTP arm (TCMS_DECOMPRESS on an RTR-compressed stream) -- crash.
add_cusz_dualcodec_pred_test(
  cusz__cesm_cldhgh__lorenzo2d__abs_1e-3__hicr_hfrev2 lorenzo hf-rev2 lc-rtr f32 abs 1e-3 3600-1800 ${CESM_FILE})

# --- LC with real outliers present (abs_1e-3 above has ~0 on this field: too loose to catch a
# dropped-outlier regression) ------------------------------------------------------------------
# At abs 1e-4 CLDHGH puts ~4.5k-103k elements/config out of radius (<10% of the global compact's
# fixed capacity, so none get capacity-dropped): a real, nonzero exercise of the codec1==LC /
# codec2==LC outlier path (splen / enable_global), across every tile-order shape (1D linear, 2D
# and 3D tile-order) and every LC archive variant (TCMS-only, HiTP, HiCR).
add_cusz_pred_test(cusz__cesm_cldhgh__lorenzo1d__abs_1e-4__lc lorenzo lc-tcms f32 abs 1e-4 6480000 ${CESM_FILE})
add_cusz_pred_test(cusz__cesm_cldhgh__lorenzo2d__abs_1e-4__lc lorenzo lc-tcms f32 abs 1e-4 3600-1800 ${CESM_FILE})
add_cusz_pred_test(cusz__cesm_cldhgh__lorenzo3d__abs_1e-4__lc lorenzo lc-tcms f32 abs 1e-4 360-180-100 ${CESM_FILE})
add_cusz_pred_test(cusz__cesm_cldhgh__spl_y25__abs_1e-4__lc spl-y25 lc-tcms f32 abs 1e-4 3600-1800 ${CESM_FILE})
# Plain `hf` (codec1==HF, no codec2) shares LC's global-compact requirement for spline (see above):
# regression for PSNR silently degrading as eb tightens (55.9/28.1 at rel 1e-3/1e-4 pre-fix, vs.
# hf-rev2's 66.7/85.4) because splen was forced to 0 and out-of-radius deltas got clamped, not kept.
add_cusz_pred_test(cusz__cesm_cldhgh__spl_y25__abs_1e-4__hf spl-y25 hf f32 abs 1e-4 3600-1800 ${CESM_FILE})
add_cusz_dualcodec_pred_test(
  cusz__cesm_cldhgh__lorenzo2d__abs_1e-4__hitp lorenzo lc-tcms lc-bitr f32 abs 1e-4 3600-1800 ${CESM_FILE})
add_cusz_dualcodec_pred_test(
  cusz__cesm_cldhgh__lorenzo2d__abs_1e-4__hicr lorenzo hf lc-rtr f32 abs 1e-4 3600-1800 ${CESM_FILE})

# 3D lorenzo eq is tile-ordered too: the 32x8x8 CTA == two 1Ki chunks, outliers routed by half.
# CESM reinterpreted as a 6.48M-element volume (boundary in all three axes); every HF variant.
foreach(C IN ITEMS hf hf-rev2 hfr-v2 hfr-v4 hfr-pbkc hfr-pbkgo lc-tcms)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_pred_test(
    cusz__cesm_cldhgh__lorenzo3d__abs_1e-3__${C_SAN} lorenzo ${C} f32 abs 1e-3 360-180-100 ${CESM_FILE})
endforeach()

# --- codec sweep on HURR Uf48 (100x500x500 f32) -----------------------------
set(HURR_FILE  "${CUSZ_TEST_DATA}/HURR/Uf48.f4")
set(HURR_DIMS  "500x500x100")
# Blockwise codecs (per-block cells + enc_id=31 incomp) hold the error bound at both ebs.
foreach(C IN ITEMS hfr hfr-pbkc)
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
foreach(C IN ITEMS hf hfr hfr-pbkc)
  string(REPLACE "-" "_" C_SAN ${C})
  add_cusz_test(cusz__nyx_velx__rel_1e-3__${C_SAN}   ${C} f32 rel 1e-3 ${NYX_DIMS} ${NYX_FILE})
endforeach()
