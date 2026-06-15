# ============================================================================
# bin_pred-driven ctest matrix (single-predictor microbench).
#
# Each row invokes `bin_pred` with --dataset (TOML registry-driven),
# --predictor, and --assert-* invariants. Pass/fail is bin_pred's exit code:
#   0 pass | 1 verification failure | 2 setup error |
#   3 assertion failure | 77 fixture absent (-> ctest "Skipped")
#
# Row name convention:    pred__<dataset_key>__<variant>__{eb<eb>,rel<eb>}
# Dataset keys come from ~/.psz_test_data.toml (or $PSZ_TEST_DATA, or
# --config). When the dataset path is missing, the row is SKIPPED via the
# 77 exit code (set up via --require-file + SKIP_RETURN_CODE 77).
#
# Develop carries the v1-only matrix (predictor name "spline"). The
# spline-evolution branch (psz-spl) extends this with spl-v3..v4r3 rows.
# ============================================================================

if(NOT DEFINED PSZ_TEST_DATA_TOML)
  set(PSZ_TEST_DATA_TOML
    "${CMAKE_CURRENT_SOURCE_DIR}/../example/src/test_lib/sample_data.toml")
endif()

function(add_bin_pred_row name dataset predictor)
  add_test(NAME ${name}
    COMMAND bin_pred1
      --config ${PSZ_TEST_DATA_TOML}
      --dataset ${dataset}
      --require-file
      --predictor ${predictor}
      ${ARGN})
  set_tests_properties(${name} PROPERTIES
    LABELS "bin_pred"
    SKIP_RETURN_CODE 77
  )
endfunction()

# ----------------------------------------------------------------------------
# Absolute-error-bound rows: spline (kernel_v1) on each registry dataset.
# ----------------------------------------------------------------------------
add_bin_pred_row(pred__RTM_0480__spline__eb1e-4
  RTM.0480 spline
  --assert-max-err-le=1.001e-4 --assert-psnr-ge=80.0)

add_bin_pred_row(pred__HURR_CLOUD48__spline__eb1e-3
  HURR.CLOUD48 spline
  --assert-max-err-le=1.001e-3 --assert-psnr-ge=30.0)

add_bin_pred_row(pred__Nyx_Baryon__spline__eb1e7
  Nyx.Baryon spline
  --assert-max-err-le=1.001e7 --assert-psnr-ge=70.0)

add_bin_pred_row(pred__CESM_CLDHGH__spline__eb1e-3
  CESM.CLDHGH spline
  --assert-max-err-le=1.001e-3 --assert-psnr-ge=30.0)

# ----------------------------------------------------------------------------
# Relative-error-bound rows: same datasets, --rel mode.
# Universal floors leave ~7 dB margin against the worst observed.
# ----------------------------------------------------------------------------
set(_PRED_DATASETS RTM.0480 HURR.CLOUD48 Nyx.Baryon CESM.CLDHGH)
set(_PRED_DATASET_TAGS RTM_0480 HURR_CLOUD48 Nyx_Baryon CESM_CLDHGH)

list(LENGTH _PRED_DATASETS _PD_N)
math(EXPR _PD_LAST "${_PD_N} - 1")

foreach(_d_idx RANGE ${_PD_LAST})
  list(GET _PRED_DATASETS    ${_d_idx} _ds)
  list(GET _PRED_DATASET_TAGS ${_d_idx} _ds_tag)
  add_bin_pred_row(pred__${_ds_tag}__spline__rel1e-4
    ${_ds} spline
    --rel --eb 1e-4
    --assert-max-err-rel-le=1.001e-4 --assert-psnr-ge=85.0)
  add_bin_pred_row(pred__${_ds_tag}__spline__rel1e-3
    ${_ds} spline
    --rel --eb 1e-3
    --assert-max-err-rel-le=1.001e-3 --assert-psnr-ge=65.0)
endforeach()
unset(_PRED_DATASETS)
unset(_PRED_DATASET_TAGS)
