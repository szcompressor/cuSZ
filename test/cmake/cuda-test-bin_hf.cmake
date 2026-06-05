# ============================================================================
# bin_hf-driven ctest matrix.
#
# Replacement for the legacy gtest-based test suite (now disabled in
# cuda-test.cmake). Each test invokes the existing `bin_hf` example binary
# with `--synth` for deterministic input + `--assert-*` for invariants;
# pass/fail is bin_hf's exit code (0=pass | 1=lossless mismatch | 2=setup |
# 3=assertion failed).
#
# Recipe:
#   bin_hf --dim3 <N>  --bklen <N>
#         <flag: --hf|--hfr|--hfr-pbkc|--hfr-pbkgo>
#         --type u2
#         --synth <spec>
#         [--assert-cr-ge=X --assert-cr-le=X --assert-incomp-le=N --assert-brnum-le=N]
#         [--emit-metrics]
#
# Add new rows by following the existing pattern; no C++ test code needed.
# ============================================================================

# Standard problem size for matrix tests: 6,480,000 elements (matches all
# existing sanity-check inputs). Small enough to run fast on any GPU.
set(BIN_HF_LEN 6480000)
set(BKLEN_U2 1024)
set(BKLEN_U2_PBK 256)  # HFR-PBK family is radius=128, so bklen 256

set(GAMMA_1 0.254763)  # mild 
set(GAMMA_2 0.039351)  # sharp 
set(SYNTH_SEED 43)     # use default

# Helper: register a test that invokes bin_hf with the given args.
# Convention: callers pass <bklen-int> as the first ARGN, then the rest.
function(add_bin_hf_test name)
  set(_args ${ARGN})
  list(GET _args 0 _bklen)
  list(REMOVE_AT _args 0)
  add_test(NAME ${name}
    COMMAND bin_hf --dim3 ${BIN_HF_LEN} --bklen ${_bklen} ${_args})
  set_tests_properties(${name} PROPERTIES
    LABELS "bin_hf"
    SKIP_RETURN_CODE 77
  )
endfunction()

# ----------------------------------------------------------------------------
# Group A: Cauchy, mild 
# CLD*-like: heavy tails 
# ----------------------------------------------------------------------------
add_bin_hf_test(hf__cauchy_mild__u2
  ${BKLEN_U2} --hf --type u2 --synth cauchy:peak=128:gamma=${GAMMA_1}:seed=${SYNTH_SEED}
  --assert-cr-ge=6.0 --assert-cr-le=10.0)

add_bin_hf_test(hfr__cauchy_mild__u2
  ${BKLEN_U2} --hfr --type u2 --synth cauchy:peak=128:gamma=${GAMMA_1}:seed=${SYNTH_SEED}
  --assert-cr-ge=6.0 --assert-cr-le=10.0)

add_bin_hf_test(hfr_pbk_compat__cauchy_mild__u2
  ${BKLEN_U2_PBK} --hfr-pbkc --type u2 --synth cauchy:peak=128:gamma=${GAMMA_1}:seed=${SYNTH_SEED})

add_bin_hf_test(hfr_pbk_go__cauchy_mild__u2
  ${BKLEN_U2_PBK} --hfr-pbkgo --type u2 --synth cauchy:peak=128:gamma=${GAMMA_1}:seed=${SYNTH_SEED})

# ----------------------------------------------------------------------------
# Group B: Cauchy, sharp
# AEROD_v-like: hi-CR, thin tails, expected smallbrnum
# ----------------------------------------------------------------------------
add_bin_hf_test(hf__cauchy_sharp__u2
  ${BKLEN_U2} --hf --type u2 --synth cauchy:peak=128:gamma=${GAMMA_2}:seed=${SYNTH_SEED}
  --assert-cr-ge=10.0 --assert-cr-le=16.0)

add_bin_hf_test(hfr__cauchy_sharp__u2
  ${BKLEN_U2} --hfr --type u2 --synth cauchy:peak=128:gamma=${GAMMA_2}:seed=${SYNTH_SEED}
  --assert-cr-ge=10.0 --assert-cr-le=16.0
  --assert-brnum-le=200000)

add_bin_hf_test(hfr_pbk_compat__cauchy_sharp__u2
  ${BKLEN_U2_PBK} --hfr-pbkc --type u2 --synth cauchy:peak=128:gamma=${GAMMA_2}:seed=${SYNTH_SEED})

add_bin_hf_test(hfr_pbk_go__cauchy_sharp__u2
  ${BKLEN_U2_PBK} --hfr-pbkgo --type u2 --synth cauchy:peak=128:gamma=${GAMMA_2}:seed=${SYNTH_SEED})

# ----------------------------------------------------------------------------
# Group C: uniform (max-entropy within bklen)
# CR ~1 expected for HF/HFR; HFR_PBK relies entirely on incomp-fallback
# ----------------------------------------------------------------------------
add_bin_hf_test(hf__uniform_256__u2
  ${BKLEN_U2} --hf --type u2 --synth uniform:max=256:seed=${SYNTH_SEED})

add_bin_hf_test(hfr__uniform_256__u2
  ${BKLEN_U2} --hfr --type u2 --synth uniform:max=256:seed=${SYNTH_SEED})

add_bin_hf_test(hfr_pbk_compat__uniform_256__u2
  ${BKLEN_U2_PBK} --hfr-pbkc --type u2 --synth uniform:max=256:seed=${SYNTH_SEED})

add_bin_hf_test(hfr_pbk_go__uniform_256__u2
  ${BKLEN_U2_PBK} --hfr-pbkgo --type u2 --synth uniform:max=256:seed=${SYNTH_SEED})

# ----------------------------------------------------------------------------
# Group D: mal-flat codebook: uniform [0, 1024). Every codeword ~10 bits
# make_altcode_single does no longer make sense.
# aggressive incomp fallback: pre-2046df9 this crashed; now CR ~ 1.
# This is the regression test for the make_altcode_single sentinel logic
# AND the dispatch-before-shuffle ordering.
# ----------------------------------------------------------------------------
add_bin_hf_test(hf__uniform_1024__u2
  ${BKLEN_U2} --hf --type u2 --synth uniform:max=1024:seed=${SYNTH_SEED})

add_bin_hf_test(hfr__uniform_1024__u2
  ${BKLEN_U2} --hfr --type u2 --synth uniform:max=1024:seed=${SYNTH_SEED})
