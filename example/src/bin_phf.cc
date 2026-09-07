#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "arg_builder.hh"
#include "compare.hh"
#include "hfd26.hh"
#include "kernel.hh"
#include "phf.hh"
#include "ptb.hh"

using std::string;
namespace utils = _ptb::utils;
using _ptb::GiBps;
using _ptb::gpu_timer;

using F = u4;

struct RunMetrics {
  bool lossless = false;
  double cr = 0.0;
  size_t arch_bytes = 0;
  size_t bs_bytes = 0;
  size_t incomp_blocks = 0;
  string codec;
  string dtype;
  size_t len = 0;
};

RunMetrics g_metrics;

// single source for the default reduce-merge pass count
constexpr int DefaultReduceTimes = (int)psz::HFR_PBK_Constants::ReduceTimes;
static const std::string DefaultReduceStr = std::to_string(DefaultReduceTimes);

template <typename E>
void load_input(const string& fname, size_t len, const string& synth_spec, E* out)
{
  if (not synth_spec.empty()) {
    auto s = _ptb::testutils::Synth::parse(synth_spec);
    s.fill((void*)out, len, _ptb::TypeSym<E>::type);
  }
  else {
    utils::fromfile_or_die(fname.c_str(), out, len);
  }
}

template <typename E>
void load_or_preload(
    const string& fname, size_t len, const string& synth_spec, E const* preloaded, E* out)
{
  if (preloaded)
    std::memcpy(out, preloaded, len * sizeof(E));
  else
    load_input<E>(fname, len, synth_spec, out);
}

extern "C" void* pbk25_r128_rvbk_d_ptr();

// flag: PHF_SELFSYNC_PROBE; offline self-sync verification with PBK
namespace selfsync_probe {

constexpr int NumBooks = (int)psz::HFR_PBK_Constants::NumBooks;
constexpr int RvbkBytesPerBook = (int)psz::HFR_PBK_Constants::RvbkBytesPerBook;
constexpr int Budget = 128;  // codewords; well past any plausible convergence

struct BookStats {
  size_t chunks = 0, tests = 0, sum_syms = 0, max_syms = 0, fails = 0;
};

using phf::cpu_ref::walk_n;

template <int Magnitude>
void run(
    u4 const* h_bitstream, size_t bitstream_words, u4 const* h_packed, int pardeg,
    u1 const* h_rvbk_pool, BookStats* stats /* [NumBooks] */)
{
  for (int gid = 0; gid < pardeg; ++gid) {
    u4 const w0 = h_packed[2 * gid];
    u4 const tree_idx = psz::unpack_par_encid<Magnitude>(w0);
    if (tree_idx >= (u4)NumBooks) continue;  // pass-through chunk

    u4 const unit_start = psz::unpack_par_entry_words<u4>(h_packed, gid);
    u4 const dense_words = psz::unpack_par_dense<Magnitude>(w0);
    u4 const n_unpred = psz::unpack_par_nunpred<Magnitude>(w0);

    // Slot layout [breaks | bitstream | unpred], same as the kernel.
    u4 const total_words =
        psz::unpack_par_end_words<u4>(h_packed, gid, pardeg, bitstream_words * sizeof(u4)) -
        unit_start;
    u4 const n_breaks = total_words - dense_words - psz::pbk_unpred_words(n_unpred);
    u4 const* bs = h_bitstream + unit_start + n_breaks;
    int const bit_end = (int)(dense_words * 32u);
    if (bit_end <= 0) continue;

    u4 const* first = reinterpret_cast<u4 const*>(h_rvbk_pool + tree_idx * RvbkBytesPerBook);

    // Ground truth: every true codeword boundary from bit 0.
    auto truth = MAKE_UNIQUE_HOST(int, (size_t)bit_end + 1);
    int truth_n = 0;
    {
      int i = 0;
      while (i < bit_end) {
        int const next = walk_n(bs, first, i, bit_end, 1);
        if (next == i) break;  // no progress; malformed tail
        truth[truth_n++] = next;
        i = next;
      }
    }
    if (truth_n == 0) continue;
    auto is_true = MAKE_UNIQUE_HOST(u1, (size_t)bit_end + 1);  // malloc_host zeroes it
    for (int k = 0; k < truth_n; ++k) {
      int const t = truth[k];
      if (t <= bit_end) is_true[t] = 1;
    }

    BookStats& st = stats[tree_idx];
    ++st.chunks;

    for (int off = 1; off <= 31 and off < bit_end; ++off) {
      ++st.tests;
      int i = off;
      bool synced = false;
      for (int step = 1; step <= Budget; ++step) {
        int const next = walk_n(bs, first, i, bit_end, 1);
        if (next == i) break;  // malformed; stop
        i = next;
        if (i <= bit_end and is_true[i]) {
          st.sum_syms += step;
          if ((size_t)step > st.max_syms) st.max_syms = step;
          synced = true;
          break;
        }
      }
      if (not synced) ++st.fails;
    }
    (void)n_unpred;
  }
}

void report(BookStats const* stats)
{
  fprintf(stderr, "[selfsync-probe] book chunks tests mean_syms max_syms fails\n");
  for (int b = 0; b < NumBooks; ++b) {
    BookStats const& st = stats[b];
    if (st.chunks == 0) continue;
    double const mean = st.tests ? (double)st.sum_syms / (double)st.tests : 0.0;
    fprintf(
        stderr, "[selfsync-probe] %2d %6zu %6zu %8.2f %8zu %5zu\n", b, st.chunks, st.tests, mean,
        st.max_syms, st.fails);
  }
}

}  // namespace selfsync_probe

namespace {

struct HFVariant {
  const char* label;
  const char* metric_name;
  psz_codec codec;
  bool is_hfr_family;
  bool skip_hist_and_book;
  bool use_HFR_buf;
  bool suppress_lago_col;
  bool use_hfd26 = false;  // route decode through HFD26_decode
};

// clang-format off
namespace hfv {
constexpr HFVariant HF      = {"Huffman",      "hf",        psz_codec::HF,        false, false, false, false};
constexpr HFVariant HF_REV2 = {"Huffman-rev2", "hf-rev2",   psz_codec::HFr2,      false, false, false, false};
constexpr HFVariant HFR     = {"HFR",          "hfr",       psz_codec::HFR,       true,  false, true,  false};
constexpr HFVariant PBKC    = {"HFR-PBKC",     "hfr-pbkc",  psz_codec::HFR_PBKC,  true,  true,  true,  false};
constexpr HFVariant PBKGO   = {"HFR-PBKGO",    "hfr-pbkgo", psz_codec::HFR_PBKGO, true,  true,  true,  true};
constexpr HFVariant HF_V3   = {"HFR-v3",        "hfr-v3",    psz_codec::HFR_V3,    true,  false, true,  false};
constexpr HFVariant HFD26       = {"HFD26",       "hfd26",       psz_codec::HFR,       true, false, true, false, true};
constexpr HFVariant HFD26_PBKC  = {"HFD26-PBKC",  "hfd26-pbkc",  psz_codec::HFR_PBKC,  true, true,  true, false, true};
constexpr HFVariant HFD26_PBKGO = {"HFD26-PBKGO", "hfd26-pbkgo", psz_codec::HFR_PBKGO, true, true,  true, true,  true};
}  // namespace hfv

static const auto bin_phf_cli =
    _ptb::arg_builder("bin_phf")
        .string ("input",             {"-i", "--input"},       "",       "input binary file (omit when --synth is set)")
        .dim3   ("dim3",              {"-l", "--dim3", "--len"},         "data dimensions: NxMxK or N (1-D)")
        .integer("bklen",             {"--bklen"},             1024,     "Huffman book length")
        .string ("path",              {"--path"},              "",       "comma-separated pipelines; load once, test all (e.g. hf,hf_rev2,hfr,hfr_pbkc)")
        .flag   ("hf",                {"--hf"},                          "plain Huffman (default)")
        .flag   ("hf_rev2",           {"--hf-rev2"},                     "Huffman rev.2: ph1+ph2 + concat + AoS bheader_backport[]")
        .flag   ("hfr",               {"--hfr"},                         "use HFR")
        .flag   ("hfr_pbkc",          {"--hfr-pbkc"},                    "use HFR-PBK-compat")
        .flag   ("hfr_pbkgo",         {"--hfr-pbkgo"},                   "use HFR-PBK-GO")
        .flag   ("hfr_pbkf",          {"--hfr-pbkf"},                    "use HFR-PBKF (gated; build needs -DPHF_ENABLE_HFR_PBKF=ON)")
        .flag   ("hfd26",             {"--hfd26"},                       "decode HFR with HFD26 instead of HFR_decode")
        .string ("type",              {"--type", "--dtype"},   "u2",     "u1|u2|u4")
        .integer("repeat",            {"--repeat"},            5,        "timed iterations (min reported; 10 untimed warmups run first)")
        .string ("reduce",            {"--rmerge-count"},      DefaultReduceStr.c_str(), "r-merge pass count (ReduceTimes), CSV e.g. 2,3,4 | auto or 0 (predict from book) | best (measure each) for HFR family; default = HFR_PBK_Constants::ReduceTimes")
        .integer("magnitude",         {"--magnitude"},         10,       "HFR block-size magnitude: 10=1Ki (default), 11=2Ki, 12=4Ki (HFR-PBKC only)")
        .integer("blockdim",          {"--blockdim"},          128,      "HFR-PBKC threadblock at 4Ki: 128 (Iters=4) | 256 (Iters=2)")
        .string ("timer",             {"--timer"},             "cupti",  "cupti | event")
        .string ("synth",             {"--synth"},             "",       "synth spec: cauchy:peak=:gamma=:seed= | uniform:max=:seed=")
        .flag   ("emit_metrics",      {"--emit-metrics"},                "machine-readable metrics")
        .number ("assert_cr_ge",      {"--assert-cr-ge"},      -1.0,     "fail if cr <  X")
        .number ("assert_cr_le",      {"--assert-cr-le"},      -1.0,     "fail if cr >  X")
        .integer("assert_incomp_le",  {"--assert-incomp-le"},  -1,       "fail if incomp > N")
        ;
// clang-format on

namespace {
std::vector<string> split_csv(const string& raw)
{
  std::vector<string> out;
  size_t start = 0;
  while (start <= raw.size()) {
    auto comma = raw.find(',', start);
    auto end = (comma == string::npos) ? raw.size() : comma;
    auto b = start, e = end;
    while (b < e and std::isspace((unsigned char)raw[b])) ++b;
    while (e > b and std::isspace((unsigned char)raw[e - 1])) --e;
    if (e > b) out.emplace_back(raw.substr(b, e - b));
    if (comma == string::npos) break;
    start = comma + 1;
  }
  return out;
}

}  // namespace

struct Arguments {
  string fname;
  int x = 0, y = 0, z = 0;
  int bklen = 1024;
  string type = "u2";
  bool use_hf_rev2 = false;
  bool use_hfr = false;
  bool use_hfr_pbk_compat = false;
  bool use_hfr_pbk_go = false;
  bool use_hfr_pbkf = false;
  bool use_hfd26 = false;
  int repeat = 5;
  std::vector<int> reduce_values{DefaultReduceTimes};
  int magnitude = 10;
  int blockdim = 128;
  bool use_cupti = true;
  string synth_spec;
  std::vector<string> paths;  // populated by --path; empty -> single-flag mode
  bool emit_metrics = false;
  double assert_cr_ge = -1.0;
  double assert_cr_le = -1.0;
  int64_t assert_incomp_le = -1;

  bool parse(int argc, char** argv)
  {
    try {
      auto a = bin_phf_cli.parse(argc, argv);
      fname = a.get<string>("input");
      auto d = a.get<_ptb_len3>("dim3");
      x = (int)d.x;
      y = (int)d.y;
      z = (int)d.z;
      bklen = (int)a.get<i8>("bklen");
      type = a.get<string>("type");
      repeat = (int)a.get<i8>("repeat");
      // --reduce: CSV of {2|3|4}; each is dispatched as its own row when the path is HFR-family.
      {
        reduce_values.clear();
        auto raw = a.get<string>("reduce");
        for (auto const& s : split_csv(raw.empty() ? DefaultReduceStr : raw)) {
          if (s == "auto" or s == "0") {
            reduce_values.push_back(-1);  // predict RT from avg code length (hist x book)
            continue;
          }
          if (s == "best") {
            reduce_values.push_back(-2);  // measure candidate RTs, pick smallest encoded_len
            continue;
          }
          int v = std::stoi(s);
          if (v < 1 or v > 4)
            throw std::runtime_error(
                "--rmerge-count: each value must be auto|best|0(=auto)|1|2|3|4, got: " + s);
          reduce_values.push_back(v);
        }
        if (reduce_values.empty()) reduce_values.push_back(DefaultReduceTimes);
      }
      magnitude = (int)a.get<i8>("magnitude");
      blockdim = (int)a.get<i8>("blockdim");
      synth_spec = a.get<string>("synth");
      emit_metrics = a.get<bool>("emit_metrics");
      assert_cr_ge = a.get<f8>("assert_cr_ge");
      assert_cr_le = a.get<f8>("assert_cr_le");
      assert_incomp_le = a.get<i8>("assert_incomp_le");
      use_hf_rev2 = a.get<bool>("hf_rev2");
      use_hfr = a.get<bool>("hfr");
      use_hfr_pbk_compat = a.get<bool>("hfr_pbkc");
      use_hfr_pbk_go = a.get<bool>("hfr_pbkgo");
      use_hfr_pbkf = a.get<bool>("hfr_pbkf");
      use_hfd26 = a.get<bool>("hfd26");
      // --path hf,hf_rev2,...: comma-separated list. Trim whitespace, drop empties.
      paths = split_csv(a.get<string>("path"));

      auto timer = a.get<string>("timer");
      if (timer == "cupti")
        use_cupti = true;
      else if (timer == "event")
        use_cupti = false;
      else
        throw std::runtime_error("--timer must be cupti|event, got: " + timer);

      if (fname.empty() and synth_spec.empty())
        throw std::runtime_error("provide one of --input <file> or --synth <spec>");
    }
    catch (std::exception const& e) {
      fprintf(stderr, "bin_phf: %s\n", e.what());
      bin_phf_cli.print_help(stderr);
      return false;
    }
    return true;
  }

  size_t total_len() const { return (size_t)x * y * z; }
};

}  // namespace

template <typename E>
void hf_run(
    Arguments const& args, size_t len, HFVariant const& v, int reduce,
    E const* preloaded_h_data = nullptr)
{
  const int bklen = args.bklen;
  const int repeat = args.repeat;

  auto h_data = MAKE_UNIQUE_HOST(E, len);
  const size_t d_data_alloc_len = ALIGN_4Ki(len);
  auto d_data = MAKE_UNIQUE_DEVICE(E, d_data_alloc_len);
  auto d_decomp = MAKE_UNIQUE_DEVICE(E, len);

  auto stream_owner = _ptb::make_gpu_stream();
  auto stream = stream_owner.get();

  load_or_preload<E>(args.fname, len, args.synth_spec, preloaded_h_data, h_data.get());
  if (d_data_alloc_len > len)
    cudaMemsetAsync(
        d_data.get() + len, 0, (d_data_alloc_len - len) * sizeof(E), (cudaStream_t)stream);
  memcpy_allkinds_async<H2D>(d_data.get(), h_data.get(), len, stream);
  sync_by_stream(stream);

  auto buf = std::make_unique<phf::Buf<E>>(
      len, bklen, -1, v.use_HFR_buf, false, v.codec == psz_codec::HFr2);

  if (not v.skip_hist_and_book) {
    auto d_hist = MAKE_UNIQUE_DEVICE(F, bklen);
    auto h_hist = MAKE_UNIQUE_HOST(F, bklen);
    int grid_dim, block_dim, shmem_use, hist_repeat;
    psz::module::GPU_histogram_generic<E>::init(
        len, bklen, grid_dim, block_dim, shmem_use, hist_repeat);
    psz::module::GPU_histogram_generic<E>::kernel(
        d_data.get(), len, d_hist.get(), bklen, grid_dim, block_dim, shmem_use, hist_repeat,
        stream);
    memcpy_allkinds_async<D2H>(h_hist.get(), d_hist.get(), bklen, stream);
    sync_by_stream(stream);
    if (v.codec == psz_codec::HFR_V3)
      phf::high_level<E>::HFR_pick_pbk(buf.get(), d_hist.get(), bklen, len, stream);
    else {
      // force runtime Radius to the book's minimum depth
      if (v.is_hfr_family and bklen > psz::HFR_PBK_Constants::Radius) {
        F maxf = 0;
        for (int i = 0; i < bklen; i++) maxf = h_hist[i] > maxf ? h_hist[i] : maxf;
        auto& f_ref = h_hist[psz::HFR_PBK_Constants::Radius];
        if (f_ref < maxf) f_ref = maxf;
      }
      phf::high_level<E>::HF_build_book(buf.get(), h_hist.get(), bklen, stream);

      // reduce==-1 is the --rmerge-count auto sentinel: predict RT from the built book.
      int const mag_min = args.magnitude >= 12 ? 2 : args.magnitude >= 11 ? 1 : 0;
      if (reduce == -1 and v.is_hfr_family) {
        using PW = HuffmanWord<4>;
        auto bk = buf->book_h();
        double wsum = 0, n = 0;
        double mass_gt[33] = {0};  // occurrence mass with code length > t
        for (int i = 0; i < bklen; i++) {
          if (h_hist[i] == 0 or bk[i] == ~(u4)0x0) continue;
          int l = (int)reinterpret_cast<PW*>(&bk[i])->bitcount;
          wsum += (double)h_hist[i] * l;
          n += (double)h_hist[i];
          for (int t = 0; t < l and t <= 32; t++) mass_gt[t] += (double)h_hist[i];
        }
        double const avg = n > 0 ? wsum / n : 32.0;
        reduce = 1;
        for (int rt = 3; rt >= 1; rt--) {
          int const thr = 32 >> rt;
          if ((double)(1 << rt) * avg <= 32.0 and mass_gt[thr] / n <= 0.01) {
            reduce = rt;
            break;
          }
        }
        if (reduce < mag_min) reduce = mag_min;
        fprintf(
            stderr, "[auto-rmerge] avg_bits=%.3f mass>%db=%.4f%% mass>%db=%.4f%% -> r%d\n", avg,
            32 >> 3, 100.0 * mass_gt[32 >> 3] / n, 32 >> 2, 100.0 * mass_gt[32 >> 2] / n, reduce);
      }

      // reduce==-2 is --rmerge-count best: trial-encode each candidate RT, keep the smallest encoded_len.
      if (reduce == -2 and v.is_hfr_family) {
        int const best_rt_floor = mag_min > 1 ? mag_min : 1;
        size_t best_len = (size_t)-1;
        int best_rt = best_rt_floor;
        for (int rt = best_rt_floor; rt <= 3; rt++) {
          u1* t_encoded = nullptr;
          size_t t_len = 0;
          phf_header t_header{};
          float t_enc = 0.0f, t_lago = 0.0f;
          phf::high_level<E>::HFR_encode(
              buf.get(), d_data.get(), len, &t_encoded, &t_len, t_header, stream, v.codec, &t_enc,
              &t_lago, HFR_Opts{rt, args.magnitude, args.blockdim});
          sync_by_stream(stream);
          fprintf(stderr, "[best-rmerge] r%d encoded_len=%zu\n", rt, t_len);
          if (t_len < best_len) {
            best_len = t_len;
            best_rt = rt;
          }
        }
        reduce = best_rt;
        fprintf(stderr, "[best-rmerge] -> r%d\n", reduce);
      }
    }
  }
  if (reduce < 0) reduce = DefaultReduceTimes;  // auto/best without a runtime book

  u1* d_encoded = nullptr;
  size_t encoded_len = 0;
  phf_header header{};

  // repeat on short kernels
  constexpr int TimesWarmup = 10;
  double ms_enc = 1e9;
  float ms_encoder_phase = 0.0f, ms_lago_phase = 0.0f;
  for (int iter = 0; iter < TimesWarmup + repeat; ++iter) {
    sync_by_stream(stream);
    gpu_timer t;
    t.start(stream);
    float ms_enc_p = 0.0f, ms_lago_p = 0.0f;
    if (v.is_hfr_family)
      phf::high_level<E>::HFR_encode(
          buf.get(), d_data.get(), len, &d_encoded, &encoded_len, header, stream, v.codec,
          &ms_enc_p, &ms_lago_p, HFR_Opts{reduce, args.magnitude, args.blockdim});
    else
      phf::high_level<E>::HF_encode(
          buf.get(), d_data.get(), len, &d_encoded, &encoded_len, header, stream, v.codec,
          &ms_enc_p, &ms_lago_p);
    double this_enc = t.stop_ms(stream);
    if (iter >= TimesWarmup and this_enc < ms_enc) {
      ms_enc = this_enc;
      ms_encoder_phase = ms_enc_p;
      ms_lago_phase = ms_lago_p;
    }
    if (v.codec != psz_codec::HF) buf->reset(stream);
  }

  // HFR-v3: patch sole PBK ID into the device archive header
  if (v.codec == psz_codec::HFR_V3) {
    memcpy_allkinds_async<D2H>(
        &header.g_encid,
        (u1*)((u1*)d_encoded + header.entry[PHFHEADER_HEADER] + offsetof(phf_header, g_encid)), 1,
        stream);
    sync_by_stream(stream);
  }

  double ms_dec = 1e9;
  for (int iter = 0; iter < TimesWarmup + repeat; ++iter) {
    sync_by_stream(stream);
    gpu_timer t;
    t.start(stream);
    if (v.use_hfd26)
      phf::high_level<E>::HFD26_decode(
          buf.get(), header, d_encoded, d_decomp.get(), stream, v.codec, args.magnitude);
    else if (v.is_hfr_family)
      phf::high_level<E>::HFR_decode(
          buf.get(), header, d_encoded, d_decomp.get(), stream, v.codec, args.magnitude);
    else
      phf::high_level<E>::HF_decode(buf.get(), header, d_encoded, d_decomp.get(), stream, v.codec);
    double this_dec = t.stop_ms(stream);
    if (iter >= TimesWarmup and this_dec < ms_dec) ms_dec = this_dec;
  }

  size_t incomp_blocks = 0;
  if (v.use_HFR_buf and header.pardeg > 0) {
    auto h_incomp_flag = MAKE_UNIQUE_HOST(u1, header.pardeg);
    memcpy_allkinds<D2H>(h_incomp_flag.get(), buf->incomp_flag_d(), header.pardeg);
    for (size_t i = 0; i < header.pardeg; i++)
      if (h_incomp_flag[i]) ++incomp_blocks;
  }

  // diagnostic split: CodeIncompBreaks vs CodeIncompUnpred
  size_t breaks_blocks = 0, unpred_blocks_direct = 0;
  if (v.is_hfr_family and header.pardeg > 0) {
    auto h_packed = MAKE_UNIQUE_HOST(u4, 2 * (size_t)header.pardeg);
    memcpy_allkinds<D2H>(
        h_packed.get(), (u4*)(d_encoded + header.entry[PHFHEADER_PBK_HEADERS]),
        2 * (size_t)header.pardeg);
    const u4 bits_unpred = (u4)args.magnitude - 7u;
    const u4 bits_breaks = (u4)args.magnitude - 4u;
    const u4 bits_encid = 5u;
    for (int i = 0; i < header.pardeg; i++) {
      u4 w0 = h_packed[2 * i];
      u4 enc_id = (w0 >> (bits_unpred + bits_breaks)) & ((1u << bits_encid) - 1u);
      if (enc_id == 30) ++breaks_blocks;
      if (enc_id == 31) ++unpred_blocks_direct;
    }
  }
  if (args.emit_metrics)
    fprintf(
        stderr, "[diag] pardeg=%d breaks30=%zu unpred31=%zu (decode-flag unpred=%zu)\n",
        header.pardeg, breaks_blocks, unpred_blocks_direct, incomp_blocks);

  if (getenv("PHF_SELFSYNC_PROBE") and
      (v.codec == psz_codec::HFR_PBKC or v.codec == psz_codec::HFR_PBKGO) and header.pardeg > 0) {
    auto h_packed = MAKE_UNIQUE_HOST(u4, 2 * (size_t)header.pardeg);
    memcpy_allkinds<D2H>(
        h_packed.get(), (u4*)(d_encoded + header.entry[PHFHEADER_PBK_HEADERS]),
        2 * (size_t)header.pardeg);
    size_t const bs_words = header.total_ncell;
    auto h_bitstream = MAKE_UNIQUE_HOST(u4, bs_words);
    memcpy_allkinds<D2H>(
        h_bitstream.get(), (u4*)(d_encoded + header.entry[PHFHEADER_BITSTREAM]), bs_words);
    constexpr int RvbkBytesPerBook = (int)psz::HFR_PBK_Constants::RvbkBytesPerBook;
    constexpr int NumBooks = (int)psz::HFR_PBK_Constants::NumBooks;
    auto h_rvbk = MAKE_UNIQUE_HOST(u1, (size_t)RvbkBytesPerBook * NumBooks);
    memcpy_allkinds<D2H>(
        h_rvbk.get(), (u1*)pbk25_r128_rvbk_d_ptr(), (size_t)RvbkBytesPerBook * NumBooks);

    selfsync_probe::BookStats stats[NumBooks];
    if (args.magnitude >= 12)
      selfsync_probe::run<12>(
          h_bitstream.get(), bs_words, h_packed.get(), header.pardeg, h_rvbk.get(), stats);
    else if (args.magnitude >= 11)
      selfsync_probe::run<11>(
          h_bitstream.get(), bs_words, h_packed.get(), header.pardeg, h_rvbk.get(), stats);
    else
      selfsync_probe::run<10>(
          h_bitstream.get(), bs_words, h_packed.get(), header.pardeg, h_rvbk.get(), stats);
    selfsync_probe::report(stats);
  }

  auto identical =
      psz::cuda::GPU_identical((void*)d_decomp.get(), (void*)d_data.get(), sizeof(E), len, stream);
  {
    auto h_decomp = MAKE_UNIQUE_HOST(E, len);
    memcpy_allkinds<D2H>(h_decomp.get(), d_decomp.get(), len);
    size_t mismatches = 0;
    size_t first_bad = (size_t)-1;
    for (size_t i = 0; i < len; i++) {
      if (h_decomp[i] != h_data[i]) {
        if (first_bad == (size_t)-1) first_bad = i;
        ++mismatches;
      }
    }
    if (mismatches > 0) {
      fprintf(
          stderr,
          "[lossless-host] FAIL: %zu/%zu mismatches; first @ idx=%zu (block=%zu, off=%zu) "
          "input=%u decoded=%u  [GPU_identical=%d — disagrees]\n",
          mismatches, len, first_bad, first_bad / 1024, first_bad % 1024,
          (unsigned)h_data[first_bad], (unsigned)h_decomp[first_bad], identical ? 1 : 0);
      identical = false;
    }
  }
  if (not identical) throw std::runtime_error(string(v.label) + ": coding-decoding FAILED.");

  const double cr = (double)(len * sizeof(E)) / encoded_len;
  char row_label[32];
  if (v.is_hfr_family)
    snprintf(row_label, sizeof(row_label), "%s/r%d", v.label, reduce);
  else
    snprintf(row_label, sizeof(row_label), "%s", v.label);
  printf("%-20s %-9.2f ", row_label, cr);
  if (ms_encoder_phase > 0)
    printf("%6.1f GiB/s (%5.3f)  ", GiBps<E>(len, ms_encoder_phase), ms_encoder_phase);
  else
    printf("%6.1f GiB/s (%5.3f)  ", GiBps<E>(len, ms_enc), ms_enc);
  if (ms_lago_phase > 0 and not v.suppress_lago_col)
    printf("%6.1f GiB/s (%5.3f)  ", GiBps<E>(len, ms_lago_phase), ms_lago_phase);
  else
    printf("           —          ");

  printf("%6.1f GiB/s (%5.3f)  ", GiBps<E>(len, ms_dec), ms_dec);
  printf("—\n");

  g_metrics.cr = cr;
  g_metrics.arch_bytes = encoded_len;
  g_metrics.bs_bytes = (size_t)header.total_ncell * 4;
  g_metrics.incomp_blocks = incomp_blocks;
  g_metrics.lossless = true;
  g_metrics.codec = v.metric_name;
  g_metrics.dtype = (sizeof(E) == 1 ? "u1" : sizeof(E) == 2 ? "u2" : "u4");
  g_metrics.len = len;
}

inline HFVariant const* lookup_variant(const string& path)
{
  if (path == "hf") return &hfv::HF;
  if (path == "hf_rev2" or path == "hf-rev2") return &hfv::HF_REV2;
  if (path == "hfr") return &hfv::HFR;
  if (path == "hfr_pbkc" or path == "hfr-pbkc") return &hfv::PBKC;
  if (path == "hfr_pbkgo" or path == "hfr-pbkgo") return &hfv::PBKGO;
  if (path == "hfr_v3" or path == "hfr-v3") return &hfv::HF_V3;
  if (path == "hfd26") return &hfv::HFD26;
  if (path == "hfd26-pbkc") return &hfv::HFD26_PBKC;
  if (path == "hfd26-pbkgo") return &hfv::HFD26_PBKGO;
  return nullptr;
}

#if PHF_ENABLE_HFR_PBKF
#include "bin_phf_pbkf.inl"
#endif

template <typename E>
void run_one_path(
    Arguments const& args, size_t len, const string& path, E const* h_raw, int reduce)
{
  if (auto const* v = lookup_variant(path)) {
    hf_run<E>(args, len, *v, reduce, h_raw);
    return;
  }
  if (path == "hfr_pbkf" or path == "hfr-pbkf") {
#if PHF_ENABLE_HFR_PBKF
    (void)reduce;  // PBKF uses its own RT internally; --reduce ignored.
    hfr_pbkf_path_row<E>(args, len, h_raw, rm, sm);
#else
    (void)reduce;
    printf(
        "%-10s %-9s %-21s %-21s %-21s %s\n", "HFR-PBKF", "—", "  (SECRET gated)    ",
        "         —          ", "         —          ", "—");
#endif
    return;
  }
  throw std::runtime_error("unknown --path entry: '" + path + "'");
}

template <typename E = u2>
int choose_pipeline(Arguments const& args, size_t len)
{
  try {
    if (not args.paths.empty()) {
      // Multi-path mode: load h_raw once, dispatch each pipeline against it.
      auto h_raw = MAKE_UNIQUE_HOST(E, len);
      load_input<E>(args.fname, len, args.synth_spec, h_raw.get());
      printf(
          "%-20s %-9s %-21s %-21s %-21s %s\n", "pipeline", "CR", "encoder", "LAGO", "decode",
          "n/incomp");
      printf(
          "%-20s %-9s %-21s %-21s %-21s %s\n", "---------", "---------", "---------------------",
          "---------------------", "---------------------", "--------");
      // per path: one row per --rmerge-count value
      for (auto const& p : args.paths) {
        auto const* v = lookup_variant(p);
        if (v and v->is_hfr_family) {
          for (auto r : args.reduce_values) run_one_path<E>(args, len, p, h_raw.get(), r);
        }
        else
          run_one_path<E>(args, len, p, h_raw.get(), args.reduce_values[0]);
      }
    }
    else if (args.use_hfr_pbkf) {
#if PHF_ENABLE_HFR_PBKF
      hfr_pbkf_verify_run<E>(args.fname, len, args.synth_spec);
#else
      fprintf(stderr, "--hfr-pbkf: build not configured with -DPHF_ENABLE_HFR_PBKF=ON\n");
      return 2;
#endif
    }
    else {
      HFVariant const* v = args.use_hfd26            ? (args.use_hfr_pbk_compat ? &hfv::HFD26_PBKC
                                                        : args.use_hfr_pbk_go   ? &hfv::HFD26_PBKGO
                                                                                : &hfv::HFD26)
                           : args.use_hfr_pbk_compat ? &hfv::PBKC
                           : args.use_hfr_pbk_go     ? &hfv::PBKGO
                           : args.use_hfr            ? &hfv::HFR
                           : args.use_hf_rev2        ? &hfv::HF_REV2
                                                     : &hfv::HF;
      hf_run<E>(args, len, *v, args.reduce_values[0]);
    }
  }
  catch (std::runtime_error const& e) {
    fprintf(stderr, "lossless verification FAILED: %s\n", e.what());
    g_metrics.lossless = false;
    if (args.emit_metrics) {
      printf("[lossless] fail\n");
      printf(
          "[encoder]  %s\n", args.use_hfr_pbk_compat ? "hfr_pbkc"
                             : args.use_hfr_pbk_go   ? "hfr_pbkgo"
                             : args.use_hfr_pbkf     ? "hfr_pbkf"
                             : args.use_hfr          ? "hfr"
                             : args.use_hf_rev2      ? "hf-rev2"
                                                     : "hf");
      printf("[dtype]    %s\n", args.type.c_str());
      printf("[len]      %zu\n", len);
    }
    return 1;
  }
  return 0;
}

int main(int argc, char** argv)
{
  Arguments args;
  if (not args.parse(argc, argv)) return 1;

  if (args.use_cupti) _ptb::timer_cupti::enable();

  size_t len = args.total_len();

  // HFR* enforces 2*Radius=256 regardless of --bklen
  if (args.type == "u1" and args.bklen > 256) {
    fprintf(stderr, "--type u1: clamping --bklen %d to 256 (u1's full range)\n", args.bklen);
    args.bklen = 256;
  }

  int rc;
  if (args.type == "u1")
    rc = choose_pipeline<u1>(args, len);
  else if (args.type == "u2")
    rc = choose_pipeline<u2>(args, len);
  else {
    fprintf(stderr, "bin_phf currently tests --type u1|u2 (got %s)\n", args.type.c_str());
    return 2;
  }
  if (rc != 0) return rc;

  // Emit machine-readable metrics block (after the human-friendly output).
  if (args.emit_metrics) {
    printf("\n");  // visual separator
    printf("[lossless] %s\n", g_metrics.lossless ? "pass" : "fail");
    printf("[cr]       %.4f\n", g_metrics.cr);
    printf("[arch]     %zu\n", g_metrics.arch_bytes);
    printf("[bs_bytes] %zu\n", g_metrics.bs_bytes);
    printf("[incomp]   %zu\n", g_metrics.incomp_blocks);
    printf("[encoder]  %s\n", g_metrics.codec.c_str());
    printf("[dtype]    %s\n", g_metrics.dtype.c_str());
    printf("[len]      %zu\n", g_metrics.len);
  }

  if (not g_metrics.lossless) return 1;
  if (args.assert_cr_ge >= 0 && g_metrics.cr < args.assert_cr_ge) {
    fprintf(stderr, "assertion failed: cr=%.4f < cr_ge=%.4f\n", g_metrics.cr, args.assert_cr_ge);
    return 3;
  }
  if (args.assert_cr_le >= 0 && g_metrics.cr > args.assert_cr_le) {
    fprintf(stderr, "assertion failed: cr=%.4f > cr_le=%.4f\n", g_metrics.cr, args.assert_cr_le);
    return 3;
  }
  if (args.assert_incomp_le >= 0 && (int64_t)g_metrics.incomp_blocks > args.assert_incomp_le) {
    fprintf(
        stderr, "assertion failed: incomp=%zu > incomp_le=%lld\n", g_metrics.incomp_blocks,
        (long long)args.assert_incomp_le);
    return 3;
  }

  return 0;
}
