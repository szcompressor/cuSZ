// Authors: Jinyang Liu, Shixun Wu, Jiannan Tian

#include <cuda_runtime.h>

#include "kernel.hh"
#include "mem/buf_comp.hh"
#include "mem/cxx_backends.h"
#include "mem/cxx_sp_gpu.h"
#include "spl_y25.cuh"

constexpr int LEVEL = 6;
constexpr int SplDim2 = 2;
constexpr int SplDim3 = 3;
constexpr int AncBlkSzX = 64;
constexpr int AncBlkSzY = 64;
constexpr int AncBlkSzZ = 1;
constexpr int NAncBlkX = 1;
constexpr int NAncBlkY = 1;
constexpr int NAncBlkZ = 1;
constexpr int ProfBlkSzX = 4;
constexpr int ProfBlkSzY = 4;
constexpr int ProfBlkSzZ = 4;
constexpr int ProfNBlkX = 4;
constexpr int ProfNBlkY = 4;
constexpr int ProfNBlkZ = 4;

#define SETUP                                                                                \
  auto div3 = [](dim3 len, dim3 sublen) {                                                    \
    return dim3(                                                                             \
        (len.x - 1) / sublen.x + 1, (len.y - 1) / sublen.y + 1, (len.z - 1) / sublen.z + 1); \
  };                                                                                         \
  auto ndim = [&]() {                                                                        \
    if (len3.z == 1 and len3.y == 1)                                                         \
      return 1;                                                                              \
    else if (len3.z == 1 and len3.y != 1)                                                    \
      return 2;                                                                              \
    else                                                                                     \
      return 3;                                                                              \
  };

template <class Types, class Features>
int psz::module::GPU_c_spline_y25<Types, Features>::kernel(
    Buf* buf, host::view<T> in, double eb, double rel_eb, uint32_t radius,
    INTERP_PARAMS& intp_param, bool enable_global, void* stream)
{
  auto data_p = in.ptr;
  auto eq_p = buf->eq_d();
  auto anchor_p = buf->anchor_d();
  auto d_ext = in.extent;
  auto a_ext = buf->anchor_len3();
  auto _outlier = (void*)buf->buf_outlier2();
  auto out_bheader = buf->buf_hf() ? (uint32_t*)buf->buf_hf()->pbk_headers_d() : nullptr;
  auto out_block_outliers = buf->block_outliers_d();
  auto d_profiled_errors = buf->profiled_errors_d();
  auto h_profiled_errors = buf->profiled_errors_h();
  auto pe_len = buf->profiled_errors_len();

  auto data = _ptb::make_view(data_p, d_ext);
  auto eq = _ptb::make_view(eq_p, d_ext);
  auto anchor = _ptb::make_view(anchor_p, a_ext);
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  using Compact = _ptb::compact_GPU_DRAM2<T, u4>;
  auto ot = (Compact*)_outlier;

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = LEN_TO_DIM3(data.extent);
  auto data_leap = LEN_TO_DIM3(data.leap);
  auto anchor_l3 = LEN_TO_DIM3(anchor.extent);
  auto anchor_leap = LEN_TO_DIM3(anchor.leap);
  auto extent = l3;

  auto auto_tuning_grid_dim = dim3(1, 1, 1);

  if (intp_param.auto_tuning > 0) {
    double a1 = 2.0, a2 = 1.75, a3 = 1.5, a4 = 1.25, a5 = 1;
    double e1 = 1e-1, e2 = 1e-2, e3 = 1e-3, e4 = 1e-4, e5 = 1e-5;

    intp_param.beta = 4.0;
    if (rel_eb >= e1)
      intp_param.alpha = a1;
    else if (rel_eb >= e2)
      intp_param.alpha = a2 + (a1 - a2) * (rel_eb - e2) / (e1 - e2);
    else if (rel_eb >= e3)
      intp_param.alpha = a3 + (a2 - a3) * (rel_eb - e3) / (e2 - e3);
    else if (rel_eb >= e4)
      intp_param.alpha = a4 + (a3 - a4) * (rel_eb - e4) / (e3 - e4);
    else if (rel_eb >= e5)
      intp_param.alpha = a5 + (a4 - a5) * (rel_eb - e5) / (e4 - e5);
    else
      intp_param.alpha = a5;
    if (intp_param.auto_tuning == 1) {
      psz::KCU_c_spl_prof_data<
          T, SplDim3, ProfBlkSzX, ProfBlkSzY, ProfBlkSzZ, ProfNBlkX, ProfNBlkY, ProfNBlkZ,
          DefaultLinBlkSz>  //
          <<<auto_tuning_grid_dim, dim3(DefaultLinBlkSz, 1, 1), 0, (cudaStream_t)stream>>>(
              data.ptr, l3, data_leap, d_profiled_errors);

      cudaStreamSynchronize((cudaStream_t)stream);

      CHECK_GPU(cudaMemcpy(
          h_profiled_errors, d_profiled_errors, pe_len * sizeof(u4), cudaMemcpyDeviceToHost));
      auto errors = h_profiled_errors;

      bool do_reverse = (errors[1] > 3 * errors[0]);
      intp_param.reverse[0] = intp_param.reverse[1] = intp_param.reverse[2] =
          intp_param.reverse[3] = do_reverse;
    }
    else if (intp_param.auto_tuning == 2) {
      if (l3.z != 1) {
        psz::KCU_c_spl_prof_data_2<
            T, SplDim3, ProfNBlkX, ProfNBlkY, ProfNBlkZ,
            DefaultLinBlkSz>  //
            <<<auto_tuning_grid_dim, dim3(DefaultLinBlkSz, 1, 1), 0, (cudaStream_t)stream>>>(
                data.ptr, l3, data_leap, d_profiled_errors);

        cudaStreamSynchronize((cudaStream_t)stream);

        CHECK_GPU(cudaMemcpy(
            h_profiled_errors, d_profiled_errors, pe_len * sizeof(u4), cudaMemcpyDeviceToHost));
        auto errors = h_profiled_errors;

        bool do_nat = errors[0] + errors[2] + errors[4] > errors[1] + errors[3] + errors[5];
        intp_param.use_natural[0] = intp_param.use_natural[1] = intp_param.use_natural[2] =
            intp_param.use_natural[3] = do_nat;
        bool do_reverse = (errors[4 + do_nat] > 3 * errors[do_nat]);
        intp_param.reverse[0] = intp_param.reverse[1] = intp_param.reverse[2] =
            intp_param.reverse[3] = do_reverse;
        intp_param.use_md[0] = intp_param.use_md[1] = intp_param.use_md[2] = intp_param.use_md[3] =
            intp_param.use_md[4] = intp_param.use_md[5] = false;
      }
      else {
        psz::KCU_c_spl_prof_data_2<T, SplDim2, ProfNBlkX, ProfNBlkY, 1, DefaultLinBlkSz>  //
            <<<auto_tuning_grid_dim, dim3(DefaultLinBlkSz, 1, 1), 0, (cudaStream_t)stream>>>(
                data.ptr, l3, data_leap, d_profiled_errors);

        cudaStreamSynchronize((cudaStream_t)stream);

        CHECK_GPU(cudaMemcpy(
            h_profiled_errors, d_profiled_errors, pe_len * sizeof(u4), cudaMemcpyDeviceToHost));
        auto errors = h_profiled_errors;
        bool do_nat = errors[0] + errors[2] > errors[1] + errors[3];
        intp_param.use_natural[0] = intp_param.use_natural[1] = intp_param.use_natural[2] =
            intp_param.use_natural[3] = do_nat;
        intp_param.use_natural[4] = intp_param.use_natural[5] = do_nat;
        bool do_reverse = (errors[2 + do_nat] > 2 * errors[do_nat]);
        intp_param.reverse[0] = intp_param.reverse[1] = intp_param.reverse[2] =
            intp_param.reverse[3] = do_reverse;
        intp_param.reverse[4] = intp_param.reverse[5] = do_reverse;
        intp_param.use_md[0] = intp_param.use_md[1] = intp_param.use_md[2] = intp_param.use_md[3] =
            intp_param.use_md[4] = intp_param.use_md[5] = false;
      }
    }
    else {
      int S_STRIDE;
      if (l3.z == 1)
        S_STRIDE = 20 * AncBlkSzX;
      else
        S_STRIDE = 8 * Blk16;

      psz::reset_errors<<<dim3(1, 1, 1), dim3(DefaultLinBlkSz, 1, 1), 0, (cudaStream_t)stream>>>(
          d_profiled_errors);

      auto calc_start_size = [&](auto dim, auto& s_start, auto& s_size, auto BLOCKSIZE) {
        auto mid = dim / 2;
        auto k = (mid - BLOCKSIZE / 2) / S_STRIDE;
        auto t = (dim - BLOCKSIZE / 2 - 1 - mid) / S_STRIDE;
        s_start = mid - k * S_STRIDE;
        s_size = k + t + 1;
      };

      int s_start_x, s_start_y, s_start_z, s_size_x, s_size_y, s_size_z;

      if (l3.z == 1) {
        calc_start_size(l3.x, s_start_x, s_size_x, AncBlkSzX);
        calc_start_size(l3.y, s_start_y, s_size_y, AncBlkSzY);
        calc_start_size(l3.z, s_start_z, s_size_z, AncBlkSzZ);
      }
      else {
        calc_start_size(l3.x, s_start_x, s_size_x, Blk16);
        calc_start_size(l3.y, s_start_y, s_size_y, Blk16);
        calc_start_size(l3.z, s_start_z, s_size_z, Blk16);
      }

      auto block_num = s_size_x * s_size_y * s_size_z;

      auto errors = h_profiled_errors;

      double best_ave_pre_error[LEVEL];
      auto calcnum = [&](auto N) { return N * (7 * N * N + 9 * N + 3); };
      T best_error;
      int best_idx;

      if (l3.z > 1) {
        psz::KCU_pa_spl_infprecis_data<
            T, float, 4, SplDim3, Blk16, Blk16, Blk16, 1, 1, 1, DefaultLinBlkSz>
            <<<dim3(s_size_x * s_size_y * s_size_z, 9, 1), dim3(DefaultLinBlkSz, 1, 1), 0,
               (cudaStream_t)stream>>>(
                data.ptr, l3, data_leap, dim3(s_start_x, s_start_y, s_start_z),
                dim3(s_size_x, s_size_y, s_size_z), dim3(S_STRIDE, S_STRIDE, S_STRIDE), eb_r, ebx2,
                intp_param, d_profiled_errors, true);

        cudaStreamSynchronize((cudaStream_t)stream);

        CHECK_GPU(cudaMemcpy(
            h_profiled_errors, d_profiled_errors, pe_len * sizeof(u4), cudaMemcpyDeviceToHost));

        if (errors[0] > errors[1]) {
          best_error = errors[1];
          intp_param.reverse[3] = true;
        }
        else {
          best_error = errors[0];
          intp_param.reverse[3] = false;
        }

        intp_param.use_md[3] = errors[2] < best_error;
        best_error = fmin(errors[2], best_error);
        best_ave_pre_error[3] = best_error / (calcnum(1) * block_num);

        if (errors[3] > errors[4]) {
          best_error = errors[4];
          intp_param.reverse[2] = true;
        }
        else {
          best_error = errors[3];
          intp_param.reverse[2] = false;
        }

        intp_param.use_md[2] = errors[5] < best_error;
        best_error = fmin(errors[5], best_error);
        best_ave_pre_error[2] = best_error / (calcnum(2) * block_num);

        best_error = errors[6];
        best_idx = 6;
        for (auto i = 6; i < 12; i++) {
          if (errors[i] < best_error) {
            best_error = errors[i];
            best_idx = i;
          }
        }
        intp_param.use_natural[1] = best_idx > 8;

        intp_param.use_md[1] = (best_idx == 8 or best_idx == 11);
        intp_param.reverse[1] = best_idx % 3;

        best_ave_pre_error[1] = best_error / (calcnum(4) * block_num);

        best_error = errors[12];
        best_idx = 12;

        for (auto i = 12; i < 18; i++) {
          if (errors[i] < best_error) {
            best_error = errors[i];
            best_idx = i;
          }
        }
        intp_param.use_natural[0] = best_idx > 14;

        intp_param.use_md[0] = (best_idx == 14 or best_idx == 17);
        intp_param.reverse[0] = best_idx % 3;

        best_ave_pre_error[0] = best_error / (calcnum(8) * block_num);
      }

      if (l3.z == 1) {  // The 2D branch

        psz::KCU_pa_spl_infprecis_data<
            T, float, LEVEL, SplDim2, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY,
            NAncBlkZ, DefaultLinBlkSz>
            <<<dim3(s_size_x * s_size_y * s_size_z, 11, 1), dim3(DefaultLinBlkSz, 1, 1), 0,
               (cudaStream_t)stream>>>(
                data.ptr, l3, data_leap, dim3(s_start_x, s_start_y, s_start_z),
                dim3(s_size_x, s_size_y, s_size_z), dim3(S_STRIDE, S_STRIDE, S_STRIDE), eb_r, ebx2,
                intp_param, d_profiled_errors, true);

        cudaStreamSynchronize((cudaStream_t)stream);

        CHECK_GPU(cudaMemcpy(
            h_profiled_errors, d_profiled_errors, pe_len * sizeof(u4), cudaMemcpyDeviceToHost));

        if (errors[0] > errors[1]) {
          best_error = errors[1];
          intp_param.reverse[5] = true;
        }
        else {
          best_error = errors[0];
          intp_param.reverse[5] = false;
        }

        intp_param.use_md[5] = errors[2] < best_error;
        best_error = fmin(errors[2], best_error);
        best_ave_pre_error[5] = best_error / (calcnum(1) * block_num);

        if (errors[3] > errors[4]) {
          best_error = errors[4];
          intp_param.reverse[4] = true;
        }
        else {
          best_error = errors[3];
          intp_param.reverse[4] = false;
        }

        intp_param.use_md[4] = errors[5] < best_error;
        best_error = fmin(errors[5], best_error);
        best_ave_pre_error[4] = best_error / (calcnum(2) * block_num);

        if (errors[6] > errors[7]) {
          best_error = errors[7];
          intp_param.reverse[3] = true;
        }
        else {
          best_error = errors[6];
          intp_param.reverse[3] = false;
        }

        intp_param.use_md[3] = errors[8] < best_error;
        best_error = fmin(errors[8], best_error);
        best_ave_pre_error[3] = best_error / (calcnum(4) * block_num);

        for (int level = 3; level < LEVEL; ++level) {
          best_error = errors[level * 6 - 9];
          best_idx = level * 6 - 9;
          int level_id = LEVEL - 1 - level;

          for (auto i = level * 6 - 9; i < level * 6 + 6 - 9; i++) {
            if (errors[i] < best_error) {
              best_error = errors[i];
              best_idx = i;
            }
          }
          intp_param.use_natural[level_id] = ((best_idx + 3) % 6) > 2;
          intp_param.use_md[level_id] = (((best_idx + 3) % 6) == 2 or ((best_idx + 3) % 6) == 5);
          intp_param.reverse[level_id] = (best_idx + 3) % 3;
          best_ave_pre_error[level_id] = best_error / (calcnum(1 << level) * block_num);
        }
      }

      if (intp_param.auto_tuning == 4) {
        psz::reset_errors<<<dim3(1, 1, 1), dim3(DefaultLinBlkSz, 1, 1), 0, (cudaStream_t)stream>>>(
            d_profiled_errors);

        if (l3.z != 1)
          psz::KCU_pa_spl_infprecis_data<
              T, float, 4, SplDim3, Blk16, Blk16, Blk16, 1, 1, 1, DefaultLinBlkSz>
              <<<dim3(s_size_x * s_size_y * s_size_z, 11, 1), dim3(DefaultLinBlkSz, 1, 1), 0,
                 (cudaStream_t)stream>>>(
                  data.ptr, l3, data_leap, dim3(s_start_x, s_start_y, s_start_z),
                  dim3(s_size_x, s_size_y, s_size_z), dim3(S_STRIDE, S_STRIDE, S_STRIDE), eb_r,
                  ebx2, intp_param, d_profiled_errors, false);
        else
          psz::KCU_pa_spl_infprecis_data<
              T, float, LEVEL, SplDim2, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY,
              NAncBlkZ, DefaultLinBlkSz>
              <<<dim3(s_size_x * s_size_y * s_size_z, 11, 1), dim3(DefaultLinBlkSz, 1, 1), 0,
                 (cudaStream_t)stream>>>(
                  data.ptr, l3, data_leap, dim3(s_start_x, s_start_y, s_start_z),
                  dim3(s_size_x, s_size_y, s_size_z), dim3(S_STRIDE, S_STRIDE, S_STRIDE), eb_r,
                  ebx2, intp_param, d_profiled_errors, false);

        cudaStreamSynchronize((cudaStream_t)stream);

        auto errors = h_profiled_errors;

        best_error = errors[0];
        auto best_idx = 0;

        for (auto i = 1; i < 11; i++) {
          if (errors[i] < best_error) {
            best_error = errors[i];
            best_idx = i;
          }
        }

        if (best_idx == 0) {
          intp_param.alpha = 1.0;
          intp_param.beta = 2.0;
        }
        else if (best_idx == 1) {
          intp_param.alpha = 1.25;
          intp_param.beta = 2.0;
        }
        else {
          intp_param.alpha = 1.5 + 0.25 * ((best_idx - 2) / 3);
          intp_param.beta = 2.0 + ((best_idx - 2) % 3);
        }
      }
      else if (intp_param.auto_tuning >= 5) {
        best_idx = intp_param.auto_tuning - 5;
        if (best_idx == 0) {
          intp_param.alpha = 1.0;
          intp_param.beta = 2.0;
        }
        else if (best_idx == 1) {
          intp_param.alpha = 1.25;
          intp_param.beta = 2.0;
        }
        else {
          intp_param.alpha = 1.5 + 0.25 * ((best_idx - 2) / 3);
          intp_param.beta = 2.0 + ((best_idx - 2) % 3);
        }
      }
    }
  }

  auto go = [&](auto global_const) {
    constexpr bool Global = decltype(global_const)::value;
    using F = psz::PredictorFeature<
        Features::UseZigZag, Features::UseH1GL,
        (Global ? 0b10 : 0b00) | (Features::UnpredIncomp & 0b01)>;
    if (l3.z == 1) {
      auto grid_dim = dim3(
          div(extent.x, AncBlkSzX * NAncBlkX), div(extent.y, AncBlkSzY * NAncBlkY),
          div(extent.z, AncBlkSzZ * NAncBlkZ));
      psz::KCU_c_spl_infprecis_data<
          T, E, float, LEVEL, SplDim2, AncBlkSzX, AncBlkSzY, AncBlkSzZ, NAncBlkX, NAncBlkY,
          NAncBlkZ, DefaultLinBlkSz, decltype(ot->val_idx_d()), uint32_t*, F>
          <<<grid_dim, dim3(DefaultLinBlkSz, 1, 1), 0, (cudaStream_t)stream>>>(
              data.ptr, extent, data_leap, eq.ptr, extent, data_leap, anchor.ptr, anchor_leap,
              ot->val_idx_d(), ot->num_d(), eb_r, ebx2, radius, intp_param, out_bheader,
              out_block_outliers, ot->max_allowed_num());
    }
    else {
      auto grid_dim = dim3(div(extent.x, Blk16), div(extent.y, Blk16), div(extent.z, Blk16));
      psz::KCU_c_spl_infprecis_data<
          T, E, float, 4, SplDim3, Blk16, Blk16, Blk16, 1, 1, 1, DefaultLinBlkSz,
          decltype(ot->val_idx_d()), uint32_t*, F>
          <<<grid_dim, dim3(DefaultLinBlkSz, 1, 1), 0, (cudaStream_t)stream>>>(
              data.ptr, extent, data_leap, eq.ptr, extent, data_leap, anchor.ptr, anchor_leap,
              ot->val_idx_d(), ot->num_d(), eb_r, ebx2, radius, intp_param, out_bheader,
              out_block_outliers, ot->max_allowed_num());
    }
  };
  if (enable_global)
    go(std::integral_constant<bool, true>{});
  else
    go(std::integral_constant<bool, false>{});

  cudaStreamSynchronize((cudaStream_t)stream);

  return 0;
}

#undef SETUP
