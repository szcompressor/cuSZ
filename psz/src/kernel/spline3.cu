/**
 * @file spline3.cu
 * @author Jinyang Liu, Shixun Wu, Jiannan Tian
 * @brief A high-level Spline3D wrapper. Allocations are explicitly out of
 * called functions.
 * @version 0.3
 * @date 2021-06-15
 *
 * (copyright to be updated)
 * (C) 2021 by Washington State University, Argonne National Laboratory
 *
 */

#include <cuda_runtime.h>

#include "cusz/type.h"
#include "detail/busyheader.hh"
#include "kernel/predictor.hh"
#include "mem/cxx_memobj.h"
#include "mem/cxx_sp_gpu.h"
#include "utils/err.hh"
//
#include "detail/spline3_md.inl"

#define BLOCK_DIM_SIZE 384
#define LEVEL 6
#define SPLINE_DIM_2 2
#define SPLINE_DIM_3 3
#define AnchorBlockSizeX 64
#define AnchorBlockSizeY 64
#define AnchorBlockSizeZ 1
#define numAnchorBlockX 1
#define numAnchorBlockY 1
#define numAnchorBlockZ 1
#define BLOCK16 16
#define PROFILE_BLOCK_SIZE_X 4
#define PROFILE_BLOCK_SIZE_Y 4
#define PROFILE_BLOCK_SIZE_Z 4
#define PROFILE_NUM_BLOCK_X 4
#define PROFILE_NUM_BLOCK_Y 4
#define PROFILE_NUM_BLOCK_Z 4

#define Z(LEN3) LEN3[2]
#define Y(LEN3) LEN3[1]
#define X(LEN3) LEN3[0]
#define TO_DIM3(LEN3) dim3(X(LEN3), Y(LEN3), Z(LEN3))

constexpr int DEFAULT_BLOCK_SIZE = BLOCK_DIM_SIZE;

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

// clang-format off

template <typename T, typename E, typename Fp>
int psz::module::GPU_spline_construct<T, E, Fp>::kernel_v1(
    T* data, stdlen3 const data_len3, T* anchor, stdlen3 const anchor_len3, E* ectrl,
    void* _outlier, double eb, double rel_eb, uint32_t radius, INTERPOLATION_PARAMS& intp_param,
    float* time, T* d_profiling_errors, T* h_profiling_errors, u4 const pe_len, void* stream)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = TO_DIM3(data_len3);  // original: auto l3 = data->len3();
  auto stride3 = dim3(1, l3.x, l3.x * l3.y);
  auto l3_anchor = TO_DIM3(anchor_len3);
  auto stride3_anchor = dim3(1, l3_anchor.x, l3_anchor.x * l3_anchor.y);

  auto auto_tuning_grid_dim = dim3(1, 1, 1);

  using Compact = _portable::compact_gpu<T>;
  auto ot = (Compact*)_outlier;

  float att_time=0;
 if(intp_param.auto_tuning>0){
   //std::cout<<"att "<<(int)intp_param.auto_tuning<<std::endl;
   double a1=2.0;
   double a2=1.75;
   double a3=1.5;
   double a4=1.25;
   double a5=1;
   double e1=1e-1;
   double e2=1e-2;
   double e3=1e-3;
   double e4=1e-4;
   double e5=1e-5;

   intp_param.beta=4.0;
   if(rel_eb>=e1)
    intp_param.alpha=a1;
   else if(rel_eb>=e2)
    intp_param.alpha=a2+(a1-a2)*(rel_eb-e2)/(e1-e2);
   else if(rel_eb>=e3)
    intp_param.alpha=a3+(a2-a3)*(rel_eb-e3)/(e2-e3);
   else if(rel_eb>=e4)
    intp_param.alpha=a4+(a3-a4)*(rel_eb-e4)/(e3-e4);
   else if(rel_eb>=e5)
    intp_param.alpha=a5+(a4-a5)*(rel_eb-e5)/(e4-e5);
   else
    intp_param.alpha=a5;

    if(intp_param.auto_tuning==1){

      CREATE_GPUEVENT_PAIR;
       START_GPUEVENT_RECORDING(stream);

      cusz::c_spline_profiling_data<
          T*, SPLINE_DIM_3, PROFILE_BLOCK_SIZE_X, PROFILE_BLOCK_SIZE_Y, PROFILE_BLOCK_SIZE_Z,
          PROFILE_NUM_BLOCK_X, PROFILE_NUM_BLOCK_Y, PROFILE_NUM_BLOCK_Z, DEFAULT_BLOCK_SIZE>  //
          <<<auto_tuning_grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
              data, l3, stride3, d_profiling_errors);
        STOP_GPUEVENT_RECORDING(stream);
        CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
        TIME_ELAPSED_GPUEVENT(&att_time);
        DESTROY_GPUEVENT_PAIR;
      // profiling_errors->control({D2H});
      // CHECK_GPU(cudaMemcpy(profiling_errors->m->h, profiling_errors->m->d, profiling_errors->m->bytes, cudaMemcpyDeviceToHost));
      memcpy_allkinds<D2H>(h_profiling_errors, d_profiling_errors, pe_len);
      auto errors = h_profiling_errors;

      //printf("host %.4f %.4f\n",errors[0],errors[1]);
      bool do_reverse=(errors[1]>3*errors[0]);
      intp_param.reverse[0]=intp_param.reverse[1]=intp_param.reverse[2]=intp_param.reverse[3]=do_reverse;
    }
    else if (intp_param.auto_tuning==2){
      if(l3.z != 1){
        CREATE_GPUEVENT_PAIR;
        START_GPUEVENT_RECORDING(stream);
        cusz::c_spline_profiling_data_2<T*, SPLINE_DIM_3, PROFILE_NUM_BLOCK_X, PROFILE_NUM_BLOCK_Y, PROFILE_NUM_BLOCK_Z, DEFAULT_BLOCK_SIZE>  //
          <<<auto_tuning_grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(data, l3, stride3, d_profiling_errors);
        STOP_GPUEVENT_RECORDING(stream);
        CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
        TIME_ELAPSED_GPUEVENT(&att_time);
        DESTROY_GPUEVENT_PAIR;
        // CHECK_GPU(cudaMemcpy(profiling_errors->m->h, profiling_errors->m->d, profiling_errors->m->bytes, cudaMemcpyDeviceToHost));
        memcpy_allkinds<D2H>(h_profiling_errors, d_profiling_errors, pe_len);

        auto errors = h_profiling_errors;

        bool do_nat = errors[0] + errors[2] + errors[4] > errors[1] + errors[3] + errors[5];
        intp_param.use_natural[0] = intp_param.use_natural[1] = intp_param.use_natural[2] = intp_param.use_natural[3] = do_nat;
          bool do_reverse=(errors[4+do_nat]>3*errors[do_nat]);
        intp_param.reverse[0]=intp_param.reverse[1]=intp_param.reverse[2]=intp_param.reverse[3]=do_reverse;
        intp_param.use_md[0] = intp_param.use_md[1] = intp_param.use_md[2] = intp_param.use_md[3] = intp_param.use_md[4] = intp_param.use_md[5] = false;
        }
        else{
          CREATE_GPUEVENT_PAIR;
          START_GPUEVENT_RECORDING(stream);
          cusz::c_spline_profiling_data_2<T*, SPLINE_DIM_2, PROFILE_NUM_BLOCK_X, PROFILE_NUM_BLOCK_Y, 1, DEFAULT_BLOCK_SIZE>  //
            <<<auto_tuning_grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(data, l3, stride3, d_profiling_errors);
          STOP_GPUEVENT_RECORDING(stream);
          CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
          TIME_ELAPSED_GPUEVENT(&att_time);
          DESTROY_GPUEVENT_PAIR;
          // CHECK_GPU(cudaMemcpy(profiling_errors->m->h, profiling_errors->m->d, profiling_errors->m->bytes, cudaMemcpyDeviceToHost));
          memcpy_allkinds<D2H>(h_profiling_errors, d_profiling_errors, pe_len);

          auto errors = h_profiling_errors;
          bool do_nat = errors[0] + errors[2]> errors[1] + errors[3];
          intp_param.use_natural[0] = intp_param.use_natural[1] = intp_param.use_natural[2] = intp_param.use_natural[3] = do_nat;
          intp_param.use_natural[4] = intp_param.use_natural[5] = do_nat;
          bool do_reverse=(errors[2+do_nat]>2*errors[do_nat]);
          intp_param.reverse[0]=intp_param.reverse[1]=intp_param.reverse[2]=intp_param.reverse[3]=do_reverse;
          intp_param.reverse[4]=intp_param.reverse[5]=do_reverse;
          intp_param.use_md[0] = intp_param.use_md[1] = intp_param.use_md[2] = intp_param.use_md[3] = intp_param.use_md[4] = intp_param.use_md[5] = false;
        }
    }
    else{
      int S_STRIDE;
      if(l3.z == 1) S_STRIDE = 20 * AnchorBlockSizeX;
      else S_STRIDE = 8 * BLOCK16;

      cusz::reset_errors<<<dim3(1, 1, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(d_profiling_errors);

      auto calc_start_size = [&](auto dim, auto & s_start, auto &s_size, auto BLOCKSIZE) {
          auto mid = dim / 2;
          auto k = (mid - BLOCKSIZE / 2) / S_STRIDE;  
          auto t = (dim - BLOCKSIZE / 2 - 1 - mid) / S_STRIDE;
          s_start = mid - k * S_STRIDE;
          s_size = k + t + 1;
      };

      int s_start_x, s_start_y, s_start_z, s_size_x, s_size_y, s_size_z;

      if(l3.z == 1){
        calc_start_size(l3.x, s_start_x, s_size_x, AnchorBlockSizeX);
        calc_start_size(l3.y, s_start_y, s_size_y, AnchorBlockSizeY);
        calc_start_size(l3.z, s_start_z, s_size_z, AnchorBlockSizeZ);
      }
      else{
        calc_start_size(l3.x, s_start_x, s_size_x, BLOCK16);
        calc_start_size(l3.y, s_start_y, s_size_y, BLOCK16);
        calc_start_size(l3.z, s_start_z, s_size_z, BLOCK16);
      }
      float temp_time = 0;
      CREATE_GPUEVENT_PAIR;
      START_GPUEVENT_RECORDING(stream);

      auto block_num = s_size_x * s_size_y * s_size_z;
      
      auto errors = h_profiling_errors;

      double best_ave_pre_error[LEVEL];
      auto calcnum  = [&](auto N){
        return N * (7 * N * N + 9 * N + 3);
      };
      T best_error;
      int best_idx;

      // if CONSTEXPR (SPLINE_DIM == 3){
      if (l3.z > 1) {
        cusz::pa_spline_infprecis_data<T*, float, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1, DEFAULT_BLOCK_SIZE>
            <<<dim3(s_size_x * s_size_y * s_size_z, 9, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
                data, l3, stride3, dim3(s_start_x, s_start_y, s_start_z),
                dim3(s_size_x, s_size_y, s_size_z), dim3(S_STRIDE, S_STRIDE, S_STRIDE), eb_r, ebx2,
                intp_param, d_profiling_errors, true);

        STOP_GPUEVENT_RECORDING(stream);
        CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
        TIME_ELAPSED_GPUEVENT(&temp_time);
        DESTROY_GPUEVENT_PAIR;
        att_time += temp_time;
        // CHECK_GPU(cudaMemcpy(profiling_errors->m->h, profiling_errors->m->d, profiling_errors->m->bytes, cudaMemcpyDeviceToHost));
        memcpy_allkinds<D2H>(h_profiling_errors, d_profiling_errors, pe_len);

        if(errors[0] > errors[1]){
          best_error = errors[1];
          intp_param.reverse[3] = true;
        }
        else{
          best_error = errors[0];
          intp_param.reverse[3] = false;
        }

        //printf("use_md[3] errors[2]=%f, best_error=%f\n", errors[2], best_error);
        intp_param.use_md[3] = errors[2] < best_error; 
        best_error = fmin(errors[2],best_error);
        best_ave_pre_error[3] = best_error / (calcnum(1) * block_num);

        if(errors[3] > errors[4]){
          best_error = errors[4];
          intp_param.reverse[2] = true;
        }
        else{
          best_error = errors[3];
          intp_param.reverse[2] = false;
        }
        //printf("use_md[2] errors[5]=%f, best_error=%f\n", errors[5], best_error);
        intp_param.use_md[2] = errors[5] < best_error; 
        best_error = fmin(errors[5],best_error);
        best_ave_pre_error[2] = best_error / (calcnum(2) * block_num);

        best_error = errors[6];
        best_idx = 6; 
        for(auto i = 6; i < 12; i++){
          if(errors[i] < best_error){
            best_error = errors[i];
            best_idx = i;
          }
        }

        intp_param.use_natural[1] = best_idx >  8;
        //printf("use_md[1] errors[8]=%f, best_error=%f\n", errors[8], best_error);
        //printf("use_md[1] errors[11]=%f, best_error=%f\n", errors[11], best_error);
        intp_param.use_md[1] = (best_idx ==  8 or best_idx ==  11) ;
        intp_param.reverse[1] = best_idx%3;

        best_ave_pre_error[1]= best_error/(calcnum(4)*block_num);

        best_error = errors[12];
        best_idx = 12; 

        for(auto i = 12;i<18;i++){
          if(errors[i]<best_error){
            best_error=errors[i];
            best_idx = i;
          }
        }

        intp_param.use_natural[0] = best_idx >  14;
        //printf("use_md[1] errors[14]=%f, best_error=%f\n", errors[14], best_error);
        //printf("use_md[1] errors[17]=%f, best_error=%f\n", errors[17], best_error);
        intp_param.use_md[0] = (best_idx ==  14 or best_idx ==  17);
        intp_param.reverse[0] = best_idx%3;

        best_ave_pre_error[0]= best_error/(calcnum(8)*block_num);
        
        //printf("BESTERROR: %.4e %.4e %.4e %.4e\n",best_ave_pre_error[3],best_ave_pre_error[2],best_ave_pre_error[1],best_ave_pre_error[0]);
      }

    // if CONSTEXPR (SPLINE_DIM == 2){
    if (l3.z == 1){
      //printf("s_size.x = %d, .y = %d, .z = %d\n", s_size_x, s_size_y, s_size_z);
      //printf("l3.x = %d, .y = %d, .z = %d\n", l3.x, l3.y, l3.z);
        cusz::pa_spline_infprecis_data<T*, float, LEVEL, SPLINE_DIM_2, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, DEFAULT_BLOCK_SIZE>
            <<<dim3(s_size_x * s_size_y * s_size_z, 11, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
                data, l3, stride3, dim3(s_start_x, s_start_y, s_start_z),
                dim3(s_size_x, s_size_y, s_size_z), dim3(S_STRIDE, S_STRIDE, S_STRIDE), eb_r, ebx2,
                intp_param, d_profiling_errors, true);

      STOP_GPUEVENT_RECORDING(stream);
      CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
      TIME_ELAPSED_GPUEVENT(&temp_time);
      DESTROY_GPUEVENT_PAIR;
      att_time += temp_time;
      // CHECK_GPU(cudaMemcpy(profiling_errors->m->h, profiling_errors->m->d, profiling_errors->m->bytes, cudaMemcpyDeviceToHost));
      memcpy_allkinds<D2H>(h_profiling_errors, d_profiling_errors, pe_len);

      if(errors[0] > errors[1]){
        best_error = errors[1];
        intp_param.reverse[5] = true;
      }
      else{
        best_error = errors[0];
        intp_param.reverse[5] = false;
      }

      //printf("use_md[5] errors[2]=%f, best_error=%f\n", errors[2], best_error);
      intp_param.use_md[5] = errors[2] < best_error; 
      best_error = fmin(errors[2],best_error);
      best_ave_pre_error[5] = best_error / (calcnum(1) * block_num);

      if(errors[3] > errors[4]){
        best_error = errors[4];
        intp_param.reverse[4] = true;
      }
      else{
        best_error = errors[3];
        intp_param.reverse[4] = false;
      }
      //printf("use_md[4] errors[5]=%f, best_error=%f\n", errors[5], best_error);

      intp_param.use_md[4] = errors[5] < best_error; 
      best_error = fmin(errors[5],best_error);
      best_ave_pre_error[4] = best_error / (calcnum(2) * block_num);

      if(errors[6] > errors[7]){
        best_error = errors[7];
        intp_param.reverse[3] = true;
      }
      else{
        best_error = errors[6];
        intp_param.reverse[3] = false;
      }
      //printf("use_md[3] errors[8]=%f, best_error=%f\n", errors[8], best_error);
      intp_param.use_md[3] = errors[8] < best_error; 
      best_error = fmin(errors[8],best_error);
      best_ave_pre_error[3] = best_error / (calcnum(4) * block_num);

      for(int level = 3; level < LEVEL; ++level){
        //printf("level=%d: ", level);
        best_error = errors[level * 6 - 9];
        best_idx = level * 6 - 9; 
        int level_id = LEVEL - 1 - level;

        for(auto i = level * 6 - 9; i < level * 6 + 6 - 9; i++){
          //printf(" errors[%d]=%.10f", i, errors[i]);
          if(errors[i] < best_error){
            best_error = errors[i];
            best_idx = i;
          }
        }
        intp_param.use_natural[level_id] = ((best_idx + 3) % 6) > 2;
        intp_param.use_md[level_id] = (((best_idx + 3) % 6) == 2 or ((best_idx + 3) % 6) ==  5) ;
        intp_param.reverse[level_id] = (best_idx + 3) % 3;
        best_ave_pre_error[level_id]= best_error / (calcnum(1 << level) * block_num);
        //printf(" BESTERROR=%f\n", best_ave_pre_error[level_id]);
      }
      //printf("\n");
    }

      // intp_param.use_md[0] = 1;
      // intp_param.use_md[1] = 1;
      // intp_param.use_md[2] = 0;
      // intp_param.use_md[3] = 1;
      
      // intp_param.use_natural[0] = 0;
      // intp_param.use_natural[1] = 0;
      // intp_param.use_natural[2] = 1;
      // intp_param.use_natural[3] = 0;

      // intp_param.reverse[0] = 0;
      // intp_param.reverse[1] = 1;
      // intp_param.reverse[2] = 0;
      // intp_param.reverse[3] = 1;
      // intp_param.use_md[3] = 1;
      // intp_param.use_md[2] = 1;
      // intp_param.use_md[1] = 0;
      // intp_param.use_md[0] = 1;
      
      // intp_param.use_natural[3] = 0;
      // intp_param.use_natural[2] = 0;
      // intp_param.use_natural[1] = 1;
      // intp_param.use_natural[0] = 0;

      // intp_param.reverse[3] = 0;
      // intp_param.reverse[2] = 1;
      // intp_param.reverse[1] = 0;
      // intp_param.reverse[0] = 1;
      // intp_param.use_md[4] = 1;
      // intp_param.use_md[5] = 1;
      // intp_param.use_md[0] = 0;
      // intp_param.use_md[1] = 0;
      // intp_param.use_md[2] = 0;
      // intp_param.use_md[3] = 0;
      // intp_param.use_md[4] = 0;
      // intp_param.use_md[5] = 0;

      if(intp_param.auto_tuning==4){
        cusz::reset_errors<<<dim3(1, 1, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(d_profiling_errors);

        float temp_time = 0;
        CREATE_GPUEVENT_PAIR;
         START_GPUEVENT_RECORDING(stream);

        if (l3.z != 1)
          cusz::pa_spline_infprecis_data<T*, float, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1, DEFAULT_BLOCK_SIZE>
              <<<dim3(s_size_x * s_size_y * s_size_z, 11, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
                  data, l3, stride3, dim3(s_start_x, s_start_y, s_start_z),
                  dim3(s_size_x, s_size_y, s_size_z), dim3(S_STRIDE, S_STRIDE, S_STRIDE), eb_r,
                  ebx2, intp_param, d_profiling_errors, false);
        else
          cusz::pa_spline_infprecis_data<T*, float, LEVEL, SPLINE_DIM_2, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, DEFAULT_BLOCK_SIZE>
              <<<dim3(s_size_x * s_size_y * s_size_z, 11, 1), dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
                  data, l3, stride3, dim3(s_start_x, s_start_y, s_start_z),
                  dim3(s_size_x, s_size_y, s_size_z), dim3(S_STRIDE, S_STRIDE, S_STRIDE), eb_r,
                  ebx2, intp_param, d_profiling_errors, false);

        STOP_GPUEVENT_RECORDING(stream);
        CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
        TIME_ELAPSED_GPUEVENT(&temp_time);
        DESTROY_GPUEVENT_PAIR;
        att_time += temp_time;

        auto errors = h_profiling_errors;
        // for(int i = 0; i < 11; i++){
        //   printf("%d %.4e\n", i, errors[i]);
        // }

        best_error = errors[0];
        auto best_idx = 0; 

        for(auto i = 1;i < 11;i++){
          if(errors[i] < best_error){
            best_error = errors[i];
            best_idx = i;
          }
        }

        if(best_idx == 0){
            intp_param.alpha = 1.0;
            intp_param.beta = 2.0;
        }
        else if (best_idx == 1){
            intp_param.alpha = 1.25;
            intp_param.beta = 2.0;
        }
        else{
            intp_param.alpha = 1.5+0.25*((best_idx-2)/3);
            intp_param.beta = 2.0+((best_idx-2)%3);
        }

      }
      else if(intp_param.auto_tuning >= 5){
        best_idx = intp_param.auto_tuning - 5;
        if(best_idx == 0){
            intp_param.alpha = 1.0;
            intp_param.beta = 2.0;
        }
        else if (best_idx == 1){
            intp_param.alpha = 1.25;
            intp_param.beta = 2.0;
        }
        else{
            intp_param.alpha = 1.5 + 0.25 * ((best_idx - 2) / 3);
            intp_param.beta = 2.0 + ((best_idx - 2) % 3);
        }

      }

    }
    /*
    if(l3.z != 1){
      printf("\nNAT:");
      for(int i = 0; i < 4; ++i) printf(" %d", intp_param.use_natural[4 - i - 1]);
      printf("\nMD:");
      for(int i = 0; i < 4; ++i) printf(" %d", intp_param.use_md[4 - i - 1]);
      printf("\nREVERSE:");
      for(int i = 0; i < 4; ++i) printf(" %d", intp_param.reverse[4 - i - 1]);
    }
    else{
      printf("\nNAT:");
      for(int i = 0; i < LEVEL; ++i) printf(" %d", intp_param.use_natural[LEVEL - i - 1]);
      printf("\nMD:");
      for(int i = 0; i < LEVEL; ++i) printf(" %d", intp_param.use_md[LEVEL - i - 1]);
      printf("\nREVERSE:");
      for(int i = 0; i < LEVEL; ++i) printf(" %d", intp_param.reverse[LEVEL - i - 1]);
    }
    */
      //printf("\nA B: %.2f %.2f\n",intp_param.alpha,intp_param.beta);
  }

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  if (l3.z == 1){
    auto grid_dim = dim3(
        div(l3.x, AnchorBlockSizeX * numAnchorBlockX),
        div(l3.y, AnchorBlockSizeY * numAnchorBlockY),
        div(l3.z, AnchorBlockSizeZ * numAnchorBlockZ));
    cusz::c_spline_infprecis_data<T*, E*, float, LEVEL, SPLINE_DIM_2, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, DEFAULT_BLOCK_SIZE>  //
        <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
            data, l3, stride3,   //
            ectrl, l3, stride3,  //
            anchor, stride3_anchor, ot->val(), ot->idx(), ot->num(), eb_r, ebx2, radius,
            intp_param);  //,d_profiling_errors);
  }
  else {
    auto grid_dim = dim3(div(l3.x, BLOCK16), div(l3.y, BLOCK16), div(l3.z, BLOCK16));
    cusz::c_spline_infprecis_data<T*, E*, float, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1, DEFAULT_BLOCK_SIZE>  //
        <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
            data, l3, stride3,   //
            ectrl, l3, stride3,  //
            anchor, stride3_anchor, ot->val(), ot->idx(), ot->num(), eb_r, ebx2, radius,
            intp_param);  //,d_profiling_errors);
  }
  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  *time += att_time;

  return 0;
}

template <typename T, typename E, typename Fp>
int psz::module::GPU_spline_reconstruct<T, E, Fp>::kernel_v1(
    T* anchor, stdlen3 const anchor_len3, E* ectrl, T* xdata, stdlen3 const xdata_len3,
    T* outlier_tmp, double eb, uint32_t radius, INTERPOLATION_PARAMS intp_param, float* time,
    void* stream)
{
  auto div = [](auto _l, auto _subl) { return (_l - 1) / _subl + 1; };

  auto ebx2 = eb * 2;
  auto eb_r = 1 / eb;

  auto l3 = TO_DIM3(xdata_len3);  // auto l3 = l3;
  auto stride3 = dim3(1, l3.x, l3.x * l3.y);
  auto l3_anchor = TO_DIM3(anchor_len3);
  auto stride3_anchor = dim3(1, l3_anchor.x, l3_anchor.x * l3_anchor.y);

  CREATE_GPUEVENT_PAIR;
  START_GPUEVENT_RECORDING(stream);

  if (l3.z == 1) {
    auto grid_dim = dim3(
        div(l3.x, AnchorBlockSizeX * numAnchorBlockX),
        div(l3.y, AnchorBlockSizeY * numAnchorBlockY),
        div(l3.z, AnchorBlockSizeZ * numAnchorBlockZ));

    cusz::x_spline_infprecis_data<E*, T*, Fp, LEVEL, SPLINE_DIM_2, AnchorBlockSizeX, AnchorBlockSizeY, AnchorBlockSizeZ, numAnchorBlockX, numAnchorBlockY, numAnchorBlockZ, DEFAULT_BLOCK_SIZE>  //
        <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
            ectrl, l3, stride3,                 //
            anchor, l3_anchor, stride3_anchor,  //
            xdata, l3, stride3,                 //
            outlier_tmp, eb_r, ebx2, radius, intp_param);
  }
  else {
    auto grid_dim = dim3(div(l3.x, BLOCK16), div(l3.y, BLOCK16), div(l3.z, BLOCK16));

    cusz::x_spline_infprecis_data<E*, T*, Fp, 4, SPLINE_DIM_3, BLOCK16, BLOCK16, BLOCK16, 1, 1, 1, DEFAULT_BLOCK_SIZE>  //
        <<<grid_dim, dim3(DEFAULT_BLOCK_SIZE, 1, 1), 0, (cudaStream_t)stream>>>(
            ectrl, l3, stride3,                 //
            anchor, l3_anchor, stride3_anchor,  //
            xdata, l3, stride3,                 //
            outlier_tmp, eb_r, ebx2, radius, intp_param);
  }
  STOP_GPUEVENT_RECORDING(stream);
  CHECK_GPU(cudaStreamSynchronize((cudaStream_t)stream));
  TIME_ELAPSED_GPUEVENT(time);
  DESTROY_GPUEVENT_PAIR;

  return 0;
}

// clang-format on

template struct psz::module::GPU_spline_construct<f4, u1>;
template struct psz::module::GPU_spline_construct<f4, u2>;
template struct psz::module::GPU_spline_construct<f4, u4>;
template struct psz::module::GPU_spline_reconstruct<f4, u1>;
template struct psz::module::GPU_spline_reconstruct<f4, u2>;
template struct psz::module::GPU_spline_reconstruct<f4, u4>;

// used too much shared memory
// template struct psz::module::GPU_spline_construct<f8, u1>;
// template struct psz::module::GPU_spline_construct<f8, u2>;
// template struct psz::module::GPU_spline_construct<f8, u4>;
// template struct psz::module::GPU_spline_reconstruct<f8, u1>;
// template struct psz::module::GPU_spline_reconstruct<f8, u2>;
// template struct psz::module::GPU_spline_reconstruct<f8, u4>;

#undef SETUP
