/*
 * main.c
 * Nazareno Bruschi <nazareno.bruschi@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "rt/rt_api.h"
#include "stats.h"
#include "pulp_nn_kernels.h"

#define STACK_SIZE      1024
#define MOUNT           1
#define UNMOUNT         0
#define CID             0

#define bitins(dst,not_mask_imm,src,mask_imm,off) __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)

#if (NUM_CORES>1)
#define PARALLEL
#endif

#define CHECKSUM

//#define VERBOSE_CHECK

#define PROFILING_MYFUNC

//#define VERBOSE_PROFILING

#define SHIFT_CLIP


${PULPNNINCLUDE}

void check_conv(rt_perf_t *perf);
#ifdef PARALLEL
void pulp_parallel_myfunc(rt_perf_t * perf)
{
  rt_team_fork(NUM_CORES, check_conv, perf);
}
#endif

void check_conv(rt_perf_t *perf)
{
  uint32_t errors = 0;

  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

#ifdef PARALLEL
  if(rt_core_id()==0)
  {
#endif
#ifdef VERBOSE_PROFILING
%if TYPE_OF_KERNEL == 'depthwise':
    printf("MACs=%d\n", DIM_KERNEL_X * DIM_KERNEL_Y * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT);
%elif TYPE_OF_KERNEL == 'pointwise':
    printf("MACs=%d\n", DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT);
%elif TYPE_OF_KERNEL == 'linear_no_quant':
    printf("MACs=%d\n", CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT);
%elif TYPE_OF_KERNEL == 'linear_quant':
    printf("MACs=%d\n", CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT);
%endif
#endif
#if INPUT == 2
%if TYPE_OF_KERNEL == 'depthwise':
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2); i++)
    {
      uint8_t c_data = bitins(IN_INT2_L2_HWC[4*i], n_mask2, IN_INT2_L2_HWC[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, IN_INT2_L2_HWC[(4*i)+2], mask4, off4);
      IN_INT8_L2_HWC[i] = bitins(c_data, n_mask6, IN_INT2_L2_HWC[(4*i)+3], mask6, off6);
    }
    for(int i=0; i<(CH_IM_IN >> 2); i++)
    {
      for(int j=0; j<(DIM_IM_IN_X * DIM_IM_IN_Y); j++)
      {
        IN_INT8_L1_CHW[(i * DIM_IM_IN_X * DIM_IM_IN_Y) + j] = IN_INT8_L2_HWC[(j * (CH_IM_IN >> 2)) + i];
      }
    }
%elif TYPE_OF_KERNEL == 'pointwise' or 'linear_no_quant' or 'linear_quant':
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 2); i++)
    {
      uint8_t c_data = bitins(IN_INT2_L2[4*i], n_mask2, IN_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, IN_INT2_L2[(4*i)+2], mask4, off4);
      IN_INT8_L1[i] = bitins(c_data, n_mask6, IN_INT2_L2[(4*i)+3], mask6, off6);
    }
%endif
#elif INPUT == 4
%if TYPE_OF_KERNEL == 'depthwise':
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1); i++)
    {
      IN_INT8_L2_HWC[i] = bitins(IN_INT4_L2_HWC[2*i], n_mask, IN_INT4_L2_HWC[(2*i)+1], mask, off);
    }
    for(int i=0; i<(CH_IM_IN >> 1); i++)
    {
      for(int j=0; j<(DIM_IM_IN_X * DIM_IM_IN_Y); j++)
      {
        IN_INT8_L1_CHW[(i * DIM_IM_IN_X * DIM_IM_IN_Y) + j] = IN_INT8_L2_HWC[(j * (CH_IM_IN >> 1)) + i];
      }
    }
%elif TYPE_OF_KERNEL == 'pointwise' or 'linear_no_quant' or 'linear_quant':
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN) >> 1); i++)
    {
      IN_INT8_L1[i] = bitins(IN_INT4_L2[2*i], n_mask, IN_INT4_L2[(2*i)+1], mask, off);
    }
%endif
#elif INPUT == 8
%if TYPE_OF_KERNEL == 'depthwise':
    for(int i=0; i<CH_IM_IN; i++)
    {
      for(int j=0; j<(DIM_IM_IN_X * DIM_IM_IN_Y); j++)
      {
        IN_INT8_L1_CHW[(i * DIM_IM_IN_X * DIM_IM_IN_Y) + j] = IN_INT8_L2_HWC[(j * CH_IM_IN) + i];
      }
    }
%elif TYPE_OF_KERNEL == 'pointwise' or 'linear_no_quant' or 'linear_quant':
    for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN); i++)
    {
      IN_INT8_L1[i] = IN_INT8_L2[i];
    }
%endif
#endif
%if TYPE_OF_KERNEL == 'depthwise':
#if WEIGHTS == 2
    for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y); i++)
    {
      for(int j=0; j<CH_IM_IN; j++)
      {
        WEIGHT_INT2_L2_HWC[(i * CH_IM_IN) + j] = WEIGHT_INT2_L2_CHW[(j * (DIM_KERNEL_X * DIM_KERNEL_Y)) + i];
      }
    }
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 2); i++)
    {
      int8_t c_data = bitins(WEIGHT_INT2_L2_HWC[4*i], n_mask2, WEIGHT_INT2_L2_HWC[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, WEIGHT_INT2_L2_HWC[(4*i)+2], mask4, off4);
      WEIGHT_INT8_L2_HWC[i] = bitins(c_data, n_mask6, WEIGHT_INT2_L2_HWC[(4*i)+3], mask6, off6);
    }
    for(int i=0; i<(CH_IM_IN >> 2); i++)
    {
      for(int j=0; j<(DIM_KERNEL_X * DIM_KERNEL_Y); j++)
      {
        WEIGHT_INT8_L1_CHW[(i * DIM_KERNEL_X * DIM_KERNEL_Y) + j] = WEIGHT_INT8_L2_HWC[(j * (CH_IM_IN >> 2)) + i];
      }
    }
#elif WEIGHTS == 4
    for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y); i++)
    {
      for(int j=0; j<CH_IM_IN; j++)
      {
        WEIGHT_INT4_L2_HWC[(i * CH_IM_IN) + j] = WEIGHT_INT4_L2_CHW[(j * (DIM_KERNEL_X * DIM_KERNEL_Y)) + i];
      }
    }
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN) >> 1); i++)
    {
      WEIGHT_INT8_L2_HWC[i] = bitins(WEIGHT_INT4_L2_HWC[2*i], n_mask, WEIGHT_INT4_L2_HWC[(2*i)+1], mask, off);
    }
    for(int i=0; i<(CH_IM_IN >> 1); i++)
    {
      for(int j=0; j<(DIM_KERNEL_X * DIM_KERNEL_Y); j++)
      {
        WEIGHT_INT8_L1_CHW[(i * DIM_KERNEL_X * DIM_KERNEL_Y) + j] = WEIGHT_INT8_L2_HWC[(j * (CH_IM_IN >> 1)) + i];
      }
    }
#elif WEIGHTS == 8
    for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN); i++)
    {
      WEIGHT_INT8_L1_CHW[i] = WEIGHT_INT8_L2_CHW[i];
    }
#endif
%elif TYPE_OF_KERNEL == 'pointwise':
#if WEIGHTS == 2
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 2); i++)
    {
      int8_t c_data = bitins(WEIGHT_INT2_L2[4*i], n_mask2, WEIGHT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, WEIGHT_INT2_L2[(4*i)+2], mask4, off4);
      WEIGHT_INT8_L1[i] = bitins(c_data, n_mask6, WEIGHT_INT2_L2[(4*i)+3], mask6, off6);
    }
#elif WEIGHTS == 4
    for(int i=0; i<((DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT) >> 1); i++)
    {
      WEIGHT_INT8_L1[i] = bitins(WEIGHT_INT4_L2[2*i], n_mask, WEIGHT_INT4_L2[(2*i)+1], mask, off);
    }
#elif WEIGHTS == 8
    for(int i=0; i<(DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * CH_IM_OUT); i++)
    {
      WEIGHT_INT8_L1[i] = WEIGHT_INT8_L2[i];
    }
#endif
%elif TYPE_OF_KERNEL == 'linear_no_quant' or 'linear_quant':
#if WEIGHTS == 2
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN * CH_IM_OUT) >> 2); i++)
    {
      int8_t c_data = bitins(WEIGHT_INT2_L2[4*i], n_mask2, WEIGHT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, WEIGHT_INT2_L2[(4*i)+2], mask4, off4);
      WEIGHT_INT8_L1[i] = bitins(c_data, n_mask6, WEIGHT_INT2_L2[(4*i)+3], mask6, off6);
    }
#elif WEIGHTS == 4
    for(int i=0; i<((DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN * CH_IM_OUT) >> 1); i++)
    {
      WEIGHT_INT8_L1[i] = bitins(WEIGHT_INT4_L2[2*i], n_mask, WEIGHT_INT4_L2[(2*i)+1], mask, off);
    }
#elif WEIGHTS == 8
    for(int i=0; i<(DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_IN * CH_IM_OUT); i++)
    {
      WEIGHT_INT8_L1[i] = WEIGHT_INT8_L2[i];
    }
#endif
%endif
#ifdef SHIFT_CLIP
    for(int i=0; i<CH_IM_OUT; i++)
    {
      KAPPA_L1[i] = KAPPA_L2[i];
    }
    for(int i=0; i<CH_IM_OUT; i++)
    {
      LAMBDA_L1[i] = LAMBDA_L2[i];
    }
#else
#if OUTPUT == 2
    for(int i=0; i<(CH_IM_OUT << 2); i++)
    {
      THR_INT2_L1[i] = THR_INT2_L2[i];
    }
#elif OUTPUT == 4
    for(int i=0; i<(CH_IM_OUT << 4); i++)
    {
      THR_INT4_L1[i] = THR_INT4_L2[i];
    }
#endif
#endif
#ifdef PARALLEL
  }
  rt_team_barrier();
#endif
#ifdef PROFILING_MYFUNC
#ifdef VERBOSE_PROFILING
  INIT_PROFILING();
  START_PROFILING();
#else
  rt_perf_t perf2;
  rt_perf_init(&perf2);
  rt_perf_conf(&perf2, (1<<RT_PERF_CYCLES));
  rt_perf_reset(&perf2);
  rt_perf_stop(&perf2);
  rt_perf_start(&perf2);
#endif
#endif
#ifdef PARALLEL
  rt_team_barrier();
#endif

${PULPNNCALL}

#ifdef PROFILING_MYFUNC
#ifdef VERBOSE_PROFILING
  STOP_PROFILING();
#else
  rt_perf_stop(&perf2);
  rt_perf_save(&perf2);
  int cid = rt_core_id();
  int perf_cyc =  rt_perf_get(&perf2, RT_PERF_CYCLES);
%if TYPE_OF_KERNEL == 'depthwise':
  int MACs = DIM_KERNEL_X * DIM_KERNEL_Y * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT;
%elif TYPE_OF_KERNEL == 'pointwise':
  int MACs = DIM_KERNEL_X * DIM_KERNEL_Y * CH_IM_IN * DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT;
%elif TYPE_OF_KERNEL == 'linear_no_quant':
  int MACs = CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT;
%elif TYPE_OF_KERNEL == 'linear_quant':
  int MACs = CH_IM_IN * DIM_IM_IN_X * DIM_IM_IN_Y * CH_IM_OUT;
%endif
  float perf_MAC =  (float)MACs/perf_cyc;
  if (cid == 0){
    printf("[%d] : num_cycles: %d\n",cid,perf_cyc);
    printf("[%d] : MACs: %d\n",cid,MACs );
    printf("[%d] : MAC/cycle: %f\n",cid,perf_MAC );
    printf("[%d] : n. of Cores: %d\n",cid,NUM_CORES);
  }
#endif
#endif
#ifdef CHECKSUM
#ifdef PARALLEL
  rt_team_barrier();
  if(rt_core_id()==0)
  {
#endif
%if TYPE_OF_KERNEL == 'linear_no_quant':
    for (int i=0; i<CH_IM_OUT; i++)
    {
      if(OUT_L1[i] != OUT_L2[i])
      {
#ifdef VERBOSE_CHECK
        printf("error at index %d, %d instead of %d\n", i, OUT_L1[i], OUT_L2[i]);
#endif
        errors++;
      }
    }
    printf("errors: %d\n", errors);
%elif TYPE_OF_KERNEL == 'linear_quant':
#if OUTPUT == 2
    for (int i=0; i<(CH_IM_OUT >> 2); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<(CH_IM_OUT >> 2); i++)
    {
      uint8_t c_data = bitins(OUT_INT2_L2[4*i], n_mask2, OUT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, OUT_INT2_L2[(4*i)+2], mask4, off4);
      OUT_INT8_L2[i] = bitins(c_data, n_mask6, OUT_INT2_L2[(4*i)+3], mask6, off6);
#elif OUTPUT == 4
    for (int i=0; i<(CH_IM_OUT >> 1); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<(CH_IM_OUT >> 1); i++)
    {
      OUT_INT8_L2[i] = bitins(OUT_INT4_L2[2*i], n_mask, OUT_INT4_L2[(2*i)+1], mask, off);
#elif OUTPUT == 8
    for (int i=0; i<CH_IM_OUT; i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<CH_IM_OUT; i++)
    {
#endif
      if(OUT_L2[i] != OUT_INT8_L2[i])
      {
#ifdef VERBOSE_CHECK
        printf("error at index %d, %d instead of %d\n", i, OUT_L2[i], OUT_INT8_L2[i]);
#endif
        errors++;
      }
    }
    printf("errors: %d\n", errors);
%else:
#if OUTPUT == 2
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 2); i++)
    {
      uint8_t c_data = bitins(OUT_INT2_L2[4*i], n_mask2, OUT_INT2_L2[(4*i)+1], mask2, off2);
      c_data = bitins(c_data, n_mask4, OUT_INT2_L2[(4*i)+2], mask4, off4);
      OUT_INT8_L2[i] = bitins(c_data, n_mask6, OUT_INT2_L2[(4*i)+3], mask6, off6);
#elif OUTPUT == 4
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<((DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT) >> 1); i++)
    {
      OUT_INT8_L2[i] = bitins(OUT_INT4_L2[2*i], n_mask, OUT_INT4_L2[(2*i)+1], mask, off);
#elif OUTPUT == 8
    for (int i=0; i<(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT); i++)
    {
      OUT_L2[i] = OUT_L1[i];
    }
    for (int i=0; i<(DIM_IM_OUT_X * DIM_IM_OUT_Y * CH_IM_OUT); i++)
    {
#endif
      if(OUT_L2[i] != OUT_INT8_L2[i])
      {
#ifdef VERBOSE_CHECK
        printf("error at index %d, %d instead of %d\n", i, OUT_L2[i], OUT_INT8_L2[i]);
#endif
        errors++;
      }
    }
    printf("errors: %d\n", errors);
%endif
#ifdef PARALLEL
  }
  rt_team_barrier();
#endif
#endif
}

///////////////////////////////////////////////////////////////////
////------------------------MAIN------------------------------/////
///////////////////////////////////////////////////////////////////

int main()
{
  rt_event_sched_t sched;
  rt_event_sched_init(&sched);
  if (rt_event_alloc(&sched, 8)) return -1;
  rt_event_t *event = rt_event_get_blocking(NULL);

  rt_cluster_mount(MOUNT, CID, 0, NULL);


  // allocate performance counters
    rt_perf_t *perf = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(rt_perf_t));
  if (perf == NULL) return -1;

  rt_perf_t * cluster_perf = rt_alloc(RT_ALLOC_L2_CL_DATA, sizeof(rt_perf_t));
  if (cluster_perf == NULL) return -1;

  #ifdef PARALLEL
    rt_cluster_call(NULL, CID, (void *)pulp_parallel_myfunc, cluster_perf, NULL,1024, 1024, NUM_CORES, NULL);
  #else
    rt_cluster_call(NULL, CID, (void *)check_conv, cluster_perf, NULL,1024, 1024, NUM_CORES, NULL);
  #endif

  rt_free(RT_ALLOC_L2_CL_DATA, (void *) perf, sizeof(rt_perf_t));
  rt_free(RT_ALLOC_L2_CL_DATA, (void *) cluster_perf, sizeof(rt_perf_t));

  rt_cluster_mount(UNMOUNT, CID, 0, NULL);

   {
       printf("\nFC last\n",rt_core_id());
   }
return 0;
}
