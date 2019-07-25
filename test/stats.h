/*
 * stats.h
 * Angelo Garofalo <angelo.garofalo@unibo.it>
 *
 * Copyright (C) 2019 University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#ifndef __PROF_STATS__
#define __PROF_STATS__
#endif

#ifdef PROFILING
#define INIT_PROFILING()       rt_perf_t perf2;

#define START_PROFILING() \
for (int k=0; k < 13; k++) { \
  if ( (k>=3))  { \
    rt_perf_init(&perf2);						\
    if(k==3) rt_perf_conf(&perf2, (1<<RT_PERF_CYCLES));			\
    if(k==4) rt_perf_conf(&perf2, (1<<RT_PERF_IMISS));			\
    if(k==5) rt_perf_conf(&perf2, (1<<RT_PERF_LD_EXT));		\
    if(k==6) rt_perf_conf(&perf2, (1<<RT_PERF_ST_EXT ));		\
    if(k==7) rt_perf_conf(&perf2, (1<<RT_PERF_TCDM_CONT ));		\
    if(k==8) rt_perf_conf(&perf2, (1<<RT_PERF_INSTR ));			\
    if(k==9) rt_perf_conf(&perf2, (1<<RT_PERF_ACTIVE_CYCLES));		\
    if(k==10) rt_perf_conf(&perf2, (1<<RT_PERF_LD_STALL));		\
    if(k==11) rt_perf_conf(&perf2, (1<<RT_PERF_JR_STALL ));		\
    if(k==12) rt_perf_conf(&perf2, (1<<RT_PERF_BRANCH ));		\
    rt_perf_reset(&perf2);						\
    rt_perf_stop(&perf2);						\
    rt_perf_start(&perf2); \
			    }

#define STOP_PROFILING() \
       if( (k>=3 )) { \
	 rt_perf_stop(&perf2);			\
	 rt_perf_save(&perf2);			\
	 int cid = rt_core_id();					\
	 if(k==3) printf("[%d] : num_cycles: %d\n",cid,rt_perf_get(&perf2, RT_PERF_CYCLES) ); \
	 if(k==4 ) printf("[%d] : num_instr_miss: %d\n",cid,rt_perf_get(&perf2, RT_PERF_IMISS) ); \
	 if(k==5) printf("[%d] : num_ext_load: %d\n",cid,rt_perf_get(&perf2, RT_PERF_LD_EXT) ); \
	 if(k==6) printf("[%d] : num_ext_Store: %d\n",cid,rt_perf_get(&perf2, RT_PERF_ST_EXT) ); \
	 if(k==7) printf("[%d] : num_tcdm_contentions: %d\n",cid,rt_perf_get(&perf2, RT_PERF_TCDM_CONT) ); \
	 if(k==8) printf("[%d] : num_instrs: %d\n",cid,rt_perf_get(&perf2, RT_PERF_INSTR) ); \
	 if(k==9) printf("[%d] : num_active_cycles: %d\n",cid,rt_perf_get(&perf2, RT_PERF_ACTIVE_CYCLES) ); \
	 if(k==10) printf("[%d] : num_load_stalls: %d\n",cid,rt_perf_get(&perf2,RT_PERF_LD_STALL ) ); \
	 if(k==11) printf("[%d] : num_jumpr_stalls: %d\n",cid,rt_perf_get(&perf2,RT_PERF_JR_STALL ) ); \
	 if(k==12) printf("[%d] : num_branch: %d\n",cid,rt_perf_get(&perf2,RT_PERF_BRANCH ) ); \
       }								\
 }
#else
#define INIT_PROFILING()
#define START_PROFILING()
#define STOP_PROFILING()
#endif
