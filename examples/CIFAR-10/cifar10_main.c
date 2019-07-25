/*
 * cifar10_main.c
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


#include "rt/rt_api.h"

#define MOUNT           1
#define UNMOUNT         0
#define CID             0


void pulp_nn_cifar10(rt_perf_t *perf);
void pulp_parallel(rt_perf_t *perf);

void pulp_parallel(rt_perf_t *perf)
{

  rt_team_fork(NUM_CORES, pulp_nn_cifar10, perf);
}



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

   // starts up the cluster

  rt_cluster_call(NULL, CID, (void *)pulp_parallel, cluster_perf, NULL,1024,1024, rt_nb_pe(), NULL);


  rt_free(RT_ALLOC_L2_CL_DATA, (void *)perf,sizeof(rt_perf_t) );
  rt_free(RT_ALLOC_L2_CL_DATA, (void *)cluster_perf,sizeof(rt_perf_t));

  rt_cluster_mount(UNMOUNT, CID, 0, NULL);

   {
       printf("\nFC last\n",rt_core_id());
   }

return 0;

}
