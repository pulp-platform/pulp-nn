/*
 * checksum_layer.h
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


#ifndef __CHECKSUM_LAYER__
#define __CHECKSUM_LAYER__
#endif

int checksum( int8_t *buffer, int dim)
{
	int check=0;
	for (int i=0 ; i < dim; i++)
		check += buffer[i];
	return check;
}
