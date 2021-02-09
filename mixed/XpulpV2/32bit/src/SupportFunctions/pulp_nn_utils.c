/*
 * pulp_nn_utils.c
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

#include "pmsis.h"
#include "pulp_nn_utils.h"

#define bitext(x,size,off)                            __builtin_pulp_bextract(x,size,off)
#define bitextu(x,size,off)                           __builtin_pulp_bextractu(x,size,off)
#define bitins(dst,not_mask_imm,src,mask_imm,off)     __builtin_pulp_binsert(dst,not_mask_imm,src,mask_imm,off)
#define pack(x,y,z,t)                                 __builtin_pulp_pack4(x,y,z,t)
#define max4(a,b)  		                                __builtin_pulp_maxu4(a,b)
#define max8(a,b)                                     __builtin_pulp_maxu8(a,b)
#define max16(a,b)                                    __builtin_pulp_maxu16(a,b)
#define avg4(a,b)                                     __builtin_pulp_avgu4(a,b)
#define avg8(a,b)                                     __builtin_pulp_avgu8(a,b)
#define avg16(a,b)                                    __builtin_pulp_avgu16(a,b)

uint8_t __attribute__((always_inline)) pulp_nn_bn_quant_u8 (
  int32_t phi,
  int32_t k,
  int32_t lambda,
  int8_t  d
) {
  int32_t integer_image_phi = (k * phi) + lambda;
  int32_t x = (integer_image_phi) >> d;
  uint8_t res = __builtin_pulp_clipu(x, 0, 255);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_quant_u8(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  int32_t x = (m * phi) >> d;
  uint8_t res = __builtin_pulp_clipu(x, 0, 255);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_add_quant_u8 (
  uint8_t pix1,
  uint8_t pix2,
  int16_t m1,
  int16_t m2,
  int8_t  d
) {
  /* Integer Batch Normalization */
  uint32_t integer_image = pix1*m1 + pix2*m2;
  /* Quantization */
  uint32_t x = (integer_image) >> d;
  uint8_t res = __builtin_pulp_clipu(x, 0, 255);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_bn_quant_u4 (
  int32_t phi,
  int32_t k,
  int32_t lambda,
  int8_t  d
) {
  int32_t integer_image_phi = (k * phi) + lambda;
  int32_t x = (integer_image_phi) >> d;
  uint8_t res = __builtin_pulp_clipu(x, 0, 15);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_quant_u4(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  int32_t x = (m * phi) >> d;
  uint8_t res = __builtin_pulp_clipu(x, 0, 15);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_add_quant_u4 (
  uint8_t pix1,
  uint8_t pix2,
  int16_t m1,
  int16_t m2,
  int8_t  d
) {
  /* Integer Batch Normalization */
  uint32_t integer_image = pix1*m1 + pix2*m2;
  /* Quantization */
  uint32_t x = (integer_image) >> d;
  uint8_t res = __builtin_pulp_clipu(x, 0, 15);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_bn_quant_u2 (
  int32_t phi,
  int32_t k,
  int32_t lambda,
  int8_t  d
) {
  int32_t integer_image_phi = (k * phi) + lambda;
  int32_t x = (integer_image_phi) >> d;
  uint8_t res = __builtin_pulp_clipu(x, 0, 3);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_quant_u2(
  int32_t phi,
  int16_t m,
  int8_t  d
) {
  int32_t x = (m * phi) >> d;
  uint8_t res = __builtin_pulp_clipu(x, 0, 3);
  return res;
}

uint8_t __attribute__((always_inline)) pulp_nn_add_quant_u2 (
  uint8_t pix1,
  uint8_t pix2,
  int16_t m1,
  int16_t m2,
  int8_t  d
) {
  /* Integer Batch Normalization */
  uint32_t integer_image = pix1*m1 + pix2*m2;
  /* Quantization */
  uint32_t x = (integer_image) >> d;
  uint8_t res = __builtin_pulp_clipu(x, 0, 3);
  return res;
}

v4s __attribute__((always_inline))pulp_nn_i4_to_i8_r( int8_t *pSrc)
{
	v4s Src = *((v4s*) pSrc);
	int8_t bext1, bext2, bext3, bext4;

	bext1 = (int8_t) bitext((int) Src, 4, 0);
	bext2 = (int8_t) bitext((int) Src, 4, 4);
	bext3 = (int8_t) bitext((int) Src, 4, 8);
	bext4 = (int8_t) bitext((int) Src, 4, 12);
	v4s res = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);

	return res;
}

v4u __attribute__((always_inline))pulp_nn_u4_to_u8_r(uint8_t *pSrc)
{
	v4u Src = *((v4u*) pSrc);
	uint8_t bext1, bext2, bext3, bext4;

	bext1 = (uint8_t) bitextu((unsigned int) Src, 4, 0);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 4, 4);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 4, 8);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 4, 12);
	v4u res = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);

	return res;
}

v4s __attribute__((always_inline))pulp_nn_i2_to_i8_r( int8_t *pSrc)
{
	v4s Src = *((v4s*) pSrc);
	int8_t bext1, bext2, bext3, bext4;

	bext1 = (int8_t) bitext((int) Src, 2, 0);
	bext2 = (int8_t) bitext((int) Src, 2, 2);
	bext3 = (int8_t) bitext((int) Src, 2, 4);
	bext4 = (int8_t) bitext((int) Src, 2, 6);
	v4s res = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);

	return res;
}

v4u __attribute__((always_inline))pulp_nn_u2_to_u8_r(uint8_t *pSrc)
{
	v4u Src = *((v4u*) pSrc);
	uint8_t bext1, bext2, bext3, bext4;

	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 0);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 2);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 4);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 6);
	v4u res = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);

	return res;
}

v4s __attribute__((always_inline))pulp_nn_i2_to_i4_r(int8_t *pSrc)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  v4s Src = *((v4s*) pSrc);
  int8_t bext1, bext2, bext3, bext4;
  int8_t out1, out2, out3, out4;

  bext1 = (int8_t) bitextu((int) Src, 2, 0);
  bext2 = (int8_t) bitextu((int) Src, 2, 2);
  bext3 = (int8_t) bitextu((int) Src, 2, 4);
  bext4 = (int8_t) bitextu((int) Src, 2, 6);

  out1 = bitins(bext1, n_mask, bext2, mask, off);
  out2 = bitins(bext3, n_mask, bext4, mask, off);

  bext1 = (int8_t) bitextu((int) Src, 2, 8);
  bext2 = (int8_t) bitextu((int) Src, 2, 10);
  bext3 = (int8_t) bitextu((int) Src, 2, 12);
  bext4 = (int8_t) bitextu((int) Src, 2, 16);

  out3 = bitins(bext1, n_mask, bext2, mask, off);
  out4 = bitins(bext3, n_mask, bext4, mask, off);

  v4s res = pack((int8_t) out1, (int8_t) out2, (int8_t) out3, (int8_t) out4);

  return res;
}

v4u __attribute__((always_inline))pulp_nn_u2_to_u4_r(uint8_t *pSrc)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  v4u Src = *((v4u*) pSrc);
  uint8_t bext1, bext2, bext3, bext4;
  uint8_t out1, out2, out3, out4;

  bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 0);
  bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 2);
  bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 4);
  bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 6);

  out1 = bitins(bext1, n_mask, bext2, mask, off);
  out2 = bitins(bext3, n_mask, bext4, mask, off);

  bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 8);
  bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 10);
  bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 12);
  bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 16);

  out3 = bitins(bext1, n_mask, bext2, mask, off);
  out4 = bitins(bext3, n_mask, bext4, mask, off);

  v4u res = pack((uint8_t) out1, (uint8_t) out2, (uint8_t) out3, (uint8_t) out4);

  return res;
}

int8_t *__attribute__((always_inline))pulp_nn_i4_to_i8( int8_t *pSrc, int8_t *pDst)
{
	v4s Src = *((v4s*) pSrc);
	int8_t bext1, bext2, bext3, bext4;

  pSrc+=4;

	bext1 = (int8_t) bitext((int) Src, 4, 0);
	bext2 = (int8_t) bitext((int) Src, 4, 4);
	bext3 = (int8_t) bitext((int) Src, 4, 8);
	bext4 = (int8_t) bitext((int) Src, 4, 12);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (int8_t) bitext((int) Src, 4, 16);
	bext2 = (int8_t) bitext((int) Src, 4, 20);
	bext3 = (int8_t) bitext((int) Src, 4, 24);
	bext4 = (int8_t) bitext((int) Src, 4, 28);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);

  return pSrc;
}

uint8_t *__attribute__((always_inline))pulp_nn_u4_to_u8(uint8_t *pSrc, uint8_t *pDst)
{
	v4u Src = *((v4u*) pSrc);
	uint8_t bext1, bext2, bext3, bext4;

  pSrc+=4;

	bext1 = (uint8_t) bitextu((unsigned int) Src, 4, 0);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 4, 4);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 4, 8);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 4, 12);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 4, 16);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 4, 20);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 4, 24);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 4, 28);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);

  return pSrc;
}

int8_t *__attribute__((always_inline))pulp_nn_i2_to_i8( int8_t * pSrc, int8_t * pDst)
{
	v4s Src = *((v4s*) pSrc);
	int8_t bext1, bext2, bext3, bext4;

  pSrc+=4;

	bext1 = (int8_t) bitext((int) Src, 2, 0);
	bext2 = (int8_t) bitext((int) Src, 2, 2);
	bext3 = (int8_t) bitext((int) Src, 2, 4);
	bext4 = (int8_t) bitext((int) Src, 2, 6);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (int8_t) bitext((int) Src, 2, 8);
	bext2 = (int8_t) bitext((int) Src, 2, 10);
	bext3 = (int8_t) bitext((int) Src, 2, 12);
	bext4 = (int8_t) bitext((int) Src, 2, 14);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (int8_t) bitext((int) Src, 2, 16);
	bext2 = (int8_t) bitext((int) Src, 2, 18);
	bext3 = (int8_t) bitext((int) Src, 2, 20);
	bext4 = (int8_t) bitext((int) Src, 2, 22);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (int8_t) bitext((int) Src, 2, 24);
	bext2 = (int8_t) bitext((int) Src, 2, 26);
	bext3 = (int8_t) bitext((int) Src, 2, 28);
	bext4 = (int8_t) bitext((int) Src, 2, 30);
	*((v4s*)pDst) = pack((int8_t) bext1, (int8_t) bext2, (int8_t) bext3, (int8_t) bext4);

  return pSrc;
}

uint8_t *__attribute__((always_inline))pulp_nn_u2_to_u8(uint8_t * pSrc, uint8_t * pDst)
{
	v4u Src = *((v4u*) pSrc);
	uint8_t bext1, bext2, bext3, bext4;

  pSrc+=4;

	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 0);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 2);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 4);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 6);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 8);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 10);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 12);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 14);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 16);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 18);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 20);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 22);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);
  	asm volatile(""::: "memory");
	pDst+=4;
	bext1 = (uint8_t) bitextu((unsigned int) Src, 2, 24);
	bext2 = (uint8_t) bitextu((unsigned int) Src, 2, 26);
	bext3 = (uint8_t) bitextu((unsigned int) Src, 2, 28);
	bext4 = (uint8_t) bitextu((unsigned int) Src, 2, 30);
	*((v4u*)pDst) = pack((uint8_t) bext1, (uint8_t) bext2, (uint8_t) bext3, (uint8_t) bext4);

  return pSrc;
}

int8_t *__attribute__((always_inline))pulp_nn_i2_to_i4( int8_t * pSrc, int8_t * pDst)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  v4s Src = *((v4s*) pSrc);
  int8_t bext1, bext2, bext3, bext4;
  int8_t out1, out2, out3, out4;

  pSrc+=4;

  bext1 = (int8_t) bitext((int) Src, 2, 0);
  bext2 = (int8_t) bitext((int) Src, 2, 2);
  bext3 = (int8_t) bitext((int) Src, 2, 4);
  bext4 = (int8_t) bitext((int) Src, 2, 6);

  out1 = bitins(bext1, n_mask, bext2, mask, off);
  out2 = bitins(bext3, n_mask, bext4, mask, off);
  asm volatile(""::: "memory");

  bext1 = (int8_t) bitext((int) Src, 2, 8);
  bext2 = (int8_t) bitext((int) Src, 2, 10);
  bext3 = (int8_t) bitext((int) Src, 2, 12);
  bext4 = (int8_t) bitext((int) Src, 2, 14);

  out3 = bitins(bext1, n_mask, bext2, mask, off);
  out4 = bitins(bext3, n_mask, bext4, mask, off);

  *((v4s*)pDst) = pack((int8_t) out1, (int8_t) out2, (int8_t) out3, (int8_t) out4);
  asm volatile(""::: "memory");

  pDst+=4;
  bext1 = (int8_t) bitext((int) Src, 2, 16);
  bext2 = (int8_t) bitext((int) Src, 2, 18);
  bext3 = (int8_t) bitext((int) Src, 2, 20);
  bext4 = (int8_t) bitext((int) Src, 2, 22);

  out1 = bitins(bext1, n_mask, bext2, mask, off);
  out2 = bitins(bext3, n_mask, bext4, mask, off);
  asm volatile(""::: "memory");

  bext1 = (int8_t) bitext((int) Src, 2, 24);
  bext2 = (int8_t) bitext((int) Src, 2, 26);
  bext3 = (int8_t) bitext((int) Src, 2, 28);
  bext4 = (int8_t) bitext((int) Src, 2, 30);

  out3 = bitins(bext1, n_mask, bext2, mask, off);
  out4 = bitins(bext3, n_mask, bext4, mask, off);

  *((v4s*)pDst) = pack((int8_t) out1, (int8_t) out2, (int8_t) out3, (int8_t) out4);

  return pSrc;
}

uint8_t *__attribute__((always_inline))pulp_nn_u2_to_u4( uint8_t * pSrc, uint8_t * pDst)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  v4u Src = *((v4u*) pSrc);
  uint8_t bext1, bext2, bext3, bext4;
  uint8_t out1, out2, out3, out4;

  pSrc+=4;

  bext1 = (uint8_t) bitext((unsigned int) Src, 2, 0);
  bext2 = (uint8_t) bitext((unsigned int) Src, 2, 2);
  bext3 = (uint8_t) bitext((unsigned int) Src, 2, 4);
  bext4 = (uint8_t) bitext((unsigned int) Src, 2, 6);

  out1 = bitins(bext1, n_mask, bext2, mask, off);
  out2 = bitins(bext3, n_mask, bext4, mask, off);
  asm volatile(""::: "memory");

  bext1 = (uint8_t) bitext((unsigned int) Src, 2, 8);
  bext2 = (uint8_t) bitext((unsigned int) Src, 2, 10);
  bext3 = (uint8_t) bitext((unsigned int) Src, 2, 12);
  bext4 = (uint8_t) bitext((unsigned int) Src, 2, 14);

  out3 = bitins(bext1, n_mask, bext2, mask, off);
  out4 = bitins(bext3, n_mask, bext4, mask, off);

  *((v4u*)pDst) = pack((uint8_t) out1, (uint8_t) out2, (uint8_t) out3, (uint8_t) out4);
  asm volatile(""::: "memory");

  pDst+=4;
  bext1 = (uint8_t) bitext((unsigned int) Src, 2, 16);
  bext2 = (uint8_t) bitext((unsigned int) Src, 2, 18);
  bext3 = (uint8_t) bitext((unsigned int) Src, 2, 20);
  bext4 = (uint8_t) bitext((unsigned int) Src, 2, 22);

  out1 = bitins(bext1, n_mask, bext2, mask, off);
  out2 = bitins(bext3, n_mask, bext4, mask, off);
  asm volatile(""::: "memory");

  bext1 = (uint8_t) bitext((unsigned int) Src, 2, 24);
  bext2 = (uint8_t) bitext((unsigned int) Src, 2, 26);
  bext3 = (uint8_t) bitext((unsigned int) Src, 2, 28);
  bext4 = (uint8_t) bitext((unsigned int) Src, 2, 30);

  out3 = bitins(bext1, n_mask, bext2, mask, off);
  out4 = bitins(bext3, n_mask, bext4, mask, off);
  
  *((v4u*)pDst) = pack((uint8_t) out1, (uint8_t) out2, (uint8_t) out3, (uint8_t) out4);

  return pSrc;
}

void __attribute__((always_inline))pulp_zero_mem(uint8_t * pBuffer, unsigned int size)
{
  int lfover = size &0x3;
  for (int i=0; i<(size>>2); i++)
  {
    *((v4u *)pBuffer) = (v4u){0,0,0,0};
    asm volatile("":::"memory");
    pBuffer+=4;
  }
  while(lfover)
  {
    *pBuffer++=0;
    lfover--;
  }
}

void __attribute__((always_inline))pulp_nn_im2col_u8_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 2u;
  int lfover = blockSize & 0x3;

  for (int i = 0; i<blkCnt; i++)
  {
    *((v4u*)pOutput) = *((v4u*) pInput);
    pInput+=4;
    pOutput+=4;
  }
  while(lfover)
  {
    *((uint8_t*)pOutput) = *((uint8_t*)pInput);
    pOutput++;
    pInput++;
    lfover--;
  }
}

void pulp_nn_im2col_u4_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 3u;
  int lfover = blockSize & 0x7;

  for (int i = 0; i<blkCnt; i++)
  {
    pInput = pulp_nn_u4_to_u8(pInput, pOutput);
    asm volatile("":::"memory");
    //pInput+=4;
    pOutput+=8;
  }
  while(lfover)
  {
	*((uint8_t*)pOutput) = (uint8_t) bitextu((unsigned int) *pInput, 4, 0);
	pOutput++;
	*((uint8_t*)pOutput) = (uint8_t) bitextu((unsigned int) *pInput, 4, 4);
	pOutput++;
	pInput++;
	lfover-=2;
  }
}

void __attribute__((always_inline))pulp_nn_im2col_u2_to_u8(uint8_t * pInput, uint8_t * pOutput, unsigned int blockSize)
{
  unsigned int blkCnt = blockSize >> 4u;
  int lfover = blockSize & 0xf;

  for(int i = 0; i<blkCnt; i++)
  {
    pInput = pulp_nn_u2_to_u8(pInput, pOutput);
    //pInput+=4;
    pOutput+=16;
  }
  while(lfover)
  {
	*((v4u*)pOutput) = pulp_nn_u2_to_u8_r(pInput);
	pInput++;
	pOutput+=4;
	lfover-=4;
  }
}

void pulp_nn_compare_and_replace_if_larger_u8(uint8_t * base,
                                                uint8_t * target,
                                                uint16_t length)
{
  uint8_t *pIn = base;
  uint8_t *pCom = target;
  v4u inp;
  v4u com;
  int cnt = length >> 2;

  while(cnt > 0u)
  {
    inp = *((v4u*)pIn);
    com = *((v4u*)pCom);

    *((v4u*)pIn) = max4(inp, com);

    pCom+=4;
    pIn+=4;
    cnt--;
  }

  int left = length & 0x3;
  while (left>0u)
  {
    if(*pIn<*pCom)
      *pIn=*pCom;
    
    pIn++;
    pCom++;
    left--;
  }
}

void pulp_nn_avg_and_replace_u8(uint8_t * base,
                                  uint8_t * target,
                                  uint16_t length)
{
  uint8_t *pIn = base;
  uint8_t *pCom = target;

  while (length>0u)
  {
    *pIn = ((*pIn + *pCom) >> 1);

    pIn++;
    pCom++;
    length--;
  }
}

void pulp_nn_compare_and_replace_if_larger_u4(uint8_t * base,
                                                uint8_t * target,
                                                uint16_t length)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  uint8_t *pIn = base;
  uint8_t *pCom = target;
  v4u inp[2];
  v4u com[2];
  uint8_t *out;
  int cnt = length >> 2;

  while(cnt > 0u)
  {
    pulp_nn_u4_to_u8(pIn, inp);
    pulp_nn_u4_to_u8(pCom, com);

    *((v4u *)out) = max4(inp[0], com[0]);

    *((uint8_t*)pIn) = bitins(*out, n_mask, *(out + 1), mask, off);
    pIn++;
    *((uint8_t*)pIn) = bitins(*(out + 2), n_mask, *(out + 3), mask, off);
    pIn++;

    *((v4u *)out) = max4(inp[1], com[1]);

    *((uint8_t*)pIn) = bitins(*out, n_mask, *(out + 1), mask, off);
    pIn++;
    *((uint8_t*)pIn) = bitins(*(out + 2), n_mask, *(out + 3), mask, off);
    pIn++;

    pCom+=4;
    cnt--;
  }

  int left = length & 0x3;
  while (left>0u)
  {
    uint8_t inA0 = (uint8_t) bitextu((unsigned int) *pIn, 4, 0);
    uint8_t inA1 = (uint8_t) bitextu((unsigned int) *pIn, 4, 4);
    uint8_t inB0 = (uint8_t) bitextu((unsigned int) *pCom, 4, 0);
    uint8_t inB1 = (uint8_t) bitextu((unsigned int) *pCom, 4, 4);

    if(inA0<inB0)
      inA0=inB0;

    if(inA1<inB1)
      inA1=inB1;

    *((uint8_t*)pIn) = bitins(inA0, n_mask, inA1, mask, off);

    pIn++;
    pCom++;
    left--;
  }
}

void xpulp_nn_compare_and_replace_if_larger_u4(uint8_t * base,
                                                uint8_t * target,
                                                uint16_t length)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  uint8_t *pIn = base;
  uint8_t *pCom = target;

  int cnt = length >> 2;

  while(cnt > 0u)
  {
    *((uint32_t *)pIn) = max8((uint32_t *)pIn, (uint32_t *)pCom);

    pIn+=4;
    pCom+=4;

    cnt--;
  }

  int left = length & 0x3;
  while (left>0u)
  {
    uint8_t inA0 = (uint8_t) bitextu((unsigned int) *pIn, 4, 0);
    uint8_t inA1 = (uint8_t) bitextu((unsigned int) *pIn, 4, 4);
    uint8_t inB0 = (uint8_t) bitextu((unsigned int) *pCom, 4, 0);
    uint8_t inB1 = (uint8_t) bitextu((unsigned int) *pCom, 4, 4);

    if(inA0<inB0)
      inA0=inB0;

    if(inA1<inB1)
      inA1=inB1;

    *((uint8_t*)pIn) = bitins(inA0, n_mask, inA1, mask, off);

    pIn++;
    pCom++;
    left--;
  }
}

void pulp_nn_avg_and_replace_u4(uint8_t * base,
                                  uint8_t * target,
                                  uint16_t length)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  uint8_t *pIn = base;
  uint8_t *pCom = target;

  while (length>0u)
  {
    uint8_t inA0 = (uint8_t) bitextu((unsigned int) *pIn, 4, 0);
    uint8_t inA1 = (uint8_t) bitextu((unsigned int) *pIn, 4, 4);
    uint8_t inB0 = (uint8_t) bitextu((unsigned int) *pCom, 4, 0);
    uint8_t inB1 = (uint8_t) bitextu((unsigned int) *pCom, 4, 4);

    inA0 = ((inA0 + inB0) >> 1);
    inA1 = ((inA1 + inB1) >> 1);

    *((uint8_t*)pIn) = bitins(inA0, n_mask, inA1, mask, off);

    pIn++;
    pCom++;
    length--;
  }
}

void xpulp_nn_avg_and_replace_u4(uint8_t * base,
                                  uint8_t * target,
                                  uint16_t length)
{
  int8_t mask = 0xf0;
  int8_t n_mask = ~ mask;
  int8_t off = 0x04;

  uint8_t *pIn = base;
  uint8_t *pCom = target;

  int cnt = length >> 2;

  while (cnt > 0u)
  {
    *((uint32_t *)pIn) = avg8((uint32_t *)pIn, (uint32_t *)pCom);

    pIn+=4;
    pCom+=4;
    cnt--;
  }

  int left = length & 0x3;

  while (left>0u)
  {
    uint8_t inA0 = (uint8_t) bitextu((unsigned int) *pIn, 4, 0);
    uint8_t inA1 = (uint8_t) bitextu((unsigned int) *pIn, 4, 4);
    uint8_t inB0 = (uint8_t) bitextu((unsigned int) *pCom, 4, 0);
    uint8_t inB1 = (uint8_t) bitextu((unsigned int) *pCom, 4, 4);

    inA0 = ((inA0 + inB0) >> 1);
    inA1 = ((inA1 + inB1) >> 1);

    *((uint8_t*)pIn) = bitins(inA0, n_mask, inA1, mask, off);

    pIn++;
    pCom++;
    length--;
  }
}

void pulp_nn_compare_and_replace_if_larger_u2(uint8_t * base,
                                                uint8_t * target,
                                                uint16_t length)
{
  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

  uint8_t *pIn = base;
  uint8_t *pCom = target;
  v4u inp[4];
  v4u com[4];
  uint8_t *out;
  int cnt = length >> 2;

  while(cnt > 0u)
  {
    pulp_nn_u2_to_u8(pIn, inp);
    pulp_nn_u2_to_u8(pCom, com);

    *((v4u*)out) = max4(inp[0], com[0]);

    uint8_t inA = (uint8_t) bitins(*out, n_mask2, *(out + 1), mask2, off2);
    inA = bitins(inA, n_mask4, *(out + 2), mask4, off4);
    *((uint8_t*)pIn) = bitins(inA, n_mask6, *(out + 3), mask6, off6);
    pIn++;

    *((v4u*)out) = max4(inp[1], com[1]);

    inA = (uint8_t) bitins(*out, n_mask2, *(out + 1), mask2, off2);
    inA = bitins(inA, n_mask4, *(out + 2), mask4, off4);
    *((uint8_t*)pIn) = bitins(inA, n_mask6, *(out + 3), mask6, off6);
    pIn++;

    *((v4u*)out) = max4(inp[2], com[2]);

    inA = (uint8_t) bitins(*out, n_mask2, *(out + 1), mask2, off2);
    inA = bitins(inA, n_mask4, *(out + 2), mask4, off4);
    *((uint8_t*)pIn) = bitins(inA, n_mask6, *(out + 3), mask6, off6);
    pIn++;

    *((v4u*)out) = max4(inp[3], com[3]);

    inA = (uint8_t) bitins(*out, n_mask2, *(out + 1), mask2, off2);
    inA = bitins(inA, n_mask4, *(out + 2), mask4, off4);
    *((uint8_t*)pIn) = bitins(inA, n_mask6, *(out + 3), mask6, off6);
    pIn++;

    pCom+=4;
    cnt--;
  }

  int left = length & 0x3;
  while (left>0u)
  {
    uint8_t inA0 = (uint8_t) bitextu((unsigned int) *pIn, 2, 0);
    uint8_t inA1 = (uint8_t) bitextu((unsigned int) *pIn, 2, 2);
    uint8_t inA2 = (uint8_t) bitextu((unsigned int) *pIn, 2, 4);
    uint8_t inA3 = (uint8_t) bitextu((unsigned int) *pIn, 2, 6);
    v4u inA4 = pack((uint8_t) inA0, (uint8_t) inA1, (uint8_t) inA2, (uint8_t) inA3);
    uint8_t inB0 = (uint8_t) bitextu((unsigned int) *pCom, 2, 0);
    uint8_t inB1 = (uint8_t) bitextu((unsigned int) *pCom, 2, 2);
    uint8_t inB2 = (uint8_t) bitextu((unsigned int) *pCom, 2, 4);
    uint8_t inB3 = (uint8_t) bitextu((unsigned int) *pCom, 2, 6);
    v4u inB4 = pack((uint8_t) inB0, (uint8_t) inB1, (uint8_t) inB2, (uint8_t) inB3);

    *((v4u*)out) = max4(inA4, inB4);

    uint8_t inA = (uint8_t) bitins(*out, n_mask2, *(out + 1), mask2, off2);
    inA = bitins(inA, n_mask4, *(out + 2), mask4, off4);
    *((uint8_t*)pIn) = bitins(inA, n_mask6, *(out + 3), mask6, off6);

    pIn++;
    pCom++;
    left--;
  }
}

void xpulp_nn_compare_and_replace_if_larger_u2(uint8_t * base,
                                                uint8_t * target,
                                                uint16_t length)
{
  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

  uint8_t *pIn = base;
  uint8_t *pCom = target;
  uint8_t *out;

  int cnt = length >> 2;

  while(cnt > 0u)
  {
    *((uint32_t *)out) = max16(pIn, pCom);

    pIn+=4;
    pCom+=4;
    cnt--;
  }

  int left = length & 0x3;
  while (left>0u)
  {
    uint8_t inA0 = (uint8_t) bitextu((unsigned int) *pIn, 2, 0);
    uint8_t inA1 = (uint8_t) bitextu((unsigned int) *pIn, 2, 2);
    uint8_t inA2 = (uint8_t) bitextu((unsigned int) *pIn, 2, 4);
    uint8_t inA3 = (uint8_t) bitextu((unsigned int) *pIn, 2, 6);
    v4u inA4 = pack((uint8_t) inA0, (uint8_t) inA1, (uint8_t) inA2, (uint8_t) inA3);
    uint8_t inB0 = (uint8_t) bitextu((unsigned int) *pCom, 2, 0);
    uint8_t inB1 = (uint8_t) bitextu((unsigned int) *pCom, 2, 2);
    uint8_t inB2 = (uint8_t) bitextu((unsigned int) *pCom, 2, 4);
    uint8_t inB3 = (uint8_t) bitextu((unsigned int) *pCom, 2, 6);
    v4u inB4 = pack((uint8_t) inB0, (uint8_t) inB1, (uint8_t) inB2, (uint8_t) inB3);

    *((v4u*)out) = max4(inA4, inB4);

    uint8_t inA = (uint8_t) bitins(*out, n_mask2, *(out + 1), mask2, off2);
    inA = bitins(inA, n_mask4, *(out + 2), mask4, off4);
    *((uint8_t*)pIn) = bitins(inA, n_mask6, *(out + 3), mask6, off6);

    pIn++;
    pCom++;
    left--;
  }
}

void pulp_nn_avg_and_replace_u2(uint8_t * base,
                                  uint8_t * target,
                                  uint16_t length)
{
  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

  uint8_t *pIn = base;
  uint8_t *pCom = target;

  while (length>0u)
  {
    uint8_t inA0 = (uint8_t) bitextu((unsigned int) *pIn, 2, 0);
    uint8_t inA1 = (uint8_t) bitextu((unsigned int) *pIn, 2, 2);
    uint8_t inA2 = (uint8_t) bitextu((unsigned int) *pIn, 2, 4);
    uint8_t inA3 = (uint8_t) bitextu((unsigned int) *pIn, 2, 6);

    uint8_t inB0 = (uint8_t) bitextu((unsigned int) *pCom, 2, 0);
    uint8_t inB1 = (uint8_t) bitextu((unsigned int) *pCom, 2, 2);
    uint8_t inB2 = (uint8_t) bitextu((unsigned int) *pCom, 2, 4);
    uint8_t inB3 = (uint8_t) bitextu((unsigned int) *pCom, 2, 6);

    inA0 = ((inA0 + inB0) >> 1);
    inA1 = ((inA1 + inB1) >> 1);
    inA2 = ((inA2 + inB2) >> 1);
    inA3 = ((inA3 + inB3) >> 1);

    uint8_t inA = (uint8_t) bitins(inA0, n_mask2, inA1, mask2, off2);
    inA = bitins(inA, n_mask4, inA2, mask4, off4);
    *((uint8_t*)pIn) = bitins(inA, n_mask6, inA3, mask6, off6);

    pIn++;
    pCom++;
    length--;
  }
}

void xpulp_nn_avg_and_replace_u2(uint8_t * base,
                                  uint8_t * target,
                                  uint16_t length)
{
  int8_t mask2 = 0x0c;
  int8_t n_mask2 = ~ mask2;
  int8_t mask4 = 0x30;
  int8_t n_mask4 = ~ mask4;
  int8_t mask6 = 0xc0;
  int8_t n_mask6 = ~ mask6;
  int8_t off2 = 2;
  int8_t off4 = 4;
  int8_t off6 = 6;

  uint8_t *pIn = base;
  uint8_t *pCom = target;
  uint8_t *out;

  int cnt = length >> 2;

  while(cnt > 0u)
  {
    *((uint32_t *)pIn) = avg16((uint32_t *)pIn, (uint32_t *)pCom);

    pIn+=4;
    pCom+=4;
    cnt--;
  }

  int left = length & 0x3;
  while (left>0u)
  {
    uint8_t inA0 = (uint8_t) bitextu((unsigned int) *pIn, 2, 0);
    uint8_t inA1 = (uint8_t) bitextu((unsigned int) *pIn, 2, 2);
    uint8_t inA2 = (uint8_t) bitextu((unsigned int) *pIn, 2, 4);
    uint8_t inA3 = (uint8_t) bitextu((unsigned int) *pIn, 2, 6);
    v4u inA4 = pack((uint8_t) inA0, (uint8_t) inA1, (uint8_t) inA2, (uint8_t) inA3);
    uint8_t inB0 = (uint8_t) bitextu((unsigned int) *pCom, 2, 0);
    uint8_t inB1 = (uint8_t) bitextu((unsigned int) *pCom, 2, 2);
    uint8_t inB2 = (uint8_t) bitextu((unsigned int) *pCom, 2, 4);
    uint8_t inB3 = (uint8_t) bitextu((unsigned int) *pCom, 2, 6);
    v4u inB4 = pack((uint8_t) inB0, (uint8_t) inB1, (uint8_t) inB2, (uint8_t) inB3);

    *((v4u*)out) = avg4(inA4, inB4);

    uint8_t inA = (uint8_t) bitins(*out, n_mask2, *(out + 1), mask2, off2);
    inA = bitins(inA, n_mask4, *(out + 2), mask4, off4);
    *((uint8_t*)pIn) = bitins(inA, n_mask6, *(out + 3), mask6, off6);

    pIn++;
    pCom++;
    left--;
  }
}

int8_t __attribute__ ((always_inline)) pulp_nn_i4_quant(int input, int16_t * pThr)
{
	if(input <= pThr[7] )
	{
		if( input <= pThr[3])
		{
			if( input <= pThr[1])
			{
				if( input <= pThr[0])
					return -8;
				else
					return -7;
			}
			else
			{
				if( input <= pThr[2])
					return -6;
				else
					return -5;
			}
		}
		else
		{
			if( input <= pThr[5])
			{
				if( input <= pThr[4])
					return -4;
				else
					return -3;
			}
			else
			{
				if( input <= pThr[6])
					return -2;
				else
					return -1;
			}
		}
	}
	else
	{
		if( input <= pThr[11])
		{
			if( input <= pThr[9])
			{
				if( input <= pThr[8])
					return 0;
				else
					return 1;
			}
			else
			{
				if( input <= pThr[10])
					return 2;
				else
					return 3;
			}
		}
		else
		{
			if( input <= pThr[13])
			{
				if( input <= pThr[12])
					return 4;
				else
					return 5;
			}
			else
			{
				if( input <= pThr[14])
					return 6;
				else
					return 7;
			}
		}
	}
}

int8_t __attribute__ ((always_inline)) pulp_nn_i2_quant(int input, int16_t * pThr)
{
	if( input <= pThr[1])
  {
		if( input <= pThr[0])
        {
			return -2;
		}
        else
        {
			return -1;
		}
	}
    else
    {
		if( input <= pThr[2])
        {
			return 0;
		}
        else
        {
			return 1;
		}
	}
}
