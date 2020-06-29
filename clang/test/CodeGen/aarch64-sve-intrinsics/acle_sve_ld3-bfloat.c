// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -D__ARM_FEATURE_SVE_BF16 -D__ARM_FEATURE_BF16_SCALAR_ARITHMETIC -DSVE_OVERLOADED_FORMS -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +bf16 -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s

#include <arm_sve.h>

#ifdef SVE_OVERLOADED_FORMS
// A simple used,unused... macro, long enough to represent any SVE builtin.
#define SVE_ACLE_FUNC(A1,A2_UNUSED,A3,A4_UNUSED) A1##A3
#else
#define SVE_ACLE_FUNC(A1,A2,A3,A4) A1##A2##A3##A4
#endif

svbfloat16x3_t test_svld3_bf16(svbool_t pg, const bfloat16_t *base)
{
  // CHECK-LABEL: test_svld3_bf16
  // CHECK: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK: %[[LOAD:.*]] = call <vscale x 24 x bfloat> @llvm.aarch64.sve.ld3.nxv24bf16.nxv8i1(<vscale x 8 x i1> %[[PG]], bfloat* %base)
  // CHECK-NEXT: ret <vscale x 24 x bfloat> %[[LOAD]]
  return SVE_ACLE_FUNC(svld3,_bf16,,)(pg, base);
}

svbfloat16x3_t test_svld3_vnum_bf16(svbool_t pg, const bfloat16_t *base, int64_t vnum)
{
  // CHECK-LABEL: test_svld3_vnum_bf16
  // CHECK-DAG: %[[PG:.*]] = call <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1> %pg)
  // CHECK-DAG: %[[BASE:.*]] = bitcast bfloat* %base to <vscale x 8 x bfloat>*
  // CHECK-DAG: %[[GEP:.*]] = getelementptr <vscale x 8 x bfloat>, <vscale x 8 x bfloat>* %[[BASE]], i64 %vnum, i64 0
  // CHECK: %[[LOAD:.*]] = call <vscale x 24 x bfloat> @llvm.aarch64.sve.ld3.nxv24bf16.nxv8i1(<vscale x 8 x i1> %[[PG]], bfloat* %[[GEP]])
  // CHECK-NEXT: ret <vscale x 24 x bfloat> %[[LOAD]]
  return SVE_ACLE_FUNC(svld3_vnum,_bf16,,)(pg, base, vnum);
}
