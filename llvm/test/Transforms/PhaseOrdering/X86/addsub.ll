; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -O3 -S                                        | FileCheck %s
; RUN: opt < %s -passes='default<O3>' -aa-pipeline=default -S | FileCheck %s

target triple = "x86_64--"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Ideally, this should reach the backend with 1 fsub, 1 fadd, and 1 shuffle.
; That may require some coordination between VectorCombine, SLP, and other passes.
; The end goal is to get a single "vaddsubps" instruction for x86 with AVX.

define <4 x float> @PR45015(<4 x float> %arg, <4 x float> %arg1) {
; CHECK-LABEL: @PR45015(
; CHECK-NEXT:    [[TMP1:%.*]] = fsub <4 x float> [[ARG:%.*]], [[ARG1:%.*]]
; CHECK-NEXT:    [[TMP2:%.*]] = fadd <4 x float> [[ARG]], [[ARG1]]
; CHECK-NEXT:    [[T16:%.*]] = shufflevector <4 x float> [[TMP1]], <4 x float> [[TMP2]], <4 x i32> <i32 0, i32 5, i32 2, i32 7>
; CHECK-NEXT:    ret <4 x float> [[T16]]
;
  %t = extractelement <4 x float> %arg, i32 0
  %t2 = extractelement <4 x float> %arg1, i32 0
  %t3 = fsub float %t, %t2
  %t4 = insertelement <4 x float> undef, float %t3, i32 0
  %t5 = extractelement <4 x float> %arg, i32 1
  %t6 = extractelement <4 x float> %arg1, i32 1
  %t7 = fadd float %t5, %t6
  %t8 = insertelement <4 x float> %t4, float %t7, i32 1
  %t9 = extractelement <4 x float> %arg, i32 2
  %t10 = extractelement <4 x float> %arg1, i32 2
  %t11 = fsub float %t9, %t10
  %t12 = insertelement <4 x float> %t8, float %t11, i32 2
  %t13 = extractelement <4 x float> %arg, i32 3
  %t14 = extractelement <4 x float> %arg1, i32 3
  %t15 = fadd float %t13, %t14
  %t16 = insertelement <4 x float> %t12, float %t15, i32 3
  ret <4 x float> %t16
}

; PR42022 - https://bugs.llvm.org/show_bug.cgi?id=42022

%struct.Vector4 = type { float, float, float, float }

define { <2 x float>, <2 x float> } @add_aggregate(<2 x float> %a0, <2 x float> %a1, <2 x float> %b0, <2 x float> %b1) {
; CHECK-LABEL: @add_aggregate(
; CHECK-NEXT:    [[TMP1:%.*]] = fadd <2 x float> [[A0:%.*]], [[B0:%.*]]
; CHECK-NEXT:    [[TMP2:%.*]] = fadd <2 x float> [[A1:%.*]], [[B1:%.*]]
; CHECK-NEXT:    [[FCA_0_INSERT:%.*]] = insertvalue { <2 x float>, <2 x float> } undef, <2 x float> [[TMP1]], 0
; CHECK-NEXT:    [[FCA_1_INSERT:%.*]] = insertvalue { <2 x float>, <2 x float> } [[FCA_0_INSERT]], <2 x float> [[TMP2]], 1
; CHECK-NEXT:    ret { <2 x float>, <2 x float> } [[FCA_1_INSERT]]
;
  %a00 = extractelement <2 x float> %a0, i32 0
  %b00 = extractelement <2 x float> %b0, i32 0
  %add = fadd float %a00, %b00
  %retval.0.0.insert = insertelement <2 x float> undef, float %add, i32 0
  %a01 = extractelement <2 x float> %a0, i32 1
  %b01 = extractelement <2 x float> %b0, i32 1
  %add4 = fadd float %a01, %b01
  %retval.0.1.insert = insertelement <2 x float> %retval.0.0.insert, float %add4, i32 1
  %a10 = extractelement <2 x float> %a1, i32 0
  %b10 = extractelement <2 x float> %b1, i32 0
  %add7 = fadd float %a10, %b10
  %retval.1.0.insert = insertelement <2 x float> undef, float %add7, i32 0
  %a11 = extractelement <2 x float> %a1, i32 1
  %b11 = extractelement <2 x float> %b1, i32 1
  %add10 = fadd float %a11, %b11
  %retval.1.1.insert = insertelement <2 x float> %retval.1.0.insert, float %add10, i32 1
  %fca.0.insert = insertvalue { <2 x float>, <2 x float> } undef, <2 x float> %retval.0.1.insert, 0
  %fca.1.insert = insertvalue { <2 x float>, <2 x float> } %fca.0.insert, <2 x float> %retval.1.1.insert, 1
  ret { <2 x float>, <2 x float> } %fca.1.insert
}

define void @add_aggregate_store(<2 x float> %a0, <2 x float> %a1, <2 x float> %b0, <2 x float> %b1, %struct.Vector4* nocapture dereferenceable(16) %r) {
; CHECK-LABEL: @add_aggregate_store(
; CHECK-NEXT:    [[TMP1:%.*]] = fadd <2 x float> [[A0:%.*]], [[B0:%.*]]
; CHECK-NEXT:    [[TMP2:%.*]] = fadd <2 x float> [[A1:%.*]], [[B1:%.*]]
; CHECK-NEXT:    [[TMP3:%.*]] = shufflevector <2 x float> [[TMP1]], <2 x float> [[TMP2]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT:    [[TMP4:%.*]] = bitcast %struct.Vector4* [[R:%.*]] to <4 x float>*
; CHECK-NEXT:    store <4 x float> [[TMP3]], <4 x float>* [[TMP4]], align 4
; CHECK-NEXT:    ret void
;
  %a00 = extractelement <2 x float> %a0, i32 0
  %b00 = extractelement <2 x float> %b0, i32 0
  %add = fadd float %a00, %b00
  %r0 = getelementptr inbounds %struct.Vector4, %struct.Vector4* %r, i64 0, i32 0
  store float %add, float* %r0, align 4
  %a01 = extractelement <2 x float> %a0, i32 1
  %b01 = extractelement <2 x float> %b0, i32 1
  %add4 = fadd float %a01, %b01
  %r1 = getelementptr inbounds %struct.Vector4, %struct.Vector4* %r, i64 0, i32 1
  store float %add4, float* %r1, align 4
  %a10 = extractelement <2 x float> %a1, i32 0
  %b10 = extractelement <2 x float> %b1, i32 0
  %add7 = fadd float %a10, %b10
  %r2 = getelementptr inbounds %struct.Vector4, %struct.Vector4* %r, i64 0, i32 2
  store float %add7, float* %r2, align 4
  %a11 = extractelement <2 x float> %a1, i32 1
  %b11 = extractelement <2 x float> %b1, i32 1
  %add10 = fadd float %a11, %b11
  %r3 = getelementptr inbounds %struct.Vector4, %struct.Vector4* %r, i64 0, i32 3
  store float %add10, float* %r3, align 4
  ret void
}
