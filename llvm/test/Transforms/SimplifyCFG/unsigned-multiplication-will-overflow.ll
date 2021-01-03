; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -simplifycfg -S | FileCheck %s
; RUN: opt < %s -simplifycfg -phi-node-folding-threshold=2 -S | FileCheck %s
; RUN: opt < %s -simplifycfg -phi-node-folding-threshold=3 -S | FileCheck %s

; This is checking that the multiplication does overflow, with a leftover
; guard against division-by-zero that was needed before InstCombine
; produced llvm.umul.with.overflow.

define i1 @will_overflow(i64 %size, i64 %nmemb) {
; CHECK-LABEL: @will_overflow(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i64 [[SIZE:%.*]], 0
; CHECK-NEXT:    [[UMUL:%.*]] = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 [[SIZE]], i64 [[NMEMB:%.*]])
; CHECK-NEXT:    [[UMUL_OV:%.*]] = extractvalue { i64, i1 } [[UMUL]], 1
; CHECK-NEXT:    [[UMUL_NOT_OV:%.*]] = xor i1 [[UMUL_OV]], true
; CHECK-NEXT:    [[TMP0:%.*]] = select i1 [[CMP]], i1 true, i1 [[UMUL_NOT_OV]]
; CHECK-NEXT:    ret i1 [[TMP0]]
;
entry:
  %cmp = icmp eq i64 %size, 0
  br i1 %cmp, label %land.end, label %land.rhs

land.rhs:                                         ; preds = %entry
  %umul = tail call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %size, i64 %nmemb)
  %umul.ov = extractvalue { i64, i1 } %umul, 1
  %umul.not.ov = xor i1 %umul.ov, true
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  %0 = phi i1 [ true, %entry ], [ %umul.not.ov, %land.rhs ]
  ret i1 %0
}

; Function Attrs: nounwind readnone speculatable
declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64) #0
