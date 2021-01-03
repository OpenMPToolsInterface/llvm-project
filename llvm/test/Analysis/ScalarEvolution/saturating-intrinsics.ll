; NOTE: Assertions have been autogenerated by utils/update_analyze_test_checks.py
; RUN: opt -analyze -enable-new-pm=0 -scalar-evolution < %s | FileCheck %s
; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

declare i32 @llvm.uadd.sat.i32(i32, i32)
declare i32 @llvm.sadd.sat.i32(i32, i32)
declare i32 @llvm.usub.sat.i32(i32, i32)
declare i32 @llvm.ssub.sat.i32(i32, i32)
declare i32 @llvm.ushl.sat.i32(i32, i32)
declare i32 @llvm.sshl.sat.i32(i32, i32)

define i32 @uadd_sat(i32 %x, i32 %y) {
; CHECK-LABEL: 'uadd_sat'
; CHECK-NEXT:  Classifying expressions for: @uadd_sat
; CHECK-NEXT:    %z = call i32 @llvm.uadd.sat.i32(i32 %x, i32 %y)
; CHECK-NEXT:    --> (((-1 + (-1 * %y)) umin %x) + %y)<nuw> U: full-set S: full-set
; CHECK-NEXT:  Determining loop execution counts for: @uadd_sat
;
  %z = call i32 @llvm.uadd.sat.i32(i32 %x, i32 %y)
  ret i32 %z
}

define i32 @sadd_sat(i32 %x, i32 %y) {
; CHECK-LABEL: 'sadd_sat'
; CHECK-NEXT:  Classifying expressions for: @sadd_sat
; CHECK-NEXT:    %z = call i32 @llvm.sadd.sat.i32(i32 %x, i32 %y)
; CHECK-NEXT:    --> %z U: full-set S: full-set
; CHECK-NEXT:  Determining loop execution counts for: @sadd_sat
;
  %z = call i32 @llvm.sadd.sat.i32(i32 %x, i32 %y)
  ret i32 %z
}

define i32 @usub_sat(i32 %x, i32 %y) {
; CHECK-LABEL: 'usub_sat'
; CHECK-NEXT:  Classifying expressions for: @usub_sat
; CHECK-NEXT:    %z = call i32 @llvm.usub.sat.i32(i32 %x, i32 %y)
; CHECK-NEXT:    --> ((-1 * (%x umin %y)) + %x) U: full-set S: full-set
; CHECK-NEXT:  Determining loop execution counts for: @usub_sat
;
  %z = call i32 @llvm.usub.sat.i32(i32 %x, i32 %y)
  ret i32 %z
}

define i32 @ssub_sat(i32 %x, i32 %y) {
; CHECK-LABEL: 'ssub_sat'
; CHECK-NEXT:  Classifying expressions for: @ssub_sat
; CHECK-NEXT:    %z = call i32 @llvm.ssub.sat.i32(i32 %x, i32 %y)
; CHECK-NEXT:    --> %z U: full-set S: full-set
; CHECK-NEXT:  Determining loop execution counts for: @ssub_sat
;
  %z = call i32 @llvm.ssub.sat.i32(i32 %x, i32 %y)
  ret i32 %z
}

define i32 @ushl_sat(i32 %x, i32 %y) {
; CHECK-LABEL: 'ushl_sat'
; CHECK-NEXT:  Classifying expressions for: @ushl_sat
; CHECK-NEXT:    %z = call i32 @llvm.ushl.sat.i32(i32 %x, i32 %y)
; CHECK-NEXT:    --> %z U: full-set S: full-set
; CHECK-NEXT:  Determining loop execution counts for: @ushl_sat
;
  %z = call i32 @llvm.ushl.sat.i32(i32 %x, i32 %y)
  ret i32 %z
}

define i32 @sshl_sat(i32 %x, i32 %y) {
; CHECK-LABEL: 'sshl_sat'
; CHECK-NEXT:  Classifying expressions for: @sshl_sat
; CHECK-NEXT:    %z = call i32 @llvm.sshl.sat.i32(i32 %x, i32 %y)
; CHECK-NEXT:    --> %z U: full-set S: full-set
; CHECK-NEXT:  Determining loop execution counts for: @sshl_sat
;
  %z = call i32 @llvm.sshl.sat.i32(i32 %x, i32 %y)
  ret i32 %z
}
