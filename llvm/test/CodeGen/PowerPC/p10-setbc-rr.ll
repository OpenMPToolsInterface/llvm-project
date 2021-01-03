; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -O2 \
; RUN:     -ppc-asm-full-reg-names -mcpu=pwr10 < %s | FileCheck %s \
; RUN:     --check-prefixes=CHECK,CHECK-LE
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -O2 \
; RUN:     -ppc-asm-full-reg-names -mcpu=pwr10 < %s | FileCheck %s \
; RUN:     --check-prefixes=CHECK,CHECK-BE

; This file does not contain many test cases involving comparisons and logical
; comparisons (cmplwi, cmpldi). This is because alternative code is generated
; when there is a compare (logical or not), followed by a sign or zero extend.
; This codegen will be re-evaluated at a later time on whether or not it should
; be emitted on P10.

@globalVal = common local_unnamed_addr global i8 0, align 1
@globalVal2 = common local_unnamed_addr global i32 0, align 4
@globalVal3 = common local_unnamed_addr global i64 0, align 8
@globalVal4 = common local_unnamed_addr global i16 0, align 2

define signext i32 @setbc1(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: setbc1:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, lt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp slt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define signext i32 @setbc2(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: setbc2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define signext i32 @setbc3(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: setbc3:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, gt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define signext i32 @setbc4(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: setbc4:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i8 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}

define void @setbc5(i8 signext %a, i8 signext %b) {
; CHECK-LE-LABEL: setbc5:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstb r3, globalVal@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc5:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    stb r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @globalVal, align 1
  ret void
}

define void @setbc6(i32 signext %a, i32 signext %b) {
; CHECK-LE-LABEL: setbc6:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstw r3, globalVal2@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc6:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC1@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC1@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    stw r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @globalVal2, align 4
  ret void
}

define signext i32 @setbc7(i64 %a, i64 %b) {
; CHECK-LABEL: setbc7:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpd r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i64 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

define signext i64 @setbc8(i64 %a, i64 %b) {
; CHECK-LABEL: setbc8:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpd r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i64 %a, %b
  %conv = zext i1 %cmp to i64
  ret i64 %conv
}


define void @setbc9(i64 %a, i64 %b) {
; CHECK-LE-LABEL: setbc9:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpd r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstd r3, globalVal3@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc9:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpd r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC2@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC2@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    std r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  store i64 %conv1, i64* @globalVal3, align 8
  ret void
}


define signext i32 @setbc10(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: setbc10:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i16 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}


define void @setbc11(i16 signext %a, i16 signext %b) {
; CHECK-LE-LABEL: setbc11:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    psth r3, globalVal4@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc11:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC3@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC3@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    sth r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i16 %a, %b
  %conv3 = zext i1 %cmp to i16
  store i16 %conv3, i16* @globalVal4, align 2
  ret void
}


define signext i32 @setbc12(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: setbc12:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i8 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}


define void @setbc13(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LE-LABEL: setbc13:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstb r3, globalVal@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc13:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    stb r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @globalVal, align 1
  ret void
}


define signext i32 @setbc14(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: setbc14:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}


define void @setbc15(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LE-LABEL: setbc15:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstw r3, globalVal2@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc15:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC1@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC1@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    stw r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @globalVal2, align 4
  ret void
}


define signext i32 @setbc16(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: setbc16:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i16 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}


define void @setbc17(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LE-LABEL: setbc17:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    psth r3, globalVal4@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc17:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC3@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC3@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    sth r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i16 %a, %b
  %conv3 = zext i1 %cmp to i16
  store i16 %conv3, i16* @globalVal4, align 2
  ret void
}


define signext i32 @setbc18(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: setbc18:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, gt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i8 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}


define void @setbc19(i8 signext %a, i8 signext %b) {
; CHECK-LE-LABEL: setbc19:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, gt
; CHECK-LE-NEXT:    pstb r3, globalVal@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc19:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, gt
; CHECK-BE-NEXT:    stb r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp sgt i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @globalVal, align 1
  ret void
}


define void @setbc20(i32 signext %a, i32 signext %b) {
; CHECK-LE-LABEL: setbc20:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, gt
; CHECK-LE-NEXT:    pstw r3, globalVal2@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc20:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC1@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC1@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, gt
; CHECK-BE-NEXT:    stw r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp sgt i32 %a, %b
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @globalVal2, align 4
  ret void
}


define signext i32 @setbc21(i64 %a, i64 %b) {
; CHECK-LABEL: setbc21:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpd r3, r4
; CHECK-NEXT:    setbc r3, gt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i64 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}


define void @setbc22(i64 %a, i64 %b) {
; CHECK-LE-LABEL: setbc22:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpd r3, r4
; CHECK-LE-NEXT:    setbc r3, gt
; CHECK-LE-NEXT:    pstd r3, globalVal3@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc22:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpd r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC2@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC2@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, gt
; CHECK-BE-NEXT:    std r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp sgt i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  store i64 %conv1, i64* @globalVal3, align 8
  ret void
}


define signext i32 @setbc23(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: setbc23:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, gt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i16 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}


define void @setbc24(i16 signext %a, i16 signext %b) {
; CHECK-LE-LABEL: setbc24:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, gt
; CHECK-LE-NEXT:    psth r3, globalVal4@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc24:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC3@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC3@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, gt
; CHECK-BE-NEXT:    sth r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp sgt i16 %a, %b
  %conv3 = zext i1 %cmp to i16
  store i16 %conv3, i16* @globalVal4, align 2
  ret void
}


define signext i32 @setbc25(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: setbc25:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, lt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp slt i8 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}


define void @setbc26(i8 signext %a, i8 signext %b) {
; CHECK-LE-LABEL: setbc26:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, lt
; CHECK-LE-NEXT:    pstb r3, globalVal@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc26:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, lt
; CHECK-BE-NEXT:    stb r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp slt i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @globalVal, align 1
  ret void
}


define void @setbc27(i32 signext %a, i32 signext %b) {
; CHECK-LE-LABEL: setbc27:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, lt
; CHECK-LE-NEXT:    pstw r3, globalVal2@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc27:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC1@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC1@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, lt
; CHECK-BE-NEXT:    stw r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp slt i32 %a, %b
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @globalVal2, align 4
  ret void
}


define signext i32 @setbc28(i64 %a, i64 %b) {
; CHECK-LABEL: setbc28:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpd r3, r4
; CHECK-NEXT:    setbc r3, lt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp slt i64 %a, %b
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}


define signext i64 @setbc29(i64 %a, i64 %b) {
; CHECK-LABEL: setbc29:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpd r3, r4
; CHECK-NEXT:    setbc r3, lt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp slt i64 %a, %b
  %conv = zext i1 %cmp to i64
  ret i64 %conv
}


define void @setbc30(i64 %a, i64 %b) {
; CHECK-LE-LABEL: setbc30:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpd r3, r4
; CHECK-LE-NEXT:    setbc r3, lt
; CHECK-LE-NEXT:    pstd r3, globalVal3@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc30:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpd r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC2@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC2@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, lt
; CHECK-BE-NEXT:    std r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp slt i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  store i64 %conv1, i64* @globalVal3, align 8
  ret void
}


define signext i32 @setbc31(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: setbc31:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, lt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp slt i16 %a, %b
  %conv2 = zext i1 %cmp to i32
  ret i32 %conv2
}


define void @setbc32(i16 signext %a, i16 signext %b) {
; CHECK-LE-LABEL: setbc32:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, lt
; CHECK-LE-NEXT:    psth r3, globalVal4@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc32:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC3@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC3@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, lt
; CHECK-BE-NEXT:    sth r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp slt i16 %a, %b
  %conv3 = zext i1 %cmp to i16
  store i16 %conv3, i16* @globalVal4, align 2
  ret void
}


define i64 @setbc33(i8 signext %a, i8 signext %b) {
; CHECK-LABEL: setbc33:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i8 %a, %b
  %conv3 = zext i1 %cmp to i64
  ret i64 %conv3
}


define void @setbc34(i8 signext %a, i8 signext %b) {
; CHECK-LE-LABEL: setbc34:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstb r3, globalVal@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc34:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    stb r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @globalVal, align 1
  ret void
}


define i64 @setbc35(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: setbc35:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i32 %a, %b
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}


define void @setbc36(i32 signext %a, i32 signext %b) {
; CHECK-LE-LABEL: setbc36:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstw r3, globalVal2@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc36:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC1@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC1@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    stw r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @globalVal2, align 4
  ret void
}


define void @setbc37(i64 %a, i64 %b) {
; CHECK-LE-LABEL: setbc37:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpd r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstd r3, globalVal3@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc37:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpd r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC2@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC2@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    std r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  store i64 %conv1, i64* @globalVal3, align 8
  ret void
}


define i64 @setbc38(i16 signext %a, i16 signext %b) {
; CHECK-LABEL: setbc38:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i16 %a, %b
  %conv3 = zext i1 %cmp to i64
  ret i64 %conv3
}


define void @setbc39(i16 signext %a, i16 signext %b) {
; CHECK-LE-LABEL: setbc39:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    psth r3, globalVal4@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc39:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC3@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC3@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    sth r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i16 %a, %b
  %conv3 = zext i1 %cmp to i16
  store i16 %conv3, i16* @globalVal4, align 2
  ret void
}


define i64 @setbc40(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LABEL: setbc40:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i8 %a, %b
  %conv3 = zext i1 %cmp to i64
  ret i64 %conv3
}


define void @setbc41(i8 zeroext %a, i8 zeroext %b) {
; CHECK-LE-LABEL: setbc41:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstb r3, globalVal@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc41:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    stb r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i8 %a, %b
  %conv3 = zext i1 %cmp to i8
  store i8 %conv3, i8* @globalVal, align 1
  ret void
}


define i64 @setbc42(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: setbc42:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i32 %a, %b
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}


define void @setbc43(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LE-LABEL: setbc43:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstw r3, globalVal2@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc43:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC1@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC1@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    stw r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i32 %a, %b
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @globalVal2, align 4
  ret void
}


define i64 @setbc44(i64 %a, i64 %b) {
; CHECK-LABEL: setbc44:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpd r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}


define void @setbc45(i64 %a, i64 %b) {
; CHECK-LE-LABEL: setbc45:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpd r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    pstd r3, globalVal3@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc45:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpd r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC2@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC2@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    std r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  store i64 %conv1, i64* @globalVal3, align 8
  ret void
}


define i64 @setbc46(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LABEL: setbc46:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpw r3, r4
; CHECK-NEXT:    setbc r3, eq
; CHECK-NEXT:    blr
entry:
  %cmp = icmp eq i16 %a, %b
  %conv3 = zext i1 %cmp to i64
  ret i64 %conv3
}


define void @setbc47(i16 zeroext %a, i16 zeroext %b) {
; CHECK-LE-LABEL: setbc47:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpw r3, r4
; CHECK-LE-NEXT:    setbc r3, eq
; CHECK-LE-NEXT:    psth r3, globalVal4@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc47:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpw r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC3@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC3@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, eq
; CHECK-BE-NEXT:    sth r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp eq i16 %a, %b
  %conv3 = zext i1 %cmp to i16
  store i16 %conv3, i16* @globalVal4, align 2
  ret void
}


define i64 @setbc48(i64 %a, i64 %b) {
; CHECK-LABEL: setbc48:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpd r3, r4
; CHECK-NEXT:    setbc r3, gt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp sgt i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}


define void @setbc49(i64 %a, i64 %b) {
; CHECK-LE-LABEL: setbc49:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpd r3, r4
; CHECK-LE-NEXT:    setbc r3, gt
; CHECK-LE-NEXT:    pstd r3, globalVal3@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setbc49:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpd r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC2@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC2@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, gt
; CHECK-BE-NEXT:    std r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp sgt i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  store i64 %conv1, i64* @globalVal3, align 8
  ret void
}


define i64 @setbc50(i64 %a, i64 %b) {
; CHECK-LABEL: setbc50:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    cmpd r3, r4
; CHECK-NEXT:    setbc r3, lt
; CHECK-NEXT:    blr
entry:
  %cmp = icmp slt i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  ret i64 %conv1
}


define void @setnbc51(i64 %a, i64 %b) {
; CHECK-LE-LABEL: setnbc51:
; CHECK-LE:       # %bb.0: # %entry
; CHECK-LE-NEXT:    cmpd r3, r4
; CHECK-LE-NEXT:    setbc r3, lt
; CHECK-LE-NEXT:    pstd r3, globalVal3@PCREL(0), 1
; CHECK-LE-NEXT:    blr
;
; CHECK-BE-LABEL: setnbc51:
; CHECK-BE:       # %bb.0: # %entry
; CHECK-BE-NEXT:    cmpd r3, r4
; CHECK-BE-NEXT:    addis r4, r2, .LC2@toc@ha
; CHECK-BE-NEXT:    ld r4, .LC2@toc@l(r4)
; CHECK-BE-NEXT:    setbc r3, lt
; CHECK-BE-NEXT:    std r3, 0(r4)
; CHECK-BE-NEXT:    blr
entry:
  %cmp = icmp slt i64 %a, %b
  %conv1 = zext i1 %cmp to i64
  store i64 %conv1, i64* @globalVal3, align 8
  ret void
}
