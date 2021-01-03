; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S < %s -loop-rotate -loop-rotate-multi=true | FileCheck %s
; RUN: opt -S < %s -passes='loop(loop-rotate)' -loop-rotate-multi=true | FileCheck %s

; Test loop rotation with multiple exits, some of them - deoptimizing.
; We should end up with a latch which exit is non-deoptimizing, so we should rotate
; more than once.

declare i32 @llvm.experimental.deoptimize.i32(...)

define i32 @test_cond_with_one_deopt_exit(i32 * nonnull %a, i64 %x) {
; Rotation done twice.
; Latch should be at the 2nd condition (for.cond2), exiting to %return.
;
; CHECK-LABEL: @test_cond_with_one_deopt_exit(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[VAL_A_IDX3:%.*]] = load i32, i32* %a, align 4
; CHECK-NEXT:    [[ZERO_CHECK4:%.*]] = icmp eq i32 [[VAL_A_IDX3]], 0
; CHECK-NEXT:    br i1 [[ZERO_CHECK4]], label %deopt.exit, label %for.cond2.lr.ph
; CHECK:       for.cond2.lr.ph:
; CHECK-NEXT:    [[FOR_CHECK8:%.*]] = icmp ult i64 0, %x
; CHECK-NEXT:    br i1 [[FOR_CHECK8]], label %for.body.lr.ph, label %return
; CHECK:       for.body.lr.ph:
; CHECK-NEXT:    br label %for.body
; CHECK:       for.cond2:
; CHECK:         [[FOR_CHECK:%.*]] = icmp ult i64 {{%.*}}, %x
; CHECK-NEXT:    br i1 [[FOR_CHECK]], label %for.body, label %for.cond2.return_crit_edge
; CHECK:       for.body:
; CHECK:         br label %for.tail
; CHECK:       for.tail:
; CHECK:         [[VAL_A_IDX:%.*]] = load i32, i32*
; CHECK-NEXT:    [[ZERO_CHECK:%.*]] = icmp eq i32 [[VAL_A_IDX]], 0
; CHECK-NEXT:    br i1 [[ZERO_CHECK]], label %for.cond1.deopt.exit_crit_edge, label %for.cond2
; CHECK:       for.cond2.return_crit_edge:
; CHECK-NEXT:    {{%.*}} = phi i32
; CHECK-NEXT:    br label %return
; CHECK:       return:
; CHECK-NEXT:    [[SUM_LCSSA2:%.*]] = phi i32
; CHECK-NEXT:    ret i32 [[SUM_LCSSA2]]
; CHECK:       for.cond1.deopt.exit_crit_edge:
; CHECK-NEXT:    {{%.*}} = phi i32
; CHECK-NEXT:    br label %deopt.exit
; CHECK:       deopt.exit:
; CHECK:         [[DEOPT_VAL:%.*]] = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 {{%.*}}) ]
; CHECK-NEXT:    ret i32 [[DEOPT_VAL]]
;
entry:
  br label %for.cond1

for.cond1:
  %idx = phi i64 [ 0, %entry ], [ %idx.next, %for.tail ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %for.tail ]
  %a.idx = getelementptr inbounds i32, i32 *%a, i64 %idx
  %val.a.idx = load i32, i32* %a.idx, align 4
  %zero.check = icmp eq i32 %val.a.idx, 0
  br i1 %zero.check, label %deopt.exit, label %for.cond2

for.cond2:
  %for.check = icmp ult i64 %idx, %x
  br i1 %for.check, label %for.body, label %return

for.body:
  br label %for.tail

for.tail:
  %sum.next = add i32 %sum, %val.a.idx
  %idx.next = add nuw nsw i64 %idx, 1
  br label %for.cond1

return:
  ret i32 %sum

deopt.exit:
  %deopt.val = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %val.a.idx) ]
  ret i32 %deopt.val
}

define i32 @test_cond_with_two_deopt_exits(i32 ** nonnull %a, i64 %x) {
; Rotation done three times.
; Latch should be at the 3rd condition (for.cond3), exiting to %return.
;
; CHECK-LABEL: @test_cond_with_two_deopt_exits(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A_IDX_DEREF4:%.*]] = load i32*, i32** %a
; CHECK-NEXT:    [[NULL_CHECK5:%.*]] = icmp eq i32* [[A_IDX_DEREF4]], null
; CHECK-NEXT:    br i1 [[NULL_CHECK5]], label %deopt.exit1, label %for.cond2.lr.ph
; CHECK:       for.cond2.lr.ph:
; CHECK-NEXT:    [[VAL_A_IDX9:%.*]] = load i32, i32* [[A_IDX_DEREF4]], align 4
; CHECK-NEXT:    [[ZERO_CHECK10:%.*]] = icmp eq i32 [[VAL_A_IDX9]], 0
; CHECK-NEXT:    br i1 [[ZERO_CHECK10]], label %deopt.exit2, label %for.cond3.lr.ph
; CHECK:       for.cond3.lr.ph:
; CHECK-NEXT:    [[FOR_CHECK14:%.*]] = icmp ult i64 0, %x
; CHECK-NEXT:    br i1 [[FOR_CHECK14]], label %for.body.lr.ph, label %return
; CHECK:       for.body.lr.ph:
; CHECK-NEXT:    br label %for.body
; CHECK:       for.cond2:
; CHECK:         [[VAL_A_IDX:%.*]] = load i32, i32*
; CHECK-NEXT:    [[ZERO_CHECK:%.*]] = icmp eq i32 [[VAL_A_IDX]], 0
; CHECK-NEXT:    br i1 [[ZERO_CHECK]], label %for.cond2.deopt.exit2_crit_edge, label %for.cond3
; CHECK:       for.cond3:
; CHECK:         [[FOR_CHECK:%.*]] = icmp ult i64 {{%.*}}, %x
; CHECK-NEXT:    br i1 [[FOR_CHECK]], label %for.body, label %for.cond3.return_crit_edge
; CHECK:       for.body:
; CHECK:         br label %for.tail
; CHECK:       for.tail:
; CHECK:         [[IDX_NEXT:%.*]] = add nuw nsw i64 {{%.*}}, 1
; CHECK:         [[NULL_CHECK:%.*]] = icmp eq i32* {{%.*}}, null
; CHECK-NEXT:    br i1 [[NULL_CHECK]], label %for.cond1.deopt.exit1_crit_edge, label %for.cond2
; CHECK:       for.cond3.return_crit_edge:
; CHECK-NEXT:    [[SPLIT18:%.*]] = phi i32
; CHECK-NEXT:    br label %return
; CHECK:       return:
; CHECK-NEXT:    [[SUM_LCSSA2:%.*]] = phi i32
; CHECK-NEXT:    ret i32 [[SUM_LCSSA2]]
; CHECK:       for.cond1.deopt.exit1_crit_edge:
; CHECK-NEXT:    br label %deopt.exit1
; CHECK:       deopt.exit1:
; CHECK-NEXT:    [[DEOPT_VAL1:%.*]] = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 0) ]
; CHECK-NEXT:    ret i32 [[DEOPT_VAL1]]
; CHECK:       for.cond2.deopt.exit2_crit_edge:
; CHECK-NEXT:    [[SPLIT:%.*]] = phi i32
; CHECK-NEXT:    br label %deopt.exit2
; CHECK:       deopt.exit2:
; CHECK-NEXT:    [[VAL_A_IDX_LCSSA:%.*]] = phi i32
; CHECK-NEXT:    [[DEOPT_VAL2:%.*]] = call i32 (...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 [[VAL_A_IDX_LCSSA]]) ]
; CHECK-NEXT:    ret i32 [[DEOPT_VAL2]]
;
entry:
  br label %for.cond1

for.cond1:
  %idx = phi i64 [ 0, %entry ], [ %idx.next, %for.tail ]
  %sum = phi i32 [ 0, %entry ], [ %sum.next, %for.tail ]
  %a.idx = getelementptr inbounds i32*, i32 **%a, i64 %idx
  %a.idx.deref = load i32*, i32** %a.idx
  %null.check = icmp eq i32* %a.idx.deref, null
  br i1 %null.check, label %deopt.exit1, label %for.cond2

for.cond2:
  %val.a.idx = load i32, i32* %a.idx.deref, align 4
  %zero.check = icmp eq i32 %val.a.idx, 0
  br i1 %zero.check, label %deopt.exit2, label %for.cond3

for.cond3:
  %for.check = icmp ult i64 %idx, %x
  br i1 %for.check, label %for.body, label %return

for.body:
  br label %for.tail

for.tail:
  %sum.next = add i32 %sum, %val.a.idx
  %idx.next = add nuw nsw i64 %idx, 1
  br label %for.cond1

return:
  ret i32 %sum

deopt.exit1:
  %deopt.val1 = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 0) ]
  ret i32 %deopt.val1
deopt.exit2:
  %deopt.val2 = call i32(...) @llvm.experimental.deoptimize.i32() [ "deopt"(i32 %val.a.idx) ]
  ret i32 %deopt.val2
}
