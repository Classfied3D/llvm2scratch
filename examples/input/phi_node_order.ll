; ModuleID = 'demo.c'
source_filename = "demo.c"
target datalayout = "e-m:o-p:32:32-p270:32:32-p271:32:32-p272:64:64-i128:128-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx12.0.0"

define noundef i32 @main() local_unnamed_addr {
  br label %loop

loop:
  %a = phi i32 [ 0, %0 ], [ %b, %loop ]
  %b = phi i32 [ 1, %0 ], [ %a, %loop ]
  br label %loop
}
