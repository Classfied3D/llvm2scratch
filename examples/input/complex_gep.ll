target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv4t-unknown-none-eabi"

%Inner = type { i8, [3 x i8], i64, i32 }
%Outer = type { i32, [4 x %Inner], i16 }

@fmt = private unnamed_addr constant [17 x i8] c"GEP offset = %d\0A\00"

define i32 @compute_gep_offset(ptr %base, i32 %outer_idx, i16 %inner_idx) {
entry:
  %gep = getelementptr inbounds %Outer, ptr %base, i32 %outer_idx, i32 1, i16 %inner_idx, i32 3
  %result = ptrtoint ptr %gep to i32
  ret i32 %result
}

define i32 @main() {
entry:
  %base = inttoptr i32 0 to ptr
  %offset = call i32 @compute_gep_offset(ptr %base, i32 3, i16 -1)
  ret i32 0
  ; returns: GEP offset = 210
}
