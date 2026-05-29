; This struct is different for every platform. For most platforms,
; it is merely a ptr.
%struct.va_list = type { ptr }

; For Unix x86_64 platforms, va_list is the following struct:
; %struct.va_list = type { i32, i32, ptr, ptr }

define i32 @test(i32 %X, ...) {
  ; Initialize variable argument processing
  %ap = alloca %struct.va_list
  call void @llvm.va_start.p0(ptr %ap)

  ; Read a single integer argument
  %tmp = va_arg ptr %ap, i32

  ; Demonstrate usage of llvm.va_copy and llvm.va_end
  %aq = alloca ptr
  call void @llvm.va_copy.p0(ptr %aq, ptr %ap)
  call void @llvm.va_end.p0(ptr %aq)

  ; Stop processing of arguments.
  call void @llvm.va_end.p0(ptr %ap)
  ret i32 %tmp
}

define i32 @main() {
  call i32 @test(i32 3, i32 1, i32 2, i32 3)
  ret i32 0
}

declare void @llvm.va_start.p0(ptr)
declare void @llvm.va_copy.p0(ptr, ptr)
declare void @llvm.va_end.p0(ptr)
