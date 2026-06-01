%Point = type { i32, i32 }
%Rect = type { %Point, %Point }

define i32 @main() {
  %p1 = insertvalue %Point undef, i32 10, 0
  %p1.1 = insertvalue %Point %p1, i32 20, 1

  %p2 = insertvalue %Point undef, i32 30, 0
  %p2.1 = insertvalue %Point %p2, i32 40, 1

  %r = insertvalue %Rect undef, %Point %p1.1, 0
  %r.1 = insertvalue %Rect %r, %Point %p2.1, 1

  %tl = extractvalue %Rect %r.1, 0
  %br = extractvalue %Rect %r.1, 1

  %x1 = extractvalue %Point %tl, 0
  %y1 = extractvalue %Point %tl, 1
  %x2 = extractvalue %Point %br, 0
  %y2 = extractvalue %Point %br, 1

  %dx = sub i32 %x2, %x1
  %dy = sub i32 %y2, %y1
  %area = mul i32 %dx, %dy

  ret i32 %area
}
