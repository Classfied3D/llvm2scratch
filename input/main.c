#include <stdio.h>

static int a = 7;

int main(void) {
  puts("hello world");
  a += 2;
  a -= 4;
  a *= 2;
  a /= -3;
  a = -340;
  a %= -60;
  a = 31;
  a <<= a;
  a >>= 3;
  u_int32_t b = 3204;
  b >>= 4;
  /*for (a = 0; a < 5; a++) {
    printf("%d", a);
  }*/
  return 0;
}