#include <stdio.h>

static int a = 7;

int add_one(int a) {
  return a + 1;
}

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

  unsigned int b = 3204;
  b >>= 2;
  b ^= 113;
  b |= 1546;
  b &= 393;

  a = add_one(a);

  //printf("%u\n", a);
  //printf("%u", b);

  /*for (a = 0; a < 5; a++) {
    printf("%d", a);
  }*/
  return 0;
}