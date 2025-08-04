#include <stdio.h>

static signed int a = 7;

int main(void) {
  puts("hello world");
  a += 2;
  a -= 4;
  a /= 3;
  a = -340;
  a %= -60;
  a = 31;
  a <<= a;
  /*for (a = 0; a < 5; a++) {
    printf("%d", a);
  }*/
  return 0;
}