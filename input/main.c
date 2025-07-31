#include <stdio.h>

static int a = 7;

int main(void) {
  puts("hello world\n");
  a += 2;
  a -= 4;
  a *= 20;
  a /= 3;
  /*for (a = 0; a < 5; a++) {
    printf("%d", a);
  }*/
  return 0;
}