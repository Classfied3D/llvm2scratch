#include <stdio.h>

static int a = 7;

int add_one(int a) {
  return a + 1;
}

int test_branch(int num) {
  int a = 3;
  //if (num != 1) {
  //  a = 4;
  //  puts("a set to 4");
  //}
  a += num != 1;

  int b = a + 1;

  return b;
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

  int c = test_branch(1);

  //printf("%u\n", a);
  //printf("%u\n", b);
  //printf("%u", c);

  //for (char c = 65; c < 70; c++) {
  //  puts(&c);
  //}
  return 0;
}