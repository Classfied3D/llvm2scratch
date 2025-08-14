/*#include <stdio.h>

static int a = 7;

int add_one(int a) {
  return a + 1;
}

int test_branch(int num) {
  int a = 3;

  if (num != 1) a = 50;

  return a + num;
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
  //printf("%u\n", b);
  //printf("%u", c);

  for (unsigned char d = 65; d != 70; d++) {
    putchar(d);
  }

  int c = test_branch(2);
  return 0;
}*/

#include <stdio.h>

int test_branch(int num) {
  int a = 3;

  if (num != 1) a = 50;

  return a + num;
}

int main(void) {
  for (unsigned char d = 65; d != 70; d++) {
    putchar(d);
  }

  int c = test_branch(2);
  return 0;
}