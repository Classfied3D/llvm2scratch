#include <stdio.h>

static int a = 7;

int add_one(int num) {
  return num + 1;
}

void do_nothing() {}

int test_branch(int num) {
  do_nothing();

  int a = 3;

  if (num != 1) a = 50;

  return a + num;
}

int factorial_recurse(int n) {
  if (n == 1) return 1;
  return factorial_recurse(n - 1) * n;
}

int sum_to_one_digit(unsigned int n) {
  unsigned int sum = 0;

  while (n > 0) {
    sum += n % 10;
    n /= 10;
  }

  if (sum >= 10) return sum_to_one_digit(sum);
  return sum;
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

  int c = test_branch(2);

  for (unsigned char d = 65; d != 70; d++) {
    putchar(d);
  }

  unsigned int e = 21;
  switch (e) {
    case 0:
      puts("0");
      break;
    case 1:
      puts("1");
      break;
    case 20:
      puts("20");
      break;
    case 21:
      puts("21");
      break;
    default:
      puts("default");
      break;
  }

  int f = factorial_recurse(10);
  int g = sum_to_one_digit(473);

  return 0;
}
