#include "sb3api.h"
#include <stdint.h>

static int a = 7;
static long long ll_a = 149876583280495765;
static char* message = "loldefault";

typedef struct SensorData {
  int temp;
  int humidity;
} SensorData;

int add_one(int num) {
  return num + 1;
}

void do_nothing(long long input) {}

int test_branch(int num) {
  do_nothing(ll_a);

  int a = 3;

  if (num != 1) a = 50;

  return a + num;
}

int factorial_recurse(int n) {
  if (n == 1) return 1;
  return factorial_recurse(n - 1) * n;
}

int sum_to_one_digit(int n) {
  int sum = 0;

  while (n > 0) {
    sum += n % 10;
    n /= 10;
  }

  if (sum >= 10) return sum_to_one_digit(sum);
  return sum;
}

void numberize(char* str) {
  for (int i = 0; str[i] != '\0'; i++) {
    switch (str[i]) {
      case 'a':
        str[i] = '4';
        break;
      case 'e':
        str[i] = '3';
        break;
      case 'l':
        str[i] = '1';
        break;
      case 'o':
        str[i] = '0';
        break;
    }
  }
}

int main(void) {
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
    SB3_say_char(d);
  }

  unsigned int e = 221;
  switch (e) {
    case 0:
      SB3_say_str("0");
      break;
    case 1:
      SB3_say_str("1");
      break;
    case 20:
      SB3_say_str("20");
      break;
    case 21:
      SB3_say_str("21");
      break;
    default:
      SB3_say_str(message + 3);
      break;
  }

  int f = factorial_recurse(10);
  int g = sum_to_one_digit(473);

  char str[] = "hello world";

  numberize(str);
  SB3_say_str(str);

  SensorData h[5];
  h[2] = (SensorData){7, 2};
  SB3_say_char('0' + h[2].temp);

  float i = 3.0f;
  i += 0.5f;
  if (i > 3.4f) {
    i -= 1.0f;
  } else {
    i += 1.0f;
  }
  i = -i;

  int j = 257;
  u_int8_t k = j;
  float l = 3.14f;
  int m = l;
  float n = m;

  long long o = 1;
  o += 2;
  o -= 5;

  ll_a += 55;

  return 0;
}
