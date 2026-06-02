#include "sb3api.h"

__attribute__((noinline)) int add_one(int num) { return num + 1; }

__attribute__((noinline)) int sum_to_one_digit(int n) {
  int sum = 0;

  while (n > 0) {
    sum += n % 10;
    n /= 10;
  }

  if (sum >= 10)
    return sum_to_one_digit(sum);

  return sum;
}

int main(void) {
  double number, output;
  SB3_ask_dbl(&number, "Enter a number");
  SB3_ask_dbl(&output, "Enter a number >0.5 to sum to one digit then add one, "
                       "otherwise add two");

  int (*fn_ptr)(int);
  if (output > 0.5) {
    fn_ptr = sum_to_one_digit;
  } else {
    fn_ptr = add_one;
  }

  number = add_one(fn_ptr(number));
  SB3_say_dbl(number);
  SB3_wait(2.0);

  double dur;
  char type[2];
  SB3_ask_dbl(&dur, "Enter a duration");
  SB3_ask_str(type, "Render during your wait? (y/n)", 2);
  void (*wait_fn)(double) = type[0] == 'y' ? SB3_wait : SB3_wait_no_render;
  wait_fn(dur);

  return 0;
}
