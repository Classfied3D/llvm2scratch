#include "sb3api.h"
#include <stdarg.h>

double mean(int count, ...) {
  va_list args;
  va_list args2;
  va_start(args, count);
  va_copy(args, args2);

  double sum = 0;
  for (int i = 0; i < count; i++) {
    sum += va_arg(args, int);
  }

  va_end(args);
  va_end(args2);

  return sum / count;
}

int main(void) {
  double a, b, c;
  SB3_ask_dbl(&a, "Enter avg (1/3)");
  SB3_ask_dbl(&b, "Enter avg (2/3)");
  SB3_ask_dbl(&c, "Enter avg (3/3)");
  SB3_say_dbl(mean(3, a, b, c));
  double (*average) (int count, ...) = mean;
  average(2, 1, 3);
}
