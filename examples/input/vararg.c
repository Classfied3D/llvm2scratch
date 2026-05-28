#include "sb3api.h"
#include <stdarg.h>

double mean(unsigned int count, ...) {
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

// N.B. requires input to be ordered
double median(unsigned int count, ...) {
  if (count == 0) {
    return 0.0 / 0.0;
  }

  va_list args;
  va_start(args, count);

  double res;
  int pos = (count + 1) / 2;
  int take_mean = count % 2 == 0;

  for (int i = 0; i < pos; i++) {
    res = va_arg(args, int);
  }

  if (take_mean) {
    res += va_arg(args, int);
    res /= 2;
  }

  return res;
}

int main(void) {
  double a, b, c;
  SB3_ask_dbl(&a, "Enter avg (1/3)");
  SB3_ask_dbl(&b, "Enter avg (2/3)");
  SB3_ask_dbl(&c, "Enter avg (3/3)");
  SB3_say_dbl(mean(3, a, b, c));
  SB3_wait(1);

  mean(0);

  SB3_ask_dbl(&a, "Enter 1 for mean and 2 for median of (1, 2, 3, 10)");
  double (*average) (unsigned int count, ...) = a == 1.0 ? mean : median;
  SB3_say_dbl(average(4, 1, 2, 3, 10));
  SB3_say_dbl(average(0));
}
