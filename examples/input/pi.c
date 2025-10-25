#include "sb3api.h"

double pi = 0.0;
double term;
int sign = 1;
unsigned int i = 0U;

void __attribute__((noinline)) calc_pi_step(void) {
  term = 1.0 / (2.0 * i + 1.0);
  pi += sign * term;
  sign = -sign;
  ++i;
}

int main(void) {
  const unsigned int print_interval = 10000U;

  for (;;) {
    calc_pi_step();

    if (i % print_interval == 0U) {
      SB3_say_dbl(100000000000 * 4.0 * pi);
    }
  }
}
