#include "sb3api.h"

int main() {
    // Unsafe (buffer overflow demo)
    char answer_unsafe[8];
    char unrelated[32];
    SB3_ask_str_unsafe(answer_unsafe, "Give me an str longer than 8 characters");
    SB3_say_str(unrelated); // Shouldn't say anything, but because of a buffer overflow, it does!

    return 0;
}
