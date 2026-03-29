// Looks
void SB3_say_char(char str);
void SB3_say_str(const char* str);
void SB3_say_dbl(double num);

// Sensing
// SB3_ask_dbl casts non-floats by using the Scratch (_ + 0) block.
// As such, numbers are unchanged, but strings become 0.
int SB3_ask_str(const char* output, const char* input, int count);
int SB3_ask_dbl(const double* output, const char* input);

// Meant for teaching about buffer overflows. Don't use this otherwise please.
// https://github.com/Classfied3D/llvm2scratch/pull/5#discussion_r3006183332
int SB3_ask_str_unsafe(const char* output, const char* input);
