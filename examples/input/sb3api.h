// Meta
// Force scratch to render a frame. Internally this uses change volume
// by 0 to force a frame to be rendered in run without screen refresh
// procedures
void SB3_render();

// Looks
void SB3_say_char(char str);
void SB3_say_str(const char *str);
void SB3_say_dbl(double num);

// Control
// Wait at least duration seconds while rendering frames. May wait up
// to an extra frametime, just wait in scratch
void SB3_wait(double duration);
// Same as SB3_wait but don't render frames while waiting. Internally
// uses the wait _ seconds block
void SB3_wait_no_render(double duration);

// Sensing
// SB3_ask_dbl casts non-floats by using the Scratch (_ + 0) block.
// As such, numbers are unchanged, but strings become 0.
int SB3_ask_str(const char *output, const char *input, int count);
int SB3_ask_dbl(const double *output, const char *input);

// Meant for teaching about buffer overflows. Don't use this otherwise please.
// https://github.com/Classfied3D/llvm2scratch/pull/5#discussion_r3006183332
int SB3_ask_str_unsafe(const char *output, const char *input);

// Returns the days since 2000 in UTC time
double SB3_days_since_2000();
