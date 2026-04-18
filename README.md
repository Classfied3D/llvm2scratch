# LLVM2Scratch

An LLVM backend to convert LLVM IR to [MIT Scratch](https://scratch.mit.edu), a block based coding language. This allows many programs written in languages which can compile to LLVM (C, C++, Rust, etc) to be ported into scratch.

## Progress

- 🆕 Stack Allocation, Deallocation, Loading + Storing
- 🔢 Integer (Up to 48 bits) and Float Operations
- 🔃 Functions + Return Values + Recursion + Function Pointers
- 🔀 Branch + Switch Instructions
- 🔁 Loops (Which unwind scratch's call stack when necessary)
- ⏺ Arrays and Structs (getelementptr support)
- 🔡 Static Variables
- 📚 [Partial cstdlib support](https://github.com/Classfied3D/newlib-scratch)
- ⚡ Optimizations (Known Value Propagation, Assignment Elision)
- 📝 Sprite3 file output

## Project Showcase

- [LLM from Scratch](https://github.com/Broyojo/llm_from_scratch) by [@Broyojo](https://github.com/Broyojo) - `llama2.c` running in scratch
- [asm2scratch](https://github.com/RetrogradeDev/asm2scratch) by [@RetrogradeDev](https://github.com/RetrogradeDev) - early llvm2scratch fork to support compiling assembly directly instead

## Examples

- [Hello World](https://scratch.mit.edu/projects/1201848279)
- [Integer Math](https://scratch.mit.edu/projects/1206058442)
- [Old Branching](https://scratch.mit.edu/projects/1206466346)
- [New Branching + Assignment Elision](https://scratch.mit.edu/projects/1208872099)
- [Recursion](https://scratch.mit.edu/projects/1211169662)
- [Arrays + Structs](https://scratch.mit.edu/projects/1226122280)
- [Pi Calculator](https://scratch.mit.edu/projects/1233764273)
- [Function Pointers](https://scratch.mit.edu/projects/1298975442)

## Installation

- Install llvm2scratch with `pip install ` followed by the path to the project root (the folder containing the pyproject.toml and llvm2scratch folder)
- Make sure to use clang 19 when compiling

## Usage

```
usage: llvm2scratch [-h] [-o OUTPUT] [-O {all,none,compiler,assignment-elision,known-value-prop}] [-M {all,none,general,break-glow}] [--hide-blocks]
                    [--memory-size MEMORY_SIZE] [--local-stack-size LOCAL_STACK_SIZE] [--max-branch-recursion MAX_BRANCH_RECURSION]
                    [--debug-scratch-code DEBUG_SCRATCH_CODE]
                    input

Compile an LLVM 19 IR (.ll) file into a scratch sprite (.sprite3)

positional arguments:
  input                 Path to the input LLVM 19 IR (.ll) file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to the output sprite3 file
  -O {all,none,compiler,assignment-elision,known-value-prop}
                        Optimizations to apply; defaults to all; see below
  -M {all,none,general,break-glow}
                        Minify settings to apply; defaults to general; see below
  --hide-blocks         Prevent blocks from rendering in the editor by setting shadow: true on top level blocks; stops editor lag
  --memory-size MEMORY_SIZE
                        Number of 'bytes' on 'memory' list; max value is 200,000; default is 1024
  --local-stack-size LOCAL_STACK_SIZE
                        Number of 'bytes' on local stack list for storing registers when recursing; max value is 200,000; default is 512
  --max-branch-recursion MAX_BRANCH_RECURSION
                        Maximum depth of scratch's call stack before resetting it; defaults to 1,000,000
  --debug-scratch-code DEBUG_SCRATCH_CODE
                        Output scratch code to a text file so it can be viewed

optimization options:
  all, none             Self-explanatory
  compiler              Enable compiler-level optimizations (e.g. addressing globals with address instead of by variable)
  assignment-elision    Reduce expensive 'Set Variable' usage by inlining variable assignments
  known-value-prop      Various transformations on values and blocks under certain values

minify options:
  all, none             Self-explanatory
  general               Optimize project.json's size by simplifing uids, removing falsy fields, etc
  break-glow            Removing the parent key when minifing prevents blocks in the same sprite from glowing correctly due to a js error - minify futher and
                        allow this error to occur
```

## Info

### How multiplication works

- Scratch uses JS' Number which can store a maximum of [2 ^ 53 - 1](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER) before the accuracy is less than 1
- This means 32 bit multiplication `(2^32 * 2^32) mod 2^32` does not give the correct result because the number calculated is 2^64 which is not accurate enough (it works with up to 26-bit integers)
- To resolve this the following maths is used:
  - Assuming `a`, `a0`, `b1`, `b`, `b0` and `b1` are positive 32-bit integers
  - Assuming `a0` and `b0` are less than `2^16` (always possible with a 32-bit `a` and `b`)
  - Where `a = a1 * 2^16 + a0`
  - And `b = b1 * 2^16 + b0`
  - Then `(2^32 * 2^32) mod 2^32 = (a1 * 2^16 + a0)(b1 * 2^16 + b0) mod 2 ^ 32`
  - If we expand the brackets of the second part:
  - `(a1b1 * 2^32 + (a0b1 + b0a1) * 2^16 + a0b0) mod 2^32`
  - Then simplify:
  - `((a0b1 + b0a1) * 2^16 + a0b0) mod 2^32`
  - Then because `a0`, `a1`, `b0` and `b1` are less than `2^16` the highest number that is calculated is
  - `((2^16)^2 * 2) * 2^16 + (2^16)^2 = 2^49`
  - It can be generalised for n bits as
  - `((a0b1 + b0a1) * 2^floor(n/2) + a0b0) mod 2^n`
  - We can calculate `a0 = a % mod 2^floor(n/2)`, `a1 = a // 2^floor(n/2)`, etc
  - This works with up to 34 bits, after which it can be rewritten as
  - `(((a0b1 + b0a1) mod (2^n / 2^floor(n/2))) * 2^floor(n/2) + a0b0) mod 2^n`
  - or `(((a0b1 + b0a1) mod 2^ceil(n/2)) * 2^floor(n/2) + a0b0) mod 2^n`

## Planning

- Opti: unused param elision
- Opti: known list (lookup table) progagation
- Opti: remove Repeat(Known(1))
- Opti: `set a (a + n)` -> `change a by n`
- Opti: `set a (a * 2)` -> `change a by a`

## Block Perf

```
Time (s) per 1,000,000 iterations:

Set Var:   7.550
Get Var:   1.538
Get Param: 1.178

Add:       0.765
Mod:       0.715
Rand:      2.473
Not:       0.725
And:       0.864
Eq:        0.929
Abs:       1.607
Join:      1.091
Letter Of: 0.737
Length Of: 0.483
Cntin Str: 1.272
Round Int: 0.304
Round Flt: 1.250

Get List:  5.814 (Unreliable)
Item:      1.679
Item #:    4.920 (Unreliable)

Counter:   0.190

Answer:    0.331

Cost Num:  0.241
Cost Name: 0.654
```
