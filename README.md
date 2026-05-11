# LLVM2Scratch

An LLVM backend to convert LLVM IR to [MIT Scratch](https://scratch.mit.edu), a block based coding language. This allows many programs written in languages which can compile to LLVM (C, C++, Rust, etc) to be ported into scratch.

## Progress

- 🆕 Stack + Heap Allocation, Deallocation, Loading + Storing
- 🔢 Integer (Up to 48 bits) and Float Operations
- 🔃 Functions + Return Values + Recursion + Function Pointers
- 🔀 Branch + Switch Instructions
- 🔁 Loops (Which unwind scratch's call stack when necessary)
- ⏺ Arrays and Structs (getelementptr support)
- 🔡 Static Variables
- 📚 [Partial cstdlib support](https://github.com/Classfied3D/newlib-scratch)
- ⚡ Optimizations (Known Value Propagation, Assignment Elision)
- 📝 .sb3 + .sprite3 + scratchblocks file output

## Project Showcase

- [LLM from Scratch](https://github.com/Broyojo/llm_from_scratch) by [@Broyojo](https://github.com/Broyojo) - `llama2.c` running in scratch
- [asm2scratch](https://github.com/RetrogradeDev/asm2scratch) by [@RetrogradeDev](https://github.com/RetrogradeDev) - early llvm2scratch fork to support compiling assembly directly instead
- [sha2scratch](https://github.com/Classfied3D/sha2scratch) by me - SHA-256 algorithm ported to scratch

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
usage: llvm2scratch [-h] [-o OUTPUT] [--format {infer,project3,sprite3}]
                    [-O {all,none,compiler,assignment-elision,known-value-prop}]
                    [-M {all,none,general,break-glow}] [--memory-size MEMORY_SIZE]
                    [--local-stack-size LOCAL_STACK_SIZE]
                    [--max-branch-recursion MAX_BRANCH_RECURSION]
                    [--no-accurate-byte-spacing] [--debug-scratch-code DEBUG_SCRATCH_CODE]
                    [--replace-hacked-blocks] [--hide-blocks]
                    input

Compile an LLVM 19 IR (.ll) file into a scratch sprite (.sprite3)

positional arguments:
  input                 Path to the input LLVM 19 IR (.ll) file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to the output file (.sb3 or .sprite3)
  --format {infer,project3,sprite3}
                        File format of output file. By default this infered by the output
                        file's extension.
  -O {all,none,compiler,assignment-elision,known-value-prop}
                        Optimizations to apply; defaults to all; see below
  -M {all,none,general,break-glow}
                        Minify settings to apply; defaults to general; see below
  --memory-size MEMORY_SIZE
                        Number of 'bytes' on 'memory' list; max value is 200,000; default is
                        4096
  --local-stack-size LOCAL_STACK_SIZE
                        Number of 'bytes' on local stack list for storing registers when
                        recursing; max value is 200,000; default is 512
  --max-branch-recursion MAX_BRANCH_RECURSION
                        Maximum depth of scratch's call stack before resetting it; defaults
                        to 2000
  --no-accurate-byte-spacing
                        Disable extra padding bytes added to each value in memory so that it
                        takes up the space it would normally in bytes. This allows byte
                        indexing to be more accurate at the cost of requiring ~3x more space
                        in the memory list. Disabling this may break programs that rely on
                        an 8-bit byte size, like memcpy on an array of i32s or optimized IR.
  --debug-scratch-text DEBUG_SCRATCH_TEXT
                        Output readable scratch code to a text file so it can be viewed
  --debug-scratchblocks DEBUG_SCRATCHBLOCKS
                        Output scratchblocks compatible code to a text file so it can be
                        viewed. See https://scratchblocks.github.io/
  --replace-hacked-blocks
                        Remove 'hacked' blocks not normally accessible from the editor such
                        as 'counter' and 'while' by replacing them with workarounds. See
                        https://en.scratch-wiki.info/wiki/Hidden_Blocks. This may lead to a
                        reduction in performance.
  --hide-blocks         Prevent blocks from rendering in the editor by setting shadow: true
                        on top level blocks; stops editor lag. Not recommended due to
                        increased project size and this seems to stop some projects from
                        running. Instead export to a project instead of a sprite and don't
                        click on the sprite.

optimization options:
  all, none             Self-explanatory
  compiler              Enable compiler-level optimizations (e.g. addressing globals with
                        address instead of by variable)
  assignment-elision    Reduce expensive 'Set Variable' usage by inlining variable
                        assignments
  known-value-prop      Various transformations on values and blocks under certain values

minify options:
  all, none             Self-explanatory
  general               Optimize project.json's size by simplifing uids, removing falsy
                        fields, etc
  break-glow            Removing the parent key when minifing prevents blocks in the same
                        sprite from glowing correctly due to a js error - minify futher and
                        allow this error to occur
```

## Block Perf

### Scratch

```
time (s) per 1,000,000 iterations:

set var:   7.550
get var:   1.538
get param: 1.178

add:       0.765
mod:       0.715
rand:      2.473
not:       0.725
and:       0.864
eq:        0.929
abs:       1.607
join:      1.091
letter of: 0.737
length of: 0.483
cntin str: 1.272
round int: 0.304
round flt: 1.250

get list:  5.814 (changes with contents)
item:      1.679
item #:    4.920 (changes with contents)
list len:  0.713

counter:   0.190

answer:    0.331

cost num:  0.241
cost name: 0.654
```

### Turbowarp

```
time (s) per 5,000,000 iterations (JIT + Warp Timer disabled):

set var:   1.307
change vr: 5.719
get var:   0.495 (interesting a lot higher when modifing the same value at 4.203)
get param: 3.796

add:       0.002
sub:       0.002
mul:       0.002
div:       0.002
mod:       6.531 (3.654 when modulus is known with my PR)
rand:      3.599
not:       0.002
and:       0.002
or:        0.002
eq:        0.418
abs:       0.847
floor:     0.843
ceil:      0.844
sqrt:      4.589
sin:       7.505
cos:       7.527
tan:       11.125
asin:      5.656
acos:      5.779
atan:      2.513
ln:        4.981
log:       5.694
e ^:       1.803
10 ^:      0.028
join:      0.006
letter of: 0.679
length of: 0.136
cntin str: 3.853 (changes with contents)
round int: 0.889
round flt: 0.583

get list:  31.636 (changes with contents)
item:      4.388
item #:    35.467 (changes with contents)
length:    0.610

counter:   0.544
incr cnt:  2.020

answer:    0.426

cost num:  0.807
cost name: 3.007
```

## Planning

- Opti: unused param elision
- Opti: known list (lookup table) progagation
- Opti: remove Repeat(Known(1))
- Opti: `set a (a + n)` -> `change a by n`
- Opti: `set a (a * 2)` -> `change a by a`

## Proofs

### Multiplication

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

### AND/OR/XOR

- AND uses a joint bitmask (via div/floor/modulo/mul) and lookup table approach when one side is known. To take advantage of this in other operations, the following equalities are used:
  - `A | B = (A & !B) + B`
  - Proved by:
    | A | B | A & !B | (A & !B) + B | A \| B |
    | --- | --- | ------ | ------------ | ------ |
    | 0 | 0 | 0 | 0 | 0 |
    | 0 | 1 | 0 | 1 | 1 |
    | 1 | 0 | 1 | 1 | 1 |
    | 1 | 1 | 0 | 1 | 1 |
  - `A ^ B = A - 2(A & B) + B`
  - Proved by:
    | A | B | A + B | 2(A & B) | A - 2(A & B) + B | A ^ B |
    | --- | --- | ----- | -------- | ---------------- | ----- |
    | 0 | 0 | 0 | 0 | 0 | 0 |
    | 0 | 1 | 1 | 0 | 1 | 1 |
    | 1 | 0 | 1 | 0 | 1 | 1 |
    | 1 | 1 | 2 | 2 | 0 | 0 |
