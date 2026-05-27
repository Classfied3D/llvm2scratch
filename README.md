![Scratch Cat Wyvern](assets/title.svg)

An LLVM backend to convert LLVM IR to [MIT Scratch](https://scratch.mit.edu), a block based coding language. This allows many programs written in languages which can compile to LLVM (C, C++, Rust, etc) to be ported into scratch.

## Progress

- 🆕 Stack + Heap Allocation, Deallocation, Loading + Storing
- 🔢 Integer (Up to 48 bits) and Float Operations
- 🔃 Functions + Return Values + Recursion + Function Pointers
- 🔀 Branch + Switch Instructions
- 🔁 Loops (Which unwind scratch's call stack when necessary)
- ⏺ Arrays and Structs (getelementptr support)
- 🔡 Static Variables
- 📚 [Partial libc support](https://github.com/Classfied3D/newlib-scratch)
- ⚡ Optimizations (Known Value Propagation, Assignment Elision)
- 📝 .sb3 + .sprite3 + scratchblocks file output

## Project Showcase

- [ELIZA-scratch](https://github.com/Callen44/ELIZA-scratch) by [@Callen44](https://github.com/Callen44) - port of the ELIZA chatbot to scratch
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
usage: llvm2scratch [-h] [-o OUTPUT] [-f {infer,project3,sprite3}] [-T TARGET [TARGET ...]]
                    [-U TARGET] [-O [OPT_OPTIONS ...]] [-M [MINIFY_OPTIONS ...]]
                    [--memory-size MEMORY_SIZE] [--local-stack-size LOCAL_STACK_SIZE]
                    [--max-branch-recursion MAX_BRANCH_RECURSION]
                    [--no-accurate-byte-spacing] [--entrypoint ENTRYPOINT]
                    [--debug-scratch-text DEBUG_SCRATCH_TEXT]
                    [--debug-scratchblocks DEBUG_SCRATCHBLOCKS] [--replace-hacked-blocks]
                    [--hide-blocks]
                    input

Compile an LLVM 19 IR (.ll) file into a scratch sprite (.sprite3)

positional arguments:
  input                 Path to the input LLVM 19 IR (.ll) file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to the output file (.sb3 or .sprite3)
  -f {infer,project3,sprite3}, --format {infer,project3,sprite3}
                        File format of output file. By default this infered by the output
                        file's extension.
  -T TARGET [TARGET ...], --targets TARGET [TARGET ...]
                        Compile code to support these targets. See list of targets below.
                        Defaults to scratch3 turbowarp3
  -U TARGET, --opt-target TARGET
                        Optimize code with this target in mind. Defaults to scratch3 if
                        available otherwise the first target listed.
  -O [OPT_OPTIONS ...]  Optimizations to apply; defaults to all; see below
  -M [MINIFY_OPTIONS ...]
                        Minify settings to apply; defaults to general; see below
  --memory-size MEMORY_SIZE
                        Number of 'bytes' on 'memory' list; max value is 200,000; default is
                        4096
  --local-stack-size LOCAL_STACK_SIZE
                        Number of 'bytes' on local stack list for storing registers when
                        recursing; max value is 200,000; default is 512
  --max-branch-recursion MAX_BRANCH_RECURSION
                        Maximum depth of scratch's call stack before resetting it; default
                        depends on targets enabled
  --no-accurate-byte-spacing
                        Disable extra padding bytes added to each value in memory so that it
                        takes up the space it would normally in bytes. This spacing allows
                        byte indexing to be more accurate at the cost of requiring ~3x more
                        space in the memory list. Disabling this may break programs that
                        rely on an 8-bit byte size, like memcpy on an array of i32s or
                        optimized IR.
  --entrypoint ENTRYPOINT
                        Specify a custom entrypoint function to run once the program starts.
                        Defaults to main.
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

targets:
  scratch3              Scratch 3.0 (https://scratch.mit.edu): The third and current major
                        version of Scratch. Interprets scratch code in javascript. Supports
                        formats: project3, sprite3
  turbowarp3            TurboWarp (https://turbowarp.org): A mod of Scratch 3.0 with
                        improved performance and other features. Compiles scratch into
                        javascript. Accurately runs scratch code with a few different
                        behaviours, namely a limited max recursion depth. Supports formats:
                        project3, sprite3

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
  gen-lut-runtime       Generate AND/OR/XOR tables at runtime rather than adding
                        pregenerated ones to the file. This reduces file size significantly
                        (by ~0.7MB) at the cost of ~0.4s spent generating the lookup tables
                        on the first time running the project (~0.01s on TurboWarp)
```

## Block Perf

See [data/targets](llvm2scratch/data/targets).

## Planning

- Opt: remove Repeat(Known(1))
- Opt: `set a (a + n)` -> `change a by n`
- Opt: `set a (a * 2)` -> `change a by a`

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
