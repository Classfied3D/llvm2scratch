# LLVM2Scratch

An LLVM backend to convert LLVM IR to [MIT Scratch](https://scratch.mit.edu), a block based coding language. This allows many programs written in languages which can compile to LLVM (C, C++, Rust, etc) to be ported into scratch.

## Progress
* 🆕 Stack Allocation, Deallocation, Loading + Storing
* 🔢 Integer (Up to 48 bits) and Float Operations
* 🔃 Functions + Return Values + Recursion
* 🔀 Branch + Switch Instructions
* 🔁 Loops (Which unwind the call stack when necessary)
* ⏺ Arrays and Structs (getelementptr support)
* 🔡 Static Variables
* ⚡ Optimizations (Known Value Propagation, Assignment Elision)
* 📝 Sprite3 file output

## Examples
* [Hello World](https://scratch.mit.edu/projects/1201848279/)
* [Math](https://scratch.mit.edu/projects/1206058442/)
* [Old Branching](https://scratch.mit.edu/projects/1206466346/)
* [New Branching + Assignment Elision](https://scratch.mit.edu/projects/1208872099/)
* [Recursion](https://scratch.mit.edu/projects/1211169662/)
* [Arrays + Structs](https://scratch.mit.edu/projects/1226122280/)

## Installation

* Install llvm2scratch with `pip install ` followed by the path to the project root (the folder containing the pyproject.toml and llvm2scratch folder)

## Info

### How multiplication works

* Scratch uses JS' Number which can store a maximum of [2 ^ 53 - 1](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER) before the accuracy is less than 1
* This means 32 bit multiplication `(2^32 * 2^32) mod 2^32` does not give the correct result because the number calculated is 2^64 which is not accurate enough (it works with up to 26-bit integers)
* To resolve this the following maths is used:
  * Assuming `a`, `a0`, `b1`, `b`, `b0` and `b1` are positive 32-bit integers
  * Assuming `a0` and `b0` are less than `2^16` (always possible with a 32-bit `a` and `b`)
  * Where `a = a1 * 2^16 + a0`
  * And `b = b1 * 2^16 + b0`
  * Then `(2^32 * 2^32) mod 2^32 = (a1 * 2^16 + a0)(b1 * 2^16 + b0) mod 2 ^ 32`
  * If we expand the brackets of the second part:
  * `(a1b1 * 2^32 + (a0b1 + b0a1) * 2^16 + a0b0) mod 2^32`
  * Then simplify:
  * `((a0b1 + b0a1) * 2^16 + a0b0) mod 2^32`
  * Then because `a0`, `a1`, `b0` and `b1` are less than `2^16` the highest number that is calculated is
  * `((2^16)^2 * 2) * 2^16 + (2^16)^2 = 2^49`
  * It can be generalised for n bits as
  * `((a0b1 + b0a1) * 2^floor(n/2) + a0b0) mod 2^n`
  * We can calculate `a0 = a % mod 2^floor(n/2)`, `a1 = a // 2^floor(n/2)`, etc
  * This works with up to 34 bits, after which it can be rewritten as
  * `(((a0b1 + b0a1) mod (2^n / 2^floor(n/2))) * 2^floor(n/2) + a0b0) mod 2^n`
  * or `(((a0b1 + b0a1) mod 2^ceil(n/2)) * 2^floor(n/2) + a0b0) mod 2^n`

## Planning

* Linking LLVM IR Files: https://stackoverflow.com/a/41181212/
* Opti: For checked branch functions, find each path of recursion then only check if the counter > max recursions for one branch in each path (otherwise just increment the counter). Sort branches most used in each trail and add the highest each time.
* Opti: unused param elision
* Opti: known list (lookup table) progagation
* Opti: (boolasint<not<cond>>) -> (1 - <cond>) (do this in scratch.py so that it doesn't interfere with other optimizations)
* Opti: remove Repeat(Known(1))
* Opti: Group allocations at start of branch, if fixed allocation then dellocate by fixed amount
* Opti: `set a (a + n)` -> `change a by n`
* Opti: `set a (a * 2)` -> `change a by a`
* Opti:
  * No branching or calling functions that branch
  * Unchecked branches, don't return to address
  * Unchecked branches, return to address
  * Checked branches, return to address
  * 8.29s for 5000000 comparisons
  * 12s for 5000000 forward traces
  * 1.00s for 5000000 backtraces
  * if log2(return addresses) * 8 + 12 > average branches then return by recursing backward

## Block Perf

```
Time (s) per 200000 iterations:

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

Item:      1.679
Item #:    4.920 (Unreliable benchmark)

Counter:   0.190
```
