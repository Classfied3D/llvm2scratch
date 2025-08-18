# LLVM2Scratch

An LLVM backend to convert LLVM IR to [MIT Scratch](https://scratch.mit.edu), a block based coding language. This allows a lot of programs written languages which can compile to LLVM (C/C++/Rust/etc) to be ported into scratch.

## Installation

* The following dependencies require manual installation:

* **llvm2py**@0.2.0b0 (installation requires llvm@19)
  * macOS - macports (recommended due to supporting precompiled binaries for older macos versions + LLVM takes several hours to compile)
    * `sudo port install llvm-19`
    * `LLVM_DIR=/opt/local/libexec/llvm-19/lib/cmake/llvm/ LLVM_CONFIG=/opt/local/libexec/llvm-19/bin/llvm-config CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install llvm2py`
  * macOS - brew
    * `brew install llvm@19`
    * `LLVM_CONFIG=/usr/local/opt/llvm@19/bin/llvm-config CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install llvm2py`

* Then, install llvm2scratch with `pip install ` followed by the path to the project root (the folder containing the pyproject.toml and llvm2scratch folder)

## Info

### How multiplication works

* Scratch uses JS' Number which can store a maximum of [2 ^ 53 - 1](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER) before the accuracy is less than 1
* This means 32 bit multiplication `(2^32 * 2^32) mod 2^32` does not give the correct result because the number calculated is 2^64 which is not accurate enough (it works with up to 26-bit integers)
* To resolve this the following maths is used:
  * Assuming `a`, `a0`, `b1`, `b,` `b0` and `b1` are positive 32-bit integers
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
  * This works with up to 36 bits, after which it can be rewritten as
  * `(((a0b1 + b0a1) mod (2^n / 2^floor(n/2))) * 2^floor(n/2) + a0b0) mod 2^n`
  * or `(((a0b1 + b0a1) mod 2^ceil(n/2)) * 2^floor(n/2) + a0b0) mod 2^n`

## Planning

* Opti:
  * No branching or calling functions that branch
  * Unchecked branches, don't return to address
  * Unchecked branches, return to address
  * Checked branches, return to address
  * 8.29s for 5000000 comparisons
  * 12s for 5000000 forward traces
  * 1.00s for 5000000 backtraces
  * if log2(return addresses) * 8 + 12 > average branches then return by recursing backward
* Opti: For checked branch functions, find each path of recursion then only check if the counter > max recursions for one branch in each path (otherwise just increment the counter). Sort branches most used in each trail and add the highest each time.
* Opti: Assignment elision
* Opti: Optimise bool as int casts
* Opti: Group allocations at start of branch, if fixed allocation then dellocate by fixed amount
* Opti: `set a (a + n)` -> `change a by n`
* Opti: `set a (a * 2)` -> `change a by a`