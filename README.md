# LLVM2Scratch

An LLVM backend to convert LLVM IR to [MIT Scratch](https://scratch.mit.edu), a block based coding language. This allows a lot of programs written languages which can compile to LLVM (C/C++/Rust/etc) to be ported into scratch.

## Dependencies

* **llvm2py**@0.2.0b0 (installation requires llvm@19)
  * If on macos, would recommend running `sudo port install llvm-19` instead of `brew install llvm@19` because brew doesn't have precompiled bottles for older versions of macos, and llvm takes hours to compile
  * Then `LLVM_DIR=/opt/local/libexec/llvm-19/lib/cmake/llvm/ LLVM_CONFIG=/opt/local/libexec/llvm-19/bin/llvm-config CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install llvm2py` (`LLVM_CONFIG=/usr/local/opt/llvm@19/bin/llvm-config` for brew)

* Because we only depend on one lib that's difficult to install, there is no need for a venv

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

* Use Scratch's call stack to store variables; when calling a different function pass all temporaries as parameters into a new function which calls the function and executes the rest of the code, rather than adding to the 'stack list' used in alloca. Could also use a variable for each local variable in each function (as long as it cannot recurse), but this might get annoying, so have an option to disable this
e.g.
fn main
  global a = 2
  global b = 2
  call main2(global a, global b, (add global a, global b))

fn main2(%a, %b, %c)
  global d = call half(%c)

fn half(%c)
  ret %c / 2

* Use broadcasts for branch instructions going backward (using fns could cause stack overflow). Use if/else etc for going forward. Use stop this script for return

* .sb3lib + .sb3exe = .sb3 OR .sprite3

* Global vars from both libs and exes should be allocated to the stack at the start

* Optimisation - No need to assign value if only used once in the next instruction
  * Make sure nothing the value depends on changes

* Optimisation - Following Code: Could change stack size by 14 instead
```
Set `@.str` to (`stack size` + 1)
Change `stack size` by 13
Change `stack size` by 1
Set `@a` to `stack size`
```

* Optimisiation - different return value for different funcs

* Opti - Variable re-use, also change by for "set a  (a + 1)" and change by self for "set a (a * 2)"

From scratch VM
```
 getCounter () {
        return this._counter;
    }

    clearCounter () {
        this._counter = 0;
    }

    incrCounter () {
        this._counter++;
    }
```

```

    forEach (args, util) {
        const variable = util.target.lookupOrCreateVariable(
            args.VARIABLE.id, args.VARIABLE.name);

        if (typeof util.stackFrame.index === 'undefined') {
            util.stackFrame.index = 0;
        }

        if (util.stackFrame.index < Number(args.VALUE)) {
            util.stackFrame.index++;
            variable.value = util.stackFrame.index;
            util.startBranch(1, true);
        }
    }
```