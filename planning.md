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