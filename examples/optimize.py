from llvm2scratch.scratch import *
import llvm2scratch as l2s
import os

def main():
  script_dir = os.path.dirname(os.path.abspath(__file__))
  os.chdir(script_dir)

  proj = Project(ScratchConfig())
  proj.code.append(BlockList([
    ProcedureDef("main", ["%1", "%2"]),
    ControlFlow("if", BoolOp("=", Known(30), Known(0)), BlockList([
      EditVar("set", "hello2", Known(10)),
    ])),
    EditVar("set", "hello3", GetCounter()),
    EditCounter("incr"),
    EditVar("set", "hello4", Op("add", GetVar("hello3"), Known(20))),
    Say(GetVar("hello4")), # Test if wont elide if used too many times for it be faster
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    Say(GetVar("hello4")),
    EditVar("set", "hello5", Op("add", GetVar("hello3"), Known(21))),
    Say(GetVar("hello5")),

    # https://github.com/Classfied3D/llvm2scratch/pull/2#discussion_r3004955618
    # Also I should stop quoting that song TwT - Heathercat123
    Ask(Known("Give me a smile")),
    Say(GetAnswer()),
    Ask(Known("Give me your name")),
    Say(GetAnswer()),
  ]))

  proj.code.append(BlockList([
    ProcedureDef("puts", ["%input"]),
    EditVar("set", "buffer", Known("")),
    EditVar("set", "ptr", GetParam("%input")),
    EditVar("set", "char", GetOfList("atindex", "!stack", GetVar("ptr"))),
    ControlFlow("until", BoolOp("=", GetVar("char"), Known(0)), BlockList([
      EditVar("set", "buffer",
        Op("join", GetVar("buffer"), GetOfList("atindex", "ASCII LOOKUP lol", GetVar("char")))),
      EditVar("change", "ptr", Known(1)),
      EditVar("set", "char", GetOfList("atindex", "!stack", GetVar("ptr"))),
    ])),
    Say(GetVar("buffer")),
    ControlFlow("if", KnownBool(True), BlockList([
      Say(Known("hello")),
    ])),
    ControlFlow("if", KnownBool(True), BlockList([
      StopScript("stopthis"),
    ])),
    EditVar("set", "return", Known(0)),
  ]))

  proj.code.append(BlockList([
    ProcedureDef("test1", []),
    EditVar("set", "b", GetVar("a")),
    ControlFlow("if_else", BoolOp("=", GetVar("a"), GetVar("a")), BlockList([
      EditVar("set", "c", GetVar("b")),
    ]), BlockList([
      EditVar("set", "a", Known(3)),
      EditVar("set", "c", GetVar("b")),
    ])),
    Say(GetVar("a")),
    Say(GetVar("c")),
  ]))

  proj.code.append(BlockList([
    ProcedureDef("test2", []),
    EditVar("set", "b", GetVar("a")),
    EditVar("set", "b", GetVar("r")),
    EditVar("set", "c", GetVar("b")),
    Say(GetVar("c")),
  ]))

  proj = l2s.optimizer.optimize(proj, l2s.getTarget("scratch3"))
  proj.export("output/out.sprite3", Format.Sprite3)

if __name__ == "__main__":
  main()
