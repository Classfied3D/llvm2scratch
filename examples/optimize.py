from llvm2scratch.scratch import *
from llvm2scratch.optimizer import optimize
import os

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
  Say(GetVar("hello4")), # Test if will elide if used too many times for it be faster
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
]))

proj.code.append(BlockList([
  ProcedureDef("puts", ["%input"]),
  EditVar("set", "buffer", Known("")),
  EditVar("set", "ptr", GetParameter("%input")),
  EditVar("set", "char", GetOfList("atindex", "!stack", GetVar("ptr"))),
  ControlFlow("until", BoolOp("=", GetVar("char"), Known(0)), BlockList([
    EditVar("set", "buffer",
      Op("join", GetVar("buffer"), GetOfList("atindex", "ASCII LOOKUP lol", GetVar("char")))),
    EditVar("change", "ptr", Known(1)),
    EditVar("set", "char", GetOfList("atindex", "!stack", GetVar("ptr"))),
  ])),
  Say(GetVar("buffer")),
  EditVar("set", "return", Known(0)),
]))

proj = optimize(proj)
proj.export("output/out.sprite3")