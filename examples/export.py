from llvm2scratch.scratch import *
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

proj = Project(ScratchConfig())
proj.code.append(BlockList([
  ProcedureDef("main", ["%1", "%2"]),
  EditList("deleteall", "hello2", None, None),
  EditList("insertat", "hello2", Known(5), Op("div", GetVar("hello3"), Known(30))),
  EditVar("set", "hello2", Known(10)),
  ControlFlow("until", BoolOp("=", Op("div", GetVar("hello3"), Known(30)), Known(0)), BlockList([
    EditVar("set", "hello2", Known(10)),
  ])),
  ProcedureCall("main", [Op("floor", GetOfList("atindex", "hello3", GetParameter("%1"))), Known(3)]),
  EditCounter("incr"),
  Say(GetCounter()),
  Broadcast(Op("length_of", GetVar("hello2")), True),
  StopScript("stopall"),
]))

proj.code.append(BlockList([
  OnBroadcast("hi"),
  Say(Known("hello")),
]))

proj.export("output/out.sprite3")
