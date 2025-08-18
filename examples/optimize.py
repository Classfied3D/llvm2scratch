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
  ]))
]))

proj = optimize(proj)
proj.export("output/out.sprite3")