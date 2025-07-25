from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

import zipfile
import hashlib
import json
import uuid

EMPTY_SVG = """<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="0" height="0" viewBox="0,0,0,0"></svg>"""
EMPTY_SVG_HASH = hashlib.md5(EMPTY_SVG.encode("utf-8")).hexdigest()
SHORT_OP_TO_OPCODE = {
  # Control
  "reptimes": "control_repeat",
  "until": "control_repeat_until",
  "while": "control_while",
  # Variables
  "set": "data_setvariableto",
  "change": "data_changevariableby",
  "addto": "data_addtolist",
  "replaceat": "data_replaceitemoflist",
  "insertat": "data_insertatlist",
  "deleteat": "data_deleteoflist",
  "deleteall": "data_deletealloflist",
  # Operators
  "add": "operator_add",
  "sub": "operator_subtract",
  "mul": "operator_multiply",
  "div": "operator_divide",
  "mod": "operator_mod",
  "rand_between": "operator_random",
  "join": "operator_join",
  "letter_n_of": "operator_letter_of",
}

Id = str

@dataclass
class ScratchContext:
  vars: dict[str, tuple[Id, KnownValue]] = field(default_factory=dict)
  lists: dict[str, tuple[Id, list[KnownValue]]] = field(default_factory=dict)
  funcs: dict[str, tuple[list[Id], bool]] = field(default_factory=dict)
  blocks: dict[Id, dict] = field(default_factory=dict)
  late_blocks: list[tuple[Id, LateBlock, BlockMeta]] = field(default_factory=list)

  def addBlock(self, id: Id, block: Block, meta: BlockMeta) -> None:
    if not isinstance(block, LateBlock):
      metaless, self = block.getRaw(id, self)
      self.blocks[id] = meta.addRawMeta(metaless)
    else:
      self.late_blocks.append((id, block, meta))

  def addBlockList(self, blocks: BlockList, parent: Id | None=None) -> Id | None:
    """Returns the id of the first block in the list"""
    lastId = parent
    firstId = currId = genId()
    nextId = genId()
    end = False

    for (i, block) in enumerate(blocks.blocks):
      if i == len(blocks.blocks) - 1: nextId = None

      assert currId is not None

      if lastId is not None and block.isStart(): raise ScratchCompException(f"Starting block {type(block)} has blocks before it")
      if end: raise ScratchCompException(f"Reached ending block {type(block)} but more blocks are left")

      end = block.isEnd()

      meta = BlockMeta(lastId, nextId)
      self.addBlock(currId, block, meta)

      lastId = currId
      currId = nextId
      nextId = genId()
    
    if len(blocks.blocks) == 0: firstId = None

    return firstId
  
  def addOrGetVar(self, var_name: str) -> Id:
    if not var_name in self.vars:
      id = genId()
      self.vars.update({var_name: (id, KnownValue(0))})
    else:
      id = self.vars[var_name][0]
    return id
  
  def addOrGetList(self, list_name: str) -> Id:
    if not list_name in self.lists:
      id = genId()
      self.lists.update({list_name: (id, [])})
    else:
      id = self.lists[list_name][0]
    return id
  
  def addFunc(self, func_name: str, param_ids: list[Id], run_without_refresh: bool) -> None:
    if func_name in self.funcs:
      raise ScratchCompException(f"Function {func_name} registered twice")
    self.funcs[func_name] = (param_ids, run_without_refresh)
  
  def getRaw(self) -> dict:
    """Returns json for blocks and vars defined"""
    while len(self.late_blocks) > 0:
      id, block, meta = self.late_blocks.pop()
      raw_block, self = block.getRawLate(id, self)
      self.addBlock(id, RawBlock(raw_block), meta)

    raw_vars = {}
    for name, (id, value) in self.vars.items():
      raw_vars.update({
        id: [name, value.getRawVarInit()]
      })
    
    raw_lists = {}
    for name, (id, values) in self.lists.items():
      raw_lists.update({
        id: [name, [value.getRawVarInit() for value in values]]
      })
    
    raw_blocks = {}
    for id, block in self.blocks.items():
      raw_blocks.update({id: block})

    return {
      "variables": raw_vars,
      "lists": raw_lists,
      "blocks": raw_blocks,
    }

@dataclass
class Block:
  def getRaw(self, _my_id: Id, _ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raise ScratchCompException("Cannot export for generic type 'Block'; must be a derived class")
  
  def isStart(self) -> bool:
    return False

  def isEnd(self) -> bool:
    return False

@dataclass
class StartBlock(Block):
  def isStart(self) -> bool:
    return True

@dataclass
class LateBlock(Block):
  """A block which requires info about the whole program to be added e.g. the id of function parameters which might not yet be defined"""
  def getRaw(self, _my_id: Id, _ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raise ScratchCompException("Cannot call getRaw on a LateBlock because it evaluates after other blocks, call getRawLate")
  
  def getRawLate(self, _my_id: Id, _ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raise ScratchCompException("Cannot export for generic type 'LateBlock'; must be a derived class")

@dataclass
class BlockMeta:
  parent: Id | None = None
  next: Id | None = None
  shadow: bool = False # True if this is the block inside a procedure definition
  x: int = 0
  y: int = 0

  def addRawMeta(self, metaless: dict) -> dict:
    metaless.update({
      "parent": self.parent, "next": self.next,
      "topLevel": self.parent is None, "shadow": self.shadow,
      "x": self.x, "y": self.y,
    })
    return metaless

@dataclass
class BlockList:
  blocks: list[Block]

@dataclass
class RawBlock(Block):
  contents: dict

  def getRaw(self, _: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return self.contents, ctx

@dataclass
class Value:
  """Something that can be in a blocks input e.g. x in Say(x)"""
  def getRawValue(self, _parent: Id, _ctx: ScratchContext) -> tuple[list, ScratchContext]:
    """Gets the json that can be put in the "inputs" field of a block"""
    raise ScratchCompException("Cannot export for generic type 'Value'; must be a derived class")

@dataclass
class BooleanValue(Value):
  """A boolean value (a diamond shaped block)"""

@dataclass
class KnownValue(Value):
  """Something that can be typed in a block input e.g. x in Say(x)"""
  known: str | float

  def getRawValue(self, _: Id, ctx: ScratchContext) -> tuple[list, ScratchContext]:
    return ([1, [10 if isinstance(self.known, str) else 4, str(self.known)]], ctx)
  
  def getRawVarInit(self) -> str | float:
    """Get the raw value to set a var to when it starts with this value"""
    try:
      return float(self.known)
    except ValueError:
      return self.known

# Control
@dataclass
class Repeat(Block):
  op: Literal["reptimes", "until", "while"]
  value: Value
  to_repeat: BlockList

  def getRaw(self, my_id: str, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raw_val, ctx = self.value.getRawValue(my_id, ctx)
    if self.op in ["until", "while"] and not isinstance(self.value, BooleanValue):
      raise ScratchCompException("A regular value cannot be placed in a boolean accepting block")
    
    to_repeat_id = ctx.addBlockList(self.to_repeat, my_id)
    
    input_name = "TIMES" if self.op == "reptimes" else "CONDITION"

    return {
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": {
        input_name: raw_val,
        "SUBSTACK": [2, to_repeat_id]
      },
    }, ctx

# Variables
@dataclass
class GetVariable(Value):
  var_name: str

  def getRawValue(self, _: Id, ctx: ScratchContext) -> tuple[list, ScratchContext]:
    id = ctx.addOrGetVar(self.var_name)
    return [3, [12, self.var_name, id], [10, ""]], ctx

@dataclass
class ModifyVariable(Block):
  op: Literal["set", "change"]
  var_name: str
  value: Value

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    id = ctx.addOrGetVar(self.var_name)
    raw_val, ctx = self.value.getRawValue(my_id, ctx)
    return {
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": {"VALUE": raw_val},
      "fields": {"VARIABLE": [self.var_name, id]}
    }, ctx

@dataclass
class ModifyList(Block):
  op: Literal["addto", "replaceat", "insertat", "deleteat", "deleteall"]
  list_name: str
  index: Value | None
  value: Value | None

  def getRaw(self, my_id: str, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    id = ctx.addOrGetList(self.list_name)
    raw_index = raw_val = None
    if self.index is not None: raw_index, ctx = self.index.getRawValue(my_id, ctx)
    if self.value is not None: raw_val, ctx = self.value.getRawValue(my_id, ctx)
    return {
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": {"INDEX": raw_index, "ITEM": raw_val},
      "fields": {"LIST": [self.list_name, id]},
    }, ctx

# Operators
@dataclass
class TwoOp(Value): # TODO: make this be able to use one input
  op: Literal["add", "sub", "mul", "div", "mod", "rand_between", "join", "letter_n_of"]
  left: Value
  right: Value

  def getRawValue(self, parent: Id, ctx: ScratchContext) -> tuple[list, ScratchContext]:
    raw_left, ctx = self.left.getRawValue(parent, ctx)
    raw_right, ctx = self.right.getRawValue(parent, ctx)

    match self.op:
      case "rand_between":
        lft_param = "FROM"
        rgt_param = "TO"
      case "join":
        lft_param = "STRING1"
        rgt_param = "STRING2"
      case "letter_n_of":
        lft_param = "LETTER"
        rgt_param = "STRING"
      case _:
        lft_param = "NUM1"
        rgt_param = "NUM2"

    id = genId()
    ctx.addBlock(id, RawBlock({
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": {
        lft_param: raw_left,
        rgt_param: raw_right
      }
    }), BlockMeta(parent))

    return [3, id, [10, ""]], ctx

# Procedures
@dataclass
class ProcedureDef(StartBlock):
  name: str
  params: list[str]
  run_without_refresh: bool = True

  def getRaw(self, my_id: str, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    proto_id = genId()
    param_ids = [genId() for _ in self.params]

    ctx.addFunc(self.name, param_ids, self.run_without_refresh)

    param_block_ids = []
    for param in self.params:
      param_block_id = genId()
      param_block_ids.append(param_block_id)
      ctx.addBlock(param_block_id, RawBlock({
        "opcode": "argument_reporter_string_number",
        "fields": {"VALUE": [param, None]}
      }), BlockMeta(proto_id))

    ctx.addBlock(proto_id, RawBlock({
      "opcode": "procedures_prototype",
      "inputs": dict(zip(param_ids, [list((1, id)) for id in param_block_ids])),
      "mutation": {
        "tagName": "mutation",
        "children": [],
        "proccode": self.name + (" %s" * len(self.params)),
        "argumentids": json.dumps(param_ids),
        "argumentnames": json.dumps([param for param in self.params]),
        "argumentdefaults": json.dumps(["" for _ in self.params]),
        "warp": json.dumps(self.run_without_refresh)
      }
    }), BlockMeta(my_id, None, True))

    return {
      "opcode": "procedures_definition",
      "inputs": {"custom_block": [1, proto_id]}
    }, ctx

@dataclass
class ProdcedureCall(LateBlock):
  proc_name: str
  arguments: list[Value]

  def getRawLate(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    values = []
    for arg in self.arguments:
      value, ctx = arg.getRawValue(my_id, ctx)
      values.append(value)

    param_ids, run_without_refresh = ctx.funcs[self.proc_name]

    return {
      "opcode": "procedures_call",
      "inputs": dict(zip(param_ids, values)),
      "mutation": {
        "tagName": "mutation",
        "children": [],
        "proccode": self.proc_name + (" %s" * len(param_ids)),
        "argumentids": json.dumps(param_ids),
        "warp": json.dumps(run_without_refresh)
      }
    }, ctx

@dataclass
class GetParameter(Value):
  param_name: str

  def getRawValue(self, parent: str, ctx: ScratchContext) -> tuple[list, ScratchContext]:
    id = genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "argument_reporter_string_number",
      "fields": {"VALUE": [self.param_name, None]}
    }), BlockMeta(parent))

    return [3, id, [10, ""]], ctx

class ScratchCompException(Exception):
  """Exception when compiling to scratch"""
  pass

def genId() -> Id:
  return uuid.uuid4().hex

def exportSpriteData(ctx: ScratchContext) -> str:
  res = {
    "isStage": False,
    "name": "Sprite",
    "variables": {},
    "lists": {},
    "broadcasts": {},
    "blocks": {},
    "currentCostume": 0,
    "costumes":[
      # A costume must be defined for the sprite to load
      {
        "name": "",
        "bitmapResolution": 1,
        "dataFormat": "svg",
        "assetId": EMPTY_SVG_HASH,
        "md5ext": f"{EMPTY_SVG_HASH}.svg",
        "rotationCenterX": 0,
        "rotationCenterY": 0
      }
    ],
    "sounds": [],
    "volume": 100,
    "visible": False,
    "x": 0,
    "y": 0,
    "size": 100,
    "direction": 90,
    "draggable": False,
    "rotationStyle": "all around"
  }

  res.update(ctx.getRaw())

  print(res)

  return json.dumps(res)

def exportSpriteFile(ctx: ScratchContext, file: zipfile.ZipFile) -> None:
  file.writestr("Sprite/sprite.json", exportSpriteData(ctx))
  file.writestr(f"Sprite/{EMPTY_SVG_HASH}.svg", EMPTY_SVG)

def main():
  ctx = ScratchContext()
  ctx.addBlockList(BlockList([
    ProcedureDef("main", ["%1", "%2"]),
    ModifyList("insertat", "hello2", KnownValue(5), TwoOp("div", GetVariable("hello3"), KnownValue(30))),
    ModifyVariable("set", "hello2", KnownValue(10)),
    Repeat("reptimes", TwoOp("div", GetVariable("hello3"), KnownValue(30)), BlockList([
      ModifyVariable("set", "hello2", KnownValue(10)),
    ])),
    ProdcedureCall("main", [TwoOp("sub", GetVariable("hello3"), GetParameter("%1")), KnownValue(3)])
  ]))
  with zipfile.ZipFile("out.sprite3", "w") as zipf:
    exportSpriteFile(ctx, zipf)

if __name__ == "__main__":
  main()