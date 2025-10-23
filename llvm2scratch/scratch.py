"""Scratch file output for the LLVM -> Scratch compiler"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, NamedTuple
from enum import Enum

import zipfile
import hashlib
import random
import json
import math

DEFAULT_BROADCAST_MESSAGE = "message1"
EMPTY_SVG = """<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="0" height="0" viewBox="0,0,0,0"></svg>"""
EMPTY_SVG_HASH = hashlib.md5(EMPTY_SVG.encode("utf-8")).hexdigest()
SHORT_OP_TO_OPCODE = {
  # Control
  "if": "control_if",
  "if_else": "control_if_else",
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
  "atindex": "data_itemoflist",
  "indexof": "data_itemnumoflist",
  # Operators
  "add": "operator_add",
  "sub": "operator_subtract",
  "mul": "operator_multiply",
  "div": "operator_divide",
  "mod": "operator_mod",
  "rand_between": "operator_random",
  "join": "operator_join",
  "letter_n_of": "operator_letter_of",
  "length_of": "operator_length",
  "round": "operator_round",
  "bool_as_int": "operator_round",
  "not": "operator_not",
  "and": "operator_and",
  "or": "operator_or",
  "=": "operator_equals",
  "<": "operator_lt",
  ">": "operator_gt",
  "contains": "operator_contains",
}

Id = str

class ScratchConfig(NamedTuple):
  # TODO add an option to replace hacked blocks with a working version that isn't hacked
  # (since they're used for better performance anyway)
  invis_blocks: bool = False # Prevent blocks from rendering; stops editor lag

@dataclass
class Project:
  cfg: ScratchConfig
  code: list[BlockList] = field(default_factory=list)
  lists: dict[str, list[int | float | str | bool]] = field(default_factory=dict)

  def export(self, filename: str) -> None:
    """Exports the project into a .sprite3 file"""
    exportScratchFile(self.getCtx(), filename)

  def getCtx(self) -> ScratchContext:
    """Converts the project into a ScratchContext which can be used to get the raw project"""
    ctx = ScratchContext(self.cfg) # pyright: ignore[reportCallIssue] this error only appears sometimes??
    for name, scratch_list in self.lists.items():
      ctx.addOrGetList(name, scratch_list)
    for block_list in self.code:
      ctx.addBlockList(block_list)
    return ctx

@dataclass
class ScratchContext:
  cfg: ScratchConfig = field(default_factory=ScratchConfig)
  vars: dict[str, tuple[Id, Known]] = field(default_factory=dict)
  lists: dict[str, tuple[Id, list[int | float | str | bool]]] = field(default_factory=dict)
  broadcasts: dict[str, Id] = field(default_factory=dict)
  funcs: dict[str, tuple[list[Id], bool]] = field(default_factory=dict)
  blocks: dict[Id, dict] = field(default_factory=dict)
  late_blocks: list[tuple[Id, LateBlock, BlockMeta]] = field(default_factory=list)
  exported: bool = False

  def addBlock(self, id: Id, block: Block, meta: BlockMeta) -> None:
    if not isinstance(block, LateBlock):
      metaless, self = block.getRaw(id, self)
      if self.cfg.invis_blocks: meta.shadow = True
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

  def addOrGetVar(self, var_name: str, default_val: Known | None = None) -> Id:
    if default_val is None: default_val = Known(0)
    if not var_name in self.vars:
      id = genId()
      self.vars.update({var_name: (id, default_val)})
    else:
      id = self.vars[var_name][0]
    return id

  def addOrGetList(self, list_name: str, default_val: list[int | float | str | bool] | None = None) -> Id:
    if default_val is None: default_val = []
    if not list_name in self.lists:
      id = genId()
      self.lists.update({list_name: (id, default_val)})
    else:
      id = self.lists[list_name][0]
      if len(default_val) > 0:
        if len(self.lists[list_name][1]) > 0: raise ScratchCompException(f"List {list_name} given default value twice")
        self.lists[list_name] = (id, default_val)
    return id

  def addFunc(self, func_name: str, param_ids: list[Id], run_without_refresh: bool) -> None:
    if func_name in self.funcs:
      raise ScratchCompException(f"Function {func_name} registered twice")
    self.funcs[func_name] = (param_ids, run_without_refresh)

  def addBroadcast(self, name: str) -> Id:
    """Adds a broadcast with the given name, returns the id of the broadcast"""
    if name in self.broadcasts:
      return self.broadcasts[name]

    id = genId()
    self.broadcasts[name] = id
    return id

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
      raw_lists[id] = [
        name,
        ["true" if v is True else "false" if v is False else v for v in values]]

    raw_broadcasts = {}
    for name, id in self.broadcasts.items():
      raw_broadcasts.update({
        id: name
      })

    raw_blocks = {}
    for id, block in self.blocks.items():
      raw_blocks.update({id: block})

    return {
      "variables": raw_vars,
      "lists": raw_lists,
      "blocks": raw_blocks,
    }

class ScratchCast(Enum):
  """How the block will cast a value. Affects number coercion and boolean casting"""
  TO_STR = 1
  TO_INT = 2

@dataclass
class Block():
  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raise ScratchCompException("Cannot export for generic type 'Block'; must be a derived class")

  def isStart(self) -> bool:
    return False

  def isEnd(self) -> bool:
    return False

class StartBlock(Block):
  def isStart(self) -> bool:
    return True

class EndBlock(Block):
  def isEnd(self) -> bool:
    return True

class LateBlock(Block):
  """A block which requires info about the whole program to be added e.g. the id of function parameters which might not yet be defined"""
  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raise ScratchCompException("Cannot call getRaw on a LateBlock because it evaluates after other blocks, call getRawLate")

  def getRawLate(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
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

class BlockList:
  blocks: list[Block]
  end: bool

  def __init__(self, blocks: list[Block] | None=None):
    if blocks is None:
      blocks = []
    self.end = False
    for block in blocks:
      if self.end: raise ScratchCompException("List of blocks contains blocks after an ending block")
      self.end |= block.isEnd()
    self.blocks = blocks

  def add(self, blocks: Block | BlockList | list[Block]) -> None:
    if isinstance(blocks, list):
      self.add(BlockList(blocks))
      return

    if self.end:
      if isinstance(blocks, Block):
        raise ScratchCompException(f"Reached ending block {self.blocks[-1]}, attempted to add {blocks}")
      elif len(blocks.blocks) > 0:
        raise ScratchCompException(f"Reached ending block {self.blocks[-1]}, attempted to add {blocks.blocks[0]}")

    if isinstance(blocks, Block):
      self.blocks.append(blocks)
    else:
      self.blocks += blocks.blocks.copy()
      self.end |= blocks.end

  def __len__(self):
    return len(self.blocks)

@dataclass
class RawBlock(Block):
  contents: dict

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return self.contents, ctx

class Value:
  """Something that can be in a blocks input e.g. x in Say(x)"""
  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    """Gets the json that can be put in the "inputs" field of a block"""
    raise ScratchCompException("Cannot export for generic type 'Value'; must be a derived class")

class BooleanValue(Value):
  """A boolean value (a diamond shaped block)"""
  def getRawBoolValue(self, parent: str, ctx: ScratchContext) -> tuple[list | None, ScratchContext]:
    raise ScratchCompException("Cannot export for generic type 'BooleanValue'; must be a derived class")

class Known(Value):
  """Something that can be typed in a block input e.g. x in Say(x)"""
  known: str | float | bool

  def __init__(self, val: str | float | bool | int):
    if not isinstance(val, int):
      self.known = val
    else:
      self.known = float(val)

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    raw = self.known
    if not isinstance(self.known, str):
      raw = float(raw)

    if raw == float("+inf"):
      raw = "Infinity"
    elif raw == float("-inf"):
      raw = "-Infinity"
    elif isinstance(raw, float) and math.isnan(raw):
      raw = "NaN"

    return [1, [(10 if isinstance(self.known, str) else 4), raw]], ctx

  def getRawVarInit(self) -> str | float | bool:
    """Get the raw value to set a var to when it starts with this value"""
    if isinstance(self.known, bool): return "true" if self.known else "false"
    try:
      return float(self.known)
    except ValueError:
      return self.known

class KnownBool(Known, BooleanValue):
  def __init__(self, known: bool):
    self.known = known

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    return Known(int(self.known)).getRawValue(parent, ctx, cast)

  def getRawBoolValue(self, parent: str, ctx: ScratchContext) -> tuple[list | None, ScratchContext]:
    if not self.known:
      return None, ctx # If false
    return BoolOp("not", KnownBool(False)).getRawBoolValue(parent, ctx)

  def getRawVarInit(self) -> str:
    """Get the raw value to set a var to when it starts with this value"""
    return "true" if self.known else "false"

# Looks
@dataclass
class Say(Block):
  value: Value

  def getRaw(self, my_id: str, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raw_msg, ctx = self.value.getRawValue(my_id, ctx, ScratchCast.TO_STR)
    return {
      "opcode": "looks_say",
      "inputs": {
        "MESSAGE": raw_msg
      }
    }, ctx

# Events
# Thank you @RetrogradeDev for this wonderful MIT licensed broadcast code which I have now stolen
@dataclass
class Broadcast(Block):
  value: Value
  wait: bool

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    opcode = "event_broadcastandwait" if self.wait else "event_broadcast"

    broadcast_name = scratchCastToStr(self.value) if isinstance(self.value, Known) else DEFAULT_BROADCAST_MESSAGE
    id = ctx.addBroadcast(broadcast_name)

    if not isinstance(self.value, Known):
      raw_input_value, ctx = self.value.getRawValue(my_id, ctx, ScratchCast.TO_STR)
      # So that if the block is removed you get a normal broadcast
      raw_value = [3, raw_input_value[1], [11, broadcast_name, id]]
    else:
      raw_value = [1, [11, broadcast_name, id]]

    return {
      "opcode": opcode,
      "inputs": {"BROADCAST_INPUT": raw_value},
    }, ctx

@dataclass
class OnBroadcast(StartBlock):
  name: str

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    id = ctx.addBroadcast(self.name)

    return {
      "opcode": "event_whenbroadcastreceived",
      "fields": {"BROADCAST_OPTION": [self.name, id]}
    }, ctx

# Control
class OnStartFlag(StartBlock):
  def getRaw(self, my_id: str, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return {
      "opcode": "event_whenflagclicked"
    }, ctx

FlowOp = Literal["if", "if_else", "reptimes", "until", "while"]
@dataclass
class ControlFlow(Block):
  op: FlowOp
  value: Value
  blocks: BlockList
  else_blocks: BlockList | None = None

  def __post_init__(self):
    if self.op in ["if", "if_else", "until", "while"] and not isinstance(self.value, BooleanValue):
      raise ScratchCompException("A regular value cannot be placed in a boolean accepting block")

    if self.op == "if_else" and self.else_blocks is None:
      raise ScratchCompException("An if-else statement must contain blocks in the else case")

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    if self.op in ["if", "if_else", "until", "while"]:
      assert isinstance(self.value, BooleanValue)
      raw_val, ctx = self.value.getRawBoolValue(my_id, ctx)
    else:
      raw_val, ctx = self.value.getRawValue(my_id, ctx, ScratchCast.TO_INT)

    blocks_id = ctx.addBlockList(self.blocks, my_id)

    input_name = "TIMES" if self.op == "reptimes" else "CONDITION"

    inputs = {"SUBSTACK": [2, blocks_id]}
    if raw_val is not None: inputs.update({input_name: raw_val})

    if self.op == "if_else":
      assert self.else_blocks is not None
      else_blocks_id = ctx.addBlockList(self.else_blocks, my_id)
      inputs.update({"SUBSTACK2": [2, else_blocks_id]})

    return {
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": inputs
    }, ctx

@dataclass
class StopScript(EndBlock):
  op: Literal["stopthis", "stopall"]

  def getRaw(self, my_id: str, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return {
      "opcode": "control_stop",
      "fields": {"STOP_OPTION": ["all" if self.op == "stopall" else "this script", None]}
    }, ctx

@dataclass
class GetCounter(Value):
  """Get the value of the special 'hacked' counter block"""

  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "control_get_counter"
    }), BlockMeta(parent))

    return [3, id, [10, ""]], ctx

@dataclass
class EditCounter(Block):
  """Increment/Assign zero to the special 'hacked' counter block"""

  op: Literal["incr", "clear"]

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return {
      "opcode": "control_incr_counter" if self.op == "incr" else "control_clear_counter",
    }, ctx

# Variables
@dataclass
class GetVar(Value):
  var_name: str

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.addOrGetVar(self.var_name)
    return [3, [12, self.var_name, id], [10, ""]], ctx

@dataclass
class EditVar(Block):
  op: Literal["set", "change"]
  var_name: str
  value: Value

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    id = ctx.addOrGetVar(self.var_name)
    # NOTE: technically set variable doesn't cast but we need to assume the worst scenario
    raw_val, ctx = self.value.getRawValue(my_id, ctx, (ScratchCast.TO_STR if self.op == "set" else ScratchCast.TO_INT))
    return {
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": {"VALUE": raw_val},
      "fields": {"VARIABLE": [self.var_name, id]}
    }, ctx

@dataclass
class EditList(Block):
  op: Literal["addto", "replaceat", "insertat", "deleteat", "deleteall"]
  list_name: str
  index: Value | None
  item: Value | None

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    list_id = ctx.addOrGetList(self.list_name)
    inputs = {}

    if self.index is not None:
      if self.op in ["addto", "deleteall"]:
        raise ScratchCompException(f"{self.op} does not support an index value")
      raw_index, ctx = self.index.getRawValue(my_id, ctx, ScratchCast.TO_INT)
      inputs.update({"INDEX": raw_index})

    if self.item is not None:
      raw_item, ctx = self.item.getRawValue(my_id, ctx, ScratchCast.TO_STR)
      if self.op in ["deleteat", "deleteall"]:
        raise ScratchCompException(f"{self.op} does not support an item")
      inputs.update({"ITEM": raw_item})

    return {
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": inputs,
      "fields": {"LIST": [self.list_name, list_id]},
    }, ctx

@dataclass
class GetOfList(Value):
  op: Literal["atindex", "indexof"]
  list_name: str
  value: Value

  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = genId()
    list_id = ctx.addOrGetList(self.list_name)

    raw_value, ctx = self.value.getRawValue(parent, ctx, (ScratchCast.TO_INT if self.op == "atindex" else ScratchCast.TO_STR))

    input_name = "INDEX" if self.op == "atindex" else "ITEM"

    ctx.addBlock(id, RawBlock({
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": {input_name: raw_value},
      "fields": {"LIST": [self.list_name, list_id]},
    }), BlockMeta(parent))

    return [3, id, [10, ""]], ctx

# Operators
OperatorsCodes = Literal["add", "sub", "mul", "div", "mod", "rand_between", "join", "letter_n_of", "length_of", "round", "bool_as_int",
                         "abs", "floor", "ceiling", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "ln", "log", "e ^", "10 ^"]
@dataclass
class Op(Value): # TODO: make this be able to use one input
  op: OperatorsCodes
  left: Value
  right: Value | None = None

  def __post_init__(self):
    takes_one_op = self.op in ["length_of", "round", "bool_as_int"] or self.op not in SHORT_OP_TO_OPCODE # if op is length_of, round or is a general op
    given_one_op = self.right is None

    if takes_one_op != given_one_op:
      raise ScratchCompException(f"{self.op} takes {1 if takes_one_op else 2} operands, given {1 if given_one_op else 2}")

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    # We don't need to round the number if it gets casted to int anyway
    if self.op == "bool_as_int" and cast == ScratchCast.TO_INT:
      return self.left.getRawValue(parent, ctx, cast)

    id = genId()

    takes_one_op = self.right is None

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
      case "length_of":
        lft_param = "STRING"
      case _:
        lft_param = "NUM1"
        rgt_param = "NUM2"
        if takes_one_op:
          lft_param = "NUM"

    opcode = SHORT_OP_TO_OPCODE.setdefault(self.op, "operator_mathop")

    casts_left_input_to = ScratchCast.TO_INT
    if self.op in ["join", "length_of"]:
      casts_left_input_to = ScratchCast.TO_STR

    raw_left, ctx = self.left.getRawValue(id, ctx, casts_left_input_to)
    inputs = {lft_param: raw_left}
    if self.right is not None:
      casts_right_input_to = casts_left_input_to
      if self.op == "letter_n_of":
        casts_right_input_to = ScratchCast.TO_STR

      raw_right, ctx = self.right.getRawValue(id, ctx, casts_right_input_to)
      inputs.update({rgt_param: raw_right})

    fields = {}
    if opcode == "operator_mathop":
      fields.update({"OPERATOR": [self.op, None]})

    ctx.addBlock(id, RawBlock({
      "opcode": opcode,
      "inputs": inputs,
      "fields": fields
    }), BlockMeta(parent))

    return [3, id, [10, ""]], ctx

BoolOpCodes = Literal["not", "and", "or", "=", "<", ">", "contains"]
@dataclass
class BoolOp(BooleanValue):
  op: BoolOpCodes
  left: Value
  right: Value | None = None

  def __post_init__(self):
    if (not isinstance(self.left, BooleanValue) and self.op in ["not", "and", "or"]) or \
       (not isinstance(self.right, BooleanValue) and self.op in ["and", "or"]):
      raise ScratchCompException(f"BoolOp {self.op} only accepts booleans")

    given_one_op = self.right is None
    takes_one_op = self.op == "not"
    if takes_one_op != given_one_op:
      raise ScratchCompException(f"{self.op} takes {1 if takes_one_op else 2} operands, given {1 if given_one_op else 2}")

  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = genId()

    raw_right = None
    if self.op in ["not", "and", "or"]:
      assert isinstance(self.left, BooleanValue)
      raw_left, ctx = self.left.getRawBoolValue(id, ctx)
      if not self.right is None:
        assert isinstance(self.right, BooleanValue)
        raw_right, ctx = self.right.getRawBoolValue(id, ctx)
    else:
      raw_left, ctx = self.left.getRawValue(id, ctx, ScratchCast.TO_STR)
      assert self.right is not None
      raw_right, ctx = self.right.getRawValue(id, ctx, ScratchCast.TO_STR)

    match self.op:
      case "not":
        lft_param = "OPERAND"
      case "contains":
        lft_param = "STRING1"
        rgt_param = "STRING2"
      case _:
        lft_param = "OPERAND1"
        rgt_param = "OPERAND2"

    inputs = {}
    if raw_left is not None: inputs.update({lft_param: raw_left})
    if raw_right is not None: inputs.update({rgt_param: raw_right})

    ctx.addBlock(id, RawBlock({
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": inputs,
    }), BlockMeta(parent))

    return [2, id], ctx

  def getRawBoolValue(self, parent: str, ctx: ScratchContext) -> tuple[list | None, ScratchContext]:
    return self.getRawValue(parent, ctx, ScratchCast.TO_INT)

# Procedures
@dataclass
class ProcedureDef(StartBlock):
  name: str
  params: list[str]
  run_without_refresh: bool = True

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    proto_id = genId()
    param_ids = [genId() for _ in self.params]

    ctx.addFunc(self.name, param_ids, self.run_without_refresh)

    param_block_ids = []
    for param in self.params:
      param_block_id = genId()
      param_block_ids.append(param_block_id)
      ctx.addBlock(param_block_id, RawBlock({
        "opcode": "argument_reporter_string_number",
        "fields": {"VALUE": [sanitizeProcName(param, True), None]}
      }), BlockMeta(proto_id))

    ctx.addBlock(proto_id, RawBlock({
      "opcode": "procedures_prototype",
      "inputs": dict(zip(param_ids, [list((1, id)) for id in param_block_ids])),
      "mutation": {
        "tagName": "mutation",
        "children": [],
        "proccode": sanitizeProcName(self.name, False) + (" %s" * len(self.params)),
        "argumentids": json.dumps(param_ids),
        "argumentnames": json.dumps([sanitizeProcName(param, True) for param in self.params]),
        "argumentdefaults": json.dumps(["" for _ in self.params]),
        "warp": json.dumps(self.run_without_refresh)
      }
    }), BlockMeta(my_id, None, True))

    return {
      "opcode": "procedures_definition",
      "inputs": {"custom_block": [1, proto_id]}
    }, ctx

@dataclass
class ProcedureCall(LateBlock):
  proc_name: str
  arguments: list[Value]

  def getRawLate(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    values = []
    for arg in self.arguments:
      # We don't know how the args will be casted so we assume the worst scenatio
      value, ctx = arg.getRawValue(my_id, ctx, ScratchCast.TO_STR)
      values.append(value)

    param_ids, run_without_refresh = ctx.funcs[self.proc_name]

    return {
      "opcode": "procedures_call",
      "inputs": dict(zip(param_ids, values)),
      "mutation": {
        "tagName": "mutation",
        "children": [],
        "proccode": sanitizeProcName(self.proc_name, False) + (" %s" * len(param_ids)),
        "argumentids": json.dumps(param_ids),
        "warp": json.dumps(run_without_refresh)
      }
    }, ctx

@dataclass
class GetParameter(Value):
  param_name: str

  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "argument_reporter_string_number",
      "fields": {"VALUE": [sanitizeProcName(self.param_name, True), None]}
    }), BlockMeta(parent))

    return [3, id, [10, ""]], ctx


class ScratchCompException(Exception):
  """Exception when compiling to scratch"""
  pass

def sanitizeProcName(name: str, is_param: bool) -> str:
  """Fixes the Bunching Blocks Bug (https://en.scratch-wiki.info/wiki/My_Blocks#Glitches)
  and the hasOwnProperty bug by replacing % with a similar unicode character when necessary"""
  if (is_param and name in ["%b", "%n"]) or (not is_param and name == "%"):
    return name.replace("%", "\uFF05")
  elif not is_param and name == "hasOwnProperty":
    return name + ":bro why"
  return name

def scratchCastToNum(value: Known) -> float:
  """Performs the same casting to number as scratch"""
  raw = value.known
  if not isinstance(raw, float):
    try:
      raw = float(raw)
    except ValueError:
      raw = math.nan

  return 0 if math.isnan(raw) else raw

def scratchCastToBool(value: Known) -> bool:
  """Performs the same casting to bool as scratch"""
  raw = value.known
  match raw:
    case str():
      return raw.lower() not in ["", "0", "false"]
    case float():
      return not (raw == 0 or math.isnan(raw))
    case bool():
      return raw
  raise AssertionError("Should be unreachable")

def scratchCastToStr(value: Known) -> str:
  """Performs the same casting to str as scratch"""
  raw = value.known
  if isinstance(raw, bool): return "true" if raw else "false"
  return str(raw)

def scratchCompare(left: Known, right: Known) -> float:
  """
  Works out the difference between two Known values like scratch does for comparison operators
  Negative number if left < right; 0 if equal; positive otherwise
  """
  try:
    left_val = float(left.known)
    right_val = float(right.known)

    # Sorry mathematicians lol
    if left_val == float("+inf") and right_val == float("+inf") or \
       left_val == float("-inf") and right_val == float("-inf"):
      return 0

    return left_val - right_val
  except ValueError:
    left_val = scratchCastToStr(left).lower()
    right_val = scratchCastToStr(right).lower()
    return 0 if left_val == right_val else (-1 if left_val < right_val else 1)

def genId() -> Id:
  return random.randbytes(16).hex()

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
    "visible": True,
    "x": 0,
    "y": 0,
    "size": 100,
    "direction": 90,
    "draggable": False,
    "rotationStyle": "all around"
  }

  res.update(ctx.getRaw())

  return json.dumps(res)

def exportScratchFile(ctx: ScratchContext, file: str) -> None:
  """Exports scratch code to a .sprite3 file"""
  with zipfile.ZipFile(file, "w") as zipf:
    zipf.writestr("Sprite/sprite.json", exportSpriteData(ctx))
    zipf.writestr(f"Sprite/{EMPTY_SVG_HASH}.svg", EMPTY_SVG)
