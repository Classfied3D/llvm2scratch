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
# https://github.com/scratchfoundation/scratch-editor/blob/develop/packages/scratch-vm/src/util/uid.js#L11
VALID_UID_CHARACTERS = "!#%()*+,-./:;=?@[]^_`{|}~ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
# List of UIDs 5 or less characters long reserved by the block palette
# https://github.com/scratchfoundation/scratch-editor/blob/develop/packages/scratch-gui/src/lib/make-toolbox-xml.js
PALETTE_UIDS = ["while", "timer", "of", "movex", "movey", "setx", "sety"]
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
  "str_to_float": "operator_add",
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

  minify: bool = True             # Optimize project.json's size by simplifing uids, removing falsy fields, etc

  minify_break_glow: bool = False # Removing the parent key when minifing prevents blocks in the same
                                  # sprite from glowing correctly due to a js error - minify futher and
                                  # allow this error to occur

  hide_blocks: bool = False       # Prevent blocks from rendering in the editor by setting shadow: true on top
                                  # level blocks: stops editor lag

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
  generated_ids: int = 0
  generated_var_ids: int = 0
  exported: bool = False

  def addBlock(self, id: Id, block: Block, meta: BlockMeta) -> None:
    if not isinstance(block, LateBlock):
      metaless, self = block.getRaw(id, self)
      # Only blocks without parents need shadow: true to hide them
      if self.cfg.hide_blocks and meta.parent is None: meta.shadow = True
      self.blocks[id] = meta.addRawMeta(metaless, self)
    else:
      self.late_blocks.append((id, block, meta))

  def addBlockList(self, blocks: BlockList, parent: Id | None=None) -> Id | None:
    """Returns the id of the first block in the list"""
    last_id = parent
    first_id = curr_id = self.genId()
    next_id = self.genId()
    end = False

    for (i, block) in enumerate(blocks.blocks):
      if i == len(blocks.blocks) - 1: next_id = None

      is_start = last_id is None

      assert curr_id is not None

      if last_id is not None and block.isStart(): raise ScratchCompException(f"Starting block {type(block)} has blocks before it")
      if end: raise ScratchCompException(f"Reached ending block {type(block)} but more blocks are left")

      end = block.isEnd()

      meta = BlockMeta(last_id, next_id)
      self.addBlock(curr_id, block, meta)

      last_id = curr_id
      curr_id = next_id
      next_id = self.genId()

    if len(blocks.blocks) == 0: first_id = None

    return first_id

  def addOrGetVar(self, var_name: str, default_val: Known | None = None) -> Id:
    if default_val is None: default_val = Known(0)
    if not var_name in self.vars:
      id = self.genId(True)
      self.vars.update({var_name: (id, default_val)})
    else:
      id = self.vars[var_name][0]
    return id

  def addOrGetList(self, list_name: str, default_val: list[int | float | str | bool] | None = None) -> Id:
    if default_val is None: default_val = []
    if not list_name in self.lists:
      id = self.genId(True)
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

    id = self.genId(True)
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
      raw_vars[id] = [name, value.getRawVarInit()]

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

  def numericToStrUID(self, n: int) -> str:
    base = len(VALID_UID_CHARACTERS)
    if n == 0:
      return VALID_UID_CHARACTERS[0]
    digits = []
    while n:
      digits.append(VALID_UID_CHARACTERS[n % base])
      n //= base
    return "".join(reversed(digits))

  def genId(self, is_var_id=False) -> Id:
    if not self.cfg.minify:
      return random.randbytes(16).hex()
    else:
      invalid = True
      while invalid:
        id = self.numericToStrUID(self.generated_var_ids if is_var_id else self.generated_ids)
        invalid = id in PALETTE_UIDS
        if is_var_id:
          self.generated_var_ids += 1
        else:
          self.generated_ids += 1
      return id

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

  def addRawMeta(self, metaless: dict, ctx: ScratchContext) -> dict:
    if self.parent is not None or not ctx.cfg.minify_break_glow:
      metaless["parent"] = self.parent

    if self.parent is None or not ctx.cfg.minify:
      metaless["topLevel"] = self.parent is None

    if self.next is not None or not ctx.cfg.minify:
      metaless["next"] = self.next

    if self.shadow or not ctx.cfg.minify:
      metaless["shadow"] = self.shadow

    if (self.x != 0 or not ctx.cfg.minify) and not ctx.cfg.hide_blocks:
      metaless["x"] = self.x

    if (self.y != 0 or not ctx.cfg.minify) and not ctx.cfg.hide_blocks:
      metaless["y"] = self.y

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

  def __repr__(self) -> str:
    return self.known.__repr__()

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    raw = self.getRawVarInit(preserve_booleans=False)

    val = [(10 if isinstance(self.known, str) else 4), raw]

    return [1, val], ctx

  def getRawVarInit(self, preserve_booleans=True) -> str | float | bool:
    """
    Get the raw value to set a var to when it starts with this value
    preserve_booleans - if enabled store booleans as strings, otherwise
    cast to int
    """
    if preserve_booleans and isinstance(self.known, bool):
      return "true" if self.known else "false"

    raw = self.known
    if not isinstance(self.known, str):
      if int(raw) == float(raw):
        raw = int(raw)
      else:
        raw = float(raw)

    if raw == float("+inf"):
      raw = "Infinity"
    elif raw == float("-inf"):
      raw = "-Infinity"
    elif isinstance(raw, float) and math.isnan(raw):
      raw = "NaN"

    return raw

class KnownBool(Known, BooleanValue):
  def __init__(self, known: bool):
    self.known = known

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    return Known(int(self.known)).getRawValue(parent, ctx, cast)

  def getRawBoolValue(self, parent: str, ctx: ScratchContext) -> tuple[list | None, ScratchContext]:
    if not self.known:
      return None, ctx # If false
    return BoolOp("not", KnownBool(False)).getRawBoolValue(parent, ctx)

  def getRawVarInit(self, preserve_booleans=True) -> str:
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
    id = ctx.genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "control_get_counter"
    }), BlockMeta(parent))

    return [3, id] + ([] if ctx.cfg.minify else [[10, ""]]), ctx

@dataclass
class EditCounter(Block):
  """Increment/Assign zero to the special 'hacked' counter block"""

  op: Literal["incr", "clear"]

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return {
      "opcode": "control_incr_counter" if self.op == "incr" else "control_clear_counter",
    }, ctx

# Sensing
@dataclass
class Ask(Block):
  value: Value

  def getRaw(self, my_id: str, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raw_msg, ctx = self.value.getRawValue(my_id, ctx, ScratchCast.TO_STR)
    return {
      "opcode": "sensing_askandwait",
      "inputs": {
        "QUESTION": raw_msg
      }
    }, ctx

@dataclass
class GetAnswer(Value):
  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "sensing_answer"
    }), BlockMeta(parent))

    return [3, id] + ([] if ctx.cfg.minify else [[10, ""]]), ctx

@dataclass
class GetOfList(Value):
  op: Literal["atindex", "indexof"]
  list_name: str
  value: Value

  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.genId()
    list_id = ctx.addOrGetList(self.list_name)

    raw_value, ctx = self.value.getRawValue(parent, ctx, (ScratchCast.TO_INT if self.op == "atindex" else ScratchCast.TO_STR))

    input_name = "INDEX" if self.op == "atindex" else "ITEM"

    ctx.addBlock(id, RawBlock({
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": {input_name: raw_value},
      "fields": {"LIST": [self.list_name, list_id]},
    }), BlockMeta(parent))

    return [3, id] + ([] if ctx.cfg.minify else [[10, ""]]), ctx

# Operators
OperatorsCodes = Literal["add", "sub", "mul", "div", "mod", "rand_between", "join", "letter_n_of", "length_of", "round", "bool_as_int", "str_to_float",
                         "abs", "floor", "ceiling", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "ln", "log", "e ^", "10 ^"]
@dataclass
class Op(Value):
  op: OperatorsCodes
  left: Value
  right: Value | None = None

  def __post_init__(self):
    takes_one_op = self.op in ["length_of", "round", "bool_as_int", "str_to_float"] or self.op not in SHORT_OP_TO_OPCODE # if op is length_of, round or is a general op
    given_one_op = self.right is None

    if takes_one_op != given_one_op:
      raise ScratchCompException(f"{self.op} takes {1 if takes_one_op else 2} operands, given {1 if given_one_op else 2}")

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    # We don't need to round the number if it gets casted to int anyway
    if self.op == "bool_as_int" and cast == ScratchCast.TO_INT:
      return self.left.getRawValue(parent, ctx, cast)

    id = ctx.genId()

    right = self.right if self.op != "str_to_float" else Known(0)

    takes_one_op = right is None

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
    if right is not None:
      casts_right_input_to = casts_left_input_to
      if self.op == "letter_n_of":
        casts_right_input_to = ScratchCast.TO_STR

      raw_right, ctx = right.getRawValue(id, ctx, casts_right_input_to)
      inputs.update({rgt_param: raw_right})

    fields = {}
    if opcode == "operator_mathop":
      fields.update({"OPERATOR": [self.op, None]})

    ctx.addBlock(id, RawBlock({
      "opcode": opcode,
      "inputs": inputs,
      "fields": fields
    }), BlockMeta(parent))

    return [3, id] + ([] if ctx.cfg.minify else [[10, ""]]), ctx

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
    id = ctx.genId()

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

# Variables
@dataclass
class GetVar(Value):
  var_name: str

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.addOrGetVar(self.var_name)
    return [3, [12, self.var_name, id]] + ([] if ctx.cfg.minify else [[10, ""]]), ctx

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

# Procedures
@dataclass
class ProcedureDef(StartBlock):
  name: str
  params: list[str]
  run_without_refresh: bool = True

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    proto_id = ctx.genId()
    param_ids = [ctx.genId() for _ in self.params]

    ctx.addFunc(self.name, param_ids, self.run_without_refresh)

    param_block_ids = []
    for param in self.params:
      param_block_id = ctx.genId()
      param_block_ids.append(param_block_id)
      ctx.addBlock(param_block_id, RawBlock({
        "opcode": "argument_reporter_string_number",
        "fields": {"VALUE": [sanitizeProcName(param, True), None]}
      }), BlockMeta(proto_id))

    data = {
      "opcode": "procedures_prototype",
      "inputs": dict(zip(param_ids, [list((1, id)) for id in param_block_ids])),
      "mutation": {
        "tagName": "mutation",
        # Seems to be necessary - while project is still able to run, loading project causes a crash
        "children": [],
        "proccode": sanitizeProcName(self.name, False) + (" %s" * len(self.params)),
        "argumentids": json.dumps(param_ids),
        "argumentnames": json.dumps([sanitizeProcName(param, True) for param in self.params]),
        "argumentdefaults": json.dumps(["" for _ in self.params]),
        "warp": json.dumps(self.run_without_refresh)
      }
    }
    if ctx.cfg.minify and len(data["inputs"]) == 0:
      del data["inputs"]
    ctx.addBlock(proto_id, RawBlock(data), BlockMeta(my_id, None, True))

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
    id = ctx.genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "argument_reporter_string_number",
      "fields": {"VALUE": [sanitizeProcName(self.param_name, True), None]}
    }), BlockMeta(parent))

    return [3, id] + ([] if ctx.cfg.minify else [[10, ""]]), ctx


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

  # Use minified json
  return json.dumps(res, separators=(",", ":"))

def exportScratchFile(ctx: ScratchContext, path: str) -> None:
  """Exports scratch code to a .sprite3 file"""
  with zipfile.ZipFile(path, "w") as zipf:
    zipf.writestr("Sprite/sprite.json", exportSpriteData(ctx))
    zipf.writestr(f"Sprite/{EMPTY_SVG_HASH}.svg", EMPTY_SVG)
