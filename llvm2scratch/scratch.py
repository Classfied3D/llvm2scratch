"""Scratch file output for the LLVM -> Scratch compiler"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
from enum import Enum

import zipfile
import hashlib
import random
import json
import math

MAIN_SPRITE_NAME = "DONT OPEN" # Incorrect grammar so it can fit in sprite name box lol
                               # Alternatively lowercase uses less horizontal space: Do Not Open could fit
EMPTY_SPRITE_NAME = "Empty"
EMPTY_SPRITE_COMMENT = f"""\
WARNING: The '{MAIN_SPRITE_NAME}' sprite may contain a lot of blocks and cause the scratch editor to crash! \
Make a backup of the project before opening! Also, opening it may cause any project.json tweaks enabled \
to break (not all projects use these so it should be fine).

This project was compiled from C, C++, Rust or other languages using llvm2scratch. The author of the \
project should have included the source code used to compile it, so check the project description! \
If you really want to read the generated scratch code (which is quite difficult to understand), the \
author may have also provided a text version.\
"""
SCRATCHBLOCKS_MESSAGE = """\
(::ring)Compiled with llvm2scratch!(::ring)::extension ring // Special blocks used internally by the compiler
(bool to float <>::extension) // This converts a boolean to an int using the round(_) block if necessary
(str to float ()::extension) // This converts a string to a float using the (_ + 0) block if necessary
<true::extension> // Known true block using the <not <>> block
<false::extension> // Known false block using an empty boolean input\
"""
DEFAULT_BROADCAST_MESSAGE = "message1"
COUNTER_REPLACEMENT_NAME = "!control:counter"
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
  "forever": "control_forever",
  "for_each": "control_for_each",

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
  "rand": "operator_random",
  "join": "operator_join",
  "letter_of": "operator_letter_of",
  "length_of": "operator_length",
  "round": "operator_round",
  "bool_to_float": "operator_round",
  "not": "operator_not",
  "and": "operator_and",
  "or": "operator_or",
  "=": "operator_equals",
  "<": "operator_lt",
  ">": "operator_gt",
  "contains": "operator_contains",
}

Id = str

class Format(Enum):
  Project3 = "project3"
  Sprite3 = "sprite3"

@dataclass
class ScratchConfig():
  minify: bool = True              # Optimize project.json's size by simplifing uids, removing falsy fields, etc

  minify_break_glow: bool = False  # Removing the parent key when minifing prevents blocks in the same
                                   # sprite from glowing correctly due to a js error - minify futher and
                                   # allow this error to occur

  hide_blocks: bool = False        # Prevent blocks from rendering in the editor by setting shadow: true on top
                                   # level blocks: stops editor lag

  allow_hacked_blocks: bool = True # Allow 'hacked' blocks not normally accessible from the editor such as 'counter'
                                   # and 'while'. See https://en.scratch-wiki.info/wiki/Hidden_Blocks. This may lead
                                   # to a performance reduction

@dataclass
class Project:
  cfg: ScratchConfig
  code: list[BlockList] = field(default_factory=list)
  lists: dict[str, list[Known]] = field(default_factory=dict)
  costumes: list[str] = field(default_factory=list)

  def export(self, filename: str, format: Format) -> None:
    """Exports the project into a .sb3/.sprite3 file"""
    exportScratchFile(self.getCtx(), filename, format)

  def stringify(self, scratchblocks: bool=False):
    """
    Convert project to readable text. If "scratchblocks" is True then
    output text compatible with scratchblocks
    """
    res = SCRATCHBLOCKS_MESSAGE + "\n\n" if scratchblocks else ""
    res += "\n\n".join(l.stringify(scratchblocks) for l in self.code)
    return res

  def addCostume(self, name: str) -> int:
    """Adds a costume and returns its costume number"""
    self.costumes.append(name)
    return len(self.costumes)

  def getCtx(self) -> ScratchContext:
    """Converts the project into a ScratchContext which can be used to get the raw project"""
    ctx = ScratchContext(self.cfg)
    for name, scratch_list in self.lists.items():
      ctx.addOrGetList(name, scratch_list)
    for block_list in self.code:
      ctx.addBlockList(block_list)
    ctx.costumes.extend(self.costumes)
    return ctx

@dataclass
class ScratchContext:
  cfg: ScratchConfig = field(default_factory=ScratchConfig)
  vars: dict[str, tuple[Id, Known]] = field(default_factory=dict)
  lists: dict[str, tuple[Id, list[Known]]] = field(default_factory=dict)
  broadcasts: dict[str, Id] = field(default_factory=dict)
  funcs: dict[str, tuple[list[Id], bool]] = field(default_factory=dict)
  blocks: dict[Id, dict] = field(default_factory=dict)
  late_blocks: list[tuple[Id, LateBlock, BlockMeta]] = field(default_factory=list)
  costumes: list[str] = field(default_factory=list)
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
    if len(blocks.blocks) == 0: return None

    last_id = parent
    curr_id = self.genId()
    first_id = curr_id
    next_id = self.genId()
    for_each_var_set = False
    end = False

    i = 0
    while i < len(blocks.blocks):
      block = blocks.blocks[i]
      if i == len(blocks.blocks) - 1: next_id = None

      assert curr_id is not None

      if last_id is not None and block.isStart():
        raise ScratchCompException(f"Starting block {type(block)} has blocks before it")
      if end: raise ScratchCompException(f"Reached ending block {type(block)} but more blocks are left")

      end = block.isEnd()
      meta = BlockMeta(last_id, next_id)

      # Temporary solution for putting multiple blocks in one for for each block when allow hacked blocks
      # is disabled
      if not self.cfg.allow_hacked_blocks and isinstance(block, ControlFlow) and block.op == "for_each":
        assert block.var is not None
        if not for_each_var_set:
          block = EditVar("set", block.var, Known(0))
          i -= 1
        for_each_var_set = not for_each_var_set

      self.addBlock(curr_id, block, meta)

      last_id = curr_id
      curr_id = next_id
      next_id = self.genId()

      i += 1

    return first_id

  def addOrGetVar(self, var_name: str, default_val: Known | None = None) -> Id:
    if default_val is None: default_val = Known(0)
    if not var_name in self.vars:
      id = self.genId()
      self.vars.update({var_name: (id, default_val)})
    else:
      id = self.vars[var_name][0]
    return id

  def addOrGetList(self, list_name: str, default_val: list[Known] | None = None) -> Id:
    if default_val is None: default_val = []
    if not list_name in self.lists:
      id = self.genId()
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

    id = self.genId()
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
        ["true" if v is True else "false" if v is False else v.getRawVarInit() for v in values]]

    raw_broadcasts = {}
    for name, id in self.broadcasts.items():
      raw_broadcasts.update({
        id: name
      })

    raw_blocks = {}
    for id, block in self.blocks.items():
      raw_blocks.update({id: block})

    raw_costumes = []
    # A costume must be defined for the sprite to load
    costume_names = [""] if len(self.costumes) == 0 else self.costumes
    for name in costume_names:
      raw_costumes.append(makeEmptyCostume(name))

    return {
      "variables": raw_vars,
      "lists": raw_lists,
      "blocks": raw_blocks,
      "costumes": raw_costumes,
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

  def genId(self) -> Id:
    if not self.cfg.minify:
      return random.randbytes(16).hex()
    else:
      invalid = True
      id = None
      while invalid:
        id = self.numericToStrUID(self.generated_ids)
        invalid = id in PALETTE_UIDS
        self.generated_ids += 1
      assert id is not None
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

  def stringify(self, sb: bool=False) -> str:
    """Convert to readable text. If "sb" is True then output text compatible with scratchblocks"""
    raise ScratchCompException("Cannot export for generic type 'Block'; must be a derived class")

@dataclass
class StartBlock(Block):
  def isStart(self) -> bool:
    return True

@dataclass
class EndBlock(Block):
  def isEnd(self) -> bool:
    return True

@dataclass
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
    # It seems like the parent property is only used from glowing
    # removing it seems to only break the glowing of scripts in the editor
    if not ctx.cfg.minify_break_glow:
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

@dataclass
class BlockList:
  blocks: list[Block]
  end: bool

  def __init__(self, blocks: list[Block] | None=None, end: bool=False):
    if blocks is None:
      blocks = []
    self.end = end
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

  def __len__(self) -> int:
    return len(self.blocks)

  def stringify(self, sb: bool=False) -> str:
    """Convert to readable text. If "sb" is True then output text compatible with scratchblocks"""
    return "\n".join(b.stringify(sb) for b in self.blocks)

@dataclass
class RawBlock(Block):
  contents: dict

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return self.contents, ctx

@dataclass
class Value:
  """Something that can be in a blocks input e.g. x in Say(x)"""
  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    """Gets the json that can be put in the "inputs" field of a block"""
    raise ScratchCompException("Cannot export for generic type 'Value'; must be a derived class")

  def stringify(self, sb: bool=False) -> str:
    """Convert to readable text. If "sb" is True than output text compatible with scratchblocks"""
    raise ScratchCompException("Cannot stringify for generic type 'Value'; must be a derived class")

@dataclass
class BooleanValue(Value):
  """A boolean value (a diamond shaped block)"""
  def getRawBoolValue(self, parent: str, ctx: ScratchContext) -> tuple[list | None, ScratchContext]:
    raise ScratchCompException("Cannot export for generic type 'BooleanValue'; must be a derived class")

@dataclass
class Known(Value):
  """Something that can be typed in a block input e.g. x in Say(x)"""
  known: str | float | bool

  def __post_init__(self):
    if isinstance(self.known, int):
      self.known = float(self.known)

  def __repr__(self) -> str:
    return self.known.__repr__()

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    raw = self.getRawVarInit(preserve_booleans=False)

    val = [(10 if isinstance(self.known, str) else 4), raw]

    return [1, val], ctx

  def getRawVarInit(self, preserve_booleans: bool=True) -> str | float | bool:
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

  def stringify(self, sb: bool=False, dropdown: bool=False) -> str:
    if not sb:
      return f'"{self.known}"' if isinstance(self.known, str) else f"{self.getRawVarInit()}"
    else:
      return knownToScratchBlocks(self.known, dropdown)

@dataclass
class KnownBool(Known, BooleanValue):
  def __post_init__(self):
    assert isinstance(self.known, bool)

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    return Known(int(self.known)).getRawValue(parent, ctx, cast)

  def getRawBoolValue(self, parent: str, ctx: ScratchContext) -> tuple[list | None, ScratchContext]:
    if not self.known:
      return None, ctx # If false
    return BoolOp("not", KnownBool(False)).getRawBoolValue(parent, ctx)

  def getRawVarInit(self, preserve_booleans=True) -> str:
    """Get the raw value to set a var to when it starts with this value"""
    return "true" if self.known else "false"

  def stringify(self, sb: bool=False, dropdown: bool=False):
    if not sb:
      return f"<{self.getRawVarInit()}>"
    else:
      return knownToScratchBlocks(self.known, dropdown)

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

  def stringify(self, sb: bool=False) -> str:
    return f"say {self.value.stringify(sb)}"

@dataclass
class SwitchCostume(Block):
  # Name of costume to switch to
  value: Value

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    if isinstance(self.value, Known):
      name_str = scratchCastToStr(self.value)

      proto_id = ctx.genId()
      # Add prototype costume
      ctx.addBlock(proto_id, RawBlock({
        "opcode": "looks_costume",
        "fields": {"COSTUME": [name_str, None]},
      }), BlockMeta(my_id, None, True))

      raw_name = [1, proto_id]

    else:
      raw_name, ctx = self.value.getRawValue(my_id, ctx, ScratchCast.TO_STR)

    return {
      "opcode": "looks_switchcostumeto",
      "inputs": {"COSTUME": raw_name},
    }, ctx

  def stringify(self, sb: bool=False) -> str:
    if sb and isinstance(self.value, Known):
      inner = self.value.stringify(sb, dropdown=True)
    else:
      inner = self.value.stringify(sb)
    return f"switch costume to {inner}"

@dataclass
class CostumeInfo(Value):
  op: Literal["name", "number"]

  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "looks_costumenumbername",
      "fields": {
        "NUMBER_NAME": [self.op, None]
      }
    }), BlockMeta(parent))

    return [3, id], ctx

  def stringify(self, sb: bool=False) -> str:
    if sb:
      return f"(costume [{self.op} v])"
    else:
      return f"(costume {self.op})"

# Sounds
# The set/change volume block is particularily useful as it causes scratch to render a frame, even in a
# run without screen refresh block
@dataclass
class EditVolume(Block):
  op: Literal["set", "change"]
  value: Value

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raw_vol, ctx = self.value.getRawValue(my_id, ctx, ScratchCast.TO_STR)

    return {
      "opcode": "sound_setvolumeto" if self.op == "set" else "sound_changevolumeby",
      "inputs": {"VOLUME": raw_vol},
    }, ctx

  def stringify(self, sb: bool=False) -> str:
    inner = self.value.stringify(sb)
    return ("set volume to " if self.op == "set" else "change volume by ") + inner

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

  def stringify(self, sb: bool=False) -> str:
    if sb and isinstance(self.value, Known):
      inner = self.value.stringify(sb, dropdown=True)
    else:
      inner = self.value.stringify(sb)
    return f"broadcast {inner}" + (" and wait" if self.wait else "")

@dataclass
class OnBroadcast(StartBlock):
  name: str

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    id = ctx.addBroadcast(self.name)

    return {
      "opcode": "event_whenbroadcastreceived",
      "fields": {"BROADCAST_OPTION": [self.name, id]}
    }, ctx

  def stringify(self, sb: bool=False) -> str:
    return f"when I recieve " + ("[" * sb) + self.name + (" v]" * sb)

# Control
@dataclass
class OnStartFlag(StartBlock):
  def getRaw(self, my_id: str, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return {
      "opcode": "event_whenflagclicked"
    }, ctx

  def stringify(self, sb: bool=False) -> str:
    return f"when green flag clicked"

FlowOp = Literal["if", "if_else", "reptimes", "until", "while", "forever", "for_each"]
@dataclass
class ControlFlow(Block):
  """
  A control flow statement (if, if else, repeat, repeat until, while, forever or for each).
  Note that the "for each" block does not guarantee the following unlike in scratch:
  * The value set is changed back if modified - it does not always work in recursion for that reason
  * The value passed in is ceiling'd - only consistent with integers
  * The counter will not be set to the value if the value is <= 0/a string/NaN

  This is due to it using the repeat block when allow_hacked_blocks is disabled
  """
  op: FlowOp
  value: Value | None
  blocks: BlockList
  else_blocks: BlockList | None = None
  var: str | None = None

  def __post_init__(self):
    if self.op == "forever" and self.value is not None:
      raise ScratchCompException("Forever cannot accept a value")
    elif self.op != "forever" and self.value is None:
      raise ScratchCompException(f"{self.op} requires a value!")

    if self.op in {"if", "if_else", "until", "while"} and not isinstance(self.value, BooleanValue):
      raise ScratchCompException("A regular value cannot be placed in a boolean accepting block")

    if self.op == "if_else" and self.else_blocks is None:
      raise ScratchCompException("An if-else statement must contain blocks in the else case")

    if self.op == "for_each" and self.var is None:
      raise ScratchCompException("A for-each block requires a variable input")

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    op, val = self.op, self.value
    blocks = self.blocks

    if not ctx.cfg.allow_hacked_blocks:
      if op == "while":
        assert val is not None
        op = "until"
        val = BoolOp("not", val)
      elif op == "for_each":
        assert self.var is not None
        op = "reptimes"
        # The inital set var is in addBlockList
        blocks = BlockList([
          EditVar("change", self.var, Known(1)),
          *blocks.blocks,
        ], blocks.end)

    opcode = SHORT_OP_TO_OPCODE[op]
    blocks_id = ctx.addBlockList(blocks, my_id)
    inputs = {"SUBSTACK": [2, blocks_id]}

    if op == "forever":
      return {
        "opcode": opcode,
        "inputs": inputs
      }, ctx

    assert val is not None

    if op in {"if", "if_else", "until", "while"}:
      assert isinstance(val, BooleanValue)
      raw_val, ctx = val.getRawBoolValue(my_id, ctx)
    else:
      raw_val, ctx = val.getRawValue(my_id, ctx, ScratchCast.TO_INT)

    match op:
      case "reptimes": input_name = "TIMES"
      case "for_each": input_name = "VALUE"
      case _:          input_name = "CONDITION"
    if raw_val is not None: inputs.update({input_name: raw_val})

    if op == "if_else":
      assert self.else_blocks is not None
      else_blocks_id = ctx.addBlockList(self.else_blocks, my_id)
      inputs.update({"SUBSTACK2": [2, else_blocks_id]})

    res = {
      "opcode": opcode,
      "inputs": inputs
    }

    if op == "for_each":
      assert self.var is not None
      id = ctx.addOrGetVar(self.var)
      res["fields"] = {"VARIABLE": [self.var, id]}

    return res, ctx

  def indent(self, input: BlockList, sb: bool=False) -> str:
    return "\n".join("  " + x for x in input.stringify(sb).split("\n"))

  def stringify(self, sb: bool=False) -> str:
    res = {
      "if": "if",
      "if_else": "if",
      "until": "repeat until",
      "while": "while",
      "reptimes": "repeat",
      "forever": "forever",
      "for_each": "for each",
    }[self.op]

    if self.op == "for_each":
      assert self.var is not None
      if not sb:
        name = self.var
      else:
        name = Known(self.var).stringify(sb, dropdown=True)
      res += f" {name} in"

    if self.value is not None:
      res += f" {self.value.stringify(sb)}"

    if sb:
      if self.op in {"while", "for_each"}:
        res += " {"
      elif self.op in {"if", "if_else"}:
        res += " then"

    res += f"\n {self.indent(self.blocks, sb)}"

    if self.else_blocks is not None:
      res += f"\nelse\n{self.indent(self.else_blocks, sb)}"

    if sb:
      if self.op in {"while", "for_each"}:
        # Techincally the for each block doesn't have a loop arrow but I don't care lol
        res += "\n}@loopArrow::control"
      else:
        res += "\nend"

    return res

  def isEnd(self) -> bool:
    return self.op == "forever"

@dataclass
class StopScript(EndBlock):
  op: Literal["stopthis", "stopall"]

  def getRaw(self, my_id: str, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return {
      "opcode": "control_stop",
      "fields": {"STOP_OPTION": ["all" if self.op == "stopall" else "this script", None]}
    }, ctx

  def stringify(self, sb: bool=False) -> str:
    op = "this script" if self.op == "stopthis" else "stop all"
    if sb: op = f"[{op} v]"
    return f"stop {op}"

@dataclass
class GetCounter(Value):
  """Get the value of the special 'hacked' counter block"""

  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    if not ctx.cfg.allow_hacked_blocks:
      return GetVar(COUNTER_REPLACEMENT_NAME).getRawValue(parent, ctx, cast)

    id = ctx.genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "control_get_counter"
    }), BlockMeta(parent))

    return [3, id], ctx

  def stringify(self, sb: bool=False) -> str:
    return "(counter)" if not sb else "(counter::control)"

@dataclass
class EditCounter(Block):
  """Increment/Assign zero to the special 'hacked' counter block"""

  op: Literal["incr", "clear"]

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    if not ctx.cfg.allow_hacked_blocks:
      op = "change" if self.op == "incr" else "set"
      val = Known(1 if self.op == "incr" else 0)
      return EditVar(op, COUNTER_REPLACEMENT_NAME, val).getRaw(my_id, ctx)

    return {
      "opcode": "control_incr_counter" if self.op == "incr" else "control_clear_counter",
    }, ctx

  def stringify(self, sb: bool=False) -> str:
    return ("increment counter" if self.op == "incr" else "clear counter") \
         + ("::control" if sb else "")

@dataclass
class Wait(Block):
  duration: Value

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raw_dur, ctx = self.duration.getRawValue(my_id, ctx, ScratchCast.TO_INT)
    return {
      "opcode": "control_wait",
      "inputs": {
        "DURATION": raw_dur
      },
    }, ctx

  def stringify(self, sb: bool=False) -> str:
    return f"wait {self.duration.stringify(sb)} seconds"

# TODO: when adding delete clone block, make sure to exclude it in optimizer from being
# inlined when in an if statement. This is because on the main sprite it does nothing

# Sensing
@dataclass
class Ask(Block):
  value: Value

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    raw_msg, ctx = self.value.getRawValue(my_id, ctx, ScratchCast.TO_STR)
    return {
      "opcode": "sensing_askandwait",
      "inputs": {
        "QUESTION": raw_msg
      }
    }, ctx

  def stringify(self, sb: bool=False) -> str:
    return f"ask {self.value.stringify(sb)} and wait"

@dataclass
class GetAnswer(Value):
  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "sensing_answer"
    }), BlockMeta(parent))

    return [3, id], ctx

  def stringify(self, sb: bool=False) -> str:
    return "(answer)"

@dataclass
class DaysSince2000(Value):
  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "sensing_dayssince2000"
    }), BlockMeta(parent))

    return [3, id], ctx

  def stringify(self, sb: bool=False) -> str:
    return "(days since 2000)"

# Operators
OperatorsCodes = Literal["add", "sub", "mul", "div", "mod", "rand", "join", "letter_of", "length_of", "round", "bool_to_float", "str_to_float",
                         "abs", "floor", "ceiling", "sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "ln", "log", "e ^", "10 ^"]
@dataclass
class Op(Value):
  op: OperatorsCodes
  left: Value
  right: Value | None = None

  def __post_init__(self):
    takes_one_op = self.op in ["length_of", "round", "bool_to_float", "str_to_float"] or self.op not in SHORT_OP_TO_OPCODE # if op is length_of, round or is a general op
    given_one_op = self.right is None

    if takes_one_op != given_one_op:
      raise ScratchCompException(f"{self.op} takes {1 if takes_one_op else 2} operands, given {1 if given_one_op else 2}")

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    # We don't need to round the number if it gets casted to int anyway
    if self.op in {"bool_to_float", "str_to_float"} and cast == ScratchCast.TO_INT:
      return self.left.getRawValue(parent, ctx, cast)

    id = ctx.genId()

    right = self.right if self.op != "str_to_float" else Known(0)

    takes_one_op = right is None

    match self.op:
      case "rand":
        lft_param = "FROM"
        rgt_param = "TO"
      case "join":
        lft_param = "STRING1"
        rgt_param = "STRING2"
      case "letter_of":
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
      if self.op == "letter_of":
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

    return [3, id], ctx

  def stringify(self, sb: bool=False) -> str:
    left = self.left.stringify(sb)
    right = self.right.stringify(sb) if self.right is not None else None
    match self.op:
      case "add" | "sub" | "mul" | "div" | "mod":
        fmt_op = {"add": "+", "sub": "-", "mul": "*", "div": "/", "mod": "mod"}[self.op]
        return f"({left} {fmt_op} {right})"
      case "rand": return f"(pick random {left} to {right})"
      case "join":         return f"(join {left} {right})"
      case "letter_of":  return f"(letter {left} of {right})"
      case "length_of" | "round" | "bool_to_float" | "str_to_float":
        force_colour = "::extension" if sb and self.op in {"bool_to_float", "str_to_float"} else ""
        fmt_op = self.op.replace("_", " ")
        return f"({fmt_op} {left}{force_colour})"
      case _:
        if sb:
          return f"([{self.op} v] of {left})"
        else:
          return f"({self.op} of {left})"

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

    rgt_param = None
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
    if raw_right is not None:
      assert rgt_param is not None
      inputs.update({rgt_param: raw_right})

    ctx.addBlock(id, RawBlock({
      "opcode": SHORT_OP_TO_OPCODE[self.op],
      "inputs": inputs,
    }), BlockMeta(parent))

    return [2, id], ctx

  def getRawBoolValue(self, parent: str, ctx: ScratchContext) -> tuple[list | None, ScratchContext]:
    return self.getRawValue(parent, ctx, ScratchCast.TO_INT)

  def stringify(self, sb: bool=False) -> str:
    if self.right is None:
      return f"<{self.op} {self.left.stringify(sb)}>"
    else:
      return f"<{self.left.stringify(sb)} {self.op} {self.right.stringify(sb)}" + "?" * (sb and self.op == "contains") + ">"

# Variables
@dataclass
class GetVar(Value):
  var_name: str

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.addOrGetVar(self.var_name)
    return [3, [12, self.var_name, id]], ctx

  def stringify(self, sb: bool=False) -> str:
    if not sb: return f"({self.var_name})"
    else:
      name = escapeScratchBlocksStr(self.var_name)

      # Make sure not treated as a number, only as a variable
      if all([char.isdigit() or char in {"-", "."} for char in name]):
        name += "::variables"

      return f"({name})"

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

  def stringify(self, sb: bool=False) -> str:
    inner = self.value.stringify(sb)
    if not sb:
      var = self.var_name
    else:
      var = Known(self.var_name).stringify(sb, dropdown=True)

    if self.op == "set" and not sb:
      return f"{var} = {inner}"
    elif self.op == "set":
      return f"set {var} to {inner}"
    else:
      return f"change {var} by {inner}"

@dataclass
class GetList(Value):
  list_name: str

  def getRawValue(self, parent: Id, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.addOrGetList(self.list_name)
    return [3, [13, self.list_name, id]], ctx

  def stringify(self, sb: bool=False) -> str:
    if not sb:
      return f"(list {self.list_name})"
    else:
      # Ensure scratchblocks doesn't treat the list as a variable
      return f"({escapeScratchBlocksStr(self.list_name)}::list)"

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

    return [3, id], ctx

  def stringify(self, sb: bool=False) -> str:
    if not sb:
      var = self.list_name
    else:
      var = Known(self.list_name).stringify(sb, dropdown=True)

    if self.op == "atindex":
      return f"(item {self.value.stringify(sb)} of {var})"
    else:
      return f"(item # of {self.value.stringify(sb)} in {var})"

@dataclass
class GetListLength(Value):
  list_name: str

  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.genId()
    list_id = ctx.addOrGetList(self.list_name)

    ctx.addBlock(id, RawBlock({
      "opcode": "data_lengthoflist",
      "fields": {"LIST": [self.list_name, list_id]},
    }), BlockMeta(parent))

    return [3, id], ctx

  def stringify(self, sb: bool=False) -> str:
    if sb:
      var = Known(self.list_name).stringify(sb, dropdown=True)
      return f"(length of {var})"
    return f"(length of list {self.list_name})"

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

  def stringify(self, sb: bool=False) -> str:
    if not sb:
      var = self.list_name
    else:
      var = Known(self.list_name).stringify(sb, dropdown=True)

    match self.op:
      case "addto":
        assert self.item is not None
        return f"add {self.item.stringify(sb)} to {var}"
      case "replaceat":
        assert self.item is not None
        assert self.index is not None
        return f"replace item {self.index.stringify(sb)} of {var} with {self.item.stringify(sb)}"
      case "insertat":
        assert self.item is not None
        assert self.index is not None
        return f"insert {self.item.stringify(sb)} at {self.item.stringify(sb)} of {var}"
      case "deleteat":
        assert self.index is not None
        return f"delete {self.index.stringify(sb)} of {var}"
      case "deleteall":
        return f"delete all of {var}"
      case _:
        raise ValueError(f"Could not find operation {self.op}")

# Procedures
@dataclass
class ProcedureDef(StartBlock):
  proc_name: str
  params: list[str]
  run_without_refresh: bool = True

  def getRaw(self, my_id: Id, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    proto_id = ctx.genId()
    param_ids = [ctx.genId() for _ in self.params]

    ctx.addFunc(self.proc_name, param_ids, self.run_without_refresh)

    param_block_ids = []
    for param in self.params:
      param_block_id = ctx.genId()
      param_block_ids.append(param_block_id)
      ctx.addBlock(param_block_id, RawBlock({
        "opcode": "argument_reporter_string_number",
        "fields": {"VALUE": [sanitizeProcName(param, True), None]}
      }), BlockMeta(proto_id))

    # Add prototype shadow
    data = {
      "opcode": "procedures_prototype",
      "inputs": dict(zip(param_ids, [list((1, id)) for id in param_block_ids])),
      "mutation": {
        "tagName": "mutation",
        # Seems to be necessary - while project is still able to run, loading project causes a crash
        "children": [],
        "proccode": sanitizeProcName(self.proc_name, False) + (" %s" * len(self.params)),
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

  def stringify(self, sb: bool=False) -> str:
    name = escapeScratchBlocksStr(self.proc_name) if sb else self.proc_name
    return " ".join(["define", name, *(f"({p})" for p in self.params)])

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

  def stringify(self, sb: bool=False) -> str:
    name = escapeScratchBlocksStr(self.proc_name) if sb else self.proc_name
    parts = [name, *(a.stringify(sb) for a in self.arguments)]
    if not sb:
      parts.insert(0, "call")
    else:
      # Force scratchblocks to treat as a custom block, even if it's not defined
      parts.append("::custom")
    return " ".join(parts)

@dataclass
class GetParam(Value):
  param_name: str

  def getRawValue(self, parent: str, ctx: ScratchContext, cast: ScratchCast) -> tuple[list, ScratchContext]:
    id = ctx.genId()

    ctx.addBlock(id, RawBlock({
      "opcode": "argument_reporter_string_number",
      "fields": {"VALUE": [sanitizeProcName(self.param_name, True), None]}
    }), BlockMeta(parent))

    return [3, id], ctx

  def stringify(self, sb: bool=False) -> str:
    if not sb:
      return f"(param {self.param_name})"
    else:
      name = escapeScratchBlocksStr(self.param_name)
      # Force scratchblocks to treat as a parameter, even if it's not in the current
      # function definition
      return f"({name}::custom)"


class ScratchCompException(Exception):
  """Exception when compiling to scratch"""
  pass

def sanitizeProcName(name: str, is_param: bool) -> str:
  """
  Fixes the Bunching Blocks Bug (https://en.scratch-wiki.info/wiki/My_Blocks#Glitches)
  and the hasOwnProperty bug by replacing % with a similar unicode character when necessary
  """
  if (is_param and name in ["%b", "%n"]) or (not is_param and name == "%"):
    return name.replace("%", "\uFF05")
  elif not is_param and name == "hasOwnProperty":
    return name + ":bro why"
  return name

def scratchCastToNum(value: Known) -> float:
  """Performs the same casting to number as scratch"""
  raw = value.known
  try:
    raw = float(raw)
  except ValueError:
    raw = math.nan

  return 0.0 if math.isnan(raw) else raw

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

def escapeScratchBlocksStr(val: str):
  # Escape special characters
  res = val.replace("::", "\\:\\:")

  special_chars = r"\()[]<>"
  for char in special_chars:
    res = res.replace(char, fr"\{char}")

  # Escape " v" so scratchblocks doesn't mistake this as a dropdown
  if res.endswith(" v"):
    res = res[:-1] + "\\v"

  return res

def knownToScratchBlocks(val: str | float | bool | int, dropdown: bool=False) -> str:
  match val:
    case bool():
      return "<" + ("true" if val else "false") + "::extension>"

    case int() | float():
      if int(val) == float(val):
        val = int(val)

      if val == float("+inf"):
        return "[Infinity]"
      elif val == float("-inf"):
        return "[-Infinity]"
      elif isinstance(val, float) and math.isnan(val):
        return "[NaN]"

      return "(" + str(val) + ")"

    case str():
      return "[" + escapeScratchBlocksStr(val) + (" v" * dropdown) + "]"

    case _:
      raise ValueError(f"Invalid type: {val}")

def makeEmptyCostume(name: str) -> dict:
  return {
    "name": name,
    "bitmapResolution": 1,
    "dataFormat": "svg",
    "assetId": EMPTY_SVG_HASH,
    "md5ext": f"{EMPTY_SVG_HASH}.svg",
    "rotationCenterX": 0,
    "rotationCenterY": 0
  }

def exportEmptySprite(name: str="", is_stage: bool=False) -> dict:
  assert len(name) > 0 or is_stage
  res: dict = {
    "isStage": is_stage,
    "name": "Stage" if is_stage else name,
    "variables": {},
    "lists": {},
    "broadcasts": {},
    "blocks": {},
    "comments": {},
    "currentCostume": 0,
    "costumes":[
      # A costume must be defined for the sprite to load
      makeEmptyCostume("")
    ],
    "sounds": [],
    "volume": 100,
    "layerOrder": 0 if is_stage else 1,
    "visible": True,
  }

  if not is_stage:
    res.update({
      "x": 0,
      "y": 0,
      "size": 100,
      "direction": 90,
      "draggable": False,
      "rotationStyle": "all around",
    })
  else:
    res.update({
      "tempo": 60,
      "videoTransparency": 50,
      "videoState": "on",
      "textToSpeechLanguage": None,
    })

  return res

def exportData(ctx: ScratchContext, format: Format) -> str:
  sprite = exportEmptySprite(MAIN_SPRITE_NAME)
  sprite.update(ctx.getRaw())

  match format:
    case Format.Sprite3:
      res = sprite
    case Format.Project3:
      buffer_sprite = exportEmptySprite(EMPTY_SPRITE_NAME)
      buffer_sprite["comments"] = {
        "coolcommentid": {
          "blockId": None,
          "x": 50,
          "y": 50,
          "width": 500,
          "height": 300,
          "minimized": False,
          "text": EMPTY_SPRITE_COMMENT
        }
      }

      stage = exportEmptySprite(is_stage=True)

      res = {
        "targets": [
          stage, buffer_sprite, sprite
        ],
        "monitors": [],
        "extensions": [],
        "meta": {
          "semver": "3.0.0",
          "vm": "13.6.10",
          "agent": "Project compiled with llvm2scratch!"
        }
      }

  # Use minified json
  return json.dumps(res, separators=(",", ":"))

def exportScratchFile(ctx: ScratchContext, path: str, format: Format) -> None:
  """Exports scratch code to a .sb3/.sprite3 file"""
  match format:
    case Format.Project3: folder, file = "Project", "project.json"
    case Format.Sprite3:  folder, file = "Sprite", "sprite.json"

  with zipfile.ZipFile(path, "w") as zipf:
    zipf.writestr(f"{folder}/{file}", exportData(ctx, format))
    zipf.writestr(f"{folder}/{EMPTY_SVG_HASH}.svg", EMPTY_SVG)
