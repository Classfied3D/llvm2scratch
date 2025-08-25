"""LLVM -> Scratch Compiler"""

from __future__ import annotations
from dataclasses import dataclass, is_dataclass, field, fields
from collections import OrderedDict
from typing import Literal

from llvm2py import parse_assembly
from llvm2py import ir
import random
import math

from . import scratch as sb3
from . import optimizer
from . import util

SCRATCH_LIST_LIMIT = 200_000

@dataclass
class DebugInfo:
  """Info about a scratch project which can be used in future compilations for optimization"""
  debug_branch_func_map: str | None = None
  debug_branch_log: str | None = None

@dataclass
class Config:
  """Config options to pass to the compiler"""
  opti: bool = True # If optimisations for scratch should be applied
  invis_blocks: bool = False # Prevent scratch editor from rendering blocks; reduces lag
  stack_size: int = 512 # Amount of 'bytes' on 'stack' list (one byte is 48 bits), max 200,000
  binop_lookup_bits: int = 8 # Amount of bits to use for AND/OR/XOR tables, creates (2**(2*n) elements per table)
  max_branch_recursion: int = 1_000_000 # Maximum amount of times a checked function can recurse before wiping scratch's call stack via a broadcast

  debug_info: DebugInfo = field(default_factory=DebugInfo) # Info about a scratch project which can be used in future compilations for optimization
  do_debug_branch_log: bool = False # If the times a function recurses should be logged

  unused_var = "!unused" # Name of the scratch variable for unused values
  return_var = "!return value" # Name of the scratch variable for returing values
  stack_var = "!stack" # Name of the scratch list for the stack list
  stack_size_var = "!stack size" # Name of the scratch variable for the stack size
  debug_branch_log_var = "!!debug_branch_log" # (using underscores to avoid spaces in filename for convenience)

  ascii_lookup_var = "!ascii lookup"
  binop_lookup_var = "!binop lookup"
  pow2_lookup_var = "!pow2 lookup"

  return_address_local = "return address" # Name of the local variable or parameter to the name to broadcast to after returning
  did_reset_local = "did reset" # Name of the local var for if the stack reset

  tmp_prefix = "tmp " # Name of temp variables before a number is added to them
  zero_indexed_suffix = " (0 indexed)"
  one_indexed_suffix = " (1 indexed)"

@dataclass
class Context:
  """Global context access when translating instructions"""
  proj: sb3.Project
  cfg: Config
  fn_info: dict[str, FuncInfo] = field(default_factory=dict)
  globvar_to_ptr: dict[str, sb3.Known] = field(default_factory=dict)
  next_fn_id: int = 0

@dataclass
class FuncInfo:
  """Info about a LLVM function"""
  name: str
  fn_id: int
  params: list[str] # The parameters the function takes (doesn't include return address)
  can_call: set[str] # Everything the function might call (may include itself)
  return_addresses: list[str] # Any functions that call this function
  returns_to_address: bool # If the function returns using a broadcast to an address
  checked_blocks: list[str] # List of label names that will contain a check to reset the stack

@dataclass
class BlockInfo:
  """Info about a LLVM block"""
  fn: FuncInfo
  is_scratch_func: bool # If the instruction can access the parameters normally (in the body of a scratch func)

@dataclass
class ValueAndBlocks:
  """A value and any blocks that come before it needed to get that value"""
  value: sb3.Value
  blocks: sb3.BlockList = field(default_factory=sb3.BlockList)

@dataclass
class IndexableValue:
  """A collection of values that can be indexed over (e.g. a string)"""
  vals: list[sb3.Value]

@dataclass
class IValueAndBlocks:
  """An indexable value and any blocks that come before it needed to get that value"""
  value: IndexableValue
  blocks: sb3.BlockList = field(default_factory=sb3.BlockList)

@dataclass
class Variable:
  """Name of a variable"""
  name: str
  is_scratch_param: bool

  def getValue(self) -> sb3.Value:
    if self.is_scratch_param:
      return sb3.GetParameter(self.name)
    return sb3.GetVar(self.name)

  def setValue(self, value: sb3.Value, op: Literal["set", "change"]="set") -> sb3.Block:
    if self.is_scratch_param: raise CompException(f"{self.name} param is read only")
    return sb3.EditVar(op, self.name, value)

class CompException(Exception):
  """Exception in the compiler"""
  pass

def astuple(obj):
  """Same as dataclasses.astuple but not recursively with dataclasses in dataclasses"""
  if not is_dataclass(obj):
    raise TypeError("Expected dataclass instance")
  return tuple(getattr(obj, f.name) for f in fields(obj))

def getByteSize(ty: ir.Type) -> int:
  match ty:
    case ir.IntegerType():
      # Scratch's fp variables can store < 52 bits per variable accurately
      return math.ceil(ty.num_bits / 48)
    case ir.ArrayType():
      return ty.elem_count * getByteSize(ty.elem_ty)
    case ir.PtrType():
      return 1
    case _:
      raise CompException(f"Unknown Type: {ty} (py type {type(ty)})")

def decodeValue(val: ir.value.Value,
                ctx: Context, bctx: BlockInfo | None) -> ValueAndBlocks | IValueAndBlocks:
  match val.val:
    case str(): # if a variable
      var = decodeVar(val, bctx)
      res = sb3.Known("") # this means no value - nothing in the block
      if var is not None:
        res = var.getValue()
        if ctx.cfg.opti and var.name in ctx.globvar_to_ptr:
          res = ctx.globvar_to_ptr[var.name]
      return ValueAndBlocks(res)
    case int():
      if not isinstance(val.ty, ir.IntegerType):
        raise CompException(f"Expected {val} to be an integer, got type {type(val.ty)}")

      # TODO FIX: allow different sizes with val.ty
      if val.ty.num_bits > 48: raise CompException(f">48 bits not yet supported, got {val.ty.num_bits}")

      # Calculate the two's complement version of the number
      num = val.val
      width = val.ty.num_bits
      if num < 0:
        num = (2 ** width) + num
      return ValueAndBlocks(sb3.Known(num))
    case bytes():
      if not (isinstance(val.ty, ir.ArrayType) and isinstance(val.ty.elem_ty, ir.IntegerType)):
        raise CompException(f"Expected bytes value {val} to be assigned to ty [N x i8], got, {type(val.ty)}")

      if val.ty.elem_ty.num_bits != 8:
        raise CompException(f"Cannot assign bytes value {val} a non byte (i8)")

      return IValueAndBlocks(
        IndexableValue([sb3.Known(str(int(byte))) for byte in val.val]))
    case _:
      raise CompException(f"Unknown Value: {val.val} (py type: {type(val.val)})")

def decodeVar(var: ir.value.Value, bctx: BlockInfo | None) -> Variable | None:
  """Used for getting the assigned variable of an instruction"""
  if not isinstance(var.val, str):
    raise CompException(f"Expected val to be a variable, got Value: {var.val} (py type: {type(var.val)})")
  if var.val == "<badref>":
    return None
  name = var.val
  is_local = name.startswith("%")
  if is_local: name = name[1:]
  return localizeVar(name, not is_local, bctx)

def localizeVar(name: str, is_global: bool, bctx: BlockInfo | None) -> Variable:
  is_param = False
  if is_global:
    name = f"@{name}"
  else:
    assert bctx is not None
    if name in bctx.fn.params and bctx.is_scratch_func:
      is_param = True
      name = localizeParameter(name)
    else:
      name = f"%{bctx.fn.name}:{name}" # Localize variables per function

  return Variable(name, is_param)

def localizeParameter(name: str) -> str:
  return "%" + name

def localizeLabel(label: str, fn: FuncInfo) -> str:
  return f"{fn.name}:{label}"

def localizeCallId(call_id: int, label: str, fn_name: str) -> str:
  return f"{fn_name}:{label}:return addr {call_id}"

def genTempVar(ctx: Context) -> str:
  return ctx.cfg.tmp_prefix + random.randbytes(12).hex()

def shouldOptimiseValueUse(val: sb3.Value, times_used: float) -> bool:
  """Returns if a value that is used multiple times should be stored"""
  return not optimizer.shouldElide(val, times_used)

def optimizeValueUse(val: sb3.Value, times_used: float, ctx: Context) -> ValueAndBlocks:
  if shouldOptimiseValueUse(val, times_used):
    tmp = genTempVar(ctx)
    return ValueAndBlocks(sb3.GetVar(tmp), sb3.BlockList([sb3.EditVar("set", tmp, val)]))
  return ValueAndBlocks(val)

def makePow2LookupTable(size: int, is_one_indexed: bool, ctx: Context) -> tuple[str, Context]:
  name = ctx.cfg.pow2_lookup_var + (ctx.cfg.one_indexed_suffix if is_one_indexed else ctx.cfg.zero_indexed_suffix)
  if name not in ctx.proj.lists or size > len(ctx.proj.lists[name]):
    pow2_lookup = []
    for x in range(int(is_one_indexed) + size):
      pow2_lookup.append(sb3.Known(2 ** x))
    ctx.proj.lists[name] = pow2_lookup
  return name, ctx

def twosComplement(width: int, val: sb3.Value) -> sb3.Value:
  return sb3.Op("mod", val, sb3.Known(2 ** width))

def undoTwosComplement(width: int, val: sb3.Value, return_var: bool, ctx: Context) -> ValueAndBlocks:
  """Calculates two's compilment on a value. If the returned value is used multiple times, it is
  prefered to be a var, which can be done with return_var=True"""
  limit = int(((2 ** width) / 2) - 1)
  decrease = 2 ** width

  if shouldOptimiseValueUse(val, 2) or return_var:
    tmp = genTempVar(ctx)
    return ValueAndBlocks(sb3.GetVar(tmp), sb3.BlockList([
      sb3.EditVar("set", tmp, val),
      sb3.ControlFlow("if", sb3.BoolOp(">", sb3.GetVar(tmp), sb3.Known(limit)), sb3.BlockList([
        sb3.EditVar("change", tmp, sb3.Known(-decrease)),
      ])),
    ]))
  return ValueAndBlocks(sb3.Op("sub", val, sb3.Op("mul", sb3.Known(decrease), sb3.BoolOp(">", val, sb3.Known(limit)))))

def intPow2(val: sb3.Value, max_val: int, ctx: Context) -> tuple[sb3.Value, Context]:
  if isinstance(val, sb3.Known):
    try:
      return sb3.Known(2 ** int(val.known)), ctx
    except ValueError:
      raise CompException("Cannot calculate pow2 of a known non-integer")
      #return sb3.Known("NaN"), ctx
  else:
    lookup, ctx = makePow2LookupTable(max_val, True, ctx) # Any value above the width will be treated as a zero
                                                        # which has no effect on the result
    return sb3.GetOfList("atindex", lookup, sb3.Op("add", val, sb3.Known(1))), ctx

def bitShift(direction: Literal["left", "right"],
             width: int, left: sb3.Value, right: sb3.Value,
             ctx: Context, can_shift_out=True) -> tuple[sb3.Value, Context]:
  right_mul, ctx = intPow2(right, width, ctx)

  if direction == "left":
    # Multipling by a power of two is safe because the internal double value scratch uses doesnt loose accuracy
    # when only the exponent part changes
    unwrapped = sb3.Op("mul", left, right_mul)
    if not can_shift_out: return unwrapped, ctx
    return sb3.Op("mod", unwrapped, sb3.Known(2 ** width)), ctx
  else:
    unwrapped = sb3.Op("div", left, right_mul)
    if not can_shift_out: return unwrapped, ctx
    return sb3.Op("floor", unwrapped), ctx

def multiplyNoWrap(width: int, left: sb3.Value, right: sb3.Value) -> sb3.Value:
  if width > 48:
    raise CompException(f"Multipling {width} bits is not supported") # TODO FIX

  return sb3.Op("mul", left, right) # Overflow is UB - we don't care if
                                    # the number overflows and gets innacurate

def multiplyWrap(width: int, left: sb3.Value, right: sb3.Value, ctx: Context) -> ValueAndBlocks:
  # TODO OPTI: if one value is a known value, wrapping behaviour could be simpilifed and
  # known info could be propagated
  # TODO OPTI: if multipling by a power of 2, there is no risk that the mantissa cannot store
  # enough to be accurate, since only the exponent changes
  if width <= 26: # Safe: (2**26) ** 2 < 9007199254740991
    return ValueAndBlocks(sb3.Op("mod", sb3.Op("mul", left, right), sb3.Known(2 ** width)))
  elif width <= 36: # Safe: (2**17 * 2**17 + 2**17 * 2**17) * 2**17 + (2**17 + 2**17) < 9007199254740991
    blocks = sb3.BlockList()
    left, lblocks = astuple(optimizeValueUse(left, 2, ctx))
    right, rblocks = astuple(optimizeValueUse(right, 2, ctx))
    blocks.add(lblocks)
    blocks.add(rblocks)

    # Use some maths to do the calculation (see README for explaination)
    half_width = width // 2
    a0 = genTempVar(ctx)
    b0 = genTempVar(ctx)
    blocks.add([
      sb3.EditVar("set", a0, sb3.Op("mod", left, sb3.Known(2 ** half_width))),
      sb3.EditVar("set", b0, sb3.Op("mod", right, sb3.Known(2 ** half_width)))
    ])
    value = sb3.Op("mod",
      sb3.Op("add",
        sb3.Op("mul",
          sb3.Op("add", # TODO FIX: modding this number by 2^(width/half_width) for width > 36
            sb3.Op("mul",
              sb3.GetVar(a0),
              sb3.Op("floor", sb3.Op("div", right, sb3.Known(2 ** half_width)))
            ),
            sb3.Op("mul",
              sb3.GetVar(b0),
              sb3.Op("floor", sb3.Op("div", left, sb3.Known(2 ** half_width)))
            ),
          ),
          sb3.Known(2 ** half_width)),
        sb3.Op("mul", sb3.GetVar(a0), sb3.GetVar(b0))
      ),
      sb3.Known(2 ** width))
    return ValueAndBlocks(value, blocks)
  else:
    raise CompException(f"Multipling {width} bits is not supported") # TODO FIX

def binarySearch(value: sb3.Value,
                 branches: OrderedDict[int, sb3.BlockList],
                 default_branch: sb3.BlockList | None = None,
                 min_poss_value: int | None = None, # Max value - we do not need to check for default values above it
                 max_poss_value: int | None = None, # Min value - likewise
                 are_branches_sorted: bool = False,
                 _lo: int=0, _hi: int | None=None) -> sb3.BlockList:
  if len(branches) == 0:
    return sb3.BlockList([]) if default_branch is None else default_branch

  if not are_branches_sorted: branches = OrderedDict(sorted(branches.items()))

  if _hi is None: _hi = len(branches.keys()) - 1
  mid = (_lo + _hi) // 2
  mid_val = list(branches.keys())[mid]

  if _lo != 0: min_poss_value = list(branches.keys())[_lo]
  if _hi != len(branches) - 1: max_poss_value = list(branches.keys())[_hi]

  if _lo == _hi:
    if default_branch is not None:
      check_below = check_above = True

      if mid != len(branches) - 1:
        # If the value above the one being checked for, there is no need to check for a default above it
        check_above = not list(branches.keys())[mid + 1] == mid_val + 1
      if mid_val == max_poss_value: # If the value is the highest, we don't need to check below
        check_above = False

      # Vice versa
      if mid != 0:
        check_below = not list(branches.keys())[mid - 1] == mid_val - 1
      if mid_val == min_poss_value:
        check_below = False

      if check_below or check_above:
        cond = sb3.BoolOp("=", value, sb3.Known(mid_val))
        return sb3.BlockList([sb3.ControlFlow("if_else", cond, list(branches.values())[mid], default_branch)])

    return list(branches.values())[mid]

  cond = sb3.BoolOp(">", value, sb3.Known(mid_val))
  return sb3.BlockList([sb3.ControlFlow("if_else", cond,
                        # Sorting already taken care of
                        binarySearch(value, branches, default_branch, min_poss_value, max_poss_value, True,
                                     _lo=mid + 1, _hi=_hi),
                        binarySearch(value, branches, default_branch, min_poss_value, max_poss_value, True,
                                     _lo=_lo,     _hi=mid))])

def transStore(value: sb3.Value | IndexableValue, address: sb3.Value, ty: ir.Type, ctx: Context) -> sb3.BlockList:
  match value:
    case sb3.Value():
      if getByteSize(ty) > 1:
        # TODO FIX: allow storing of larger values
        raise CompException("Only 1 scratch byte can be stored per stack value at the moment")

      return sb3.BlockList([sb3.EditList("replaceat", ctx.cfg.stack_var, address, value)])
    case IndexableValue():
      if not (isinstance(ty, ir.ArrayType) and isinstance(ty.elem_ty, ir.IntegerType)):
        raise CompException(f"Expected stored type {ty} to be [Y x iN]")

      if getByteSize(ty.elem_ty) > 1:
        raise CompException("Only 1 scratch byte can be stored per stack value at the moment")

      blocks = sb3.BlockList()
      for (i, ival) in enumerate(value.vals):
        # TODO OPTI: when adding operator blocks, calc at compile time if values are known
        blocks.add(sb3.EditList("replaceat", ctx.cfg.stack_var,
                                  sb3.Op("add", address, sb3.Known(i)), ival))

      return blocks
    case _:
      raise CompException("Unmatched")

def transCall(name: str, arguments: list[sb3.Value],
              output: Variable | None, ctx: Context) -> tuple[sb3.BlockList, sb3.BlockList]:
  """The first block list returned is any blocks to call the function, the second any needed to assign the return value to the output"""

  call_blocks = sb3.BlockList([sb3.ProcedureCall(name, arguments)])

  set_value_blocks = sb3.BlockList()
  if output is not None:
    set_value_blocks.add(output.setValue(sb3.GetVar(ctx.cfg.return_var)))

  return call_blocks, set_value_blocks

def transInstr(instr: ir.Instruction, ctx: Context, bctx: BlockInfo) -> tuple[sb3.BlockList, Context]:
  blocks = sb3.BlockList()
  match instr:
    case ir.Alloca(): # Allocate space on the stack and return ptr
      var = decodeVar(instr.result, bctx)
      assert var is None or var.is_scratch_param is False

      blocks = sb3.BlockList()
      size = getByteSize(instr.allocated_ty)

      if var is not None:
        blocks.add(var.setValue(sb3.GetVar(ctx.cfg.stack_size_var)))

      # TODO OPTI: can skip increasing the size if we know the function does not allocate or call another func
      blocks.add(sb3.EditVar("change", ctx.cfg.stack_size_var, sb3.Known(size)))

    case ir.Load(): # Load a value from an address on the stack
      address = decodeValue(instr.address, ctx, bctx)
      blocks.add(address.blocks)

      if isinstance(address.value, IndexableValue):
        raise CompException(f"Address to load cannot be an indexable value in {instr}")

      var = decodeVar(instr.result, bctx)
      if var is not None:
        blocks.add(sb3.EditVar("set", var.name,
                                      sb3.GetOfList("atindex", ctx.cfg.stack_var, address.value)))

    case ir.Store(): # Copy a value to an address on the stack
      value = decodeValue(instr.value, ctx, bctx)
      blocks.add(value.blocks)

      address = decodeValue(instr.address, ctx, bctx)
      blocks.add(address.blocks)

      if isinstance(address.value, IndexableValue):
        raise CompException(f"Address to store cannot be an indexable value in {instr}")

      blocks.add(transStore(value.value, address.value, instr.value.ty, ctx))

    case ir.BinOp(): # Do a calculation with two values
      left = decodeValue(instr.fst_operand, ctx, bctx)
      blocks.add(left.blocks)
      left = left.value

      right = decodeValue(instr.snd_operand, ctx, bctx)
      blocks.add(right.blocks)
      right = right.value

      res_var = decodeVar(instr.result, bctx)
      assert res_var is None or res_var.is_scratch_param is False
      res_val = None

      if isinstance(left, IndexableValue) or \
         isinstance(right, IndexableValue):
        raise CompException(f"Indexable value not supported in binop {instr}")

      match instr.opcode:
        case "add" | "sub": # Add/Sub two values
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values by using multiple vars and carrying
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          if instr.is_nsw and instr.is_nuw and ctx.cfg.opti:
            # If no wrapping behaviour is required then under/overflowing is ub so can be ignored
            res_val = sb3.Op(instr.opcode, left, right)
          else:
            res_val = sb3.Op("mod", sb3.Op(instr.opcode, left, right),
                             sb3.Known(2 ** width))

        case "mul": # Multiply two values
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          width = instr.fst_operand.ty.num_bits

          if instr.is_nsw and instr.is_nuw and ctx.cfg.opti:
            res_val = multiplyNoWrap(width, left, right)
          else:
            blocks_and_value = multiplyWrap(width, left, right, ctx)
            blocks.add(blocks_and_value.blocks)
            res_val = blocks_and_value.value

        case "udiv": # Divide one value by another (unsigned)
          # TODO OPTI: optimise for known values
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          # Division by zero is UB
          if not instr.is_exact:
            res_val = sb3.Op("floor", sb3.Op("div", left, right))
          else:
            res_val = sb3.Op("div", left, right) # Value is poison if one is not a multiple of another

        case "sdiv": # Divide one value by another (signed)
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          width = instr.fst_operand.ty.num_bits

          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          if instr.is_exact:
            signed_left, lblocks = astuple(undoTwosComplement(width, left, False, ctx))
            signed_right, rblocks = astuple(undoTwosComplement(width, right, False, ctx))
            blocks.add(lblocks)
            blocks.add(rblocks)

            res_val = twosComplement(width, sb3.Op("div", signed_left, signed_right))
          else:
            if res_var is not None:
              left, lblocks = astuple(optimizeValueUse(left, 2, ctx))
              right, rblocks = astuple(optimizeValueUse(right, 2, ctx))
              blocks.add(lblocks)
              blocks.add(rblocks)

              point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
              change = 2 ** width

              # TODO: optimise for known values

              # Undo two's complement, divide, round towards zero using floor or ceiling and calculate two's complement
              blocks.add([
                sb3.ControlFlow("if_else", sb3.BoolOp("<", left, sb3.Known(point_of_neg)), sb3.BlockList([
                  sb3.ControlFlow("if_else", sb3.BoolOp("<", right, sb3.Known(point_of_neg)), sb3.BlockList([
                    # If left + right are pos
                    sb3.EditVar("set", res_var.name, sb3.Op("floor", sb3.Op("div", left, right))),
                  ]), sb3.BlockList([
                    # If left is pos and right is neg
                    sb3.EditVar("set", res_var.name, sb3.Op("add",
                                                      sb3.Op("ceiling",
                                                        sb3.Op("div",
                                                          left,
                                                          sb3.Op("sub", right, sb3.Known(change)))),
                                                      sb3.Known(change))),
                  ]))
                ]), sb3.BlockList([
                  sb3.ControlFlow("if_else", sb3.BoolOp("<", right, sb3.Known(point_of_neg)), sb3.BlockList([
                    # If left is neg and right is pos
                    sb3.EditVar("set", res_var.name, sb3.Op("add",
                                                      sb3.Op("ceiling",
                                                        sb3.Op("div",
                                                          sb3.Op("sub", left, sb3.Known(change)),
                                                          right)),
                                                      sb3.Known(change))),
                  ]), sb3.BlockList([
                    # If left + right are neg
                    sb3.EditVar("set", res_var.name, sb3.Op("floor",
                                                      sb3.Op("div",
                                                        sb3.Op("sub", left, sb3.Known(change)),
                                                        sb3.Op("sub", right, sb3.Known(change))))),
                  ]))
                ]))
              ])
            res_val = False # We set res_var ourselves

        case "urem": # Calculate remainder (unsigned)
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          # mod 0 is UB, can ignore
          res_val = sb3.Op("mod", left, right)

        case "srem": # Calculate remainder (signed)
          # TODO OPTI: optimise for known values
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          if res_var is not None:
            # TODO: Reuse if statement to work out if a / b > 0
            left, lblocks = astuple(optimizeValueUse(left, 2, ctx))
            right_is_temp = shouldOptimiseValueUse(right, 3)
            right, rblocks = astuple(optimizeValueUse(right, 3, ctx))
            if right_is_temp:
              assert isinstance(right, sb3.GetVar)
            blocks.add(lblocks)
            blocks.add(rblocks)

            point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
            change = 2 ** width

            if not right_is_temp:
              # Undo Two's Complement
              right_sub_change = sb3.Op("sub", right, sb3.Known(change))

              right_sub_change, pos_neg_block = astuple(optimizeValueUse(right, 2, ctx))
              pos_neg_block.add(sb3.BlockList([
                # If left is pos and right is neg - remainder = (l mod r) - r
                sb3.EditVar("set", res_var.name, sb3.Op("sub",
                                                  sb3.Op("mod",
                                                    left,
                                                    right_sub_change),
                                                  right_sub_change))]))
            else:
              # Re-use the generated temp var
              pos_neg_block = sb3.BlockList([
                sb3.EditVar("change", right.var_name, sb3.Known(-change)),
                sb3.EditVar("set", res_var.name, sb3.Op("sub",
                                                  sb3.Op("mod",
                                                    left,
                                                    right),
                                                  right))])

            # Undo two's complement, calculate modulo, then adjust for differences with llvm's remainder operation
            # (different when one side is negative)
            blocks.add([
              sb3.ControlFlow("if_else", sb3.BoolOp("<", left, sb3.Known(point_of_neg)), sb3.BlockList([
                sb3.ControlFlow("if_else", sb3.BoolOp("<", right, sb3.Known(point_of_neg)), sb3.BlockList([
                    # Modulus and remainder operations do the same
                    sb3.EditVar("set", res_var.name, sb3.Op("mod", left, right)),
                  ]),
                  # If left is pos and right is neg - remainder = (l mod r) - r
                  pos_neg_block
                )
              ]), sb3.BlockList([
                sb3.ControlFlow("if_else", sb3.BoolOp("<", right, sb3.Known(point_of_neg)), sb3.BlockList([
                  # If left is neg and right is pos - remainder = (l mod r) - r
                  sb3.EditVar("set", res_var.name, sb3.Op("add",
                                                    sb3.Op("sub",
                                                      sb3.Op("mod",
                                                        sb3.Op("sub", left, sb3.Known(change)),
                                                        right),
                                                      right),
                                                    sb3.Known(change))),
                ]), sb3.BlockList([
                  # If left + right are neg
                  sb3.EditVar("set", res_var.name, sb3.Op("add",
                                                    sb3.Op("mod",
                                                      sb3.Op("sub", left, sb3.Known(change)),
                                                      sb3.Op("sub", right, sb3.Known(change))),
                                                    sb3.Known(change))),
                ]))
              ]))
            ])

          res_val = False # We set res_var ourselves

        case "shl": # Calculate left shift
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          can_shift_out = not (instr.is_nsw and instr.is_nuw)
          res_val, ctx = bitShift("left", width, left, right, ctx, can_shift_out)

        case "lshr": # Calculate right shift (unsigned)
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          can_shift_out = not instr.is_exact
          res_val, ctx = bitShift("right", width, left, right, ctx, can_shift_out)

        case "ashr": # Calculate right shift (signed)
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          if res_var is not None:
            point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
            change = 2 ** width

            right_mul, ctx = intPow2(right, width, ctx)

            unwrapped_pos = sb3.Op("div", left, right_mul)
            val_pos = unwrapped_pos if instr.is_exact else sb3.Op("floor", unwrapped_pos)

            unwrapped_neg = sb3.Op("div", sb3.Op("sub", left, sb3.Known(change)), right_mul)
            val_neg = sb3.Op("add", unwrapped_neg if instr.is_exact else sb3.Op("ceiling", unwrapped_neg), sb3.Known(change))

            blocks.add([
              sb3.ControlFlow("if_else", sb3.BoolOp("<", left, sb3.Known(point_of_neg)), sb3.BlockList([
                sb3.EditVar("set", res_var.name, val_pos),
              ]), sb3.BlockList([
                sb3.EditVar("set", res_var.name, val_neg),
              ])),
            ])

          res_val = False # We set res_var ourselves

        case "and" | "or" | "xor": # Calculate binary operation
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          if instr.opcode == "or" and instr.is_disjoint:
            # If there would be no carry
            res_val = sb3.Op("add", left, right)
          else:
            # TODO OPTI: gen (11-bit) tables/use mod for known values

            if width > ctx.cfg.binop_lookup_bits:
              uses = width / ctx.cfg.binop_lookup_bits
              left, lblocks = astuple(optimizeValueUse(left, uses, ctx))
              right, rblocks = astuple(optimizeValueUse(right, uses, ctx))
              blocks.add(lblocks)
              blocks.add(rblocks)

            lookup_size = 2 ** ctx.cfg.binop_lookup_bits
            name = f"{ctx.cfg.binop_lookup_var} {instr.opcode}{ctx.cfg.zero_indexed_suffix}"
            lookup = []
            for l in range(0, lookup_size):
              for r in range(0, lookup_size):
                lookup.append(sb3.Known({"and": l & r, "or": l | r, "xor": l ^ r}[instr.opcode]))
            ctx.proj.lists[name] = lookup[1:] # since 0 &/|/^ 0 is 0, an empty value being treated as zero is fine

            results = []
            for offset in range(0, width, ctx.cfg.binop_lookup_bits):
              left_index = left
              right_index = right
              if offset > 0:
                left_index = sb3.Op("floor", sb3.Op("div", left_index, sb3.Known(2 ** offset)))
                # No floor instruction needed because scratch rounds down with atindex
                right_index = sb3.Op("div", right_index, sb3.Known(2 ** offset))
              if offset + ctx.cfg.binop_lookup_bits < width:
                left_index = sb3.Op("mod", left_index, sb3.Known(lookup_size))
                right_index = sb3.Op("mod", right_index, sb3.Known(lookup_size))
              left_index = sb3.Op("mul", left_index, sb3.Known(lookup_size))

              result = sb3.GetOfList("atindex", name, sb3.Op("add", left_index, right_index))
              if offset > 0: result = sb3.Op("mul", result, sb3.Known(2 ** offset))
              results.append(result)

            if len(results) > 1:
              res_val = sb3.Op("add", results.pop(), results.pop())
              while len(results) > 0:
                res_val = sb3.Op("add", res_val, results.pop())
            else:
              res_val = results[0]

        case _:
          raise CompException(f"Unknown instruction opcode {instr} (type BinOp)")

      if res_val is not False: # If the binop sets res_var itself
        if res_var is not None:
          blocks.add(res_var.setValue(res_val))
        else:
          if not ctx.cfg.opti: # Values have no effect on state in scratch, only state has effect on values
            blocks.add(sb3.EditVar("set", ctx.cfg.unused_var, res_val))

    case ir.ICmp(): # Compare two values
      left = decodeValue(instr.fst_operand, ctx, bctx)
      blocks.add(left.blocks)
      left = left.value

      right = decodeValue(instr.snd_operand, ctx, bctx)
      blocks.add(right.blocks)
      right = right.value

      res_var = decodeVar(instr.result, bctx)
      assert res_var is None or res_var.is_scratch_param is False

      if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
        raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
      assert isinstance(left, sb3.Value) and isinstance(right, sb3.Value)

      width = instr.fst_operand.ty.num_bits
      # TODO FIX: support larger values
      if width > 48:
        raise CompException(f"Instruction icmp currently supports integers with <= 48 bits")

      # TODO FIX: boolean values 'true' or 'false' could potentially create issues, use a new class called BoolIntCast
      match instr.cond:
        case "eq":
          res_val = sb3.BoolOp("=", left, right)
        case "ne":
          res_val = sb3.BoolOp("not", sb3.BoolOp("=", left, right))
        case _:
          # TODO FIX: ugt, uge, ult, ule, and signed counterparts
          raise CompException(f"icmp does not support comparsion mode {instr.cond}")

      # Bool as int will cast to an int if needed (so the bool isn't treated as 'true')
      res_val = sb3.Op("bool_as_int", res_val)

      if res_var is not None:
        blocks.add(res_var.setValue(res_val))

    case ir.Conversion(): # Convert a value from one type to another
      value = decodeValue(instr.value, ctx, bctx)
      blocks.add(value.blocks)
      value = value.value

      res_var = decodeVar(instr.result, bctx)
      assert res_var is None or res_var.is_scratch_param is False

      if not isinstance(instr.value.ty, ir.IntegerType): # TODO: add vector support
        raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.value.ty)}")
      assert isinstance(value, sb3.Value)

      width = instr.value.ty.num_bits
      # TODO FIX: support larger values
      if width > 48:
        raise CompException(f"Instruction icmp currently supports integers with <= 48 bits")

      res_val = None

      match instr.opcode:
        case "zext":
          res_val = value
        case _:
          raise CompException(f"Unknown instruction opcode {instr} (type Conversion)")

      if res_val is not None and res_var is not None:
        blocks.add(res_var.setValue(res_val))

    case _:
      raise CompException(f"Unsupported instruction opcode {instr} (type {type(instr)})")
  return blocks, ctx

def assignParameters(func: FuncInfo) -> sb3.BlockList:
  # TODO OPTI: work out if values are needed before assigning them
  blocks = sb3.BlockList()
  for param_name in func.params:
    param = localizeVar(param_name, False, BlockInfo(func, True))
    var = localizeVar(param_name, False, BlockInfo(func, False))
    blocks.add(var.setValue(param.getValue()))
  return blocks

def getUncheckedProcedureStart(proc_name: str, params: list[str], fn: FuncInfo,
                               ctx: Context, is_counted: bool=False) -> tuple[sb3.BlockList, Context]:
  blocks = sb3.BlockList([sb3.ProcedureDef(proc_name, params)])

  if ctx.cfg.do_debug_branch_log:
    # Increment the branch counter in the log
    index = sb3.Known((fn.fn_id * 2) + 2)
    blocks.add(sb3.BlockList([
      sb3.EditList("replaceat", ctx.cfg.debug_branch_log_var, index, sb3.Op("add",
        sb3.Known(1), sb3.GetOfList("atindex", ctx.cfg.debug_branch_log_var, index)
    ))]))

  if is_counted:
    blocks.add(sb3.EditCounter("incr")) # The 'hacked' counter blocks are 20x faster than incrementing a number

  return blocks, ctx

def getCheckedProcedureStart(proc_name: str, params: list[str], fn: FuncInfo,
                             ctx: Context) -> tuple[sb3.BlockList, Context]:
  """Returns the blocks needed to return a branch instruction (procedure) that will reset the scratch's stack
  if reaching a max amount of permutations, preventing scratch from running out of memory"""

  reset_broadcast = f"{proc_name}:reset stack"

  arguments = [localizeVar(name, False, BlockInfo(fn, False)).getValue() for name in params]
  ctx.proj.code.append(sb3.BlockList([
    sb3.OnBroadcast(reset_broadcast),
    sb3.ProcedureCall(proc_name, arguments),
  ]))

  blocks, ctx = getUncheckedProcedureStart(proc_name, params, fn, ctx, is_counted=True)

  on_reset = sb3.BlockList()
  if len(params) > 0:
    # FUTURE OPTI: the assigned params could potentially used elsewhere but there's hardly a perf gain
    on_reset = assignParameters(fn)
  on_reset.add(sb3.BlockList([
    sb3.Broadcast(sb3.Known(reset_broadcast), False), # While broadcasts are slow, this is the fastest option in this rare case
    sb3.EditCounter("clear"),
    sb3.StopScript("stopthis")]))

  blocks.add(sb3.ControlFlow("if",
    sb3.BoolOp(">", sb3.GetCounter(), sb3.Known(ctx.cfg.max_branch_recursion)), on_reset))

  return blocks, ctx

def transTerminatorInstr(instr: ir.Instruction,
                         ctx: Context, bctx: BlockInfo) -> tuple[sb3.BlockList, Context]:
  blocks = sb3.BlockList()
  match instr:
    case ir.Unreacheble(): # Never reached if not UB
      pass

    case ir.Ret(): # Return from a func
      if instr.value is not None:
        value = decodeValue(instr.value, ctx, bctx)
        if isinstance(value.value, IndexableValue):
          raise CompException(f"Returning multiple values not supported in {instr}")
        blocks.add(value.blocks)
        blocks.add(sb3.EditVar("set", ctx.cfg.return_var, value.value))

      if bctx.fn.returns_to_address:
        return_address = localizeVar(ctx.cfg.return_address_local, False, bctx)
        return_to_addr_code = OrderedDict()
        for i, addr in enumerate(bctx.fn.return_addresses):
          return_to_addr_code.update({i: sb3.BlockList([sb3.ProcedureCall(addr, [])])})

        return_table = binarySearch(return_address.getValue(), return_to_addr_code, are_branches_sorted=True)
        blocks.add(return_table)

      # TODO FIX: deallocate on ret instruction (only if function can allocate)
      blocks.end = True

    case ir.Br(): # Jump to a label, either known or dependent on a condition
      if bctx.is_scratch_func:
        # Allow the parameters to be accessed later
        blocks.add(assignParameters(bctx.fn))

      if instr.cond is None:
        label = instr.label_false.val
        assert isinstance(label, str)
        proc_name = localizeLabel(label, bctx.fn)
        blocks.add(sb3.ProcedureCall(proc_name, []))
      else:
        cond = decodeValue(instr.cond, ctx, bctx)
        if isinstance(cond.value, IndexableValue):
          raise CompException(f"Indexable value not supported for brach condition {instr}")
        blocks.add(cond.blocks)

        assert instr.label_true is not None
        label_true, label_false = instr.label_true.val, instr.label_false.val
        assert isinstance(label_true, str) and isinstance(label_false, str)
        true_proc_name, false_proc_name = localizeLabel(label_true, bctx.fn), localizeLabel(label_false, bctx.fn)

        # TODO: use broadcast(join + condition) where possible
        blocks.add(sb3.ControlFlow("if_else", sb3.BoolOp("=", cond.value, sb3.Known(1)), sb3.BlockList([
          sb3.ProcedureCall(true_proc_name, [])
        ]), sb3.BlockList([
          sb3.ProcedureCall(false_proc_name, [])
        ])))

    case ir.Switch(): # Jump to many labels depending on a value
      assert isinstance(instr.cond.ty, ir.IntegerType)
      width = instr.cond.ty.num_bits
      if getByteSize(instr.cond.ty) > 1:
        raise CompException("Cannot currently switch with an integer more than 48 bits (would take multiple vars to store)")

      val, val_blocks = astuple(decodeValue(instr.cond, ctx, bctx))
      blocks.add(val_blocks)
      if len(instr.cases) > 1:
        val, opti_val_blocks = astuple(optimizeValueUse(val, math.log2(len(instr.cases)), ctx))
        blocks.add(opti_val_blocks)

      case_vs_label: OrderedDict[int, sb3.BlockList] = OrderedDict()
      for case, label in instr.cases:
        case_val = decodeValue(case, ctx, bctx)
        # Switch cases should be constant and unique
        assert isinstance(case_val, ValueAndBlocks)
        assert len(case_val.blocks.blocks) == 0
        assert isinstance(case_val.value, sb3.Known)
        assert isinstance(case_val.value.known, float)
        assert case_val.value.known == int(case_val.value.known)
        assert int(case_val.value.known) not in case_vs_label
        case_val = int(case_val.value.known)

        label_name = label.val
        assert isinstance(label_name, str)
        label_proc_name = localizeLabel(label_name, bctx.fn)
        label_block_call = sb3.ProcedureCall(label_proc_name, [])

        case_vs_label[case_val] = sb3.BlockList([label_block_call])

      assert isinstance(instr.label_default.val, str)
      default_proc_name = localizeLabel(instr.label_default.val, bctx.fn)
      default_label_call = sb3.BlockList([sb3.ProcedureCall(default_proc_name, [])])

      lowest_poss = 0
      highest_poss = (2 ** width) - 1 # FUTURE OPTI: if the value called from was zero-extended (e.g. from an i8)
                                      # then could set this value to the max of the type before

      blocks.add(binarySearch(val, case_vs_label, default_label_call, lowest_poss, highest_poss))

    case _:
      raise CompException(f"Unsupported terminator instruction opcode {instr} (type {type(instr)})")
  return blocks, ctx

def transGlobals(mod: ir.Module, ctx: Context) -> tuple[sb3.BlockList, Context]:
  blocks = sb3.BlockList()

  # TODO: not necessary if done beforehand (only need to change stack size)
  blocks.add(sb3.EditList("deleteall", ctx.cfg.stack_var, None, None))

  if ctx.cfg.stack_size > SCRATCH_LIST_LIMIT:
    raise CompException("Stack is too large to fit in one list, multiple lists not implemented")

  ptr = 1

  # Set up static values
  for glob in mod.global_vars.values():
    # TODO OPTI: Don't use allocate and store and set to the var, instead just remember the
    # ptr value in the decoder and replace any uses of the variable with it (only when opti=true)
    globvar = decodeVar(glob.value, None)
    if globvar is None:
      raise CompException(f"Expected static var {glob} to be named")

    blocks_and_value = decodeValue(glob.initializer, ctx, None)
    unknown = len(blocks_and_value.blocks) > 0
    value = blocks_and_value.value

    if isinstance(value, IndexableValue):
      unknown |= not all([isinstance(val, sb3.Known) for val in value.vals])
    else:
      unknown |= not isinstance(value, sb3.Known)

    if unknown: raise CompException(f"Expected static value {glob} to have a compile time known value")

    total_size = getByteSize(glob.initializer.ty)
    values = []
    match value:
      case sb3.Known():
        size = total_size
        values.append(value)
      case IndexableValue():
        assert isinstance(glob.initializer.ty, ir.ArrayType)
        size = getByteSize(glob.initializer.ty.elem_ty)
        values.extend(value.vals)

    if size != 1: raise CompException("Cannot create value or values with a size in bytes > 1")

    ctx.globvar_to_ptr[globvar.name] = sb3.Known(ptr)
    if not ctx.cfg.opti:
      blocks.add(sb3.EditVar("set", globvar.name, sb3.Known(ptr)))
    for value in values:
      blocks.add(sb3.EditList("addto", ctx.cfg.stack_var, None, value))
    ptr += total_size

  blocks.add(sb3.BlockList([
    sb3.EditVar("set", ctx.cfg.stack_size_var, sb3.Known(ptr)),
    sb3.ControlFlow("reptimes", sb3.Known(ctx.cfg.stack_size - (ptr - 1)), sb3.BlockList([
      sb3.EditList("addto", ctx.cfg.stack_var, None, sb3.Known(0))
    ])),
  ]))

  return blocks, ctx

def getFnInfo(mod: ir.Module, ctx: Context) -> Context:
  # Get info about what a function calls and how it returns
  call_graph: dict[str, tuple[set[str], bool]] = {}
  return_addresses: dict[str, list[str]] = {}
  returns_to_address: dict[str, bool] = {}
  all_check_locations: dict[str, list[str]] = {}
  for func in mod.funcs.values():
    fn_name = func.value.val
    assert isinstance(fn_name, str)

    if func.has_no_body():
      if fn_name not in ctx.fn_info:
        raise CompException("Externally defined function must have info about it")
      info = ctx.fn_info[fn_name]
      call_graph[fn_name] = info.can_call, True

    else:
      calls: set[str] = set() # What the function calls
      for block in func.blocks.values():
        # Find every function the function could call
        block_label = block.value.val
        assert isinstance(block_label, str)
        call_id = 0
        for instr in block.instrs:
          if isinstance(instr, ir.Call) or isinstance(instr, ir.CallBr): # TODO FIX: tail recursion treated differently?
            called_name = instr.callee.val
            assert isinstance(called_name, str)
            calls.add(called_name)
            return_addresses.setdefault(called_name, list())
            return_addresses[called_name].append(localizeCallId(call_id, block_label, fn_name))
            call_id += 1
      call_graph[fn_name] = calls, False

  for start_func in call_graph:
    if call_graph[start_func][1]: continue

    visited = set()
    stack = list(call_graph[start_func][0])

    while stack:
      func = stack.pop()
      if func in visited: continue
      visited.add(func)

      callees, searched = call_graph[func]
      if searched:
        visited.update(callees)
      else:
        stack.extend(callees) # OPTI: re-use results if computed here first

    call_graph[start_func] = visited, True

  for func in mod.funcs.values():
    fn_name = func.value.val
    assert isinstance(fn_name, str)

    if func.has_no_body():
      info = ctx.fn_info[fn_name]
      returns_to_address[fn_name] = info.returns_to_address
      all_check_locations[fn_name] = info.checked_blocks
    else:
      branches: dict[str, list[str]] = {"ret": []} # Where different branches could lead
      for block in func.blocks.values():
        # Find every function the function could call
        block_label = block.value.val
        assert isinstance(block_label, str)

        # Find what each block in the function could branch to
        term_instr = block.instrs[-1]
        match term_instr:
          case ir.Unreacheble():
            pass
          case ir.Ret():
            branches[block_label] = ["ret"]
          case ir.Br():
            assert isinstance(term_instr.label_false.val, str)
            branches[block_label] = [term_instr.label_false.val]
            if term_instr.label_true is not None:
              assert isinstance(term_instr.label_true.val, str)
              branches[block_label].append(term_instr.label_true.val)
          case ir.Switch():
            assert isinstance(term_instr.label_default.val, str)
            branches[block_label] = [term_instr.label_default.val]
            for _, label in term_instr.cases:
              assert isinstance(label.val, str)
              branches[block_label].append(label.val)
          case ir.CallBr():
            assert isinstance(term_instr.fallthrough_label.val, str)
            if term_instr.indirect_labels is not None:
              raise CompException("Indirect (exceptional) labels not supported")
            branches[block_label].append(term_instr.fallthrough_label.val)
          case _:
            raise CompException(f"Unsupported terminator instruction opcode {instr} (type {type(instr)})")

      fn_check_locations = util.select_minimum_checks_scc(branches)
      could_recurse = len(fn_check_locations) > 0
      # If the branches could create a loop, we must place stack checks, so we should return to an
      # address. Furthermore, a binary search and a call is usually faster than potentially
      # hundreds of recursions backward.
      returns_to_address[fn_name] = could_recurse
      all_check_locations[fn_name] = fn_check_locations

      # FUTURE OPTI: could check to see if in a non-recursive environment, there are enough
      # branches that it is faster to recurse backward. However this happens rarely, it is
      # difficult to gauge if it will actually have a net positive impact (will force callers
      # to return to an address, the longest path may only be taken some of the times).
      #if not could_recurse:
      #  first_label = list(func.blocks.values())[0].value.val
      #  assert isinstance(first_label, str)
      #  longest_path, max_branch_count = util.longest_path_dag(branches, first_label, "ret")
      #  max_branch_count -= 1 # Ignore "ret" path
      #  can_return = longest_path is not None
      #  return_address_count = len(return_addresses.get(fn_name, []))

  for fn_name in returns_to_address:
    # If the function returns to an address, then callers must also return to an address
    returns_to_address[fn_name] |= any(returns_to_address[call] for call in call_graph[fn_name][0])

  for func in mod.funcs.values():
    fn_name = func.value.val
    assert isinstance(fn_name, str)

    param_names = []
    for arg in func.args:
      assert isinstance(arg.val, str)
      param_names.append(arg.val[1:])

    if returns_to_address[fn_name]: param_names.append(ctx.cfg.return_address_local)

    ctx.fn_info[fn_name] = FuncInfo(fn_name, ctx.next_fn_id, param_names, call_graph[fn_name][0],
                                    return_addresses.get(fn_name, list()), returns_to_address[fn_name],
                                    all_check_locations.get(fn_name, list()))
    ctx.next_fn_id += 1

  return ctx

def transFuncs(mod: ir.Module, ctx: Context) -> Context:
  ctx = getFnInfo(mod, ctx)

  for func in mod.funcs.values():
    if func.has_no_body(): continue

    assert isinstance(func.value.ty, ir.FunctionType)
    fn_name = func.value.val
    assert isinstance(fn_name, str)

    info = ctx.fn_info[fn_name]

    for param_ty in func.value.ty.param_tys:
      if getByteSize(param_ty) > 1:
        raise CompException("Parameters can only be one scratch byte in size") # TODO FIX

    first_block = True
    for block in func.blocks.values():
      call_id = 0
      block_label = block.value.val
      assert isinstance(block_label, str)

      if first_block:
        proc_name = fn_name
        localized_params = [localizeParameter(param) for param in info.params]
      else:
        proc_name = localizeLabel(block_label, info)
        localized_params = []

      if block_label not in info.checked_blocks:
        block_code, ctx = getUncheckedProcedureStart(proc_name, localized_params, info, ctx,
                                                     is_counted=info.returns_to_address)
      else:
        block_code, ctx = getCheckedProcedureStart(proc_name, localized_params, info, ctx)

      if first_block and ctx.cfg.do_debug_branch_log:
        # Increment the function counter in the log
        index = sb3.Known((info.fn_id * 2) + 1)
        block_code.add(sb3.BlockList([
          sb3.EditList("replaceat", ctx.cfg.debug_branch_log_var, index, sb3.Op("add",
            sb3.Known(1), sb3.GetOfList("atindex", ctx.cfg.debug_branch_log_var, index)
        ))]))

      # After the first block we can no longer access the parameters in the function
      bctx = BlockInfo(info, first_block)
      # Translate everything except the terminator operation
      for instr in block.instrs[:-1]:
        if isinstance(instr, ir.Call): # Call instructions handled here because might need to return at an address
          callee_name = instr.callee.val
          assert isinstance(callee_name, str)

          # TODO FIX: in llvm getElementPtr is used but seems to be ignored in this code
          arguments: list[sb3.Value] = []
          for arg in instr.args:
            value = decodeValue(arg, ctx, bctx)
            if isinstance(value, IValueAndBlocks):
              raise CompException(f"Function argument cannot be an indexable value in {instr}")
            block_code.add(value.blocks)
            arguments.append(value.value)

          output = decodeVar(instr.result, bctx)

          callee_info = ctx.fn_info[callee_name]
          callee_returns_to_address = callee_info.returns_to_address

          if callee_returns_to_address:
            return_proc_name = localizeCallId(call_id, block_label, fn_name)
            return_addr_id = callee_info.return_addresses.index(return_proc_name)
            arguments.append(sb3.Known(return_addr_id)) # Pass return address into function

            # If adding a broadcast then make sure parameters can be accessed later
            if first_block: block_code.add(assignParameters(info))

          call_blocks, assign_blocks = transCall(callee_name, arguments, output, ctx)
          block_code.add(call_blocks)

          if not callee_returns_to_address:
            block_code.add(assign_blocks)
          else:
            # Start new block list
            ctx.proj.code.append(block_code)
            first_block = False
            bctx = BlockInfo(info, first_block)

            # Add code for callback
            block_code, ctx = getUncheckedProcedureStart(return_proc_name, [], info, ctx,
                                                         is_counted=info.returns_to_address)
            block_code.add(assign_blocks)

          call_id += 1
        else:
          instr_code, ctx = transInstr(instr, ctx, bctx)
          block_code.add(instr_code)

      # TODO OPTI: work out where if statements can be placed etc
      terminator_code, ctx = transTerminatorInstr(block.instrs[-1], ctx, bctx)
      block_code.add(terminator_code)

      ctx.proj.code.append(block_code)

      first_block = False

  return ctx

def addFunc(name: str, params: list[str], can_call: set[str],
            contents: sb3.BlockList, ctx: Context) -> Context:
  #if returns_to_address: params.append(ctx.cfg.return_address_local)
  localized_params = [localizeParameter(param) for param in params]
  blocks = sb3.BlockList([sb3.ProcedureDef(name, localized_params)])
  blocks.add(contents)
  ctx.proj.code.append(blocks)
  ctx.fn_info[name] = FuncInfo(name, ctx.next_fn_id, params, can_call, list(), False, list())
  ctx.next_fn_id += 1
  return ctx

def compile(llvm: str | ir.Module, cfg: Config | None=None) -> tuple[sb3.Project, DebugInfo]:
  """Compile LLVM IR to a scratch project. Returns a project and any debug info generated."""
  if cfg is None: cfg = Config()
  debug_info = cfg.debug_info
  scfg = sb3.ScratchConfig(
    invis_blocks=cfg.invis_blocks)

  ctx = Context(sb3.Project(scfg), cfg)

  # Parse llvm
  mod: ir.Module = parse_assembly(llvm) if isinstance(llvm, str) else llvm

  # Starting code
  initblocks = sb3.BlockList([sb3.OnStartFlag()])

  # Reset call stack
  initblocks.add(sb3.EditCounter("clear"))

  # Setup stack
  globblocks, ctx = transGlobals(mod, ctx)
  initblocks.add(globblocks)

  # Add foreign functions
  ascii_lookup = []
  for x in range(1, 256): # Ignore zero; improves perf as scratch lists are 1 indexed and zero signifies end of string
    char = chr(x)
    if char.encode("unicode_escape").decode("ascii").startswith("\\") and char != "\\":
      ascii_lookup.append(sb3.Known(f"\\{x:02X}"))
    else:
      ascii_lookup.append(sb3.Known(char))
  ctx.proj.lists[cfg.ascii_lookup_var + cfg.zero_indexed_suffix] = ascii_lookup

  ctx = addFunc("puts", ["input"], set(), sb3.BlockList([
    sb3.EditVar("set", "buffer", sb3.Known("")),
    sb3.EditVar("set", "ptr", sb3.GetParameter(localizeParameter("input"))),
    sb3.EditVar("set", "char", sb3.GetOfList("atindex", cfg.stack_var, sb3.GetVar("ptr"))),
    sb3.ControlFlow("until", sb3.BoolOp("=", sb3.GetVar("char"), sb3.Known(0)), sb3.BlockList([
      sb3.EditVar("set", "buffer",
        sb3.Op("join", sb3.GetVar("buffer"), sb3.GetOfList("atindex", (cfg.ascii_lookup_var + cfg.zero_indexed_suffix), sb3.GetVar("char")))),
      sb3.EditVar("change", "ptr", sb3.Known(1)),
      sb3.EditVar("set", "char", sb3.GetOfList("atindex", cfg.stack_var, sb3.GetVar("ptr"))),
    ])),
    sb3.Say(sb3.GetVar("buffer")),
    sb3.EditVar("set", cfg.return_var, sb3.Known(0)),
  ]), ctx)

  ctx = addFunc("putchar", ["input"], set(), sb3.BlockList([
    sb3.Say(sb3.GetOfList("atindex", (cfg.ascii_lookup_var + cfg.zero_indexed_suffix), sb3.GetParameter(localizeParameter("input")))),
    sb3.EditVar("set", cfg.return_var, sb3.Known(0)),
  ]), ctx)

  # Translate functions
  ctx = transFuncs(mod, ctx)

  # Reset debug info on initialisation
  if cfg.do_debug_branch_log:
    initblocks.add(sb3.EditList("deleteall", cfg.debug_branch_log_var, None, None))

    debug_branch_list_length = (ctx.next_fn_id * 2) + 10 # Adding 10 for safety + bc it doesn't really matter
    if debug_branch_list_length > SCRATCH_LIST_LIMIT:
      raise CompException("Too many functions; could not add them all to the recursive branch debug list")

    initblocks.add(sb3.ControlFlow("reptimes", sb3.Known(debug_branch_list_length), sb3.BlockList([
      sb3.EditList("addto", cfg.debug_branch_log_var, None, sb3.Known(0))
    ])))

  # Export debug info
  if cfg.do_debug_branch_log:
    branch_debug_info: list[tuple[int, str]] = []
    for info in ctx.fn_info.values():
      branch_debug_info.append((info.fn_id, info.name))
    branch_debug_info.sort(key=lambda info: info[0])

    debug_info.debug_branch_func_map = "\n".join([info[1] for info in branch_debug_info])

  # Call main func call to init code
  if not "main" in ctx.fn_info:
    raise CompException("No main function") # TODO FIX: add libs
  main_params_len = len(ctx.fn_info["main"].params) + int(ctx.fn_info["main"].returns_to_address)
  initblocks.add(sb3.ProcedureCall("main", [sb3.Known("")] * main_params_len))

  # Add init code
  ctx.proj.code.append(initblocks)

  # Optimise scratch project
  if cfg.opti: ctx.proj = optimizer.optimize(ctx.proj)

  return ctx.proj, debug_info
