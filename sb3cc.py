"""The LLVM -> Scratch Compiler"""

from __future__ import annotations
from dataclasses import dataclass, is_dataclass, field, fields
from typing import Literal

from llvm2py import parse_assembly
from llvm2py import ir
import math
import os

import sb3

@dataclass
class Config:
  """Config options to pass to the sb3cc compiler"""
  file: str = "input/main.c" # C file to compile
  c_opti: str = "0" # Level of optimisation to compile with, passes -O<value> to clang
  
  opti: bool = True # If optimisations for scratch should be applied
  invis_blocks: bool = False # Prevent scratch editor from rendering blocks; reduces lag
  stack_size: int = 512 # Amount of 'bytes' on 'stack' list (one byte is 48 bits), max 200,000
  binop_lookup_bits: int = 8 # Amount of bits to use for AND/OR/XOR tables, creates (2**(2*n) elements per table)
  
  unused_var = "!unused" # Name of the scratch variable for unused values
  return_var = "!return value" # Name of the scratch variable for returing values
  stack_list_var = "!stack" # Name of the scratch list for the stack list
  stack_size_var = "!stack size" # Name of the scratch variable for the stack size
  tmp_prefix = "!temp " # Name of temp variables before a number is added to them
  ascii_lookup_var = "!ascii lookup"
  binop_lookup_var = "!binop lookup"
  pow2_lookup_var = "!pow2 lookup"
  zero_indexed_suffix = " (0 indexed)"
  one_indexed_suffix = " (1 indexed)"

cfg = Config() # TODO maybe not a global lol, add cfg: Config to Context
highest_tmp = 0

@dataclass
class Context:
  """Global context access when translating instructions"""
  funcs: dict[str, tuple[int, sb3.BlockList]] = field(default_factory=dict)
  lists: dict[str, list[sb3.Known]] = field(default_factory=dict)
  globvar_to_ptr: dict[str, sb3.Known] = field(default_factory=dict)

@dataclass
class FuncContext:
  """Context about the function which instructions are running in"""
  name: str
  params: list[str]
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
  ty: ir.Type

  def getValue(self) -> sb3.Value:
    if self.is_scratch_param:
      return sb3.GetParameter(self.name)
    return sb3.GetVar(self.name)
  
  def setValue(self, value: sb3.Value, op: Literal["set", "change"]="set") -> sb3.Block:
    if self.is_scratch_param: raise CompException(f"{self.name} param is read only")
    return sb3.EditVar(op, self.name, value)

class CompException(Exception):
  """Exception in the sb3cc compiler"""
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

def decodeValue(val: ir.value.Value, ctx: Context, fctx: FuncContext | None) -> ValueAndBlocks | IValueAndBlocks:
  match val.val:
    case str(): # if a variable
      var = decodeVar(val, fctx)
      res = sb3.Known("") # this means no value - nothing in the block
      if var is not None:
        res = var.getValue()
        if cfg.opti and var.name in ctx.globvar_to_ptr:
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

def decodeVar(var: ir.value.Value, fctx: FuncContext | None) -> Variable | None:
  """Used for getting the assigned variable of an instruction"""
  if not isinstance(var.val, str):
    raise CompException(f"Expected val to be a variable, got Value: {var.val} (py type: {type(var.val)})")
  if var.val == "<badref>":
    return None
  
  name = var.val
  is_param = False
  if not name.startswith("%"):
    name = f"@{name}"
  elif fctx is not None:
    if name in fctx.params and fctx.is_scratch_func:
      is_param = True
    else:
      name = f"%{fctx.name}:{name[1:]}" # Localise variables for functions

  return Variable(name, is_param, var.ty)

def genTempVar() -> str:
  global highest_tmp
  highest_tmp += 1
  return cfg.tmp_prefix + str(highest_tmp)

def shouldOptimiseValueUse(val: sb3.Value) -> bool:
  """Returns if a value that is used more than once should be stored"""
  match val:
    case sb3.Known() | sb3.GetVar() | sb3.GetParameter():
      return False
    case sb3.GetOfList(op="atindex"):
      return not isinstance(val.value, sb3.Known)
    case _:
      return True

def optimiseValueUse(val: sb3.Value) -> ValueAndBlocks:
  if shouldOptimiseValueUse(val):
    tmp = genTempVar()
    return ValueAndBlocks(sb3.GetVar(tmp), sb3.BlockList([sb3.EditVar("set", tmp, val)]))
  
  return ValueAndBlocks(val)

def makePow2LookupTable(size: int, is_one_indexed: bool, ctx: Context) -> tuple[str, Context]:
  name = cfg.pow2_lookup_var + (cfg.one_indexed_suffix if is_one_indexed else cfg.zero_indexed_suffix)
  if name not in ctx.lists or size > len(ctx.lists[name]):
    pow2_lookup = []
    for x in range(int(is_one_indexed) + size):
      pow2_lookup.append(sb3.Known(2 ** x))
    ctx.lists[name] = pow2_lookup
  return name, ctx

def twosComplement(width: int, val: sb3.Value) -> sb3.Value:
  return sb3.Op("mod", val, sb3.Known(2 ** width))

def undoTwosComplement(width: int, val: sb3.Value, return_var: bool) -> ValueAndBlocks:
  """Calculates two's compilment on a value. If the returned value is used multiple times, it is
  prefered to be a var, which can be done with return_var=True"""
  limit = int(((2 ** width) / 2) - 1)
  decrease = 2 ** width
  
  if shouldOptimiseValueUse(val) or return_var:
    tmp = genTempVar()
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

def bitShift(direction: Literal["left", "right"], width: int, left: sb3.Value, right: sb3.Value, ctx: Context, can_shift_out=True) -> tuple[sb3.Value, Context]:
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
    raise CompException(f"Multipling {width} bits is not supported") # TODO FIX
  
  return sb3.Op("mul", left, right) # Overflow is UB - we don't care if
                                    # the number overflows and gets innacurate

def multiplyWrap(width: int, left: sb3.Value, right: sb3.Value) -> ValueAndBlocks:
  # TODO OPTI: if one value is a known value, wrapping behaviour could be simpilifed and
  # known info could be propagated
  # TODO OPTI: if multipling by a power of 2, there is no risk that the mantissa cannot store
  # enough to be accurate, since only the exponent changes
  if width <= 26: # Safe: (2**26) ** 2 < 9007199254740991
    return ValueAndBlocks(sb3.Op("mod", sb3.Op("mul", left, right), sb3.Known(2 ** width)))
  elif width <= 36: # Safe: (2**17 * 2**17 + 2**17 * 2**17) * 2**17 + (2**17 + 2**17) < 9007199254740991
    blocks = sb3.BlockList()
    left, lblocks = astuple(optimiseValueUse(left))
    right, rblocks = astuple(optimiseValueUse(right))
    blocks.add(lblocks)
    blocks.add(rblocks)

    # Use some maths to do the calculation (see README for explaination)
    half_width = width // 2
    a0 = genTempVar()
    b0 = genTempVar()
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

def transAlloca(var: Variable | None, ty: ir.Type) -> sb3.BlockList:
  assert var is None or var.is_scratch_param is False

  blocks = sb3.BlockList()
  size = getByteSize(ty)

  if var is not None:
    blocks.add(var.setValue(sb3.GetVar(cfg.stack_size_var)))
    
  # TODO OPTI: can skip increasing the size if we know the function does not allocate or call another func
  blocks.add(sb3.EditVar("change", cfg.stack_size_var, sb3.Known(size)))
  return blocks

def transStore(value: sb3.Value | IndexableValue, address: sb3.Value, ty: ir.Type) -> sb3.BlockList:
  match value:
    case sb3.Value():
      if getByteSize(ty) > 1:
        # TODO FIX: allow storing of larger values
        raise CompException("Only 1 scratch byte can be stored per stack value at the moment")
      
      return sb3.BlockList([sb3.EditList("replaceat", cfg.stack_list_var, address, value)])
    case IndexableValue():
      if not (isinstance(ty, ir.ArrayType) and isinstance(ty.elem_ty, ir.IntegerType)):
        raise CompException(f"Expected stored type {ty} to be [Y x iN]")

      if getByteSize(ty.elem_ty) > 1:
        raise CompException("Only 1 scratch byte can be stored per stack value at the moment")

      blocks = sb3.BlockList()
      for (i, ival) in enumerate(value.vals):
        # TODO OPTI: when adding operator blocks, calc at compile time if values are known
        blocks.add(sb3.EditList("replaceat", cfg.stack_list_var,
                                  sb3.Op("add", address, sb3.Known(i)), ival))
      
      return blocks

def transInstr(instr: ir.Instruction, ctx: Context, fctx: FuncContext) -> tuple[sb3.BlockList, Context]:
  blocks = sb3.BlockList()
  match instr:
    case ir.instruction.Alloca(): # Allocate space on the stack
      blocks.add(transAlloca(decodeVar(instr.result, fctx), instr.allocated_ty))

    case ir.instruction.Load(): # Copy a value from an address on the stack
      address = decodeValue(instr.address, ctx, fctx)
      blocks.add(address.blocks)

      if isinstance(address.value, IndexableValue):
        raise CompException(f"Address to load cannot be an indexable value in {instr}")

      var = decodeVar(instr.result, fctx)
      if var is not None:
        blocks.add(sb3.EditVar("set", var.name, 
                                      sb3.GetOfList("atindex", cfg.stack_list_var, address.value)))

    case ir.instruction.Store(): # Copy a value to an address on the stack
      value = decodeValue(instr.value, ctx, fctx)
      blocks.add(value.blocks)

      address = decodeValue(instr.address, ctx, fctx)
      blocks.add(address.blocks)

      if isinstance(address.value, IndexableValue):
        raise CompException(f"Address to store cannot be an indexable value in {instr}")

      blocks.add(transStore(value.value, address.value, instr.value.ty))
    
    case ir.instruction.Call(): # call a func
      # TODO FIX: stack vars being reused (use scratch's 'stack')
      fn_name = instr.callee.val

      if not isinstance(fn_name, str):
        raise CompException(f"Expected {instr.callee} to be a string")

      # TODO FIX: in llvm getElementPtr is used but ignored in this code
      values = []
      for arg in instr.args:
        value = decodeValue(arg, ctx, fctx)
        blocks.add(value.blocks)
        values.append(value.value)
      
      blocks.add(sb3.ProdcedureCall(fn_name, values))
      
      var = decodeVar(instr.result, fctx)
      if var is not None:
        blocks.add(sb3.EditVar("set", var.name, sb3.GetVar(cfg.return_var)))
    
    case ir.instruction.Ret():
      if instr.value is not None:
        value = decodeValue(instr.value, ctx, fctx)
        if isinstance(value.value, IndexableValue):
          raise CompException(f"Returning multiple values not supported in {instr}")
        blocks.add(value.blocks)
        blocks.add(sb3.EditVar("set", cfg.return_var, value.value))
      # TODO FIX: deallocate on ret instruction (only if function can allocate)
      blocks.end = True

    case ir.instruction.BinOp(): # do something with two vars
      left = decodeValue(instr.fst_operand, ctx, fctx)
      blocks.add(left.blocks)
      left = left.value

      right = decodeValue(instr.snd_operand, ctx, fctx)
      blocks.add(right.blocks)
      right = right.value

      res_var = decodeVar(instr.result, fctx)
      assert res_var is None or res_var.is_scratch_param is False
      res_val = None

      if isinstance(left, IndexableValue) or \
         isinstance(right, IndexableValue):
        raise CompException(f"Indexable value not supported in binop {instr}")

      match instr.opcode:
        case "add" | "sub": # add/sub two vars
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          
          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values by using multiple vars and carrying
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")
          
          if instr.is_nsw and instr.is_nuw and cfg.opti:
            # If no wrapping behaviour is required then under/overflowing is ub so can be ignored
            res_val = sb3.Op(instr.opcode, left, right)
          else:
            res_val = sb3.Op("mod", sb3.Op(instr.opcode, left, right),
                             sb3.Known(2 ** width))
        
        case "mul":
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          width = instr.fst_operand.ty.num_bits
          
          if instr.is_nsw and instr.is_nuw and cfg.opti:
            res_val = multiplyNoWrap(width, left, right)
          else:
            blocks_and_value = multiplyWrap(width, left, right)
            blocks.add(blocks_and_value.blocks)
            res_val = blocks_and_value.value
        
        case "udiv":
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

        case "sdiv":
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          width = instr.fst_operand.ty.num_bits

          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")
          
          if instr.is_exact:
            signed_left, lblocks = astuple(undoTwosComplement(width, left, False))
            signed_right, rblocks = astuple(undoTwosComplement(width, right, False))
            blocks.add(lblocks)
            blocks.add(rblocks)

            res_val = twosComplement(width, sb3.Op("div", signed_left, signed_right))
          else:
            if res_var is not None:
              left, lblocks = astuple(optimiseValueUse(left))
              right, rblocks = astuple(optimiseValueUse(right))
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
        
        case "urem":
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")
          
          # mod 0 is UB, can ignore
          res_val = sb3.Op("mod", left, right)
        
        case "srem":
          # TODO OPTI: optimise for known values
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          
          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")
          
          if res_var is not None:
            # TODO: Reuse if statement to work out if a / b > 0
            left, lblocks = astuple(optimiseValueUse(left))
            right_is_temp = shouldOptimiseValueUse(right)
            right, rblocks = astuple(optimiseValueUse(right))
            if right_is_temp:
              assert isinstance(right, sb3.GetVar)
            blocks.add(lblocks)
            blocks.add(rblocks)
            
            point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
            change = 2 ** width

            if not right_is_temp:
              right_minus_change = genTempVar()
            # Undo two's complement, calculate modulo, then adjust for differences with llvm's remainder operation
            # (different when one side is negative)
            blocks.add([
              sb3.ControlFlow("if_else", sb3.BoolOp("<", left, sb3.Known(point_of_neg)), sb3.BlockList([
                sb3.ControlFlow("if_else", sb3.BoolOp("<", right, sb3.Known(point_of_neg)), sb3.BlockList([
                  # Modulus and remainder operations do the same
                  sb3.EditVar("set", res_var.name, sb3.Op("mod", left, right)),
                ]), sb3.BlockList([
                  # If left is pos and right is neg - remainder = (l mod r) - r
                  sb3.EditVar("set", right_minus_change, sb3.Op("sub", right, sb3.Known(change))),
                  sb3.EditVar("set", res_var.name, sb3.Op("sub",
                                                    sb3.Op("mod",
                                                      left,
                                                      sb3.GetVar(right_minus_change)),
                                                    sb3.GetVar(right_minus_change))),
                ]) if not right_is_temp else sb3.BlockList([
                  sb3.EditVar("change", right.var_name, sb3.Known(-change)),
                  sb3.EditVar("set", res_var.name, sb3.Op("sub",
                                                    sb3.Op("mod",
                                                      left,
                                                      right),
                                                    right)),
                ]))
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
        
        case "shl":
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          
          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")

          can_shift_out = not (instr.is_nsw and instr.is_nuw)
          res_val, ctx = bitShift("left", width, left, right, ctx, can_shift_out)
        
        case "lshr":
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          
          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")
          
          can_shift_out = not instr.is_exact
          res_val, ctx = bitShift("right", width, left, right, ctx, can_shift_out)
        
        case "ashr":
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
        
        case "and" | "or" | "xor":
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

            if width > cfg.binop_lookup_bits:
              left, lblocks = astuple(optimiseValueUse(left))
              right, rblocks = astuple(optimiseValueUse(right))
              blocks.add(lblocks)
              blocks.add(rblocks)

            lookup_size = 2 ** cfg.binop_lookup_bits
            name = f"{cfg.binop_lookup_var} {instr.opcode}{cfg.zero_indexed_suffix}"
            lookup = []
            for l in range(0, lookup_size):
              for r in range(0, lookup_size):
                lookup.append(sb3.Known({"and": l & r, "or": l | r, "xor": l ^ r}[instr.opcode]))
            ctx.lists[name] = lookup[1:] # since 0 &/|/^ 0 is 0, an empty value being treated as zero is fine

            results = []
            for offset in range(0, width, cfg.binop_lookup_bits):
              left_index = left
              right_index = right
              if offset > 0:
                left_index = sb3.Op("floor", sb3.Op("div", left_index, sb3.Known(2 ** offset)))
                # No floor instruction needed because scratch rounds down with atindex
                right_index = sb3.Op("div", right_index, sb3.Known(2 ** offset))
              if offset + cfg.binop_lookup_bits < width:
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
          if not cfg.opti: # Values have no effect on state in scratch, only state has effect on values
            blocks.add(sb3.EditVar("set", cfg.unused_var, res_val))

    case _:
      raise CompException(f"Unknown instruction opcode {instr} (type {type(instr)})")
  return blocks, ctx

def main():
  global ctx, cfg, highest_tmp
  ctx = Context()
  cfg = Config()
  highest_tmp = 0
  
  # TODO SEC passing raw parmas kinda unsafe, use subprocess
  os.system(f"clang -S -emit-llvm -O{cfg.c_opti} {cfg.file}")

  with open("main.ll", "r") as file:
    ll = file.read()
  
  mod: ir.Module = parse_assembly(ll)

  # Blocks that make up the code to setup the stack at the start of the program
  initsblocks = sb3.BlockList([
    sb3.OnStartFlag()
  ])

  # Set up stack
  # TODO: not necessary if done beforehand (only need to change stack size)
  initsblocks.add(sb3.EditList("deleteall", cfg.stack_list_var, None, None))

  ptr = 1

  # Set up static values
  for instr in mod.global_vars.values():
    # TODO OPTI: Don't use allocate and store and set to the var, instead just remember the
    # ptr value in the decoder and replace any uses of the variable with it (only when opti=true)
    globvar = decodeVar(instr.value, None)
    if globvar is None:
      raise CompException(f"Expected static var {instr} to be named")
    
    blocks_and_value = decodeValue(instr.initializer, ctx, None)
    unknown = len(blocks_and_value.blocks) > 0
    value = blocks_and_value.value

    if isinstance(value, IndexableValue):
      unknown |= not all([isinstance(val, sb3.Known) for val in value.vals])
    else:
      unknown |= not isinstance(value, sb3.Known)

    if unknown: raise CompException(f"Expected static value {instr} to have a compile time known value")

    total_size = getByteSize(instr.initializer.ty)
    values = []
    match value:
      case sb3.Known():
        size = total_size
        values.append(value)
      case IndexableValue():
        assert isinstance(instr.initializer.ty, ir.ArrayType)
        size = getByteSize(instr.initializer.ty.elem_ty)
        values.extend(value.vals)
    
    if size != 1: raise CompException("Cannot create value or values with a size in bytes > 1")

    ctx.globvar_to_ptr[globvar.name] = sb3.Known(ptr)
    if not cfg.opti:
      initsblocks.add(sb3.EditVar("set", globvar.name, sb3.Known(ptr)))
    for value in values:
      initsblocks.add(sb3.EditList("addto", cfg.stack_list_var, None, value))
    ptr += total_size

  initsblocks.add(sb3.BlockList([
    sb3.EditVar("set", cfg.stack_size_var, sb3.Known(ptr)),
    sb3.ControlFlow("reptimes", sb3.Known(cfg.stack_size - (ptr - 1)), sb3.BlockList([
      sb3.EditList("addto", cfg.stack_list_var, None, sb3.Known(0))
    ])),
  ]))

  for func in mod.funcs.values():
    assert isinstance(func.value.ty, ir.FunctionType)
    name = func.value.val
    assert isinstance(name, str)

    for param_ty in func.value.ty.param_tys:
      if getByteSize(param_ty) > 1:
        raise CompException("Parameters can only be one scratch byte in size") # TODO FIX
    
    param_names = []
    for arg in func.args:
      assert isinstance(arg.val, str)
      param_names.append(arg.val)

    print(f"LLVM FUNC {name} start")
    funcblocks = sb3.BlockList([
      sb3.ProcedureDef(name, param_names)
    ])

    if len(func.blocks) > 1:
      raise CompException("Cannot support more than 1 LLVM blocks") # TODO FIX

    fctx = FuncContext(name, param_names, True)
    for block in func.blocks.values():
      print("LLVM BLOCK start")
      for instr in block.instrs:
        blocks, ctx = transInstr(instr, ctx, fctx)
        funcblocks.add(blocks)
    
    ctx.funcs[name] = (len(param_names), funcblocks)

  ascii_lookup = []
  for x in range(1, 256): # Ignore zero; improves perf as scratch lists are 1 indexed and zero signifies end of string
    char = chr(x)
    if char.encode("unicode_escape").decode("ascii").startswith("\\") and char != "\\":
      ascii_lookup.append(sb3.Known(f"\\{x:02X}"))
    else:
      ascii_lookup.append(sb3.Known(char))
  ctx.lists[cfg.ascii_lookup_var + cfg.zero_indexed_suffix] = ascii_lookup

  ctx.funcs["puts"] = (1, sb3.BlockList([
    sb3.ProcedureDef("puts", ["input"]),
    sb3.EditVar("set", "buffer", sb3.Known("")),
    sb3.EditVar("set", "ptr", sb3.GetParameter("input")),
    sb3.EditVar("set", "char", sb3.GetOfList("atindex", cfg.stack_list_var, sb3.GetVar("ptr"))),
    sb3.ControlFlow("until", sb3.BoolOp("=", sb3.GetVar("char"), sb3.Known(0)), sb3.BlockList([
      sb3.EditVar("set", "buffer",
        sb3.Op("join", sb3.GetVar("buffer"), sb3.GetOfList("atindex", (cfg.ascii_lookup_var + cfg.zero_indexed_suffix), sb3.GetVar("char")))),
      sb3.EditVar("change", "ptr", sb3.Known(1)),
      sb3.EditVar("set", "char", sb3.GetOfList("atindex", cfg.stack_list_var, sb3.GetVar("ptr"))),
    ])),
    sb3.Say(sb3.GetVar("buffer")),
    sb3.EditVar("set", cfg.return_var, sb3.Known(0)),
  ]))

  if not "main" in ctx.funcs:
    raise CompException("No main function") # TODO FIX: allow libs
  initsblocks.add(sb3.ProdcedureCall("main", [sb3.Known("")] * ctx.funcs["main"][0]))

  sctx = sb3.ScratchContext(sb3.ScratchConfig(cfg.invis_blocks))
  sctx.addBlockList(initsblocks)
  for name, scratchlist in ctx.lists.items():
    sctx.addOrGetList(name, scratchlist)
  for _, (_, func) in ctx.funcs.items():
    sctx.addBlockList(func)

  sb3.exportSpriteFile(sctx, "out.sprite3")

if __name__ == "__main__":
  main()