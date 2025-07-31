"""The LLVM -> Scratch Compiler"""

from __future__ import annotations
from dataclasses import dataclass, is_dataclass, field, fields

from llvm2py import parse_assembly
from llvm2py import ir
import math
import os

import sb3

@dataclass
class Config:
  """Config options to pass to the sb3cc compiler"""
  c_opti: str = "0" # Level of optimisation to compile with, passes -O<value> to clang
  opti: bool = True # If optimisations for scratch should be applied
  stack_size: int = 512 # Amount of 'bytes' on 'stack' list (one byte is 48 bits), max 200,000
  file: str = "input/main.c" # C file to compile
  unused_var = "!unused" # Name of the scratch variable for unused values
  return_var = "!return value" # Name of the scratch variable for returing values
  stack_list_var = "!stack" # Name of the scratch list for the stack list
  stack_size_var = "!stack size" # Name of the scratch variable for the stack size
  tmp_prefix = "!temp " # Name of temp variables before a number is added to them

cfg = Config() # TODO maybe not a global lol, add cfg: Config to Context
highest_tmp = 0

@dataclass
class Context:
  """Global context access when translating instructions"""
  globvar_to_ptr: dict[str, sb3.Known] = field(default_factory=dict)

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
  ty: ir.Type

class CompException(Exception):
  """Exception in the sb3cc compiler"""
  pass

def astuple(obj):
  """Same as dataclasses.astuple but not recursive because thats just annoying"""
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

def decodeValue(ctx: Context, val: ir.value.Value) -> ValueAndBlocks | IValueAndBlocks:
  match val.val:
    case str(): # if a variable
      var = decodeVar(val)
      res = sb3.Known("") # this means no value - nothing in the block
      if var is not None:
        res = sb3.GetVar(var.name)
        if cfg.opti and var.name in ctx.globvar_to_ptr:
          res = ctx.globvar_to_ptr[var.name]
      return ValueAndBlocks(res)
    case int():
      # TODO FIX: allow different sizes with val.ty
      return ValueAndBlocks(sb3.Known(val.val))
    case bytes():
      if not (isinstance(val.ty, ir.ArrayType) and isinstance(val.ty.elem_ty, ir.IntegerType)):
        raise CompException(f"Expected bytes value {val} to be assigned to ty [N x i8], got, {type(val.ty)}")
      
      if val.ty.elem_ty.num_bits != 8:
        raise CompException(f"Cannot assign bytes value {val} a non byte (i8)")
      
      return IValueAndBlocks(
        IndexableValue([sb3.Known(str(int(byte))) for byte in val.val]))
    case _:
      raise CompException(f"Unknown Value: {val.val} (py type: {type(val.val)})")

def decodeVar(var: ir.value.Value) -> Variable | None:
  """Used for getting the assigned variable of an instruction"""
  if not isinstance(var.val, str):
    raise CompException(f"Expected val to be a variable, got Value: {var.val} (py type: {type(var.val)})")
  if var.val == "<badref>":
    return None
  return Variable(decodeVarName(var.val), var.ty)

def decodeVarName(name: str) -> str:
  if not name.startswith("%"):
    return "@" + name
  return name

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

def multiplyNoWrap(width: int, left: sb3.Value, right: sb3.Value) -> sb3.Value:
  if width > 48:
    raise CompException(f"Multipling {width} bits is not supported") # TODO FIX
  
  return sb3.Op("mul", left, right) # Overflow is UB - we don't care if
                                    # the number overflows and gets innacurate

def multiplyWrap(width: int, left: sb3.Value, right: sb3.Value) -> ValueAndBlocks:
  # TODO OPTI: if one value is a known value, wrapping behaviour could be simpilifed and
  # known info could be propagated
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
  blocks = sb3.BlockList()
  size = getByteSize(ty)

  if var is not None:
    blocks.add(sb3.EditVar("set", var.name, sb3.GetVar(cfg.stack_size_var)))
    
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

def transInstr(ctx: Context, instr: ir.Instruction) -> sb3.BlockList:
  blocks = sb3.BlockList()
  match instr:
    case ir.instruction.Alloca(): # Allocate space on the stack
      blocks.add(transAlloca(decodeVar(instr.result), instr.allocated_ty))

    case ir.instruction.Load(): # Copy a value from an address on the stack
      address = decodeValue(ctx, instr.address)
      blocks.add(address.blocks)

      if isinstance(address.value, IndexableValue):
        raise CompException(f"Address to load cannot be an indexable value in {instr}")

      var = decodeVar(instr.result)
      if var is not None:
        blocks.add(sb3.EditVar("set", var.name, 
                                      sb3.GetOfList("atindex", cfg.stack_list_var, address.value)))

    case ir.instruction.Store(): # Copy a value to an address on the stack
      value = decodeValue(ctx, instr.value)
      blocks.add(value.blocks)

      address = decodeValue(ctx, instr.address)
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
        value = decodeValue(ctx, arg)
        blocks.add(value.blocks)
        values.append(value.value)
      
      blocks.add(sb3.ProdcedureCall(fn_name, values))
      
      var = decodeVar(instr.result)
      if var is not None:
        blocks.add(sb3.EditVar("set", var.name, sb3.GetVar(cfg.return_var)))
    
    case ir.instruction.Ret():
      if instr.value is not None:
        value = decodeValue(ctx, instr.value)
        if isinstance(value.value, IndexableValue):
          raise CompException(f"Returning multiple values not supported in {instr}")
        blocks.add(value.blocks)
        blocks.add(sb3.EditVar("set", cfg.return_var, value.value))
      # TODO: OPTI: not needed if last instr
      blocks.add(sb3.StopScript("stopthis"))
      # TODO FIX: deallocate on ret instruction (only if function can allocate)
      blocks.end = True

    case ir.instruction.BinOp(): # do something with two vars
      first_val = decodeValue(ctx, instr.fst_operand)
      blocks.add(first_val.blocks)
      first_val = first_val.value

      second_val = decodeValue(ctx, instr.snd_operand)
      blocks.add(second_val.blocks)
      second_val = second_val.value

      res_var = decodeVar(instr.result)
      res_val = None

      if isinstance(first_val, IndexableValue) or \
         isinstance(second_val, IndexableValue):
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
            res_val = sb3.Op(instr.opcode, first_val, second_val)
          else:
            res_val = sb3.Op("mod", sb3.Op(instr.opcode, first_val, second_val),
                             sb3.Known(2 ** width))
        
        case "mul":
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          width = instr.fst_operand.ty.num_bits
          
          if instr.is_nsw and instr.is_nuw and cfg.opti:
            res_val = multiplyNoWrap(width, first_val, second_val)
          else:
            blocks_and_value = multiplyWrap(width, first_val, second_val)
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
            res_val = sb3.Op("floor", sb3.Op("div", first_val, second_val))
          else:
            res_val = sb3.Op("div", first_val, second_val) # Value is poison if one is not a multiple of another

        case "sdiv":
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          width = instr.fst_operand.ty.num_bits

          # TODO FIX: support larger values
          if width > 48:
            raise CompException(f"Instruction {instr} currently supports integers with <= 48 bits")
          
          if instr.is_exact:
            signed_left, lblocks = astuple(undoTwosComplement(width, first_val, False))
            signed_right, rblocks = astuple(undoTwosComplement(width, second_val, False))
            blocks.add(lblocks)
            blocks.add(rblocks)

            res_val = twosComplement(width, sb3.Op("div", signed_left, signed_right))
          else:
            left, lblocks = astuple(optimiseValueUse(first_val))
            right, rblocks = astuple(optimiseValueUse(second_val))
            blocks.add(lblocks)
            blocks.add(rblocks)

            point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
            change = 2 ** width

            if res_var is not None:
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

        case _:
          raise CompException(f"Unknown BinOp Opcode: {instr.opcode} in {instr}")

      if res_val is not False: # If the binop sets res_var itself
        if res_var is not None:
          blocks.add(sb3.EditVar("set", res_var.name, res_val))
        else:
          if not cfg.opti: # Values have no effect on state in scratch, only state has effect on values
            blocks.add(sb3.EditVar("set", cfg.unused_var, res_val))

    case _:
      raise CompException(f"Unknown instruction opcode {instr} (type {type(instr)})")
  return blocks

def main():
  # TODO SEC passing raw parmas kinda unsafe, use subprocess
  os.system(f"clang -S -emit-llvm -O{cfg.c_opti} {cfg.file}")

  with open("main.ll", "r") as file:
    ll = file.read()
  
  mod: ir.Module = parse_assembly(ll)

  ctx = Context()

  # Blocks that make up the code to setup the stack at the start of the program
  initsblocks = sb3.BlockList([
    sb3.OnStartFlag()
  ])

  funcs: dict[str, tuple[int, sb3.BlockList]] = {}
  lists: dict[str, list[sb3.Known]] = {}

  # Set up stack
  # TODO: not necessary if done beforehand (only need to change stack size)
  initsblocks.add(sb3.EditList("deleteall", cfg.stack_list_var, None, None))

  ptr = 1

  # Set up static values
  for instr in mod.global_vars.values():
    # TODO OPTI: Don't use allocate and store and set to the var, instead just remember the
    # ptr value in the decoder and replace any uses of the variable with it (only when opti=true)
    globvar = decodeVar(instr.value)
    if globvar is None:
      raise CompException(f"Expected static var {instr} to be named")
    
    blocks_and_value = decodeValue(ctx, instr.initializer)
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

    for block in func.blocks.values():
      print("LLVM BLOCK start")
      for instr in block.instrs:
        funcblocks.add(transInstr(ctx, instr))
    
    funcs[name] = (len(param_names), funcblocks)

  ascii_lookup = []
  for x in range(1, 256): # Ignore zero; improves perf as scratch lists are 1 indexed and zero signifies end of string
    char = chr(x)
    if char.encode("unicode_escape").decode("ascii").startswith("\\") and char != "\\":
      ascii_lookup.append(sb3.Known(f"\\{x:02X}"))
    else:
      ascii_lookup.append(sb3.Known(char))
  lists["ASCII Lookup (0 Indexed)"] = ascii_lookup

  funcs["puts"] = (1, sb3.BlockList([
    sb3.ProcedureDef("puts", ["input"]),
    sb3.EditVar("set", "buffer", sb3.Known("")),
    sb3.EditVar("set", "ptr", sb3.GetParameter("input")),
    sb3.EditVar("set", "char", sb3.GetOfList("atindex", cfg.stack_list_var, sb3.GetVar("ptr"))),
    sb3.ControlFlow("until", sb3.BoolOp("=", sb3.GetVar("char"), sb3.Known(0)), sb3.BlockList([
      sb3.EditVar("set", "buffer",
        sb3.Op("join", sb3.GetVar("buffer"), sb3.GetOfList("atindex", "ASCII Lookup (0 Indexed)", sb3.GetVar("char")))),
      sb3.EditVar("change", "ptr", sb3.Known(1)),
      sb3.EditVar("set", "char", sb3.GetOfList("atindex", cfg.stack_list_var, sb3.GetVar("ptr"))),
    ])),
    sb3.Say(sb3.GetVar("buffer")),
    sb3.EditVar("set", cfg.return_var, sb3.Known(0)),
  ]))

  if not "main" in funcs:
    raise CompException("No main function") # TODO FIX: allow libs
  initsblocks.add(sb3.ProdcedureCall("main", [sb3.Known("")] * funcs["main"][0]))

  sctx = sb3.ScratchContext()
  sctx.addBlockList(initsblocks)
  for name, scratchlist in lists.items():
    sctx.addOrGetList(name, scratchlist)
  for _, (_, func) in funcs.items():
    sctx.addBlockList(func)

  sb3.exportSpriteFile(sctx, "out.sprite3")

if __name__ == "__main__":
  main()