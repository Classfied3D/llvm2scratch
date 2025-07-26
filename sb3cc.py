from __future__ import annotations
from dataclasses import dataclass, field

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
  unused_var = "unused" # Name of the scratch variable for unused values
  return_var = "return value" # Name of the scratch variable for returing values
  stack_list_var = "stack" # Name of the scratch list for the stack list
  stack_size_var = "stack size" # Name of the scratch variable for the stack size

cfg = Config() # TODO maybe not a global lol

@dataclass
class BlocksAndValue:
  """A value and any blocks that come before it needed to get that value"""
  value: sb3.Value
  blocks: sb3.BlockList = field(default_factory=sb3.BlockList)

@dataclass
class IndexableValue:
  """A collection of values that can be indexed over (e.g. a string)"""
  vals: list[sb3.Value]

@dataclass
class BlocksAndIValue:
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

def getStackByteSize(ty: ir.Type) -> int:
  match ty:
    case ir.IntegerType():
      # Scratch's fp variables can store >48 bits per variable accurately
      return math.ceil(ty.num_bits / 48)
    case ir.ArrayType():
      return ty.elem_count * getStackByteSize(ty.elem_ty)
    case ir.PtrType():
      return 1
    case _:
      raise CompException(f"Unknown Type: {ty} (py type {type(ty)})")

def decodeValue(val: ir.value.Value) -> BlocksAndValue | BlocksAndIValue:
  match val.val:
    case str(): # if a variable
      var = decodeVar(val)
      res = sb3.KnownValue("") # this means no value - nothing in the block
      if var is not None:
        res = sb3.GetVariable(var.name)
      return BlocksAndValue(res)
    case int():
      # TODO FIX: allow different sizes with val.ty
      return BlocksAndValue(sb3.KnownValue(val.val))
    case bytes():
      if not (isinstance(val.ty, ir.ArrayType) and isinstance(val.ty.elem_ty, ir.IntegerType)):
        raise CompException(f"Expected bytes value {val} to be assigned to ty [N x i8], got, {type(val.ty)}")
      
      if val.ty.elem_ty.num_bits != 8:
        raise CompException(f"Cannot assign bytes value {val} a non byte (i8)")
      
      return BlocksAndIValue(
        IndexableValue([sb3.KnownValue(str(int(byte))) for byte in val.val]))
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

def transAlloca(var: Variable | None, ty: ir.Type) -> sb3.BlockList:
  blocks = sb3.BlockList()
  size = getStackByteSize(ty)

  inc_size_block = sb3.ModifyVariable("change", cfg.stack_size_var, sb3.KnownValue(size))

  if var is not None:
    if cfg.opti and size == 1:
      # Optimisation if only allocating 1 byte
      return sb3.BlockList([
        inc_size_block,
        sb3.ModifyVariable("set", var.name, sb3.GetVariable(cfg.stack_size_var))
      ])
    
    blocks.add(sb3.ModifyVariable("set", var.name, 
      sb3.TwoOp("add", sb3.GetVariable(cfg.stack_size_var), sb3.KnownValue(1))))
    
  # TODO OPTI: can skip increasing the size if we know the function does not allocate or call another func
  blocks.add(inc_size_block)
  return blocks

def transStore(value: sb3.Value | IndexableValue, address: sb3.Value, ty: ir.Type) -> sb3.BlockList:
  match value:
    case sb3.Value():
      if getStackByteSize(ty) > 1:
        # TODO FIX: allow storing of larger values
        raise CompException("Only 1 scratch byte can be stored per stack value at the moment")
      
      return sb3.BlockList([sb3.ModifyList("replaceat", cfg.stack_list_var, address, value)])
    case IndexableValue():
      if not (isinstance(ty, ir.ArrayType) and isinstance(ty.elem_ty, ir.IntegerType)):
        raise CompException(f"Expected stored type {ty} to be [Y x iN]")

      if getStackByteSize(ty.elem_ty) > 1:
        raise CompException("Only 1 scratch byte can be stored per stack value at the moment")

      blocks = sb3.BlockList()
      for (i, ival) in enumerate(value.vals):
        # TODO OPTI: when adding operator blocks, calc at compile time if values are known
        blocks.add(sb3.ModifyList("replaceat", cfg.stack_list_var,
                                  sb3.TwoOp("add", address, sb3.KnownValue(i)), ival))
      
      return blocks

def transInstr(instr: ir.Instruction) -> sb3.BlockList:
  blocks = sb3.BlockList()
  match instr:
    case ir.instruction.Alloca(): # Allocate space on the stack
      blocks.add(transAlloca(decodeVar(instr.result), instr.allocated_ty))

    case ir.instruction.Load(): # Copy a value from an address on the stack
      address = decodeValue(instr.address)
      blocks.add(address.blocks)

      if isinstance(address.value, IndexableValue):
        raise CompException(f"Address to load cannot be an indexable value in {instr}")

      var = decodeVar(instr.result)
      if var is not None:
        blocks.add(sb3.ModifyVariable("set", var.name, 
                                      sb3.GetOfList("atindex", cfg.stack_list_var, address.value)))

    case ir.instruction.Store(): # Copy a value to an address on the stack
      value = decodeValue(instr.value)
      blocks.add(value.blocks)

      address = decodeValue(instr.address)
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
        value = decodeValue(arg)
        blocks.add(value.blocks)
        values.append(value.value)
      
      blocks.add(sb3.ProdcedureCall(fn_name, values))
      
      var = decodeVar(instr.result)
      if var is not None:
        blocks.add(sb3.ModifyVariable("set", var.name, sb3.GetVariable(cfg.return_var)))
    
    case ir.instruction.Ret():
      if instr.value is not None:
        value = decodeValue(instr.value)
        if isinstance(value.value, IndexableValue):
          raise CompException(f"Returning multiple values not supported in {instr}")
        blocks.add(value.blocks)
        blocks.add(sb3.ModifyVariable("set", cfg.return_var, value.value))
      # TODO: OPTI: not needed if last instr
      blocks.add(sb3.StopScript("stopthis"))
      # TODO FIX: deallocate on ret instruction (only if function can allocate)
      blocks.end = True

    case ir.instruction.BinOp(): # do something with two vars
      first_val = decodeValue(instr.fst_operand)
      blocks.add(first_val.blocks)

      second_val = decodeValue(instr.snd_operand)
      blocks.add(second_val.blocks)

      res_var = decodeVar(instr.result)
      res_val = None

      if isinstance(first_val.value, IndexableValue) or \
         isinstance(second_val.value, IndexableValue):
        raise CompException(f"Indexable value not supported in binop {instr}")

      match instr.opcode:
        case "add": # add two vars
          if not isinstance(instr.fst_operand.ty, ir.IntegerType):
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          
          # TODO FIX: support larger values by using multiple vars and carrying
          if instr.fst_operand.ty.num_bits > 48:
            raise CompException(f"Add currently supports integers with <= 48 bits")
          
          if instr.is_nsw and instr.is_nuw and cfg.opti:
            # If no wrapping behaviour is required then overflowing is ub so can be ignored
            res_val = sb3.TwoOp("add", first_val.value, second_val.value)
          else:
            res_val = sb3.TwoOp("mod", sb3.TwoOp("add", first_val.value, second_val.value),
                                sb3.KnownValue(2 ** instr.fst_operand.ty.num_bits))
        case _:
          raise CompException(f"Unknown BinOp Opcode: {instr.opcode} in {instr}")
      
      if res_var is not None:
        blocks.add(sb3.ModifyVariable("set", res_var.name, res_val))
      else:
        if not cfg.opti: # Values have no effect on state in scratch, only state has effect on values
          blocks.add(sb3.ModifyVariable("set", cfg.unused_var, res_val))

    case _:
      raise CompException(f"Unknown instruction opcode {instr} (type {type(instr)})")
  return blocks

def main():
  # TODO SEC passing raw parmas kinda unsafe, use subprocess
  os.system(f"clang -S -emit-llvm -O{cfg.c_opti} {cfg.file}")

  with open("main.ll", "r") as file:
    ll = file.read()
  
  mod: ir.Module = parse_assembly(ll)

  # Blocks that make up the code to setup the stack at the start of the program
  initsblocks = sb3.BlockList([
    sb3.OnStartFlag()
  ])
  funcs: dict[str, tuple[int, sb3.BlockList]] = {}
  lists: dict[str, list[sb3.KnownValue]] = {}

  # Set up stack
  # TODO: not necessary if done beforehand (only need to change stack size)
  initsblocks.add(sb3.BlockList([
    sb3.ModifyVariable("set", cfg.stack_size_var, sb3.KnownValue(0)),
    sb3.ModifyList("deleteall", cfg.stack_list_var, None, None),
    sb3.Repeat("reptimes", sb3.KnownValue(cfg.stack_size), sb3.BlockList([
      sb3.ModifyList("addto", cfg.stack_list_var, None, sb3.KnownValue(0))
    ])),
  ]))

  # Set up static values
  for instr in mod.global_vars.values():
    # TODO OPTI: Don't use allocate and store and set to the var, instead just remember the
    # ptr value in the decoder and replace any uses of the variable with it (only when opti=true)
    ptr = decodeVar(instr.value)
    if ptr is None:
      raise CompException(f"Expected static var {instr} to be named")
    value = decodeValue(instr.initializer)
    initsblocks.add(value.blocks)

    initsblocks.add(transAlloca(ptr, instr.initializer.ty))
    initsblocks.add(transStore(value.value, sb3.GetVariable(ptr.name), instr.initializer.ty))

  for func in mod.funcs.values():
    assert isinstance(func.value.ty, ir.FunctionType)
    name = func.value.val
    assert isinstance(name, str)

    for param_ty in func.value.ty.param_tys:
      if getStackByteSize(param_ty) > 1:
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
        funcblocks.add(transInstr(instr))
    
    funcs[name] = (len(param_names), funcblocks)

  ascii_lookup = []
  for x in range(1, 256): # Ignore zero; improves perf as scratch lists are 1 indexed and zero signifies end of string
    char = chr(x)
    if char.encode("unicode_escape").decode("ascii").startswith("\\") and char != "\\":
      ascii_lookup.append(sb3.KnownValue(f"\\{x:02X}"))
    else:
      ascii_lookup.append(sb3.KnownValue(char))
  lists["ASCII Lookup"] = ascii_lookup

  funcs["puts"] = (1, sb3.BlockList([
    sb3.ProcedureDef("puts", ["input"]),
    sb3.ModifyVariable("set", "buffer", sb3.KnownValue("")),
    sb3.ModifyVariable("set", "ptr", sb3.GetParameter("input")),
    sb3.ModifyVariable("set", "char", sb3.GetOfList("atindex", cfg.stack_list_var, sb3.GetVariable("ptr"))),
    sb3.Repeat("until", sb3.BoolOp("=", sb3.GetVariable("char"), sb3.KnownValue(0)), sb3.BlockList([
      sb3.ModifyVariable("set", "buffer",
        sb3.TwoOp("join", sb3.GetVariable("buffer"), sb3.GetOfList("atindex", "ASCII Lookup", sb3.GetVariable("char")))),
      sb3.ModifyVariable("change", "ptr", sb3.KnownValue(1)),
      sb3.ModifyVariable("set", "char", sb3.GetOfList("atindex", cfg.stack_list_var, sb3.GetVariable("ptr"))),
    ])),
    sb3.Say(sb3.GetVariable("buffer")),
    sb3.ModifyVariable("set", cfg.return_var, sb3.KnownValue(0)),
  ]))

  if not "main" in funcs:
    raise CompException("No main function") # TODO FIX: allow libs
  initsblocks.add(sb3.ProdcedureCall("main", [sb3.KnownValue("")] * funcs["main"][0]))

  ctx = sb3.ScratchContext()
  ctx.addBlockList(initsblocks)
  for name, scratchlist in lists.items():
    ctx.addOrGetList(name, scratchlist)
  for _, (_, func) in funcs.items():
    ctx.addBlockList(func)
  sb3.exportSpriteFile(ctx, "out.sprite3")

if __name__ == "__main__":
  main()