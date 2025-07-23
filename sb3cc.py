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
  file: str = "main.c" # C file to compile

cfg = Config() # TODO maybe not a global lol

@dataclass
class SBlocks:
  """Blocks that can be placed between other blocks unless end is True"""
  blocks: list[str] = field(default_factory=list)
  end: bool = False # If no other blocks can be placed after

  def add(self, other: str | SBlocks) -> None:
    if self.end:
      raise CompException(f"Cannot add blocks \n\n{self}\n\nand\n\n{other}\n\n together because the first is an ending block")
    match other:
      case str():
        self.blocks.append(other)
      case SBlocks():
        self.blocks += other.blocks
        self.end |= other.end
  
  def __str__(self) -> str:
    return "\n".join(self.blocks)

@dataclass
class SValue:
  """A value block (operator, variable, boolean, etc)"""
  val: str

  def __str__(self) -> str:
    return self.val
  
@dataclass
class SIndexableValue:
  """A collection of values that can be indexed over (e.g. a string)"""
  vals: list[SValue]

  def __str__(self) -> str:
    return "[" + ", ".join([str(val) for val in self.vals]) + "]"

@dataclass
class SBlocksAndVal:
  """A value block (operator, variable, boolean, etc) and any blocks that come before it needed to get that value"""
  val: SValue
  blocks: SBlocks = field(default_factory=SBlocks)

  @staticmethod
  def new(val: str, blocks: SBlocks | None = None) -> SBlocksAndVal:
    if blocks is None: blocks = SBlocks()
    return SBlocksAndVal(SValue(val), blocks)

@dataclass
class SBlocksAndIVal:
  """An indexable value and any blocks that come before it needed to get that value"""
  val: SIndexableValue
  blocks: SBlocks = field(default_factory=SBlocks)

@dataclass
class Variable:
  """Name of a variable"""
  name: str
  ty: ir.Type

  def as_value(self) -> SValue:
    return SValue(f"`{self.name}`")

class CompException(Exception):
  """Exception in the sb3cc compiler"""
  pass

def makeCharList() -> list[str]:
  """Makes a lookup table from bytes to ascii chars"""
  res = []
  for x in range(256):
    char = chr(x)
    if char.encode("unicode_escape").decode("ascii").startswith("\\") and char != "\\":
      res.append(f"\\{x:02X}")
    else:
      res.append(char)
  return res

def getStackByteSize(ty: ir.Type) -> int:
  match ty:
    case ir.IntegerType():
      # Scratch's fp variables can store >48 bits per variable accurately
      return math.ceil(ty.num_bits / 48)
    case ir.ArrayType():
      return ty.elem_count * getStackByteSize(ty.elem_ty)
    case _:
      raise CompException(f"Unknown Type: {ty} (py type {type(ty)})")

def decodeValue(val: ir.value.Value) -> SBlocksAndVal | SBlocksAndIVal:
  match val.val:
    case str(): # if a variable
      var = decodeVar(val)
      res = "" # this means no value - nothing in the block
      if var is not None:
        res = f"`{var.name}`"
      return SBlocksAndVal.new(res)
    case int():
      # TODO FIX: allow different sizes with val.ty
      return SBlocksAndVal.new(f"{val.val}")
    case bytes():
      if not (isinstance(val.ty, ir.ArrayType) and isinstance(val.ty.elem_ty, ir.IntegerType)):
        raise CompException(f"Expected bytes value {val} to be assigned to ty [N x i8], got, {type(val.ty)}")
      
      if val.ty.elem_ty.num_bits != 8:
        raise CompException(f"Cannot assign bytes value {val} a non byte (i8)")
      
      return SBlocksAndIVal(SIndexableValue([SValue(str(int(byte))) for byte in val.val]))
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

def transAlloca(var: Variable | None, ty: ir.Type) -> SBlocks:
  blocks = []
  size = getStackByteSize(ty)

  inc_size_block = f"change `stack size` by {size}"

  if var is not None:
    if cfg.opti and size == 1:
      # Optimisation if only allocating 1 byte
      return SBlocks([
        inc_size_block,
        f"set `{var.name}` to `stack size`"
      ])
    
    blocks.append(f"set `{var.name}` to (`stack size` + 1)")
    
  # TODO OPTI: can skip increasing the size if we know the function does not allocate or call another func
  blocks.append(inc_size_block)
  return SBlocks(blocks)

def transStore(value: SValue | SIndexableValue, address: SValue, ty: ir.Type) -> SBlocks:
  match value:
    case SValue():
      if getStackByteSize(ty) > 1:
        # TODO FIX: allow storing of larger values
        raise CompException("Only 1 scratch byte can be stored per stack value at the moment")
      
      return SBlocks([f"replace item {address.val} of `stack` with {value.val}"])
    case SIndexableValue():
      if not (isinstance(ty, ir.ArrayType) and isinstance(ty.elem_ty, ir.IntegerType)):
        raise CompException(f"Expected stored type {ty} to be [Y x iN]")

      if getStackByteSize(ty.elem_ty) > 1:
        raise CompException("Only 1 scratch byte can be stored per stack value at the moment")

      blocks = SBlocks()
      for (i, ival) in enumerate(value.vals):
        # TODO OPTI: when adding operator blocks, calc at compile time if values are known
        blocks.add(f"replace item ({address.val} + {i}) of `stack` with {ival.val}")
      
      return blocks

def transInstr(instr: ir.Instruction) -> SBlocks:
  blocks = SBlocks()
  match instr:
    case ir.instruction.Alloca(): # Allocate space on the stack
      blocks.add(transAlloca(decodeVar(instr.result), instr.allocated_ty))

    case ir.instruction.Load(): # Copy a value from an address on the stack
      address = decodeValue(instr.address)
      blocks.add(address.blocks)

      var = decodeVar(instr.result)
      if var is not None:
        blocks.add(f"set `{var.name}` to (item {address.val} of `stack`)")

    case ir.instruction.Store(): # Copy a value to an address on the stack
      value = decodeValue(instr.value)
      blocks.add(value.blocks)

      address = decodeValue(instr.address)
      blocks.add(address.blocks)

      if isinstance(address.val, SIndexableValue): raise CompException("Did not expect ptr to be a vector")
      blocks.add(transStore(value.val, address.val, instr.value.ty))
    
    case ir.instruction.Call(): # call a func
      # TODO: fix stack vars being reused (use scratch's 'stack')
      fn_name = instr.callee.val
      call_block = f"call {fn_name}"
      for param in instr.args:
        call_block += f" ({decodeValue(param).val})"
      blocks.add(call_block)
      
      var = decodeVar(instr.result)
      if var is not None:
        blocks.add(f"set `{var.name}` to `return value`")
    
    case ir.instruction.Ret():
      if instr.value is not None:
        value = decodeValue(instr.value)
        blocks.add(f"set `return value` to {value.val}")
      # TODO: OPTI: not needed if last instr
      blocks.add("stop this script")
      # TODO FIX: deallocate on ret instruction (only if function can allocate)
      blocks.end = True

    case ir.instruction.BinOp(): # do something with two vars
      first_val = decodeValue(instr.fst_operand)
      blocks.add(first_val.blocks)

      second_val = decodeValue(instr.snd_operand)
      blocks.add(second_val.blocks)

      res_var = decodeVar(instr.result)
      res_val = None

      match instr.opcode:
        case "add": # add two vars
          if not isinstance(instr.fst_operand.ty, ir.IntegerType):
            raise CompException(f"Instruction {instr} with opcode add only supports integers, got type {type(instr.fst_operand.ty)}")
          
          # TODO FIX: support larger values by using multiple vars and carrying
          if instr.fst_operand.ty.num_bits > 48:
            raise CompException(f"Add currently supports integers with <= 48 bits")
          
          if instr.is_nsw and instr.is_nuw and cfg.opti:
            # If no wrapping behaviour is required then overflowing is ub so can be ignored
            res_val = f"({first_val.val} + {second_val.val})"
          else:
            res_val = f"(({first_val.val} + {second_val.val}) mod {2 ** instr.fst_operand.ty.num_bits})"
        case _:
          raise CompException(f"Unknown BinOp Opcode: {instr.opcode} in {instr}")
      
      if res_var is not None:
        blocks.add(f"set `{res_var.name}` to {res_val}")
      else:
        blocks.add(f"set `unused` to {res_val}")

    case _:
      print("Unknown: {")
      print(instr)
      print(type(instr))
      print("}")
      # TODO FUTURE: Throw CompExecption on unknown instruction
  return blocks

def main():
  # TODO SEC passing raw parmas kinda unsafe, use subprocess
  os.system(f"clang -S -emit-llvm -O{cfg.c_opti} {cfg.file}")

  with open("main.ll", "r") as file:
    ll = file.read()
  
  mod: ir.Module = parse_assembly(ll)
  
  print("LOGS:")

  # Blocks that make up the code to setup the stack at the start of the program
  initsblocks = SBlocks()
  # Blocks that make up main program
  sblocks = SBlocks()

  # Set up stack
  # TODO: not necessary if done beforehand (only need to change stack size)
  initsblocks.add(f"set `stack size` to {cfg.stack_size}")
  initsblocks.add(f"delete all of `stack`")
  initsblocks.add(f"repeat {cfg.stack_size}")
  initsblocks.add(f"  add 0 to `stack`")

  # Set up static values
  for instr in mod.global_vars.values():
    ptr = decodeVar(instr.value)
    if ptr is None:
      raise CompException(f"Expected static var {instr} to be named")
    value = decodeValue(instr.initializer)
    initsblocks.add(value.blocks)

    initsblocks.add(transAlloca(ptr, instr.initializer.ty))
    initsblocks.add(transStore(value.val, ptr.as_value(), instr.initializer.ty))

  for func in mod.funcs.values():
    print("LLVM FUNC:")
    for block in func.blocks.values():
      print("LLVM BLOCK:")
      for instr in block.instrs:
        sblocks.add(transInstr(instr))
  
  print("INIT:")
  for sblock in initsblocks.blocks:
    print(sblock)
  print("MAIN:")
  for sblock in sblocks.blocks:
    print(sblock)

if __name__ == "__main__":
  main()