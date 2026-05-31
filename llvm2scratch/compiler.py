"""LLVM -> Scratch Compiler"""

from __future__ import annotations
from dataclasses import dataclass, is_dataclass, field, fields
from collections import defaultdict
from ordered_set import OrderedSet
from typing import Literal, cast
from copy import deepcopy

import warnings
import random
import math

from . import graph_util as util
from . import scratch as sb3
from . import optimizer as opt
from . import target
from . import parser
from . import ir

INTERMEDIATE_MAX_BITS = 53   # Max bits scratch's doubles can store
VARIABLE_MAX_BITS = 48       # Maximum amount of bits to store in a fp variable. Maximum is 48 because while scratch's doubles support
                             # up to 53 bits, some operations require extra precision. This should be more than the regular byte size (8)
                             # and more than PTR_SIZE_BITS (currently 32, see below)
SCRATCH_LIST_LIMIT = 200_000 # Max elements in a scratch list (without importing a larger list or project.json hacks)
PTR_WIDTH_BITS = 32          # Bits per pointer - this is 32 as we are compiling for a 32 bit system
BINOP_LOOKUP_BITS = 8        # Amount of bits to use for AND/OR/XOR tables, creates (2**(2*n) elements per table)
EXIT_CALL_ID = 0             # Jump table call ID of corresponding to exit
ENTRY_CALL_ID = 1            # Jump table call ID of corresponding to entry
START_STACK_RESET_ID = 2     # Starting jump table call ID of stack reset

@dataclass
class DebugInfo:
  """Info about a scratch project which can be used in future compilations for optimization"""
  debug_branch_func_map: str | None = None
  debug_branch_log: str | None = None

@dataclass
class Config:
  """Config options to pass to the compiler"""
  targets: list[target.Target] = field(default_factory=lambda: [target.getTarget(t) for t in target.DEFAULT_TARGETS])

  compiler_opt: bool = True # If compiler should be applied
  compiler_minify: bool = True # If the compiler should minify expressions (e.g. sub 1 instead of adding 2^N-1)
  opt_passes: set[opt.Optimization] = field(default_factory=lambda: opt.ALL_OPTIMIZATIONS) # Set of opt passes to apply
  opt_target: target.Target = field(default_factory=lambda: target.getTarget(target.DEFAULT_OPT_TARGET))

  memory_size: int = 4096 # Number of 'bytes' on 'memory' list; max value is 200,000
  local_stack_size: int = 512 # Number of 'bytes' on local stack list for storing registers when recursing; max value is 200,000
  use_branch_jump_table: bool = False # If branching should be done via jump tables in each function
  max_branch_recursion: int = 2000 # Maximum depth of scratch's call stack before resetting it
  # If extra padding bytes should be added to each value in memory so that it takes up the
  # space it would normally in bytes. This allows byte indexing to be more accurate at the
  # cost of requiring ~3x more space in the memory list. Disabling this may break programs
  # that rely on an 8-bit byte size, like memcpy on an array of i32s or optimized IR
  accurate_byte_spacing: bool = True
  entrypoint: str = "main" # Name of function to call to start the program
  gen_lut_runtime: bool = False # If AND/OR/XOR lookup tables should be generated at runtime. This reduces resultant file size
                                # significantly at the cost of ~0.4s of time on first start spent generating them
  scratch_config: sb3.ScratchConfig = field(default_factory=sb3.ScratchConfig) # Config options for the scratch serializer
  do_debug_branch_log: bool = False # If the times a function recurses should be logged
  # Any functions which might call a non-existent fn ptr in an unreachable branch.
  # This defaults to a few clib funcs with this property to avoid warnings
  no_warn_missing_fn_sig: set[str] = field(default_factory=lambda: {"exit", "__call_exitprocs"})

  return_var = "!return value" # Variable for returing values
  mem_var = "!mem" # List for memory
  init_mem_var = "!mem init" # List to store initial memory
  stack_pointer_var = "!stack pointer" # Variable for the stack pointer
  heap_pointer_var = "!heap pointer" # Variable for the heap pointer
  local_stack_var = "!local stack" # List to store locals to for later (i.e. when recursing)
  local_stack_size_var = "!local stack size" # Variable to store the label stack's size
  jump_table_id_var = "!call stack reset id" # Variable to store the ID of the current branch on a stack reset
  debug_branch_log_var = "!!debug_branch_log" # list to store debug info (using underscores
                                              # to avoid spaces in exported filename for convenience)

  ascii_lookup_var = "!ASCII lookup"
  pow2_lookup_var = "!POW2 lookup"
  lowercase_var = "!lowercase"

  return_address_local = "return address" # local variable or parameter to the id of func the return to
  vararg_ptr_local = "vararg ptr" # local variable or parameter to the pointer to varargs
  previous_stack_size_local = "prev stack size" # Local variable to store the previous stack size
  branch_jump_table_addr_local = "branch jump table addr" # Local variable to store current branch jump table location to
  special_locals = {return_address_local, vararg_ptr_local, previous_stack_size_local, branch_jump_table_addr_local} # All special local vars

  func_ptr_parameter = "func ptr addr" # Name of the parameter to pass func ptr addr to

  tmp_prefix = "%!tmp:" # Name of temp variables before a number is added to them
  zero_indexed_suffix = " (0 indexed)"
  one_indexed_suffix = " (1 indexed)"

@dataclass
class Context:
  """Global context access when translating instructions"""
  proj: sb3.Project
  cfg: Config
  fn_info: dict[str, FuncInfo] = field(default_factory=dict)
  fn_ptr_sig_info: list[FuncPtrSigInfo] = field(default_factory=list)
  fn_ptr_sigs: list[tuple[ir.FuncTy, list[str]]] = field(default_factory=list)
  globvar_to_ptr: dict[str, int] = field(default_factory=dict)
  highest_return_size: int | None = None
  next_fn_id: int = 0
  # Starting ptr addr for functions. Could be independent from stack
  # using LLVM datalayout
  min_func_ptr_addr: int = 0
  # [(func_name, branch_name)]. List of all locations which contain a stack check
  all_check_locations: list[tuple[str, str]] = field(default_factory=list)
  # Get which lookup tables are needed
  needs_and_lut: bool = False
  needs_or_lut: bool = False
  needs_xor_lut: bool = False
  # Info about a scratch project which can be used in future compilations for optimization
  debug_info: DebugInfo = field(default_factory=DebugInfo)

@dataclass
class FuncPtrSigInfo:
  # ID for the signature of the function pointer when being called
  signature_id: int
  # Descriptions for following are in FuncInfo
  can_call: set[str]
  value_param_count: int
  is_variadic: bool
  return_addresses: list[str]
  returns_to_address: bool
  takes_return_address: bool
  could_recurse: bool

@dataclass
class FuncInfo:
  """Info about a LLVM function"""
  name: str
  fn_id: int
  # The parameters the function takes (doesn't include return address)
  params: list[Variable]
  # The types of the parameters
  param_sizes: list[int]
  # The amount of parameters which are ir values the function accepts (i.e. excluding
  # return address and vararg pointer)
  value_param_count: int
  # If the function is variadic
  is_variadic: bool = False
  # Everything the function might call (may include itself)
  can_call: set[str] = field(default_factory=set)
  # Any functions that call this function
  return_addresses: list[str] = field(default_factory=list)
  # If the function returns using an id to an address
  returns_to_address: bool = False
  # If the function takes a return address as a parameter. This can be false while
  # returns_to_address is true if this function only can return to one possible "address"
  takes_return_address: bool = False
  # List of label names that will contain a check to reset the stack
  checked_blocks: list[str] = field(default_factory=list)
  # Amount allocated per branch
  block_alloca_size: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
  # Amount allocated total. None if this amount is not known
  total_alloca_size: int | None = 0
  # Whether to skip increasing the stack size because other functions don't rely on it
  skip_stack_size_change: bool = False
  # What a block depends on and modifies
  block_var_use: dict[str, BlockVarUse] = field(default_factory=dict)
  # If any branches in the function can go to the first block
  branches_to_first: bool = False
  # Info about phi instructions in each branch
  phi_info: defaultdict[str, defaultdict[str, list[tuple[Variable, ir.Value]]]] = \
    field(default_factory=lambda: defaultdict(lambda: defaultdict(lambda: list())))

@dataclass
class BlockInfo:
  """Info about a LLVM block"""
  fn: FuncInfo
  available_params: list[Variable] # All params that can be accessed from the function
  available_param_sizes: list[int] # All sizes of above params
  code: sb3.BlockList = field(default_factory=sb3.BlockList) # The current code instructions are being added to
  label: str | None = None # Name/Label of the block
  allocated: int = 0 # Out of the amount allocated for the branch beforehand, how much has been translated into addresses
  next_call_id: int = 0 # The id to give to the nth function call

@dataclass
class BlockVarUse:
  depends: set[str] = field(default_factory=set)
  modifies: set[str] = field(default_factory=set)
  branches: set[str] = field(default_factory=set)
  depends_var_sizes: dict[str, int] = field(default_factory=dict)

@dataclass
class ValueAndBlocks:
  """A value and any blocks that come before it needed to get that value"""
  value: sb3.Value
  blocks: sb3.BlockList = field(default_factory=sb3.BlockList)

@dataclass
class IdxbleValue:
  """A collection of values that can be indexed over (e.g. a string)"""
  vals: list[sb3.Value] = field(default_factory=list)

  def stringify(self, sb: bool=False):
    """Convert to readable text. If "sb" is True then output text compatible with scratchblocks"""
    return str([v.stringify(sb) for v in self.vals])

@dataclass
class IdxbleValueAndBlocks:
  """An indexable value and any blocks that come before it needed to get that value"""
  value: IdxbleValue = field(default_factory=IdxbleValue)
  blocks: sb3.BlockList = field(default_factory=sb3.BlockList)

@dataclass
class Variable:
  var_name: str
  var_type: Literal["global", "param", "var", "special_var"]
  fn_name: str | None

  def getUnidxedRawVarName(self) -> str:
    match self.var_type:
      case "global":
        return f"@{self.var_name}"
      case "param":
        return localizeParam(self.var_name)
      case "var":
        assert self.fn_name is not None
        return f"%{self.fn_name}:{self.var_name}" # Localize variables per function
      case "special_var":
        return f"{self.var_name}"
      case _:
        raise CompException("Unmatched")

  def getRawVarName(self, index: int | None = None) -> str:
    unidxed = self.getUnidxedRawVarName()
    if index is None: return unidxed
    return f"{unidxed}:{index}"

  def getValue(self, index: int | None = None) -> sb3.Value:
    name = self.getRawVarName(index)
    if self.var_type == "param":
      return sb3.GetParam(name)
    return sb3.GetVar(name)

  def getAllValues(self, value_len: int) -> IdxbleValue:
    assert value_len > 1
    values = []
    for i in range(value_len):
      values.append(self.getValue(i))
    return IdxbleValue(values)

  def setValue(self, value: sb3.Value, op: Literal["set", "change"]="set", index: int | None = None) -> sb3.Block:
    if self.var_type == "param": raise CompException(f"{self.var_name} param is read only")
    return sb3.EditVar(op, self.getRawVarName(index), value)

  def setAllValues(self, values: IdxbleValue) -> sb3.BlockList:
    assert len(values.vals) > 1
    blocks = sb3.BlockList()
    for i, val in enumerate(values.vals):
      blocks.add(self.setValue(val, index=i))
    return blocks

class CompException(Exception):
  """Exception in the compiler"""
  pass

class CompWarning(Warning):
  """Warning in the compiler"""
  pass

def flatAsTuple(obj):
  """Same as dataclasses.astuple but not recursively with dataclasses in dataclasses"""
  if not is_dataclass(obj):
    raise TypeError("Expected dataclass instance")
  return tuple(getattr(obj, f.name) for f in fields(obj))

def getByteSize(ty: ir.Type, include_padding: bool=False) -> int:
  """
  Gets the size in bytes of a type. If include_padding is False, then this will return the amount of
  variables needed to store the type. If it is True, then it will return the size in memory it will take
  up (so that byte offsets created by LLVM are respected).
  """

  match ty:
    case ir.IntegerTy():
      # Scratch's fp variables can store < 52 bits per variable accurately
      # If include_spacing is enabled, then calculate what the size would be in bytes
      return math.ceil(ty.width / (8 if include_padding else VARIABLE_MAX_BITS))

    case ir.FloatingPointTy():
      if not include_padding:
        # All floats are stored with scratch's double precision variables
        return 1

      match ty:
        case ir.HalfTy():   return 2  # 16 bit float; 2 bytes
        case ir.FloatTy():  return 4  # 32 bit float; 4 bytes
        case ir.DoubleTy(): return 8  # 64 bit float; 8 bytes
        case ir.Fp128Ty():  return 16 # 128 bit float; 16 bytes

      raise CompException(f"Unknown floating point type: {ty}")

    case ir.ArrayTy():
      return ty.size * getByteSize(ty.inner, include_padding)

    case ir.StructTy():
      return sum(getByteSize(mem, include_padding) for mem in ty.members)

    case ir.PointerTy():
      return 1 if not include_padding else math.ceil(PTR_WIDTH_BITS / 8)

  raise CompException(f"Unknown type: {ty}")

def getGepOffsets(base_ptr_type: ir.Type, indices: list[tuple[sb3.Value, int]], ctx: Context) -> tuple[int, list[tuple[sb3.Value, int, int]]]:
  known_offset: int = 0
  unknown_offsets: list[tuple[sb3.Value, int, int]] = []

  is_arr_offset = True
  inner_type = base_ptr_type
  for i, (index_val, index_width) in enumerate(indices):
    if is_arr_offset:
      # An array offset
      if i != 0:
        assert isinstance(inner_type, ir.ArrayTy)
        inner_type = inner_type.inner

      # Since GEP interfaces with pointers to memory, it will need to include any padding that is enabled
      offset_size = getByteSize(inner_type, include_padding=ctx.cfg.accurate_byte_spacing)
      if isinstance(index_val, sb3.Known):
        assert isinstance(index_val.known, (int, float)) and index_val.known.is_integer()
        # Account for negative indices
        no_twos_comp_known = comptimeUndoTwosComplement(index_val.known, index_width)
        known_offset += int(no_twos_comp_known) * offset_size
      else:
        unknown_offsets.append((index_val, index_width, offset_size))
    else:
      # A struct offset
      assert isinstance(inner_type, ir.StructTy)
      assert isinstance(index_val, sb3.Known)
      assert isinstance(index_val.known, int) or \
            (isinstance(index_val.known, float) and index_val.known.is_integer())

      members = inner_type.members
      member_offset = int(index_val.known)
      for member in members[:member_offset]:
        # Since GEP interfaces with pointers to memory, it will need to include any padding that is enabled
        known_offset += getByteSize(member, include_padding=ctx.cfg.accurate_byte_spacing)

      inner_type = inner_type.members[member_offset]

    is_arr_offset = isinstance(inner_type, ir.ArrayTy)

  return known_offset, unknown_offsets

def applyGepOffsets(base: sb3.Value, known_offset: int, unknown_offsets: list[tuple[sb3.Value, int, int]], is_nuw: bool, ctx: Context) -> sb3.Value:
  """
  Applies GEP offsets to a base pointer. Accepts unknown_offsets of a list of (index_val, index_width, multiplier) where multiplier
  is the size of the value multiplied by in the GEP.
  """

  final_val = base

  # Don't multiply by one
  multiply_offset = lambda offset, multiplier: sb3.Op("mul", sb3.Known(multiplier), offset) \
                                               if multiplier != 1 else offset

  if is_nuw:
    # No unsigned wrap - we do not need to account for negative indices
    if known_offset != 0:
      final_val = sb3.Op("add", final_val, sb3.Known(known_offset))

    for index_val, _, multiplier in unknown_offsets:
      final_val = sb3.Op("add", final_val, multiply_offset(index_val, multiplier))

  else:
    # We need to account for negative indices
    max_intermediate = 2**INTERMEDIATE_MAX_BITS

    twos_comp_sum: list[tuple[sb3.Value, int, int]] = []
    rev_twos_comp: list[tuple[sb3.Value, int, int]] = []

    for item in unknown_offsets:
      _, index_width, multiplier = item
      if index_width != PTR_WIDTH_BITS or 2**PTR_WIDTH_BITS + multiplier * 2**index_width >= max_intermediate:
        # If the index is too large to multiply by multiplier without unsigned
        # overflow for -ve values OR the index is not 32 bits wide we should
        # reverse two's complement
        rev_twos_comp.append(item)
      else:
        # We can use the fact that multiplication and additions with a 32 bit
        # two's complement number is the same as it's reversed value under mod 2^32
        twos_comp_sum.append(item)

    # Sort from smallest 2^width * multiplier upward, meaning that fewer offset
    # magnitude reductions are required
    sort_func = lambda k: 2**k[1] * k[2]
    twos_comp_sum.sort(key=sort_func)
    rev_twos_comp.sort(key=sort_func)

    cuml_offset = known_offset

    for index_val, index_width, multiplier in rev_twos_comp:
      rev, rev_offset = undoTwosComplementWithOffset(index_val, index_width)
      this_offset = rev_offset * multiplier

      # If the multiplier * min value due to reverse two's comp is too large,
      # we need to add before multipling
      if multiplier * 2**index_width >= max_intermediate:
        rev = sb3.Op("add", rev, sb3.Known(rev_offset))
        this_offset = 0

      # If the offset is too large that it would make the final offset larger
      # than the intermediate value, then add that offset early and bring
      # it back to a reasonable value
      if cuml_offset + this_offset >= max_intermediate:
        final_val = sb3.Op("add", final_val, sb3.Known(cuml_offset))
        cuml_offset = 0

      cuml_offset += this_offset
      final_val = sb3.Op("add", final_val, multiply_offset(rev, multiplier))

    # Add any leftover offset
    if cuml_offset != 0:
      final_val = sb3.Op("add", final_val, sb3.Known(cuml_offset))

    # Sum of index * width should index appropriately into memory, so it is safe
    # to assume it takes a reasonable value with reversed two's complement values
    # This is guaranteed by any inbounds GEP (which is most of them)
    cuml_max_val = (10 * ctx.cfg.memory_size) if len(rev_twos_comp) > 0 else 0
    # If we are adding two's comp numbers, we need to ensure that we use mod at the
    # end in order to wrap it back around
    final_mod_step = len(twos_comp_sum) > 0

    for index_val, index_width, multiplier in twos_comp_sum:
      this_max_val = multiplier * 2**index_width

      # If the value would be too large, then add a mod step earlier to keep it within
      # the intermediate range
      if cuml_max_val + this_max_val >= max_intermediate:
        final_val = sb3.Op("mod", final_val, sb3.Known(2**PTR_WIDTH_BITS))
        this_max_val = 2**PTR_WIDTH_BITS

      cuml_max_val += this_max_val
      final_val = sb3.Op("add", final_val, multiply_offset(index_val, multiplier))

    if final_mod_step:
      # Make the final number in bounds of two's complement
      final_val = sb3.Op("mod", final_val, sb3.Known(2**PTR_WIDTH_BITS))

  return final_val

def padValue(val: ValueAndBlocks | IdxbleValueAndBlocks, size: int) -> ValueAndBlocks | IdxbleValueAndBlocks:
  """
  Apply padding to a Value/IdxbleValue + Blocks so that it matches "size"
  """
  originally_idxable = isinstance(val, IdxbleValueAndBlocks)
  blocks = val.blocks
  values = val.value.vals if isinstance(val, IdxbleValueAndBlocks) else [val.value]
  padding_len = size - len(values)
  assert padding_len >= 0
  values.extend(sb3.Known(0) for _ in range(padding_len))
  assert len(values) == size

  if originally_idxable or size > 1:
    return IdxbleValueAndBlocks(IdxbleValue(values), blocks)
  else:
    return ValueAndBlocks(values[0], blocks)

def transValue(val: ir.Value,
               ctx: Context, bctx: BlockInfo | None,
               is_global_init: bool=False,
               include_padding: bool=False) -> ValueAndBlocks | IdxbleValueAndBlocks:

  """
  Convert an IR value into a scratch value + blocks used to generate it. If is_global_init is
  True, then create a value made up of sb3.Knowns
  """

  if include_padding:
    # We should only be adding padding if we are in a global initializer
    assert is_global_init

  match val:
    case ir.LocalVarVal() | ir.ArgumentVal() | ir.GlobalPtrVal():
      var = transVar(val, bctx)
      res = var.getValue()

      if is_global_init:
        assert isinstance(val, ir.GlobalPtrVal)

      if isinstance(val, ir.GlobalPtrVal) and (ctx.cfg.compiler_opt or is_global_init):
        # Global variables store their address in their variable
        # when optimizations are enabled we use this address directly
        res = sb3.Known(ctx.globvar_to_ptr[val.name])

      size = getByteSize(val.type, include_padding)
      if isinstance(val, (ir.LocalVarVal, ir.ArgumentVal)) and size > 1:
        return IdxbleValueAndBlocks(var.getAllValues(size))

      return padValue(ValueAndBlocks(res), size)

    case ir.KnownIntVal():
      # Calculate the two's complement version of the number
      num = val.value
      width = val.width
      assert num >= 0 # Previously two's complement parsing existed here,
                      # but now the parser should handle it

      if val.width <= VARIABLE_MAX_BITS:
        res = ValueAndBlocks(sb3.Known(num))
      else:
        # Little endian ordering of values
        values: list[sb3.Value] = []
        while width > 0:
          values.append(sb3.Known(num % (2 ** VARIABLE_MAX_BITS)))
          num = num // (2 ** VARIABLE_MAX_BITS)
          width -= VARIABLE_MAX_BITS
        res = IdxbleValueAndBlocks(IdxbleValue(values))

      return padValue(res, getByteSize(val.type, include_padding))

    case ir.KnownFloatVal():
      return padValue(
        ValueAndBlocks(sb3.Known(val.value)),
        getByteSize(val.type, include_padding))

    case ir.KnownArrVal() | ir.KnownStructVal():
      values: list[sb3.Value] = []
      blocks = sb3.BlockList()
      for element in val.values:
        el_val, el_blocks = flatAsTuple(transValue(element, ctx, bctx, is_global_init, include_padding))
        if isinstance(el_val, sb3.Value):
          values.append(el_val)
        else:
          values.extend(el_val.vals)
        blocks.add(el_blocks)

      # Arrays/Structs don't need padding
      return IdxbleValueAndBlocks(IdxbleValue(values), blocks)

    case ir.NullPtrVal():
      # Since pointers start from one anyway (because lists start from one in scratch), zero can be used for null
      return padValue(
        ValueAndBlocks(sb3.Known(0)),
        getByteSize(val.type, include_padding))

    case ir.FunctionVal():
      # Return a pointer corresponding to an id given to each func ptr reference
      return padValue(
        ValueAndBlocks(sb3.Known(getFuncPtrAddr(val.name, ctx))),
        getByteSize(val.type, include_padding))

    case ir.ConstExprVal():
      expr = val.expr
      match expr:
        case ir.GetElementPtr():
          assert isinstance(expr.base_ptr, ir.GlobalPtrVal)

          indices: list[tuple[sb3.Value, int]] = []
          for index_val in expr.indices:
            assert isinstance(index_val, ir.KnownIntVal)
            indices.append((sb3.Known(index_val.value), index_val.width))

          known_offset, unknown_offsets = getGepOffsets(expr.base_ptr_type, indices, ctx)
          assert len(unknown_offsets) == 0

          base_ptr = ctx.globvar_to_ptr[expr.base_ptr.name]

          # Pointer needs padding applied
          return padValue(
            ValueAndBlocks(sb3.Known(base_ptr + known_offset)),
            getByteSize(val.type, include_padding))

        case ir.Conversion():
          match expr.opcode:
            case ir.ConvOpcode.PtrToInt | ir.ConvOpcode.PtrToAddr | ir.ConvOpcode.IntToPtr:
              # TODO: truncation and zero extension if int type is different width to pointer, also
              # make sure to padValue if zero extending
              int_ty = expr.value.type if expr.opcode == ir.ConvOpcode.IntToPtr else expr.res_type
              assert isinstance(int_ty, ir.IntegerTy)
              assert int_ty.width == PTR_WIDTH_BITS

              # No-op
              return transValue(expr.value, ctx, bctx, is_global_init, include_padding)

            case ir.ConvOpcode.BitCast | ir.ConvOpcode.AddrSpaceCast:
              raise CompException(f"Unsupported constexpr conv opcode {expr.opcode}")

            case _:
              raise CompException(f"Conv opcode {expr.opcode} is invalid for a constexpr")

        case _:
          raise CompException(f"Unsupported constant expression type: {expr}")

    case _:
      raise CompException(f"Unknown Value {val}")

def transVar(var: ir.Value | ir.ResultLocalVar | str, bctx: BlockInfo | None) -> Variable:
  """Used for getting the assigned variable of an instruction"""
  match var:
    case str():
      return localizeVar(var, False, bctx)
    case ir.LocalVarVal() | ir.ArgumentVal() | ir.ResultLocalVar():
      return localizeVar(var.name, False, bctx)
    case ir.GlobalPtrVal():
      return localizeVar(var.name, True, bctx)
    case _:
      raise CompException(f"Invalid type for var {var}: {type(var)}")

def localizeVar(name: str, is_global: bool, bctx: BlockInfo | None) -> Variable:
  if is_global:
    var_type = "global"
    fn_name = None
  else:
    assert bctx is not None
    fn_name = bctx.fn.name

    if name in [param.var_name for param in bctx.available_params]:
      var_type = "param"
    else:
      var_type = "var"

  return Variable(name, var_type, fn_name)

def localizeParam(name: str) -> str:
  return "%" + name

def localizeLabel(label: str, fn_name: str) -> str:
  return f"{fn_name}:{label}"

def localizeCallId(call_id: int, label: str, fn_name: str, recursive: bool = False) -> str:
  if not recursive:
    return f"{fn_name}:{label}:return addr {call_id}"
  return f"{fn_name}:{label}:recursive call {call_id}"

def localizeFuncPtrSig(signature_id: int) -> str:
  return f"!fn pointer signature:{signature_id}"

def localizeFuncPtrSigCallback(signature_id: int) -> str:
  return f"{localizeFuncPtrSig(signature_id)}:callback"

def localizeSizedParameters(params: list[Variable], sizes: list[int]) -> list[str]:
  assert len(params) == len(sizes)
  res: list[str] = []
  for param, size in zip(params, sizes):
    for i in range(max(size, 1)):
      res.append(param.getRawVarName(None if size == 1 else i))
  return res

def combineIdxbleValues(vals: list[sb3.Value | IdxbleValue]) -> IdxbleValue:
  res = []
  for val in vals:
    if isinstance(val, sb3.Value): res.append(val)
    else:                          res.extend(val.vals)
  return IdxbleValue(res)

def genTempVar(ctx: Context) -> str:
  return ctx.cfg.tmp_prefix + random.randbytes(12).hex()

def shouldOptimiseValueUse(val: sb3.Value, times_used: float, ctx: Context) -> bool:
  """Returns if a value that is used multiple times should be stored"""
  return not opt.shouldElide(val, times_used, ctx.cfg.opt_target.perf)

def optimizeValueUse(val: sb3.Value, times_used: float, ctx: Context) -> ValueAndBlocks:
  if shouldOptimiseValueUse(val, times_used, ctx):
    tmp = genTempVar(ctx)
    return ValueAndBlocks(sb3.GetVar(tmp), sb3.BlockList([sb3.EditVar("set", tmp, val)]))
  return ValueAndBlocks(val)

def chooseFastestValue(vals: list[sb3.Value], ctx: Context) -> sb3.Value:
  return min(vals, key=lambda val: opt.getValueCost(val, ctx.cfg.opt_target.perf))

def scratchRuntimeError(message: str) -> sb3.BlockList:
  # TODO: this would be logged to stdout
  return sb3.BlockList([
    sb3.ControlFlow("until", sb3.BoolOp("=", sb3.GetAnswer(), sb3.Known("ignore")), sb3.BlockList([
      sb3.Ask(sb3.Known(f"L2S ERROR: {message}")),
    ]))
  ])

def getPow2Offset() -> int:
  # To calculate 2^x with x = -VARIABLE_MAX_BITS corresponds to the 1st item in the list (index of 1)
  # Then f(x) = 1 => -VARIABLE_MAX_BITS + offset = 1 => offset = VARIABLE_MAX_BITS + 1
  return VARIABLE_MAX_BITS + 1

def makePow2LookupTable(ctx: Context) -> tuple[str, int, Context]:
  """
  Creates a pow2 lookup table for 2^n for -VARIABLE_MAX_BITS <= n <= VARIABLE_MAX_BITS.
  Returns the name of the list and the offset needed to add to n to get the index
  """
  name = ctx.cfg.pow2_lookup_var
  if name not in ctx.proj.lists:
    ctx.proj.lists[name] = [sb3.Known(2 ** x) for x in range(-VARIABLE_MAX_BITS, VARIABLE_MAX_BITS + 1)]

  return name, getPow2Offset(), ctx

def twosComplement(val: sb3.Value, width: int) -> sb3.Value:
  return sb3.Op("mod", val, sb3.Known(2 ** width))

def undoTwosComplementWithOffset(val: sb3.Value, width: int) -> tuple[sb3.Value, int]:
  """
  Returns (value, offset), where value + offset is the reverse two's complement of the input value with
  a two's complement of width. Offset is always equal to 2^(width - 1) - 1.
  """

  # e.g. with 8 bits: output = ( ( input + 129 ) mod -256 ) + 127
  # this reverses two's complement in 3 values and only uses the input value once, as well as providing
  # an offset which can be optimized somewhere else with the power of maths

  # ( ( input + 2^(N/2) ) mod 2^N ) - 2^(N/2) would also work but this is way cooler so I'm keeping it

  return (
    sb3.Op("mod", sb3.Op("add", val, sb3.Known(2 ** (width - 1) + 1)), sb3.Known(-(2 ** width))),
    (2 ** (width - 1) - 1)
  )

def undoTwosComplement(val: sb3.Value, width: int) -> sb3.Value:
  """Reverses a two's complement of width on value"""

  value, offset = undoTwosComplementWithOffset(val, width)

  return sb3.Op("add", value, sb3.Known(offset))

def comptimeUndoTwosComplement(val: int | float, width: int) -> int | float:
  if val >= 2**(width-1):
    val -= 2**width
  return val

def intPow2(val: sb3.Value, ctx: Context, manual_offset: int=0) -> tuple[sb3.Value, Context]:
  """
  Calculates pow2 of a value. Only valid for values -VARIABLE_MAX_BITS <= n <= VARIABLE_MAX_BITS,
  but empty result can be used in a bitshift as zero, having no effect on the result.
  manual_offset can be combined with the offset used with pow2 to calculate 2^(n-1) at no extra cost.
  """
  if isinstance(val, sb3.Known):
    try:
      return sb3.Known(2 ** (int(val.known) + manual_offset)), ctx
    except ValueError:
      raise CompException("Cannot calculate pow2 of a known non-integer")
  else:
    lookup, offset, ctx = makePow2LookupTable(ctx) # Any value above the width will be treated as a zero
                                                   # which has no effect on the result
    return sb3.GetOfList("atindex", lookup, sb3.Op("add", val, sb3.Known(offset + manual_offset))), ctx

def bitShift(direction: Literal["left", "right"],
             width: int, left: sb3.Value, right: sb3.Value,
             ctx: Context, can_shift_out=True) -> tuple[sb3.Value, Context]:
  right_mul, ctx = intPow2(right, ctx)

  if direction == "left":
    # Multipling by a power of two is safe because the internal double value scratch uses doesn't lose accuracy
    # when only the exponent part changes
    unwrapped = sb3.Op("mul", left, right_mul)
    if not can_shift_out: return unwrapped, ctx
    return sb3.Op("mod", unwrapped, sb3.Known(2 ** width)), ctx
  else:
    unwrapped = sb3.Op("div", left, right_mul)
    if not can_shift_out: return unwrapped, ctx
    return sb3.Op("floor", unwrapped), ctx

def multiplyNoWrap(width: int, left: sb3.Value, right: sb3.Value) -> sb3.Value:
  if width > VARIABLE_MAX_BITS:
    raise CompException(f"Multipling {width} bits is not supported") # TODO FIX

  return sb3.Op("mul", left, right) # Overflow is UB - we don't care if
                                    # the number overflows and gets innacurate

def multiplyWrap(width: int, left: sb3.Value, right: sb3.Value, ctx: Context) -> ValueAndBlocks:
  # TODO OPTI: if one value is a known value, wrapping behaviour could be simpilifed and
  # known info could be propagated
  # TODO OPTI: if multipling by a power of 2, there is no risk that the mantissa cannot store
  # enough to be accurate, since only the exponent changes
  if width > VARIABLE_MAX_BITS:
    raise CompException(f"Multipling {width} bits not supported")

  if width <= 26: # Safe: (2**26) ** 2 < 9007199254740991
    return ValueAndBlocks(sb3.Op("mod", sb3.Op("mul", left, right), sb3.Known(2 ** width)))
  elif width <= 50: # Safe (with extra mod step): 2**25 * 2**25 + 2**25 * 2**25 < 9007199254740991
    blocks = sb3.BlockList()
    left, lblocks = flatAsTuple(optimizeValueUse(left, 3, ctx))
    right, rblocks = flatAsTuple(optimizeValueUse(right, 3, ctx))
    blocks.add(lblocks)
    blocks.add(rblocks)

    # Use some maths to do the calculation (see README for explaination)
    half_width = width // 2
    a0 = sb3.Op("mod", left, sb3.Known(2 ** half_width))
    b0 = sb3.Op("mod", right, sb3.Known(2 ** half_width))

    a0, a0blocks = flatAsTuple(optimizeValueUse(a0, 2, ctx))
    b0, b0blocks = flatAsTuple(optimizeValueUse(b0, 2, ctx))
    blocks.add(a0blocks)
    blocks.add(b0blocks)

    a0b1_plus_b0a1 = sb3.Op("add",
      sb3.Op("mul",
        a0,
        sb3.Op("floor", sb3.Op("div", right, sb3.Known(2 ** half_width)))
      ),
      sb3.Op("mul",
        b0,
        sb3.Op("floor", sb3.Op("div", left, sb3.Known(2 ** half_width)))
      ),
    )

    # 34 bits or less: no mod step needed: (2**17 * 2**17 + 2**17 * 2**17) * 2**17 + (2**17 + 2**17) < 9007199254740991
    # 50 bits or less: mod step is safe:   2**25 * 2**25 + 2**25 * 2**25 < 9007199254740991
    extra_mod_step = width > 34
    if extra_mod_step:
      a0b1_plus_b0a1 = sb3.Op("mod", a0b1_plus_b0a1, sb3.Known(2 ** math.ceil(width / 2)))

    value = sb3.Op("mod",
      sb3.Op("add",
        sb3.Op("mul",
          a0b1_plus_b0a1,
          sb3.Known(2 ** half_width)),
        sb3.Op("mul", a0, b0)
      ),
      sb3.Known(2 ** width))
    return ValueAndBlocks(value, blocks)
  else:
    raise CompException(f"Multipling {width} bits is not supported")

# TODO: use a linear search if it is faster (or maybe even use a linear search in a binary search)
# this is due to TW perf
def binarySearch(value: sb3.Value,
                 branches: dict[int, sb3.BlockList],
                 default_branch: sb3.BlockList | None = None,
                 min_poss_value: int | None = None, # Max value - we do not need to check for default values above it
                 max_poss_value: int | None = None, # Min value - likewise
                 are_branches_sorted: bool = False,
                 _lo: int=0, _hi: int | None=None) -> sb3.BlockList:
  if len(branches) == 0:
    return sb3.BlockList([]) if default_branch is None else default_branch

  if not are_branches_sorted: branches = dict(sorted(branches.items()))

  if _hi is None: _hi = len(branches.keys()) - 1
  mid = (_lo + _hi) // 2
  mid_val = list(branches.keys())[mid]

  if _lo == _hi:
    if default_branch is not None:
      # If there is only one possible value in the range then skip the equality check
      skip_check = min_poss_value is not None and min_poss_value == max_poss_value

      if not skip_check:
        cond = sb3.BoolOp("=", value, sb3.Known(mid_val))
        return sb3.BlockList([sb3.ControlFlow("if_else", cond, list(branches.values())[mid], default_branch)])

    return list(branches.values())[mid]

  cond = sb3.BoolOp(">", value, sb3.Known(mid_val))
  return sb3.BlockList([sb3.ControlFlow("if_else", cond,
                        # Sorting already taken care of
                        binarySearch(value, branches, default_branch, mid_val + 1, max_poss_value, True,
                                     _lo=mid + 1, _hi=_hi),
                        binarySearch(value, branches, default_branch, min_poss_value, mid_val, True,
                                     _lo=_lo,     _hi=mid))])

def paritialSumDiff(op: Literal["add", "sub"], left: sb3.Value, right: sb3.Value, prev_sum: sb3.Value, ctx: Context) -> sb3.Value:
  # Binary subtraction is exactly the same as addition but using subtraction to apply the carry/borrow bit and subtraction
  # checks for negative instead. The modulus used with add also functions as a "borrow"
  raw_sum = sb3.Op(op, left, right)

  if op == "add":
    carry = sb3.BoolOp(">", prev_sum, sb3.Known((2 ** VARIABLE_MAX_BITS) - 1))
  else:
    carry = sb3.BoolOp("<", prev_sum, sb3.Known(0))

  carried_sum = sb3.Op(op, raw_sum, carry)
  # When using known values, the optimizer can subtract values from both sides
  if ctx.cfg.compiler_opt: carried_sum = opt.simplifyValue(carried_sum)

  return carried_sum

def calculateSumDiff(op: Literal["add", "sub"], left: IdxbleValue, right: IdxbleValue, width: int, ctx: Context) \
    -> IdxbleValueAndBlocks:
  steps = len(left.vals)
  assert steps == len(right.vals)
  assert steps == math.ceil(width / VARIABLE_MAX_BITS)
  if steps == 0:
    return IdxbleValueAndBlocks()

  if ctx.cfg.compiler_opt:
    # Optimize the values beforehand - helps with finding optimial calculation after optimizations
    left = IdxbleValue([opt.simplifyValue(l) for l in left.vals])
    right = IdxbleValue([opt.simplifyValue(r) for r in right.vals])

  best_cost = float("inf")
  best_blocks = sb3.BlockList()
  best_sum_nodes: list[sb3.Value] = []

  max_spacing = min(steps, 10) # Tends to be about 3, no need to search much higher

  for spacing in range(1, max_spacing + 1):
    start_index = spacing
    for omit_last in (False, True):
      checkpoint_indices = set(range(start_index, steps, spacing))
      if omit_last and (steps - 1) in checkpoint_indices:
        checkpoint_indices.remove(steps - 1)

      cost = 0.0
      blocks = sb3.BlockList()
      sum_nodes: list[sb3.Value] = []
      stored_temp_names: dict[int, str] = {}

      for i in range(steps):
        modulus = 2 ** (VARIABLE_MAX_BITS if i != steps - 1 else width % VARIABLE_MAX_BITS)

        if i == 0:
          raw = sb3.Op(op, left.vals[0], right.vals[0])
          cost += opt.getValueCost(raw, ctx.cfg.opt_target.perf)
        else:
          earlier_stored = [idx for idx in stored_temp_names.keys() if idx < i]
          prev_stored = max(earlier_stored) if earlier_stored else None

          if prev_stored is None:
            start = 1
            prev = sb3.Op(op, left.vals[0], right.vals[0])
          else:
            start = prev_stored + 1
            prev = sb3.GetVar(stored_temp_names[prev_stored])

          for j in range(start, i + 1):
            prev = paritialSumDiff(op, left.vals[j], right.vals[j], prev, ctx)

          raw = prev

        if (i in checkpoint_indices) and (i >= start_index):
          temp_name = genTempVar(ctx)
          stored_temp_names[i] = temp_name
          blocks.add(sb3.EditVar("set", temp_name, raw))
          cost += ctx.cfg.opt_target.perf.set_var + opt.getValueCost(raw, ctx.cfg.opt_target.perf)

        if i in stored_temp_names:
          expr_for_mod = sb3.GetVar(stored_temp_names[i])
        else:
          expr_for_mod = raw

        mod_node = sb3.Op("mod", expr_for_mod, sb3.Known(modulus))
        cost += opt.getValueCost(mod_node, ctx.cfg.opt_target.perf)

        sum_nodes.append(mod_node)

      if cost < best_cost:
        best_cost = cost
        best_blocks = blocks
        best_sum_nodes = sum_nodes

  return IdxbleValueAndBlocks(IdxbleValue(best_sum_nodes), best_blocks)

def sumValueParts(parts: list[sb3.Value], default: int | None=None) -> sb3.Value:
  if len(parts) == 0:
    assert default is not None
    return sb3.Known(default)

  res = parts[0]
  for i, part in enumerate(parts):
    if i == 0: continue
    res = sb3.Op("add", res, part)

  return res

def extractBits(
    val: sb3.Value, width: int, start: int, bits: int,
    in_place: bool=False, skip_floor: bool=False
  ) -> tuple[sb3.Value, int]:
  """
  Extract the first 'bits' bits after the 'start + 1'th MSB on an integer 'val' of width 'width'
  If in_place is True then shift the bits back afterward so they end up back in the same place they started
  If skip_floor is True the floor of the retured value is equal to the extracted bits
  Returns (result, shift) where shift is the bit shift used internally
  """
  end = start + bits
  shift = width - end
  assert start >= 0 and end <= width

  # If we skip floor, than the error will not be corrected by an external floor
  assert not (in_place and skip_floor)

  res = val
  # Shift right if necessary
  if shift != 0:
    res = sb3.Op("div", val, sb3.Known(2 ** shift))
    if not skip_floor:
      res = sb3.Op("floor", res)
  # Bit mask any extra bits with modulo. It is fine that the input may not be floored
  if start != 0: res = sb3.Op("mod", res, sb3.Known(2 ** bits))
  # Shift left if in place
  if in_place and shift != 0: res = sb3.Op("mul", res, sb3.Known(2 ** shift))

  return res, shift

def getKnownBitGroups(val: int, width: int) -> list[tuple[int, int]]:
  """Returns [(bit, length)] for each group of similar bits in val, starting from the MSB"""
  current_bit = None
  current_len = 0
  bit_groups: list[tuple[int, int]] = []

  # e.g. with 64 bits [63, 62, ..., 1, 0]
  for i in range(width - 1, -1, -1):
    bit = (val >> i) & 1
    if bit != current_bit:
      if current_bit is not None: bit_groups.append((current_bit, current_len))
      current_bit = bit
      current_len = 0
    current_len += 1

  assert current_bit is not None
  bit_groups.append((current_bit, current_len))
  return bit_groups

def getOpLookupTableName(op: Literal["and", "or", "xor"], ctx: Context) -> str:
  return f"!{op.upper()} lookup{ctx.cfg.zero_indexed_suffix}"

def binOpPartViaLookupTable(
    op: Literal["and", "or", "xor"], lft: sb3.Value, rgt: sb3.Value,
    width: int, start: int, ctx: Context, bits=BINOP_LOOKUP_BITS,
  ) -> sb3.Value:
  """
  Calculates 'bits' of a AND, OR or XOR on two values using a lookup table from the
  'start'th MSB. The caller is responsible for ensuring the lookup table is created if the result
  is used. 'bits' must be less than or equal to BINOP_LOOKUP_BITS
  """

  assert bits <= BINOP_LOOKUP_BITS

  table_name = getOpLookupTableName(op, ctx)

  # It is faster for the known value to be left because this means
  # a multiplication can be skipped
  if isinstance(rgt, sb3.Known):
    tmp = lft
    lft = rgt
    rgt = tmp

  if width >= bits:
    # Extract the bits used
    extracted_lft, shift_l = extractBits(lft, width, start, bits)
    # We can skip flooring this value as scratch's "get item _ of _" block
    # floors the index internally
    extracted_rgt, shift_r = extractBits(rgt, width, start, bits, skip_floor=True)
    assert shift_l == shift_r

  else:
    # If the size is small enough that extracting the bits would cause an error,
    # the extracted values are the original values
    extracted_lft, extracted_rgt, shift_l = lft, rgt, 0

  # e.g. A & B = and_lookup[A * 256 + B]
  lookup_index = sb3.Op("add",
    sb3.Op("mul", extracted_lft, sb3.Known(2 ** BINOP_LOOKUP_BITS)),
    extracted_rgt,
  )

  # If the index is 0, "" is returned. We need to ensure this is casted to zero,
  # so use str_to_float. Most of the time this should be optimized away.
  res = sb3.Op("str_to_float", sb3.GetOfList("atindex", table_name, lookup_index))
  # Shift left the result back to put it back in place
  if shift_l != 0: res = sb3.Op("mul", res, sb3.Known(2 ** shift_l))

  return res

def binOpWithKnownViaLookupTable(
    op: Literal["and", "or", "xor"], unknown: sb3.Value, known: int,
    width: int, ctx: Context
  ) -> sb3.Value:
  """
  Calculates AND/OR/XOR against a known value using its lookup table. The caller is
  responsible for ensuring the respective lookup table is created.
  """

  i = 0
  final_offset = 0
  parts = []
  while i < width:
    if op != "xor":
      # Skip over 0s in AND, skip over 1s in OR
      skip_over = 0 if op == "and" else 1

      get_shift = lambda i: width - i - 1
      while i < width and (known >> get_shift(i)) & 1 == skip_over:
        # If we skipped over a 1 we need to add it to the offset at the end
        if op == "or":
          final_offset += 1 << get_shift(i)
        i += 1

      if i >= width: break

    bits = min(BINOP_LOOKUP_BITS, width - i)
    parts.append(binOpPartViaLookupTable(op, unknown, sb3.Known(known), width, i, ctx, bits))
    i += BINOP_LOOKUP_BITS

  if final_offset > 0: parts.append(sb3.Known(final_offset))

  return sumValueParts(parts, default=0)

def binOpWithUnknownViaLookupTable(op: Literal["and", "or", "xor"], lft: sb3.Value, rgt: sb3.Value,
    width: int, ctx: Context
  ) -> sb3.Value:
  """
  Calculates AND/OR/XOR between two unknown values using a lookup table. The caller is
  responsible for ensuring the respective lookup table is created
  """
  parts = []
  for i in range(0, width, BINOP_LOOKUP_BITS):
    bits = min(BINOP_LOOKUP_BITS, width - i)
    parts.append(binOpPartViaLookupTable(op, lft, rgt, width, i, ctx, bits))

  return sumValueParts(parts, default=0)

def andWithKnownMaskParts(unknown: sb3.Value, known: int, width: int, ctx: Context) -> tuple[list[sb3.Value], bool]:
  """
  Calculates AND of a known and unknown value. Returns the ([val], needs_lut) where the
  sum of the list of values is the result and needs_lut means if a lookup table needs to
  be created.
  """

  needs_lut = False
  current_bit_idx = 0
  parts = []
  groups = getKnownBitGroups(known, width)
  i = 0
  while current_bit_idx < width:
    bit, length = groups[i]
    # X & 0 = 0
    if bit == 0:
      current_bit_idx += length
      i += 1
      continue

    # Get the next regions in the BINOP_LOOKUP_BITS bits (not including regions cut off by it)
    j = i
    next_regions: list[tuple[int, int, int]] = []
    current_reg_bit_idx = current_bit_idx

    # While the current bit is within the region and we're not out of range
    reg_len = None
    region_end = min(current_bit_idx + BINOP_LOOKUP_BITS, width)
    while current_reg_bit_idx < region_end:
      reg_bit, reg_len = groups[j]

      # Set the next region, which will be added if it fits
      current_region = (current_reg_bit_idx, reg_len, j)
      current_reg_bit_idx += reg_len
      if reg_bit == 1 and current_reg_bit_idx <= region_end:
        next_regions.append(current_region)

      j += 1
      if j >= len(groups): break

    lookup_table_method = None
    extract_region = lambda idx, r_len: extractBits(unknown, width, idx, r_len, in_place=True)[0]

    # There is no advantage to using a lookup table because the region is too large (or zero)
    if reg_len is None or len(next_regions) == 0:
      use_lut_method = False
    else:
      # Calculate the cost of extracting the values in each region
      extracted_regions: list[sb3.Value] = []
      for reg_start, r_len, _ in next_regions:
        extracted_regions.append(extract_region(reg_start, r_len))
      extract_region_method = sumValueParts(extracted_regions, default=0)

      # Don't AND more bits than exist
      bits = min(BINOP_LOOKUP_BITS, width - current_bit_idx)
      # Create a new region for the binop lookup table equal to the size of regions it skips over
      lookup_table_method = opt.simplifyValue(
        binOpPartViaLookupTable("and", unknown, sb3.Known(known), width, current_bit_idx, ctx, bits))
      # If it's faster to use a lookup table, then use it!
      perf = ctx.cfg.opt_target.perf
      use_lut_method = opt.getValueCost(lookup_table_method, perf) < opt.getValueCost(extract_region_method, perf)

    if use_lut_method:
      # This should not occur as reg_len is always int if len(next_regions) == 0
      assert reg_len is not None

      last_region_index = j - 1
      last_region_cut_off_by = reg_len - (current_reg_bit_idx - region_end)
      assert last_region_cut_off_by >= 0
      needs_lut = True

      assert lookup_table_method is not None
      parts.append(lookup_table_method)

      # The lookup table might have AND'd some extra bits not inside the last region. Correct
      # this by increasing the length of bits in the region before and decreasing the length
      # of bits in that region
      old_region = groups[last_region_index]
      groups[last_region_index] = (old_region[0], old_region[1] - last_region_cut_off_by)

      if last_region_index > 0:
        old_region_before = groups[last_region_index - 1]
        groups[last_region_index - 1] = (old_region_before[0], old_region_before[1] + last_region_cut_off_by)

      # Set the index to the index after the final region
      reg_start, _, reg_i = next_regions[-1]
      i = reg_i + 1
      current_bit_idx = reg_start + groups[reg_i][1]
      # If when we changed groups the regions' start was shifted forward, then shift the current
      # bit forward too
      if reg_i >= last_region_index:
        current_bit_idx += last_region_cut_off_by

    else:
      parts.append(extract_region(current_bit_idx, length))
      current_bit_idx += length
      i += 1

  return parts, needs_lut

def andWithKnownMask(unknown: sb3.Value, known: int, width: int, ctx: Context) -> tuple[sb3.Value, bool]:
  """
  Returns (result, needs_and_lut) where 'result' is the the result and needs_and_lut
  is if a AND lookup table is required
  """
  parts, needs_lut = andWithKnownMaskParts(unknown, known, width, ctx)
  # Don't break if ANDing with zero
  return sumValueParts(parts, default=0), needs_lut

def orWithKnownMask(unknown: sb3.Value, known: int, width: int, ctx: Context) -> tuple[sb3.Value, bool, bool]:
  """
  Returns (result, needs_and_lut, needs_or_lut) where 'result' is the result and
  needs_abc_lut is if a abc lookup table is required
  """
  # a | b = (a & !b) + b, see README for proof
  not_b = (2 ** width - 1) - known
  a_and_b, needs_and_lut = andWithKnownMask(unknown, not_b, width, ctx)
  a_or_b_with_and = sb3.Op("add", a_and_b, sb3.Known(known))
  a_or_b_specialized_lut = opt.simplifyValue(binOpWithKnownViaLookupTable("or", unknown, known, width, ctx))
  perf = ctx.cfg.opt_target.perf
  if opt.getValueCost(a_or_b_with_and, perf) < opt.getValueCost(a_or_b_specialized_lut, perf):
    return a_or_b_with_and, needs_and_lut, False
  else:
    return a_or_b_specialized_lut, False, True

def xorWithKnownMask(unknown: sb3.Value, known: int, width: int, ctx: Context) -> tuple[sb3.Value, bool, bool]:
  """
  Returns (result, needs_and_lut, needs_xor_lut) where 'result' is the result and
  needs_abc_lut is if a abc lookup table is required
  """

  # Special case: a ^ -1. Via the AND path this would be a - 2a + b, but the optimizer
  # is not smart enough yet to rearrange this to b - a
  if known == 2 ** width - 1:
    return sb3.Op("sub", sb3.Known(known), unknown), False, False

  a_and_b_parts, needs_and_lut = andWithKnownMaskParts(unknown, known, width, ctx)
  # Multiply each coefficient by 2 if there is already a multiplication for each part
  # otherwise, multiply the sum by two
  use_parts = True
  a_and_b_times_2_parts: list[sb3.Value] = []
  for part in a_and_b_parts:
    if isinstance(part, sb3.Op) and part == "mul" and \
       isinstance(part.right, sb3.Known) and isinstance(part.right.known, float):
      part_times_2 = sb3.Op(part.op, part.left, sb3.Known(float(part.right.known) * 2))
      a_and_b_times_2_parts.append(part_times_2)
    else:
      use_parts = False
      break

  if use_parts:
    a_and_b_times_2 = sumValueParts(a_and_b_times_2_parts, default=0)
  else:
    a_and_b_times_2 = sb3.Op("mul", sumValueParts(a_and_b_parts, default=0), sb3.Known(2))

  # a ^ b = a - 2 * (a & b) + b, see README for proof
  a_xor_b_with_and = sb3.Op("add", sb3.Op("sub", unknown, a_and_b_times_2), sb3.Known(known))
  a_xor_b_specialized_lut = opt.simplifyValue(binOpWithKnownViaLookupTable("xor", unknown, known, width, ctx))
  perf = ctx.cfg.opt_target.perf
  if opt.getValueCost(a_xor_b_with_and, perf) < opt.getValueCost(a_xor_b_specialized_lut, perf):
    return a_xor_b_with_and, needs_and_lut, False
  else:
    return a_xor_b_specialized_lut, False, True

def binOp(op: Literal["and", "or", "xor"], lft: sb3.Value, rgt: sb3.Value,
    width: int, ctx: Context, is_disjoint: bool=False
  ) -> tuple[sb3.Value, Context]:
  """
  Calculates AND/OR/XOR of two values, taking advantage of one of the values being known. May create
  lookup tables which are saved to ctx. is_disjoint - if no bits in lft and rgt overlap. Only allowed
  on an OR operation.
  """

  lft_is_known = isinstance(lft, sb3.Known)
  rgt_is_known = isinstance(rgt, sb3.Known)

  # No overlap, therefore a | b = a + b
  if is_disjoint:
    assert op == "or"
    return sb3.Op("add", lft, rgt), ctx

  # For 1 bit values, more simple and/or/xor methods can be used
  if width == 1:
    if op == "and":
      # lft * rgt
      return sb3.Op("mul", lft, rgt), ctx
    elif op == "or":
      # lft + rgt > 0
      # the bool to float is usually optimized away here
      return sb3.Op("bool_to_float", sb3.BoolOp(">", sb3.Op("add", lft, rgt), sb3.Known(0))), ctx
    elif op == "xor":
      if lft_is_known or rgt_is_known:
        unknown = rgt if lft_is_known else lft
        # 1 - unknown, the optimizer wouldn't be able to optimize the generic case to this
        return sb3.Op("sub", sb3.Known(1), unknown), ctx
      else:
        # lft + rgt mod 2
        return sb3.Op("mod", sb3.Op("add", lft, rgt), sb3.Known(2)), ctx

  if lft_is_known or rgt_is_known:
    known = lft if lft_is_known else rgt
    assert isinstance(known, sb3.Known)

    if isinstance(known.known, str) or int(known.known) != known.known:
      raise CompException(f"Cannot {op} against invalid known value: {known.known}")

    known = int(known.known)
    unknown = rgt if lft_is_known else lft

    needs_and_lut = needs_or_lut = needs_xor_lut = False
    if op == "and":
      res, needs_and_lut = andWithKnownMask(unknown, known, width, ctx)
    elif op == "or":
      res, needs_and_lut, needs_or_lut = orWithKnownMask(unknown, known, width, ctx)
    else:
      res, needs_and_lut, needs_xor_lut = xorWithKnownMask(unknown, known, width, ctx)

    ctx.needs_and_lut |= needs_and_lut
    ctx.needs_or_lut  |= needs_or_lut
    ctx.needs_xor_lut |= needs_xor_lut

    return res, ctx

  else:
    ctx.needs_and_lut |= op == "and"
    ctx.needs_or_lut  |= op == "or"
    ctx.needs_xor_lut |= op == "xor"
    return binOpWithUnknownViaLookupTable(op, lft, rgt, width, ctx), ctx

def intCompare(left: sb3.Value, right: sb3.Value, width: int, mode: ir.ICmpCond, cfg: Config) -> sb3.BooleanValue:
  special_signed_handling = cfg.compiler_opt and \
                            mode in {ir.ICmpCond.Sge, ir.ICmpCond.Sle} and \
                            not isinstance(left, sb3.Known) and \
                            not isinstance(right, sb3.Known)

  modulus = -(2 ** width)

  # Like undoTwosComplement but can remove the extra addition at the end because algebra
  # Simplify because instructions are optimized for known values
  reverse_twos_complement = lambda val: opt.simplifyValue(
    sb3.Op("mod", sb3.Op("add", val, sb3.Known(2 ** (width - 1) + 1)), sb3.Known(modulus)))

  # Subtracting half the modulus after reversing two's complement
  reverse_twos_complement_and_sub_half = lambda val: \
    sb3.Op("mod", sb3.Op("add", val, sb3.Known(2 ** (width - 1) + 0.5)), sb3.Known(modulus))

  if not special_signed_handling:
    if mode in {ir.ICmpCond.Sgt, ir.ICmpCond.Sge, ir.ICmpCond.Slt, ir.ICmpCond.Sle}:
      left = reverse_twos_complement(left)
      right = reverse_twos_complement(right)

    match mode:
      case ir.ICmpCond.Eq:
        return sb3.BoolOp("=", left, right)
      case ir.ICmpCond.Ne:
        return sb3.BoolOp("not", sb3.BoolOp("=", left, right))
      case ir.ICmpCond.Ugt | ir.ICmpCond.Sgt:
        return sb3.BoolOp(">", left, right)
      case ir.ICmpCond.Ult | ir.ICmpCond.Slt:
        return sb3.BoolOp("<", left, right)
      case ir.ICmpCond.Uge | ir.ICmpCond.Sge:
        if isinstance(left, sb3.Known):
          return sb3.BoolOp(">", sb3.Known(sb3.scratchCastToNum(left) + 1), right)
        elif isinstance(right, sb3.Known):
          return sb3.BoolOp(">", left, sb3.Known(sb3.scratchCastToNum(right) - 1))
        else:
          return sb3.BoolOp("not", sb3.BoolOp("<", left, right))
      case ir.ICmpCond.Ule | ir.ICmpCond.Sle:
        if isinstance(left, sb3.Known):
          return sb3.BoolOp("<", sb3.Known(sb3.scratchCastToNum(left) - 1), right)
        elif isinstance(right, sb3.Known):
          return sb3.BoolOp("<", left, sb3.Known(sb3.scratchCastToNum(right) + 1))
        else:
          return sb3.BoolOp("not", sb3.BoolOp(">", left, right))
      case _:
        raise CompException(f"icmp does not support comparsion mode {mode}")
  else:
    # We can skip adding a number for greater equal/less equal by adjusting the values
    # of the two's complement reversal
    if mode is ir.ICmpCond.Sge:
      return sb3.BoolOp(">", reverse_twos_complement(left), reverse_twos_complement_and_sub_half(right))
    else:
      return sb3.BoolOp("<", reverse_twos_complement_and_sub_half(left), reverse_twos_complement(right))

def offsetStackSize(stack_size_var: str, offset: int) -> sb3.Value:
  ptr = sb3.GetVar(stack_size_var)
  if offset > 0:
    ptr = sb3.Op("add", ptr, sb3.Known(offset))
  elif offset < 0:
    # Subtract instead... because it looks nicer lol
    ptr = sb3.Op("sub", ptr, sb3.Known(-offset))
  return ptr

def transLoad(result: Variable, address: sb3.Value, loaded_type: ir.Type, ctx: Context) -> sb3.BlockList:
  blocks = sb3.BlockList()

  # TODO FIX: properly skip over padding bytes
  if ctx.cfg.accurate_byte_spacing and isinstance(loaded_type, (ir.ArrayTy, ir.StructTy)):
    raise CompException(f"Loading aggregates with accurate padding not supported yet")

  # Don't include padding - we don't care about loading padded bytes, they only exist to offset pointers,
  # not variables
  var_size = getByteSize(loaded_type, include_padding=False)

  if var_size == 1:
    blocks.add(result.setValue(sb3.GetOfList("atindex", ctx.cfg.mem_var, address)))
  else:
    blocks.add(result.setAllValues(IdxbleValue([
      sb3.GetOfList("atindex", ctx.cfg.mem_var, sb3.Op("add", address, sb3.Known(i))) for i in range(var_size)
    ])))

  return blocks

def transStore(value: sb3.Value | IdxbleValue, address: sb3.Value, stored_type: ir.Type, ctx: Context) -> sb3.BlockList:
  # TODO FIX: properly skip over padding bytes when storing
  if ctx.cfg.accurate_byte_spacing and isinstance(stored_type, (ir.ArrayTy, ir.StructTy)):
    raise CompException(f"Storing aggregates with accurate padding not supported yet")

  blocks = sb3.BlockList()
  if isinstance(value, sb3.Value):
    blocks.add(sb3.EditList("replaceat", ctx.cfg.mem_var, address, value))
  else:
    for offset, val in enumerate(value.vals):
      offset_val = sb3.Known(offset)
      blocks.add(sb3.EditList("replaceat", ctx.cfg.mem_var, sb3.Op("add", address, offset_val), val))
  return blocks

def storeOnStack(stack_var: str, stack_size_var: str, offset: int, size: int, value: sb3.Value | IdxbleValue) -> sb3.BlockList:
  blocks = sb3.BlockList()
  if isinstance(value, sb3.Value):
    blocks.add(sb3.EditList("replaceat", stack_var, offsetStackSize(stack_size_var, offset), value))
  else:
    for i in range(size):
      blocks.add(sb3.EditList("replaceat", stack_var, offsetStackSize(stack_size_var, offset + i), value.vals[i]))
  return blocks

def loadFromStack(stack_var: str, stack_size_var: str, offset: int, size: int) -> sb3.Value | IdxbleValue:
  res: list[sb3.Value] = [sb3.GetOfList("atindex", stack_var, offsetStackSize(stack_size_var, offset + i)) for i in range(size)]
  if size == 1: return res[0]
  return IdxbleValue(res)

def assignParameters(params: list[Variable], param_sizes: list[int], next_var_use_depends: set[str], ctx: Context) -> sb3.BlockList:
  assert len(params) == len(param_sizes)

  blocks = sb3.BlockList()
  # We never need to assign parameters as everything is in one function when using
  # the branch jump table method
  if ctx.cfg.use_branch_jump_table:
    return blocks

  for param, size in zip(params, param_sizes):
    var = deepcopy(param)
    var.var_type = "var"
    # Don't assign anything we depend upon in future
    if var.var_name in next_var_use_depends:
      if size == 1:
        blocks.add(var.setValue(param.getValue()))
      else:
        blocks.add(var.setAllValues(param.getAllValues(size)))
  return blocks

def assignPhiNodes(phi_info: list[tuple[Variable, ir.Value]], ctx: Context, bctx: BlockInfo) -> sb3.BlockList:
  end_assignments: list[tuple[Variable, ValueAndBlocks | IdxbleValueAndBlocks]] = []
  to_resolve: dict[str, str] = {}
  set_by: dict[str, str] = {}
  resolved: list[tuple[str, str]] = []
  var_lookup: dict[str, Variable] = {}
  val_lookup: dict[str, ValueAndBlocks | IdxbleValueAndBlocks] = {}

  for res_var, ir_val in phi_info:
    # Undef values do not need to be assigned
    if isinstance(ir_val, ir.UndefVal):
      continue

    val = transValue(ir_val, ctx, bctx)

    if isinstance(ir_val, ir.LocalVarVal):
      # Other variables in this phi assignment might depend on this one, ensure it gets resolved correctly
      to_resolve[res_var.var_name] = ir_val.name
      var_lookup[res_var.var_name] = res_var
      val_lookup[ir_val.name] = val
    else:
      # It is safe to put this at the end because it does not rely on anything that might be changed
      end_assignments.append((res_var, val))

  while to_resolve:
    # Anything that has a dependency on it cannot be set
    cant_set = OrderedSet(to_resolve.values())
    to_set = OrderedSet(to_resolve.keys()) - cant_set

    for var in to_set:
      set_by[to_resolve[var]] = var
      resolved.append((var, to_resolve[var]))
      del to_resolve[var]

    # If there is a dependency cycle and we need to create a temporary (or use an existing variable)
    if len(to_set) == 0:
      # We can use order of operations to our advantage. For example:
      # 2 = 3;
      # temp1 = 3;
      # 3 = 4;
      # 4 = 5;
      # 5 = temp1;
      # Can be optimized to:
      # 2 = 3;
      # 3 = 4;
      # 4 = 5;
      # 5 = 2;
      already_set = OrderedSet(to_resolve.keys()) & OrderedSet(set_by.keys())
      if already_set:
        # Use an existing variable set to the same value
        to_make_temp = next(iter(already_set))
        temp_name = set_by[to_make_temp]
      else:
        to_make_temp = next(iter(to_resolve.keys()))
        to_make_temp_val = val_lookup[to_make_temp]

        # Create a temporary
        temp_name = genTempVar(ctx)
        temp_var = Variable(temp_name, "special_var", None)
        var_lookup[temp_name] = temp_var
        if isinstance(to_make_temp_val, ValueAndBlocks):
          temp_val = ValueAndBlocks(temp_var.getValue())
        else:
          temp_val = IdxbleValueAndBlocks(temp_var.getAllValues(len(to_make_temp_val.value.vals)))
        val_lookup[temp_name] = temp_val

        resolved.append((temp_name, to_make_temp))

      resolved.append((to_make_temp, to_resolve[to_make_temp]))
      del to_resolve[to_make_temp]

      # Update references to value with the new temporary or existing value
      for var, deps in to_resolve.items():
        if to_make_temp == deps:
          to_resolve[var] = temp_name

  assignments: list[tuple[Variable, ValueAndBlocks | IdxbleValueAndBlocks]] = []
  for var_name, val_name in resolved:
    assignments.append((var_lookup[var_name], val_lookup[val_name]))
  assignments.extend(end_assignments)

  blocks = sb3.BlockList()
  for res_var, val in assignments:
    blocks.add(val.blocks)
    if isinstance(val, ValueAndBlocks):
      blocks.add(res_var.setValue(val.value))
    else:
      blocks.add(res_var.setAllValues(val.value))

  return blocks

def getCallArguments(
    args: list[ir.Value], vararg_ptr: sb3.Value | None,
    ret_addr: int | None, ctx: Context, bctx: BlockInfo
  ) -> tuple[list[sb3.Value], sb3.BlockList]:

  arguments: list[sb3.Value] = []
  blocks = sb3.BlockList()
  for arg in args:
    value = transValue(arg, ctx, bctx)
    blocks.add(value.blocks)
    if not isinstance(value, IdxbleValueAndBlocks):
      arguments.append(value.value)
    else:
      for val in value.value.vals: arguments.append(val)

  if vararg_ptr is not None: arguments.append(vararg_ptr)
  if ret_addr is not None:   arguments.append(sb3.Known(ret_addr))

  return arguments, blocks

def getUncheckedProcedureStart(proc_name: str, params: list[Variable], param_sizes: list[int], fn: FuncInfo,
                               ctx: Context, is_counted: bool=False) -> tuple[sb3.BlockList, Context]:
  assert len(params) == len(param_sizes)
  blocks = sb3.BlockList([sb3.ProcedureDef(proc_name, localizeSizedParameters(params, param_sizes))])

  if ctx.cfg.do_debug_branch_log:
    # Increment the branch counter in the log
    index = sb3.Known((fn.fn_id * 2) + 2)
    blocks.add(sb3.BlockList([
      sb3.EditList("replaceat", ctx.cfg.debug_branch_log_var, index, sb3.Op("add",
        sb3.Known(1), sb3.GetOfList("atindex", ctx.cfg.debug_branch_log_var, index)
    ))]))

  if is_counted:
    blocks.add(sb3.EditCounter("incr")) # The 'hacked' counter blocks are 20x faster than incrementing
                                        # a number

  return blocks, ctx

def getCheckedProcedureStart(proc_name: str, params: list[Variable], param_sizes: list[int],
                             next_var_use_depends: set[str], block_label: str, fn: FuncInfo,
                             ctx: Context) -> tuple[sb3.BlockList, Context]:
  """
  Returns the blocks needed to return a branch instruction (procedure)
  that will reset the scratch's stack if reaching a max amount of recursions,
  preventing scratch from running out of memory
  """
  assert len(params) == len(param_sizes)

  blocks, ctx = getUncheckedProcedureStart(proc_name, params, param_sizes, fn, ctx, is_counted=True)

  # Get the ID of this branch in the stack reset jump table so it can jump back to this branch once
  # the stack is reset
  reset_id = START_STACK_RESET_ID + ctx.all_check_locations.index((fn.name, block_label))

  blocks.add(sb3.ControlFlow("if",
    sb3.BoolOp(">", sb3.GetCounter(), sb3.Known(ctx.cfg.max_branch_recursion)), sb3.BlockList([
      # This should never be called as it is not possible to branch to the first branch, but will be kept
      # in case parameters are used between blocks in future
      *assignParameters(params, param_sizes, next_var_use_depends, ctx).blocks,

      sb3.EditVar("set", ctx.cfg.jump_table_id_var, sb3.Known(reset_id)),
      sb3.StopScript("stopthis")
    ])))

  return blocks, ctx

def transSimpleCall(name: str, arguments: list[sb3.Value],
                    result: Variable | None, result_size: int | None,
                    ctx: Context) -> tuple[sb3.BlockList, sb3.BlockList]:
  """
  Translates simple function calls. Deals with passing parameters and
  return values. The first block list returned is any blocks to call
  the function, the second any needed to assign the return value to
  the output.
  """

  call_blocks = sb3.BlockList([sb3.ProcedureCall(name, arguments)])

  set_value_blocks = sb3.BlockList()
  if result is not None:
    assert result_size is not None
    return_var = Variable(ctx.cfg.return_var, "special_var", None)
    if result_size == 1:
      set_value_blocks.add(result.setValue(return_var.getValue()))
    else:
      set_value_blocks.add(result.setAllValues(return_var.getAllValues(result_size)))

  return call_blocks, set_value_blocks

def transComplexCall(caller: FuncInfo, callee: FuncInfo | FuncPtrSigInfo,
                     args: list[ir.Value], result: Variable | None,
                     result_size: int | None, following_instrs: list[ir.Instr],
                     ctx: Context, bctx: BlockInfo) -> tuple[Context, BlockInfo]:
  """
  Translates a function call. Deals with functions with return
  addresses, recursion and function pointers. May change the function
  that instructions are being added to.
  """

  assert bctx.label is not None

  param_count = callee.value_param_count
  if isinstance(callee, FuncPtrSigInfo):
    # Account for the function pointer address
    param_count += 1

  assert callee.is_variadic or param_count == len(args)
  args, varargs = args[:param_count], args[param_count:]

  total_alloc_size = 0
  vararg_ptr = None
  if callee.is_variadic:
    # Allocate vararg memory
    total_alloc_size = sum(getByteSize(arg.type, include_padding=ctx.cfg.accurate_byte_spacing) for arg in varargs)
    if total_alloc_size != 0:
      vararg_ptr = sb3.GetVar(ctx.cfg.stack_pointer_var)
      bctx.code.add(sb3.EditVar("change", ctx.cfg.stack_pointer_var, sb3.Known(-total_alloc_size)))
    else:
      vararg_ptr = sb3.Known(0)

    # Store varargs
    offset = 0
    for arg in varargs:
      arg_val, arg_blocks = flatAsTuple(transValue(arg, ctx, bctx))
      bctx.code.add(arg_blocks)
      bctx.code.add(transStore(arg_val, sb3.Op("add", vararg_ptr, sb3.Known(offset)), arg.type, ctx))
      offset += getByteSize(arg.type, include_padding=ctx.cfg.accurate_byte_spacing)
  else:
    assert len(varargs) == 0

  # If a function pointer, call the 'signature' corresponding to how we called the function
  callee_name = callee.name if isinstance(callee, FuncInfo) else localizeFuncPtrSig(callee.signature_id)

  # Include the return value in variables which aren't depended on
  starting_var_use = BlockVarUse() if result is None else BlockVarUse(modifies={result.var_name})
  # All variables that might be depended on/modified after the function is called
  next_var_use = getBlockVarUse(following_instrs, bctx.fn.phi_info[bctx.label], caller.block_var_use, starting_var_use)

  poss_recursive = caller.name in callee.can_call
  # Get all variables which are used later after the recursion
  must_store: list[Variable] = []
  must_store_sizes: list[int] = []
  if poss_recursive:
    # Include the return value in variables which aren't depended on
    for var in next_var_use.depends:
      # We don't need to store parameters for later in a branch table
      # because parameters are tied to scope and therefore always available
      if ctx.cfg.use_branch_jump_table and var in bctx.available_params:
        continue

      decoded_var = transVar(var, bctx)
      assert decoded_var is not None

      must_store.append(decoded_var)

    # Sort the parameters in numeric then alphabetical order for better readability
    must_store.sort(key=lambda var: (0, int(var.var_name)) if var.var_name.isdigit() else (1, var.var_name))

    # Work out sizes of must stores
    must_store_sizes = [next_var_use.depends_var_sizes[var.var_name] for var in must_store]

    must_store_special = []
    if caller.total_alloca_size is None and not caller.skip_stack_size_change:
      must_store_special.append(ctx.cfg.previous_stack_size_local)
    if caller.takes_return_address:
      must_store_special.append(ctx.cfg.return_address_local)
    if caller.is_variadic:
      must_store_special.append(ctx.cfg.vararg_ptr_local)
    if ctx.cfg.use_branch_jump_table:
      must_store_special.append(ctx.cfg.branch_jump_table_addr_local)

    for n in must_store_special:
      must_store.append(localizeVar(n, False, bctx))
      must_store_sizes.append(1)

    # If we don't need to store any parameters for later we don't need to do anything special when we recurse
    poss_recursive = len(must_store) > 0

  return_addr_id = return_proc_name = None
  if callee.returns_to_address:
    return_proc_name = localizeCallId(bctx.next_call_id, bctx.label, caller.name)
  if callee.takes_return_address:
    assert return_proc_name is not None
    return_addr_id = callee.return_addresses.index(return_proc_name)

  if not poss_recursive: # TODO OPTI: this can also be used if possibly recusive but we don't depend on anything after
    arguments, arg_value_blocks = getCallArguments(args, vararg_ptr, return_addr_id, ctx, bctx)

    if callee.returns_to_address:
      # Make sure parameters can be accessed later
      bctx.code.add(assignParameters(
        bctx.available_params, bctx.available_param_sizes,
        next_var_use.depends | ctx.cfg.special_locals,
        ctx
      ))

    call_blocks, assign_blocks = transSimpleCall(callee_name, arguments, result, result_size, ctx)
    bctx.code.add(arg_value_blocks)
    bctx.code.add(call_blocks)

    if not callee.returns_to_address:
      bctx.code.add(assign_blocks)
    else:
      # If the function we called returns to an address, it will call our function
      # back when it is done
      assert not ctx.cfg.use_branch_jump_table

      # Start new block list
      ctx.proj.code.append(bctx.code)
      bctx.available_params = []
      bctx.available_param_sizes = []

      # Add code for callback
      assert return_proc_name is not None
      bctx.code, ctx = getUncheckedProcedureStart(return_proc_name, [], [], caller, ctx,
                                                  is_counted=callee.returns_to_address)
      bctx.code.add(assign_blocks)
  else:
    if not callee.returns_to_address and not ctx.cfg.use_branch_jump_table:
      # Use the parameters for procedures to use scratch's stack to store any variables needed later
      recurse_proc_name = localizeCallId(bctx.next_call_id, bctx.label, caller.name, True)
      bctx.code.add(sb3.ProcedureCall(recurse_proc_name, [var.getValue() for var in must_store]))

      for i, var in enumerate(must_store):
        must_store[i].var_type = "param"

      # Start new block list
      ctx.proj.code.append(bctx.code)
      # Make sure that these parameters are assigned back to variables if needed later
      bctx.available_params = must_store
      bctx.available_param_sizes = must_store_sizes
      bctx.code = sb3.BlockList([sb3.ProcedureDef(recurse_proc_name, [var.getRawVarName() for var in must_store])])

      arguments, arg_value_blocks = getCallArguments(args, vararg_ptr, return_addr_id, ctx, bctx)
      call_blocks, assign_blocks = transSimpleCall(callee_name, arguments, result, result_size, ctx)
      bctx.code.add(arg_value_blocks)
      bctx.code.add(call_blocks)
      bctx.code.add(assign_blocks)
    else:
      # Store variables that will be needed later on the local var stack
      # We cannot save these as parameters because the stack might reset,
      # or this is a jump table and everything must be able to be put
      # into the same function
      total_size = sum(must_store_sizes)

      bctx.code.add(sb3.EditVar("change", ctx.cfg.local_stack_size_var, sb3.Known(total_size)))

      offset = 0
      for i, (var, size) in enumerate(zip(must_store, must_store_sizes)):
        val = var.getValue() if size == 1 else var.getAllValues(size)
        bctx.code.add(storeOnStack(ctx.cfg.local_stack_var, ctx.cfg.local_stack_size_var, -offset - (size - 1), size, val))
        offset += size
        must_store[i].var_type = "var"

      arguments, arg_value_blocks = getCallArguments(args, vararg_ptr, return_addr_id, ctx, bctx)

      call_blocks, assign_blocks = transSimpleCall(callee_name, arguments, result, result_size, ctx)
      bctx.code.add(arg_value_blocks)
      bctx.code.add(call_blocks)

      # Only create a return address target if it should be created
      if callee.returns_to_address:
        assert return_proc_name is not None
        ctx.proj.code.append(bctx.code)
        bctx.available_params = []
        bctx.available_param_sizes = []

        # Add code for callback
        bctx.code, ctx = getUncheckedProcedureStart(return_proc_name, [], [], caller, ctx,
                                                    is_counted=callee.returns_to_address)

      bctx.code.add(assign_blocks)

      offset = 0
      for var, size in zip(must_store, must_store_sizes):
        val = loadFromStack(ctx.cfg.local_stack_var, ctx.cfg.local_stack_size_var, -offset - (size - 1), size)
        if isinstance(val, sb3.Value):
          bctx.code.add(var.setValue(val))
        else:
          bctx.code.add(var.setAllValues(val))
        offset += size

      bctx.code.add(sb3.EditVar("change", ctx.cfg.local_stack_size_var, sb3.Known(-total_size)))

  # Deallocate vararg memory
  if callee.is_variadic and total_alloc_size != 0:
    bctx.code.add(sb3.EditVar("change", ctx.cfg.stack_pointer_var, sb3.Known(total_alloc_size)))

  bctx.next_call_id += 1

  return ctx, bctx

def transInstr(instr: ir.Instr, ctx: Context, bctx: BlockInfo) -> tuple[sb3.BlockList, Context, BlockInfo]:
  blocks = sb3.BlockList()
  match instr:
    case ir.Alloca(): # Allocate space on the stack and return ptr
      assert isinstance(instr.num_elements, ir.KnownIntVal)
      assert instr.num_elements.value == 1

      var = transVar(instr.result, bctx)
      assert var is None or var.var_type != "param"

      blocks = sb3.BlockList()
      # Include padding in allocation space to prevent bytes being overriden (i.e. with memcpy)
      size = getByteSize(instr.allocated_type, include_padding=ctx.cfg.accurate_byte_spacing)

      assert bctx.label is not None
      if bctx.fn.skip_stack_size_change:
        # If we skip increasing the stack pointer, don't subtract to componsate for the extra memory
        offset = 0
      elif bctx.fn.total_alloca_size is None:
        # Go back to the offset before adding the total size
        offset = -bctx.fn.block_alloca_size[bctx.label]
      else:
        offset = -bctx.fn.total_alloca_size
      offset += bctx.allocated + size - 1 # Add size so that there is enough space. For example if stack pointer is 100
                                          # and we want to allocate 10 bytes, then change stack pointer to 90 to reserve
                                          # bytes 91, 92, ..., 100. Subtract one to allow some allocations to be set to
                                          # the stack pointer to skip an add as well as include the inital stack pointer
      bctx.allocated += size

      # Negate offset as the stack grows backward
      blocks.add(var.setValue(offsetStackSize(ctx.cfg.stack_pointer_var, -offset)))

    case ir.Load(): # Load a value from an address on the stack
      address = transValue(instr.address, ctx, bctx)
      blocks.add(address.blocks)

      var = transVar(instr.result, bctx)

      if isinstance(address, IdxbleValueAndBlocks):
        raise CompException(f"Address to load cannot be an indexable value in {instr}")

      blocks.add(transLoad(var, address.value, instr.loaded_type, ctx))

    case ir.Store(): # Copy a value to an address on the stack
      address = transValue(instr.address, ctx, bctx)
      blocks.add(address.blocks)
      address = address.value

      # TODO: technically the store should still happen if volatile
      if not isinstance(instr.value, ir.UndefVal):
        value = transValue(instr.value, ctx, bctx)
        blocks.add(value.blocks)
        value = value.value

        if isinstance(address, IdxbleValue):
          raise CompException(f"Address to store cannot be an indexable value in {instr}")

        blocks.add(transStore(value, address, instr.value.type, ctx))

    case ir.UnaryOp(): # Do a calculation with one value
      operand = transValue(instr.operand, ctx, bctx)
      blocks.add(operand.blocks)
      operand = operand.value

      res_var = transVar(instr.result, bctx)
      assert res_var.var_type != "param"

      assert instr.opcode == ir.UnaryOpcode.FNeg # Only fneg exists in llvm ir

      if isinstance(operand, IdxbleValue):
        raise CompException(f"Indexable value not supported in unary op {instr}")
      assert isinstance(instr.operand.type, ir.FloatingPointTy)

      blocks.add(res_var.setValue(sb3.Op("sub", sb3.Known(0), operand)))

    case ir.BinaryOp(): # Do a calculation with two values
      lft = transValue(instr.left, ctx, bctx)
      blocks.add(lft.blocks)
      lft = lft.value

      rgt = transValue(instr.right, ctx, bctx)
      blocks.add(rgt.blocks)
      rgt = rgt.value

      res_var = transVar(instr.result, bctx)
      assert res_var.var_type != "param"
      res_val = None

      match instr.opcode:
        case ir.BinaryOpcode.Add | ir.BinaryOpcode.Sub | ir.BinaryOpcode.And | ir.BinaryOpcode.Or | ir.BinaryOpcode.Xor:
          pass
        case _:
          if isinstance(lft, IdxbleValue) or \
             isinstance(rgt, IdxbleValue):
            raise CompException(f"Indexable value not supported in binop {instr}")

      match instr.opcode:
        case ir.BinaryOpcode.Add | ir.BinaryOpcode.Sub: # Add/Sub two values
          op = "add" if instr.opcode == ir.BinaryOpcode.Add else "sub"

          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add/sub only supports "
                                f"integers, got type {type(instr.left.type)}")

          width = instr.left.type.width

          if width > VARIABLE_MAX_BITS:
            assert isinstance(lft, IdxbleValue) and isinstance(rgt, IdxbleValue)

            sum = calculateSumDiff(op, lft, rgt, width, ctx)
            blocks.add(sum.blocks)
            blocks.add(res_var.setAllValues(sum.value))

            res_val = False

          else:
            # 1 variable wide addition/subtraction
            assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))

            if instr.is_nuw:
              # Cannot overflow - result is guaranteed range [0, 256) for 8 bit
              res_val = sb3.Op(op, lft, rgt)
            else:
              known, unknown, lft_is_known = (lft, rgt, True) if isinstance(lft, sb3.Known) else (rgt, lft, False)
              has_known = isinstance(known, sb3.Known)

              perf = ctx.cfg.opt_target.perf
              val_reference_cost = opt.getValueCost(lft, perf) + opt.getValueCost(rgt, perf)
              # res mod 2^N
              mod_cost = perf.mod
              # res +/- N(res </> N)
              alt_cost = perf.add + perf.mul + perf.gt + perf.add * int(not has_known) + val_reference_cost
              mod_is_faster = mod_cost < alt_cost
              mod_base: int = 2 ** width

              # Not compatible with nuw
              # e.g. 1 + -128 -> 1 + 128 (twos comp) -> 129 (twos comp) => no mod step needed
              #                  1 - 128 (twos comp) -> -127 (intemediate) -> 129 (mod 256) => mod step needed
              if ctx.cfg.compiler_minify and has_known:
                assert isinstance(known, sb3.Known)
                known_val = int(known.known)

                # a + b mod N = a + (b - N) mod N = a - (N - b) mod N
                # a - b mod N = a - (b - N) mod N = a + (N - b) mod N
                # b - a mod N = (b - N) - a mod N
                # so if b is known N - b is smaller then use it and swap operation
                # the 3rd equality is not available when mod_faster is False
                # because the min value is lower (0 - N) - N = -2N < -N
                alt_op = op
                alt_known = known_val
                if op == "add": # x + c
                  alt_op = "sub"
                  alt_known = mod_base - known_val

                elif not lft_is_known: # x - c
                  alt_op = "add"
                  alt_known = mod_base - known_val

                elif mod_is_faster is True: # c - x
                  alt_known -= mod_base

                # Choose the value that takes up less space in project.json
                if len(str(alt_known)) < len(str(known_val)):
                  known = sb3.Known(alt_known)
                  op = alt_op
                  if lft_is_known: lft = known
                  else:            rgt = known

              unwrapped = sb3.Op(op, lft, rgt)
              if mod_is_faster:
                res_val = sb3.Op("mod", unwrapped, sb3.Known(mod_base))
              else:
                # The highest magnitude adjustment we can make is N, so we never need to adjust more than once, so we can
                # replace mod with this if it is faster to do so
                comp_op, k_comp_val, adjustment = (">", mod_base - 1, -mod_base) if op == "add" else ("<", 0, mod_base)

                if not has_known:
                  if op == "add":
                    lft_comp_val = unwrapped
                    rgt_comp_val = sb3.Known(k_comp_val)
                  else:
                    # l - r < 0 -> l < r
                    assert comp_op == "<"
                    lft_comp_val = lft
                    rgt_comp_val = rgt
                else:
                  assert isinstance(known, sb3.Known)
                  known_val = int(known.known)
                  if op == "add" or (op == "sub" and not lft_is_known):
                    # x + c > N
                    # x > N - c
                    k_comp_val -= (-1) ** int(op == "sub") * known_val
                  else:
                    # c - x < N
                    # -x < N - c
                    # x > c - N
                    assert comp_op == "<"
                    k_comp_val = known_val - k_comp_val
                    comp_op = ">"
                  lft_comp_val = unknown
                  rgt_comp_val = sb3.Known(k_comp_val)

                res_val = sb3.Op("add", unwrapped,
                  sb3.Op("mul", sb3.BoolOp(comp_op, lft_comp_val, rgt_comp_val), sb3.Known(adjustment)))

        case ir.BinaryOpcode.Mul: # Multiply two values
          assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.left.type)}")
          width = instr.left.type.width

          if instr.is_nsw and instr.is_nuw and ctx.cfg.compiler_opt:
            res_val = multiplyNoWrap(width, lft, rgt)
          else:
            blocks_and_value = multiplyWrap(width, lft, rgt, ctx)
            blocks.add(blocks_and_value.blocks)
            res_val = blocks_and_value.value

        case ir.BinaryOpcode.UDiv: # Divide one value by another (unsigned)
          assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
          # TODO OPTI: optimise for known values
          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.left.type)}")

          width = instr.left.type.width
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          # Division by zero is UB
          if not instr.is_exact:
            res_val = sb3.Op("floor", sb3.Op("div", lft, rgt))
          else:
            res_val = sb3.Op("div", lft, rgt) # Value is poison if one is not a multiple of another

        case ir.BinaryOpcode.SDiv: # Divide one value by another (signed)
          assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.left.type)}")
          width = instr.left.type.width

          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports integers with <= {VARIABLE_MAX_BITS} bits")

          if instr.is_exact:
            signed_lft = undoTwosComplement(lft, width)
            signed_rgt = undoTwosComplement(rgt, width)

            res_val = twosComplement(sb3.Op("div", signed_lft, signed_rgt), width)
          else:
            lft, lblocks = flatAsTuple(optimizeValueUse(lft, 2, ctx))
            rgt, rblocks = flatAsTuple(optimizeValueUse(rgt, 2, ctx))
            blocks.add(lblocks)
            blocks.add(rblocks)

            point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
            change = 2 ** width

            # TODO: optimise for known values

            # Undo two's complement, divide, round towards zero using floor or ceiling and calculate two's complement
            blocks.add([
              sb3.ControlFlow("if_else", sb3.BoolOp("<", lft, sb3.Known(point_of_neg)), sb3.BlockList([
                sb3.ControlFlow("if_else", sb3.BoolOp("<", rgt, sb3.Known(point_of_neg)), sb3.BlockList([
                  # If left + right are pos
                  res_var.setValue(sb3.Op("floor", sb3.Op("div", lft, rgt))),
                ]), sb3.BlockList([
                  # If left is pos and right is neg
                  res_var.setValue(sb3.Op("add",
                                          sb3.Op("ceiling",
                                            sb3.Op("div",
                                              lft,
                                              sb3.Op("sub", rgt, sb3.Known(change)))),
                                          sb3.Known(change))),
                ]))
              ]), sb3.BlockList([
                sb3.ControlFlow("if_else", sb3.BoolOp("<", rgt, sb3.Known(point_of_neg)), sb3.BlockList([
                  # If left is neg and right is pos
                  res_var.setValue(sb3.Op("add",
                                          sb3.Op("ceiling",
                                            sb3.Op("div",
                                              sb3.Op("sub", lft, sb3.Known(change)),
                                              rgt)),
                                          sb3.Known(change))),
                ]), sb3.BlockList([
                  # If left + right are neg
                  res_var.setValue(sb3.Op("floor",
                                          sb3.Op("div",
                                            sb3.Op("sub", lft, sb3.Known(change)),
                                            sb3.Op("sub", rgt, sb3.Known(change))))),
                ]))
              ]))
            ])
          res_val = False # We set res_var ourselves

        case ir.BinaryOpcode.URem: # Calculate remainder (unsigned)
          assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.left.type)}")

          width = instr.left.type.width
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports"
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          # mod 0 is UB, can ignore
          res_val = sb3.Op("mod", lft, rgt)

        case ir.BinaryOpcode.SRem: # Calculate remainder (signed)
          assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
          # TODO OPTI: optimise for known values
          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.left.type)}")

          width = instr.left.type.width
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          # TODO: Reuse if statement to work out if a / b > 0
          left, lblocks = flatAsTuple(optimizeValueUse(lft, 2, ctx))
          right_is_temp = shouldOptimiseValueUse(rgt, 3, ctx)
          right, rblocks = flatAsTuple(optimizeValueUse(rgt, 3, ctx))
          if right_is_temp:
            assert isinstance(right, sb3.GetVar)
          blocks.add(lblocks)
          blocks.add(rblocks)

          point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
          change = 2 ** width

          if not right_is_temp:
            # Undo Two's Complement
            right_sub_change = sb3.Op("sub", right, sb3.Known(change))

            right_sub_change, pos_neg_block = flatAsTuple(optimizeValueUse(right, 2, ctx))
            pos_neg_block.add(sb3.BlockList([
              # If left is pos and right is neg - remainder = (l mod r) - r
              res_var.setValue(sb3.Op("sub",
                                      sb3.Op("mod",
                                        left,
                                        right_sub_change),
                                      right_sub_change))]))
          else:
            # Re-use the generated temp var
            pos_neg_block = sb3.BlockList([
              sb3.EditVar("change", right.var_name, sb3.Known(-change)),
              res_var.setValue(sb3.Op("sub",
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
                  res_var.setValue(sb3.Op("mod", left, right)),
                ]),
                # If left is pos and right is neg - remainder = (l mod r) - r
                pos_neg_block
              )
            ]), sb3.BlockList([
              sb3.ControlFlow("if_else", sb3.BoolOp("<", right, sb3.Known(point_of_neg)), sb3.BlockList([
                # If left is neg and right is pos - remainder = (l mod r) - r
                res_var.setValue(sb3.Op("add",
                                       sb3.Op("sub",
                                         sb3.Op("mod",
                                           sb3.Op("sub", left, sb3.Known(change)),
                                           right),
                                         right),
                                       sb3.Known(change))),
              ]), sb3.BlockList([
                # If left + right are neg
               res_var.setValue(sb3.Op("add",
                                       sb3.Op("mod",
                                         sb3.Op("sub", left, sb3.Known(change)),
                                         sb3.Op("sub", right, sb3.Known(change))),
                                       sb3.Known(change))),
              ]))
            ]))
          ])

          res_val = False # We set res_var ourselves

        case ir.BinaryOpcode.Shl: # Calculate left shift
          assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.left.type)}")

          width = instr.left.type.width
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          can_shift_out = not (instr.is_nsw and instr.is_nuw)
          res_val, ctx = bitShift("left", width, lft, rgt, ctx, can_shift_out)

        case ir.BinaryOpcode.LShr: # Calculate right shift (unsigned)
          assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.left.type)}")

          width = instr.left.type.width
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          can_shift_out = not instr.is_exact
          res_val, ctx = bitShift("right", width, lft, rgt, ctx, can_shift_out)

        case ir.BinaryOpcode.AShr: # Calculate right shift (signed)
          assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.left.type)}")

          width = instr.left.type.width
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
          change = 2 ** width

          right_mul, ctx = intPow2(rgt, ctx)

          unwrapped_pos = sb3.Op("div", lft, right_mul)
          val_pos = unwrapped_pos if instr.is_exact else sb3.Op("floor", unwrapped_pos)

          unwrapped_neg = sb3.Op("div", sb3.Op("sub", lft, sb3.Known(change)), right_mul)
          val_neg = sb3.Op("add",
                      unwrapped_neg if instr.is_exact else sb3.Op("ceiling", unwrapped_neg),
                      sb3.Known(change))

          blocks.add([
            sb3.ControlFlow("if_else", sb3.BoolOp("<", lft, sb3.Known(point_of_neg)), sb3.BlockList([
              res_var.setValue(val_pos),
            ]), sb3.BlockList([
              res_var.setValue(val_neg),
            ])),
          ])

          res_val = False # We set res_var ourselves

        case ir.BinaryOpcode.And | ir.BinaryOpcode.Or | ir.BinaryOpcode.Xor: # Calculate binary operation
          if not isinstance(instr.left.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.left.type)}")

          width = instr.left.type.width

          match instr.opcode:
            case ir.BinaryOpcode.And: op = "and"
            case ir.BinaryOpcode.Or:  op = "or"
            case ir.BinaryOpcode.Xor: op = "xor"

          if width > VARIABLE_MAX_BITS:
            assert isinstance(lft, IdxbleValue) and isinstance(rgt, IdxbleValue)

            res_vals = IdxbleValue()
            for i in range(len(lft.vals)):
              val, ctx = binOp(op, lft.vals[i], rgt.vals[i], width, ctx, instr.is_disjoint)
              res_vals.vals.append(val)

            blocks.add(res_var.setAllValues(res_vals))
            res_val = False

          else:
            assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
            res_val, ctx = binOp(op, lft, rgt, width, ctx, instr.is_disjoint)

        case ir.BinaryOpcode.FAdd | ir.BinaryOpcode.FSub | \
             ir.BinaryOpcode.FMul | ir.BinaryOpcode.FDiv: # Basic float operations
           assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))

           op_lookup: dict[ir.BinaryOpcode, Literal["add", "sub", "mul", "div"]] = {
            ir.BinaryOpcode.FAdd: "add", ir.BinaryOpcode.FSub: "sub",
            ir.BinaryOpcode.FMul: "mul", ir.BinaryOpcode.FDiv: "div"
           }

           if not isinstance(instr.left.type, ir.FloatingPointTy):
             raise CompException(f"Instruction {instr} with opcode add only supports "
                                 f"floats, got type {type(instr.left.type)}")

           res_val = sb3.Op(op_lookup[instr.opcode], lft, rgt)

        case ir.BinaryOpcode.FRem: # Float remainder
          assert not (isinstance(lft, IdxbleValue) or isinstance(rgt, IdxbleValue))
          if not isinstance(instr.left.type, ir.FloatingPointTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"floats, got type {type(instr.left.type)}")

          if not (isinstance(lft, sb3.Known) or isinstance(rgt, sb3.Known)):
            # Result is negative if left/right have different signs
            cond = sb3.BoolOp("<", sb3.Op("mul", lft, rgt), sb3.Known(0))
          else:
            known, unknown = (lft, rgt) if isinstance(lft, sb3.Known) else (rgt, lft)
            assert isinstance(known, sb3.Known)

            if sb3.scratchCastToNum(known) > 0:
              # Result is negative if unknown is negative
              cond = sb3.BoolOp("<", unknown, sb3.Known(0))
            else:
              # Result is negative if unknown is positive
              cond = sb3.BoolOp(">", unknown, sb3.Known(0))

          res_val = sb3.Op("sub",
            sb3.Op("mod", lft, rgt),
            sb3.Op("mul", rgt, cond))

        case _:
          raise CompException(f"Unknown instruction opcode {instr} (type BinOp)")

      if res_val is not False: # If the binop sets res_var itself
        blocks.add(res_var.setValue(res_val))

    case ir.Conversion(): # Convert a value from one type to another
      value = transValue(instr.value, ctx, bctx)
      blocks.add(value.blocks)
      value = value.value

      from_ty, to_ty = instr.value.type, instr.res_type

      res_var = transVar(instr.result, bctx)
      assert res_var.var_type != "param"

      assert isinstance(value, sb3.Value)

      match instr.opcode:
        case ir.ConvOpcode.Trunc | ir.ConvOpcode.ZExt | ir.ConvOpcode.SExt | \
             ir.ConvOpcode.UIToFP | ir.ConvOpcode.SIToFP | ir.ConvOpcode.IntToPtr:
          if not isinstance(instr.value.type, ir.IntegerTy):
            raise CompException(f"Instruction {instr} with opcode add only supports "
              f"integers, got type {type(instr.value.type)}")

          # TODO FIX: support larger values
          if instr.value.type.width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

        case ir.ConvOpcode.FPTrunc | ir.ConvOpcode.FPExt:
          if not isinstance(instr.value.type, ir.FloatingPointTy):
            raise CompException(f"Instruction {instr} only supports "
                                f"floats, got type {type(instr.value.type)}")

        case ir.ConvOpcode.FPToUI | ir.ConvOpcode.FPToSI:
          if not isinstance(instr.value.type, ir.FloatingPointTy):
            raise CompException(f"Instruction {instr} only supports "
                                f"floats, got type {type(instr.value.type)}")

          assert isinstance(to_ty, ir.IntegerTy)
          if to_ty.width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports converting to "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

        case ir.ConvOpcode.PtrToInt | ir.ConvOpcode.PtrToAddr:
          assert isinstance(instr.value.type, ir.PointerTy)
          assert isinstance(instr.res_type, ir.IntegerTy)
          if instr.res_type.width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports converting to "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

        case ir.ConvOpcode.BitCast:
          if isinstance(instr.value.type, ir.IntegerTy):
            assert isinstance(instr.res_type, ir.IntegerTy)
            if instr.value.type.width > VARIABLE_MAX_BITS:
              raise CompException(f"Instruction icmp currently supports "
                                  f"integers with <= {VARIABLE_MAX_BITS} bits")
          elif isinstance(instr.value.type, ir.PointerTy):
            assert isinstance(instr.res_type, ir.PointerTy)
          # FUTURE FIX: float -> int and vice versa bitcasts, would require the IEEE
          # representation in bits

        case _: pass

      res_val = None

      match instr.opcode:
        case ir.ConvOpcode.Trunc:
          assert isinstance(to_ty, ir.IntegerTy)
          res_val = sb3.Op("mod", value, sb3.Known(2 ** to_ty.width))

        case ir.ConvOpcode.SExt:
          assert isinstance(from_ty, ir.IntegerTy) and isinstance(to_ty, ir.IntegerTy)
          from_bits, to_bits = from_ty.width, to_ty.width

          # Two's complement
          limit = 2 ** (from_bits - 1) - 1
          is_neg = sb3.BoolOp(">", value, sb3.Known(limit))

          # e.g. i2 -> i4: 1111 - 0011 = 1100
          diff = (2 ** to_bits - 1) - (2 ** from_bits - 1)
          res_val = sb3.Op("add", value, sb3.Op("mul", sb3.Known(diff), is_neg))

        case ir.ConvOpcode.FPToUI:
          res_val = sb3.Op("floor", value)

        case ir.ConvOpcode.FPToSI:
          assert isinstance(to_ty, ir.IntegerTy)
          res_val = sb3.Op("mod",
            sb3.Op("mul",
              sb3.Op("floor", sb3.Op("abs", value)),
              sb3.Op("sub",
                sb3.Op("mul",
                  sb3.BoolOp(">", value, sb3.Known(0)),
                  sb3.Known(2)),
                sb3.Known(1))),
            sb3.Known(2 ** to_ty.width))

        case ir.ConvOpcode.SIToFP:
          assert isinstance(from_ty, ir.IntegerTy)
          res_val = undoTwosComplement(value, from_ty.width)

        case ir.ConvOpcode.PtrToInt | ir.ConvOpcode.PtrToAddr | ir.ConvOpcode.IntToPtr:
          # TODO: truncation and zero extension if int type is different width to pointer
          int_ty = instr.value.type if instr.opcode == ir.ConvOpcode.IntToPtr else instr.res_type
          assert isinstance(int_ty, ir.IntegerTy)
          assert int_ty.width == PTR_WIDTH_BITS

          # No-op
          res_val = value

        case ir.ConvOpcode.ZExt | ir.ConvOpcode.FPTrunc | ir.ConvOpcode.FPExt | \
             ir.ConvOpcode.UIToFP | ir.ConvOpcode.BitCast:
          # No-op
          res_val = value

        case _:
          raise CompException(f"Unknown instruction opcode {instr} (type Conversion)")

      assert res_val is not None
      blocks.add(res_var.setValue(res_val))

    case ir.ICmp(): # Compare two values
      lft = transValue(instr.left, ctx, bctx)
      blocks.add(lft.blocks)
      lft = lft.value

      rgt = transValue(instr.right, ctx, bctx)
      blocks.add(rgt.blocks)
      rgt = rgt.value

      res_var = transVar(instr.result, bctx)
      assert res_var.var_type != "param"

      if not (isinstance(instr.left.type, ir.IntegerTy) or isinstance(instr.left.type, ir.PointerTy)):
        raise CompException(f"Instruction {instr} with opcode add only supports "
                            f"integers or pointers, got type {type(instr.left.type)}")

      width = instr.left.type.width if isinstance(instr.left.type, ir.IntegerTy) else PTR_WIDTH_BITS
      # TODO FIX: support larger values
      if width > VARIABLE_MAX_BITS:
        raise CompException(f"Instruction icmp currently supports "
                            f"integers with <= {VARIABLE_MAX_BITS} bits")

      assert isinstance(lft, sb3.Value) and isinstance(rgt, sb3.Value)

      result = intCompare(lft, rgt, width, instr.cond, ctx.cfg)

      # Bool to float will cast to an int if needed (so the bool is treated as 1 instead of 'true')
      casted_result = sb3.Op("bool_to_float", result)

      blocks.add(res_var.setValue(casted_result))

    case ir.FCmp(): # Compare two float values
      lft = transValue(instr.left, ctx, bctx)
      blocks.add(lft.blocks)
      lft = lft.value

      rgt = transValue(instr.right, ctx, bctx)
      blocks.add(rgt.blocks)
      right = rgt.value

      res_var = transVar(instr.result, bctx)
      assert res_var.var_type != "param"

      if not isinstance(instr.left.type, ir.FloatingPointTy):
        raise CompException(f"Instruction {instr} with opcode add only supports "
                            f"floats, got type {type(instr.left.type)}")
      assert isinstance(lft, sb3.Value) and isinstance(right, sb3.Value)

      # NaN > Infinity in scratch
      is_nan = lambda val: sb3.BoolOp(">", val, sb3.Known(float("inf")))
      # Use slower lexical string comparison (Infinity > j > NaN due to alphabetical order)
      is_not_nan = lambda val: sb3.BoolOp("<", val, sb3.Known("j"))

      match instr.cond:
        case ir.FCmpCond.TrueCond | ir.FCmpCond.FalseCond:
          res_val = sb3.KnownBool(instr.cond is ir.FCmpCond.TrueCond)

        case ir.FCmpCond.Oeq:
          # Anything = NaN is false in scratch apart from NaN = NaN - only need to check one side
          res_val = sb3.BoolOp("and", is_not_nan(right), sb3.BoolOp("=", lft, right))

        case ir.FCmpCond.One | ir.FCmpCond.Ogt | ir.FCmpCond.Oge | ir.FCmpCond.Olt | ir.FCmpCond.Ole:
          op = "=" if instr.cond is ir.FCmpCond.One else \
              (">" if instr.cond in {ir.FCmpCond.Ogt, ir.FCmpCond.Ole} else "<")
          inverted = instr.cond in {ir.FCmpCond.One, ir.FCmpCond.Oge, ir.FCmpCond.Ole}
          invert_val_if_necessary = lambda val: sb3.BoolOp("not", val) if inverted else val

          res_val = sb3.BoolOp("and",
            sb3.BoolOp("and", is_not_nan(lft), is_not_nan(right)),
            invert_val_if_necessary(sb3.BoolOp(op, lft, right)))

        case ir.FCmpCond.Uno:
          res_val = sb3.BoolOp("and", is_not_nan(lft), is_not_nan(right))

        case ir.FCmpCond.Ueq | ir.FCmpCond.Une | ir.FCmpCond.Ugt | ir.FCmpCond.Uge | ir.FCmpCond.Ult | ir.FCmpCond.Ule:
          op = "=" if instr.cond in {ir.FCmpCond.Ueq, ir.FCmpCond.Une} else \
              (">" if instr.cond in {ir.FCmpCond.Ogt, ir.FCmpCond.Ole} else "<")
          inverted = instr.cond in {ir.FCmpCond.One, ir.FCmpCond.Oge, ir.FCmpCond.Ole}
          invert_val_if_necessary = lambda val: sb3.BoolOp("not", val) if inverted else val

          res_val = sb3.BoolOp("or",
            sb3.BoolOp("or", is_nan(lft), is_nan(right)),
            invert_val_if_necessary(sb3.BoolOp(op, lft, right)))

        case ir.FCmpCond.Ord:
          res_val = sb3.BoolOp("or", is_nan(lft), is_nan(right))

        case _:
          raise CompException(f"fcmp does not support comparsion mode {instr.cond}")

      # Bool to float will cast to an int if needed (so the bool isn't treated as 'true')
      res_val = sb3.Op("bool_to_float", res_val)

      blocks.add(res_var.setValue(res_val))

    case ir.Select(): # Select between two values based on a condition
      cond = transValue(instr.cond, ctx, bctx)
      true_val = transValue(instr.true_value, ctx, bctx)
      false_val = transValue(instr.false_value, ctx, bctx)

      assert isinstance(instr.cond.type, ir.IntegerTy)
      assert instr.cond.type.width == 1
      assert isinstance(cond.value, sb3.Value)

      res_var = transVar(instr.result, bctx)
      assert res_var.var_type != "param"

      blocks.add(cond.blocks)

      if isinstance(true_val, ValueAndBlocks) and isinstance(true_val.value, sb3.Known) and \
         isinstance(false_val.value, sb3.Known) and isinstance(true_val.value.known, float) and \
         isinstance(false_val.value.known, float):
        # TODO: Support multi-width vars for this, if it is faster than the if method
        assert len(true_val.blocks) == 0 and len(false_val.blocks) == 0

        # Result = false_value + (true_value - false_value) * cond
        diff = true_val.value.known - false_val.value.known
        offset = cond.value if diff == 1 else sb3.Op("mul", cond.value, sb3.Known(diff))
        blocks.add(res_var.setValue(sb3.Op("add", false_val.value, offset)))
      else:
        true_blocks = true_val.blocks
        false_blocks = false_val.blocks
        if isinstance(true_val, ValueAndBlocks):
          assert isinstance(false_val, ValueAndBlocks)
          true_blocks.add(res_var.setValue(true_val.value))
          false_blocks.add(res_var.setValue(false_val.value))
        else:
          assert isinstance(true_val, IdxbleValueAndBlocks)
          assert isinstance(false_val, IdxbleValueAndBlocks)
          true_blocks.add(res_var.setAllValues(true_val.value))
          false_blocks.add(res_var.setAllValues(false_val.value))

        blocks.add(sb3.ControlFlow("if_else", sb3.BoolOp("=", cond.value, sb3.Known(1)), true_blocks, false_blocks))

    case ir.Phi(): # Choose a value based on which block control came from
      pass # Phi assignments are dealt with in the brancher function

    case ir.GetElementPtr(): # Offset a pointer to get an address in an array/struct
      base_ptr = transValue(instr.base_ptr, ctx, bctx)
      assert isinstance(base_ptr, ValueAndBlocks)
      blocks.add(base_ptr.blocks)

      indices: list[tuple[sb3.Value, int]] = []
      for index_val in instr.indices:
        assert isinstance(index_val.type, ir.IntegerTy)
        index = transValue(index_val, ctx, bctx)
        assert isinstance(index, ValueAndBlocks)
        blocks.add(index.blocks)
        indices.append((index.value, index_val.type.width))

      known_offset, unknown_offsets = getGepOffsets(instr.base_ptr_type, indices, ctx)

      offset_ptr = applyGepOffsets(base_ptr.value, known_offset, unknown_offsets, instr.is_nuw, ctx)

      res_var = transVar(instr.result, bctx)
      assert res_var.var_type != "param"
      blocks.add(res_var.setValue(offset_ptr))

    case ir.Freeze(): # Freeze a value to replace poison/undef with a valid value
      # Essentially a no-op, but for some types (e.g. integers)
      # poison can exist in invalid states e.g. -3 or 1.2 or NaN
      value = transValue(instr.value, ctx, bctx)
      blocks.add(value.blocks)
      value = value.value
      assert isinstance(value, sb3.Value)

      res_var = transVar(instr.result, bctx)
      assert res_var.var_type != "param"

      match instr.value.type:
        case ir.IntegerTy() | ir.PointerTy():
          width = instr.value.type.width if isinstance(instr.value.type, ir.IntegerTy) \
            else PTR_WIDTH_BITS

          # Works with NaN, Infinity, invalid values (mod turns infinity to NaN but
          # floor doesn't, so doesn't work other way around)
          res_val = res_var.setValue(sb3.Op("floor", sb3.Op("mod", value, sb3.Known(2 ** width))))

        case ir.FloatingPointTy():
          # No-op
          res_val = res_var.setValue(value)

        case _:
          raise CompException(f"Instruction {instr} with opcode add only supports "
                              f"integers and floats, got type {type(instr.value.type)}")

      blocks.add(res_val)

    case ir.Call(): # Call a function
      pass # Calls are handled in transFuncs

    case ir.VaArg(): # Load an argument in a va_list and point the va_list to the next argument
      arglist = transValue(instr.arglist, ctx, bctx)
      blocks.add(arglist.blocks)
      arglist = arglist.value

      res_var = transVar(instr.result, bctx)

      assert isinstance(instr.arglist.type, ir.PointerTy)
      assert isinstance(arglist, sb3.Value)

      arg_ptr = sb3.GetOfList("atindex", ctx.cfg.mem_var, arglist)
      arg_ptr, opt_blocks = flatAsTuple(optimizeValueUse(arg_ptr, 2, ctx))
      arg_size = sb3.Known(getByteSize(instr.argty, include_padding=ctx.cfg.accurate_byte_spacing))

      blocks.add(opt_blocks)
      blocks.add(transLoad(res_var, arg_ptr, instr.argty, ctx))
      blocks.add(sb3.EditList("replaceat", ctx.cfg.mem_var, arglist, sb3.Op("add", arg_ptr, arg_size)))

    case _:
      raise CompException(f"Unsupported instruction opcode {instr} (type {type(instr)})")

  return blocks, ctx, bctx

def getTerminatorInstrLabels(instr: ir.Instr) -> set[str]:
  """
  Returns every label a terminator instruction could branch to.
  Returns the string "ret" to indictate a return
  """
  match instr:
    case ir.Unreachable():
      return set()
    case ir.Ret():
      return {"ret"}
    case ir.UncondBr():
      return {instr.branch.label}
    case ir.CondBr():
      return {instr.branch_true.label, instr.branch_false.label}
    case ir.Switch():
      branches = {instr.branch_default.label}
      for _, label in instr.branch_table:
        branches.add(label.label)
      return branches
    case _:
      raise CompException(f"Unsupported terminator instruction "
                          f"opcode {instr} (type {type(instr)})")

def getInstrValues(instr: ir.Instr, include_called_funcs: bool=True) -> list[ir.Value]:
  match instr:
    case ir.Unreachable() | ir.Alloca() | ir.Phi():
      return []
    case ir.Ret() | ir.Conversion() | ir.Freeze():
      return [] if instr.value is None else [instr.value]
    case ir.Load():
      return [instr.address]
    case ir.Store():
      return [instr.address, instr.value]
    case ir.Call():
      return [instr.func, *instr.args] if include_called_funcs else [*instr.args]
    case ir.UnaryOp():
      return [instr.operand]
    case ir.BinaryOp() | ir.ICmp() | ir.FCmp():
      return [instr.left, instr.right]
    case ir.Br() | ir.Switch():
      return [instr.cond] if isinstance(instr, (ir.CondBr, ir.Switch)) else []
    case ir.Select():
      return [instr.cond, instr.true_value, instr.false_value]
    case ir.GetElementPtr():
      return [instr.base_ptr, *instr.indices]
    case ir.ExtractValue():
      return [instr.agg, *instr.indices]
    case ir.InsertValue():
      return [instr.agg, instr.element, *instr.indices]
    case ir.VaArg():
      return [instr.arglist]
    case _:
      raise CompException(f"Unknown instruction {instr} (type {type(instr)})")

def getInstrVarUse(instr: ir.Instr,
    block_phi_info: defaultdict[str, list[tuple[Variable, ir.Value]]]
  ) -> tuple[set[str], set[str], dict[str, int]]:
  """Returns what the instruction depends on and modifies, and the var sizes of what it depends on"""
  depends: set[str] = set()
  modifies: set[str] = set()
  depends_var_sizes: dict[str, int] = {}

  vals = getInstrValues(instr)

  if isinstance(instr, (ir.HasResult, ir.MaybeHasResult)):
    if instr.result is not None:
      modifies.add(instr.result.name)

  # Extend used values to values downstream of the branch instr
  # n.b. For phi, even though it might depend on a value, the
  # branch instruction is made responsible instead
  if isinstance(instr, (ir.Br, ir.Switch)):
    poss_labels = getTerminatorInstrLabels(instr) - {"ret"}
    for label in poss_labels:
      vals.extend([value for _, value in block_phi_info[label]])

  for val in vals:
    match val:
      case ir.ArgumentVal() | ir.LocalVarVal():
        depends.add(val.name)
        depends_var_sizes[val.name] = getByteSize(val.type)
      case ir.KnownVal():
        pass
      case _:
        raise CompException(f"Unknown Value: {val}")

  return depends, modifies, depends_var_sizes

def getBlockVarUse(instrs: list[ir.Instr],
                   block_phi_info: defaultdict[str, list[tuple[Variable, ir.Value]]],
                   block_var_use: dict[str, BlockVarUse] | None = None,
                   starting_var_use: BlockVarUse | None = None) -> BlockVarUse:
  """
  Accepts a list of instructions. The last should be an terminator instruction.
  Also accepts information about what other branches depend on/use which it will
  apply to all possible branches.
  """
  if len(instrs) == 0: return BlockVarUse()

  res = BlockVarUse() if starting_var_use is None else starting_var_use

  for instr in instrs:
    instr_depends, instr_modifies, instr_depends_var_sizes = getInstrVarUse(instr, block_phi_info)
    # If we modify something before using it then we don't depend on it
    res.depends |= instr_depends - res.modifies
    res.modifies |= instr_modifies
    res.depends_var_sizes.update(instr_depends_var_sizes)

  res.branches = getTerminatorInstrLabels(instrs[-1]) - {"ret"}
  if block_var_use is not None:
    for label in res.branches:
      res.depends |= block_var_use[label].depends - res.modifies
      res.modifies |= block_var_use[label].modifies
      res.depends_var_sizes.update(block_var_use[label].depends_var_sizes)

  return res

def getFuncBranchesVarUse(func: ir.Function,
    phi_info: defaultdict[str, defaultdict[str, list[tuple[Variable, ir.Value]]]]
  ) -> dict[str, BlockVarUse]:

  entrypoint = list(func.blocks.keys())[0]

  total_depends_var_sizes: dict[str, int] = {}
  label_info: dict[str, util.NodeInfo] = {}
  for name, block in func.blocks.items():
    var_use = getBlockVarUse(block.instrs, phi_info[name])
    total_depends_var_sizes.update(var_use.depends_var_sizes)
    label_info[name] = util.NodeInfo(
      depends=var_use.depends,
      modifies=var_use.modifies,
      calls=var_use.branches,
      direct_modifies=deepcopy(var_use.modifies),
      direct_calls=deepcopy(var_use.branches),
    )

  ana = util.CallGraphAnalysis(entrypoint, label_info)
  ana.analyze()

  return {
    name: BlockVarUse(
      info.depends, info.modifies, info.calls, total_depends_var_sizes
    ) for name, info in ana.info.items()
  }

def getValueFuncPtrRefs(value: ir.Value, global_names: list[str]) -> set[str]:
  match value:
    case ir.FunctionVal():
      return {value.name}
    case ir.KnownArrVal() | ir.KnownStructVal() | ir.KnownVecVal():
      return set().union(*(getValueFuncPtrRefs(mem, global_names) for mem in value.values))
    case ir.ConstExprVal():
      return set().union(*(getValueFuncPtrRefs(val, global_names) for val in getInstrValues(value.expr, False)))
    case _:
      # Nothing else contains a func ptr or nested values
      return set()

def getFuncPtrRefs(mod: ir.Module) -> list[tuple[ir.FuncTy, list[str]]]:
  global_names = list(mod.global_vars.keys())
  all_refs = set()

  for glob in mod.global_vars.values():
    all_refs |= getValueFuncPtrRefs(glob.init, global_names)

  for func in mod.functions.values():
    for block in func.blocks.values():
      for instr in block.instrs:
        for val in getInstrValues(instr, False):
          all_refs |= getValueFuncPtrRefs(val, global_names)

  # Sort alphabetically to get a consistent compiled result
  sorted_refs = sorted(list(all_refs))

  func_ptrs: list[tuple[ir.FuncTy, list[str]]] = []
  for fn_name in sorted_refs:
    fn = mod.functions[fn_name]
    signature = ir.FuncTy(fn.return_type, [arg.type for arg in fn.params], fn.variadic)
    found = -1
    for index, (signature_other, _) in enumerate(func_ptrs):
      if signature == signature_other:
        found = index
        break
    if found == -1:
      func_ptrs.append((signature, [fn_name]))
    else:
      func_ptrs[found][1].append(fn_name)

  return func_ptrs

def getFuncPtrAddr(ptr: str, ctx: Context) -> int:
  addr = ctx.min_func_ptr_addr
  for (_, ptrs) in ctx.fn_ptr_sigs:
    if ptr in ptrs:
      return addr + ptrs.index(ptr)
    addr += len(ptrs)

  raise CompException(f"Could not find function pointer for {ptr}")

def getFuncPtrSignatureInfo(signature: ir.FuncTy, caller_name: str, ctx: Context,
    warn_no_matching: bool=True
  ) -> tuple[int, list[str]] | None:
  """
  Returns (func ptr signature id, func ptrs) for a func ptr signature (inclusive)
  Returns None if no matching signature was found - this is valid because the path
  may not necessarily be followed
  warn_no_matching - whether to warn if there is no matching signature
  """
  for signature_id, (signature_other, ptrs) in enumerate(ctx.fn_ptr_sigs):
    if signature_other == signature:
      return signature_id, ptrs

  # TODO config to disable these warnings
  if warn_no_matching and caller_name not in ctx.cfg.no_warn_missing_fn_sig:
    warnings.warn(f"Could not find function signature for {signature} in {caller_name}", CompWarning)
  return None

def transReturnAddr(return_address: sb3.Value, info: FuncInfo | FuncPtrSigInfo, ctx: Context) -> sb3.BlockList:
  """
  Returns instructions to return back to the caller function from the "address" it passed in
  """
  assert info.returns_to_address

  blocks = sb3.BlockList()
  if info.takes_return_address:
    return_to_addr_code = {}
    for i, addr in enumerate(info.return_addresses):
      return_to_addr_code[i] = sb3.BlockList([sb3.ProcedureCall(addr, [])])

    return_table = binarySearch(return_address, return_to_addr_code, are_branches_sorted=True)
    blocks.add(return_table)
  elif len(info.return_addresses) > 0:
    # If the function doesn't take a return address then it must always return to the same place
    assert len(info.return_addresses) == 1
    blocks.add(sb3.BlockList([sb3.ProcedureCall(info.return_addresses[0], [])]))

  blocks.end = True

  return blocks

def transTerminatorInstr(instr: ir.Instr,
                         ctx: Context, bctx: BlockInfo) -> tuple[sb3.BlockList, Context]:
  # Work out what variables might be depended on in future
  poss_branch = getTerminatorInstrLabels(instr) - {"ret"}
  poss_depends: set[str] = set()
  for branch in poss_branch:
    poss_depends |= bctx.fn.block_var_use[branch].depends
  poss_depends |= ctx.cfg.special_locals

  assert bctx.label is not None
  phi_info = bctx.fn.phi_info[bctx.label]

  blocks = sb3.BlockList()
  match instr:
    case ir.Unreachable(): # Never reached if not UB
      # TODO config to disable these errors
      blocks.add(scratchRuntimeError("Reached unreachable branch"))
      blocks.end = True

    case ir.Ret(): # Return from a func
      if instr.value is not None:
        value = transValue(instr.value, ctx, bctx)
        blocks.add(value.blocks)

        return_var = Variable(ctx.cfg.return_var, "special_var", None)
        if isinstance(value.value, sb3.Value):
          blocks.add(return_var.setValue(value.value))
        else:
          blocks.add(return_var.setAllValues(value.value))

        # Indicate to the optimizer that more numbered return values have been used which should not be optimized away
        return_size = getByteSize(instr.value.type)
        ctx.highest_return_size = return_size if ctx.highest_return_size is None else \
                                  max(ctx.highest_return_size, return_size)

      # Change the stack size after setting the return value because some return values might be optimized to use
      # the stack size var
      if not bctx.fn.skip_stack_size_change:
        if bctx.fn.total_alloca_size is not None:
          if bctx.fn.total_alloca_size != 0:
            blocks.add(sb3.EditVar("change", ctx.cfg.stack_pointer_var, sb3.Known(bctx.fn.total_alloca_size)))
        else:
          blocks.add(sb3.EditVar("set", ctx.cfg.stack_pointer_var, localizeVar(ctx.cfg.previous_stack_size_local,
                                                                            False, bctx).getValue()))

      # Branch jump table does not use a global jump table
      if bctx.fn.name == ctx.cfg.entrypoint and not ctx.cfg.use_branch_jump_table:
        # Exit the program
        blocks.add(sb3.EditVar("set", ctx.cfg.jump_table_id_var, sb3.Known(EXIT_CALL_ID)))

        # TODO allow calling the entrypoint function with a return address corresponding to
        # exiting the function
        assert not bctx.fn.takes_return_address
        assert (not bctx.fn.returns_to_address) or len(bctx.fn.return_addresses) == 0

      if bctx.fn.returns_to_address:
        return_addr = localizeVar(ctx.cfg.return_address_local, False, bctx)
        blocks.add(transReturnAddr(return_addr.getValue(), bctx.fn, ctx))

      # Need to escape the forever loop
      if ctx.cfg.use_branch_jump_table:
        blocks.add(sb3.StopScript("stopthis"))

      blocks.end = True

    case ir.UncondBr():
      # Allow the parameters to be accessed later
      blocks.add(assignParameters(bctx.available_params, bctx.available_param_sizes, poss_depends, ctx))

      # Assign phi nodes
      blocks.add(assignPhiNodes(phi_info[instr.branch.label], ctx, bctx))

      # Jump to label
      proc_name = localizeLabel(instr.branch.label, bctx.fn.name)
      blocks.add(sb3.ProcedureCall(proc_name, []))

    case ir.CondBr(): # Jump to a label, either known or dependent on a condition
      # Allow the parameters to be accessed later
      blocks.add(assignParameters(bctx.available_params, bctx.available_param_sizes, poss_depends, ctx))

      cond = transValue(instr.cond, ctx, bctx)
      if isinstance(cond.value, IdxbleValue):
        raise CompException(f"Indexable value not supported for branch condition {instr}")
      blocks.add(cond.blocks)

      label_true, label_false = instr.branch_true.label, instr.branch_false.label
      phi_blocks_true = assignPhiNodes(phi_info[label_true], ctx, bctx)
      phi_blocks_false = assignPhiNodes(phi_info[label_false], ctx, bctx)
      true_proc_name, false_proc_name = localizeLabel(label_true, bctx.fn.name), localizeLabel(label_false, bctx.fn.name)

      blocks.add(sb3.ControlFlow("if_else", sb3.BoolOp("=", cond.value, sb3.Known(1)), sb3.BlockList(
        phi_blocks_true.blocks + [sb3.ProcedureCall(true_proc_name, [])]
      ), sb3.BlockList(
        phi_blocks_false.blocks + [sb3.ProcedureCall(false_proc_name, [])]
      )))

    case ir.Switch(): # Jump to many labels depending on a value
      # Allow the parameters to be accessed later
      blocks.add(assignParameters(bctx.available_params, bctx.available_param_sizes, poss_depends, ctx))

      assert isinstance(instr.cond.type, ir.IntegerTy)
      width = instr.cond.type.width
      if getByteSize(instr.cond.type) > 1:
        raise CompException(f"Cannot currently switch with an integer more "
                            f"than {VARIABLE_MAX_BITS} bits (would take multiple vars to store)")

      val, val_blocks = flatAsTuple(transValue(instr.cond, ctx, bctx))
      blocks.add(val_blocks)
      if len(instr.branch_table) > 1:
        val, opti_val_blocks = flatAsTuple(optimizeValueUse(val, math.log2(len(instr.branch_table)), ctx))
        blocks.add(opti_val_blocks)

      case_vs_label: dict[int, sb3.BlockList] = {}
      for case, label in instr.branch_table:
        case_val = transValue(case, ctx, bctx)
        # Switch cases should be constant and unique
        assert isinstance(case_val, ValueAndBlocks)
        assert len(case_val.blocks.blocks) == 0
        assert isinstance(case_val.value, sb3.Known)
        assert isinstance(case_val.value.known, float)
        assert case_val.value.known.is_integer()
        assert int(case_val.value.known) not in case_vs_label
        case_val = int(case_val.value.known)

        label_name = label.label
        label_proc_name = localizeLabel(label_name, bctx.fn.name)

        case_vs_label[case_val] = sb3.BlockList([
          *assignPhiNodes(phi_info[label_name], ctx, bctx).blocks,
          sb3.ProcedureCall(label_proc_name, []),
        ])

      default_label_name = instr.branch_default.label
      default_proc_name = localizeLabel(default_label_name, bctx.fn.name)

      default_label_call = sb3.BlockList([
        *assignPhiNodes(phi_info[default_label_name], ctx, bctx).blocks,
        sb3.ProcedureCall(default_proc_name, []),
      ])

      lowest_poss = 0
      highest_poss = (2 ** width) - 1 # FUTURE OPTI: if the value called from was zero-extended (e.g. from an i8)
                                      # then could set this value to the max of the type before

      blocks.add(binarySearch(val, case_vs_label, default_label_call, lowest_poss, highest_poss))

    case _:
      raise CompException(f"Unsupported terminator instruction opcode {instr} (type {type(instr)})")

  return blocks, ctx

def transIntrinsic(intrinsic: ir.Intrinsic, args: list[ir.Value], result: Variable | None, \
                   ctx: Context, bctx: BlockInfo) -> tuple[sb3.BlockList, Context]:
  blocks = sb3.BlockList()

  # For some intrinsics, they are no-op, etc and we don't need to translate args
  match intrinsic:
    case ir.Intrinsic.VaEnd | ir.Intrinsic.LifetimeStart | ir.Intrinsic.LifetimeEnd | ir.Intrinsic.NoAliasScopeDecl | \
         ir.Intrinsic.Expect | ir.Intrinsic.ExpectWithProbability | ir.Intrinsic.Assume:
      return blocks, ctx

    case _: pass

  metadata: list[ir.MetadataVal] = []
  values: list[sb3.Value] = []
  for arg in args:
    if not isinstance(arg, ir.MetadataVal):
      val = transValue(arg, ctx, bctx)
      assert isinstance(val, ValueAndBlocks)
      blocks.add(val.blocks)
      values.append(val.value)
    else:
      metadata.append(arg)

  match intrinsic:
    case ir.Intrinsic.VaStart:
      # arglist_ptr is a va_list*, so it points to a va_list. For our target, va_list
      # is just a pointer so it is a pointer to a pointer
      arglist_ptr, = values
      assert isinstance(arglist_ptr, sb3.Value)

      # Store vararg_ptr at arglist_ptr
      vararg_ptr = localizeVar(ctx.cfg.vararg_ptr_local, False, bctx)
      blocks.add(sb3.EditList("replaceat", ctx.cfg.mem_var, arglist_ptr, vararg_ptr.getValue()))

    # Note: VaEnd is a no-op on our target

    case ir.Intrinsic.VaCopy:
      # Copy one vararg pointer to another
      dest, src = values
      assert isinstance(dest, sb3.Value) and isinstance(src, sb3.Value)
      src_vararg_ptr = sb3.GetOfList("atindex", ctx.cfg.mem_var, src)
      blocks.add(sb3.EditList("replaceat", ctx.cfg.mem_var, dest, src_vararg_ptr))

    case ir.Intrinsic.Abs:
      val, _ = values
      ty = args[0].type
      assert isinstance(val, sb3.Value)
      assert isinstance(ty, ir.IntegerTy)
      assert result is not None

      # Note: this method transfoms int_min into int_min as 128 * -1 mod 256 = -128 mod 256 = 128,
      # so we can simply ignore the is_int_min_poison flag
      is_pos = sb3.BoolOp("<", val, sb3.Known(2**(ty.width - 1)))
      sign = sb3.Op("sub", sb3.Op("mul", is_pos, sb3.Known(2)), sb3.Known(1))
      blocks.add(result.setValue(sb3.Op("mod", sb3.Op("mul", val, sign), sb3.Known(2**ty.width))))

    case ir.Intrinsic.UMin | ir.Intrinsic.UMax | ir.Intrinsic.SMin | ir.Intrinsic.SMax:
      match intrinsic:
        case ir.Intrinsic.UMin: mode = ir.ICmpCond.Ult
        case ir.Intrinsic.UMax: mode = ir.ICmpCond.Ugt
        case ir.Intrinsic.SMin: mode = ir.ICmpCond.Slt
        case ir.Intrinsic.SMax: mode = ir.ICmpCond.Sgt

      lft, rgt = values
      assert isinstance(lft, sb3.Value) and isinstance(rgt, sb3.Value)
      ty = args[0].type
      assert isinstance(ty, ir.IntegerTy)
      assert result is not None

      cond = intCompare(lft, rgt, ty.width, mode, ctx.cfg)
      blocks.add(sb3.BlockList([
        sb3.ControlFlow("if_else", cond, sb3.BlockList([result.setValue(lft)]), sb3.BlockList([result.setValue(rgt)]))
      ]))

    case ir.Intrinsic.MemCpy:
      dest, src, length, _volatile = values

      # If the length is unknown and large, we'll use a loop like when it is known
      known_length = isinstance(length, sb3.Known) and \
        (isinstance(length.known, (int, float)) and length.known < 12)

      if known_length:
        assert isinstance(length, sb3.Known)
        assert isinstance(length.known, (int, float))
        assert length.known.is_integer()
        for offset in range(int(length.known)):
          offset_val = sb3.Known(offset)

          get_ptr = sb3.GetOfList("atindex", ctx.cfg.mem_var, sb3.Op("add", src, offset_val))
          set_ptr = sb3.EditList("replaceat", ctx.cfg.mem_var, sb3.Op("add", dest, offset_val), get_ptr)

          blocks.add(set_ptr)
      else:
        uses = 40 # Reasonable default for unknown lengths
        if isinstance(length, sb3.Known) and isinstance(length.known, (int, float)):
          uses = int(length.known)
        src, src_blocks = flatAsTuple(optimizeValueUse(src, uses, ctx))
        dest, dest_blocks = flatAsTuple(optimizeValueUse(dest, uses, ctx))

        ptr_offset = genTempVar(ctx)

        blocks.add(src_blocks)
        blocks.add(dest_blocks)
        blocks.add(sb3.BlockList([
          sb3.EditVar("set", ptr_offset, sb3.Known(0)),
          sb3.ControlFlow("reptimes", length, sb3.BlockList([
            sb3.EditList("replaceat", ctx.cfg.mem_var,
              sb3.Op("add", dest, sb3.GetVar(ptr_offset)),
              sb3.GetOfList("atindex", ctx.cfg.mem_var, sb3.Op("add", src, sb3.GetVar(ptr_offset)))),
            sb3.EditVar("change", ptr_offset, sb3.Known(1))
          ]))
        ]))

    case ir.Intrinsic.FShl | ir.Intrinsic.FShr:
      lft, rgt, shift = values
      assert isinstance(lft, sb3.Value) and isinstance(rgt, sb3.Value) and isinstance(shift, sb3.Value)
      ty = args[0].type
      assert isinstance(ty, ir.IntegerTy)
      assert result is not None

      # If there is enough space so that the bits can be concatenated to fit in one value
      short_version = ty.width * 2 < INTERMEDIATE_MAX_BITS

      # e.g. in 8 bits a rotation of 9 bits is equivalent to a rotation of 1 bit
      # simplify early so that intPow2 can use known values to lookup pow2 beforehand
      shift = opt.simplifyValue(sb3.Op("mod", shift, sb3.Known(ty.width)))
      shift_op = "mul" if intrinsic == ir.Intrinsic.FShl else "div"
      pow2_width = sb3.Known(2 ** ty.width)

      if short_version:
        # Short version where there is enough space to concatinate bits e.g
        # for 8 bit FSHL: floor((a + b / 256) * 2^(shift MOD 8)) MOD 256

        pow2_shift, ctx = intPow2(shift, ctx)

        # Shift either left or right by a fixed value
        preshifted_lft = sb3.Op("mul", lft, pow2_width) if intrinsic == ir.Intrinsic.FShl else lft
        preshifted_rgt = sb3.Op("div", rgt, pow2_width) if intrinsic == ir.Intrinsic.FShr else rgt

        blocks.add(result.setValue(
          sb3.Op("mod",
            sb3.Op("floor",
              sb3.Op(shift_op,
                sb3.Op("add",
                  preshifted_lft,
                  preshifted_rgt,
                ),
                pow2_shift, # MUL/DIV
              ) # FLOOR
            ),
            pow2_width # MOD
        )))

      else:
        # Solution with highest intermediate bits the same as the original bits e.g
        # for 8 bit FSHL: (a / 2^(shift MOD 8) MOD 256) + floor((b / 256) / 2^(shift MOD 8))
        # since intPow gives us an offset with no extra cost, we can use this to multiply
        # or divide by 2^n here

        # Shift length is referenced twice, store if faster
        shift, shift_blocks = flatAsTuple(optimizeValueUse(shift, 2, ctx))
        blocks.add(shift_blocks)

        if intrinsic == ir.Intrinsic.FShl:
          lft_offset = 0
          # Same as dividing by 2^n beforehand
          rgt_offset = -ty.width
        else:
          # Same as multipling lft by 2^n beforehand, as we divide by 2^(shift + offset)
          lft_offset = -ty.width
          rgt_offset = 0

        pow2_shift_plus_offset_lft, ctx = intPow2(shift, ctx, lft_offset)
        pow2_shift_plus_offset_rgt, ctx = intPow2(shift, ctx, rgt_offset)

        blocks.add(result.setValue(
          sb3.Op("add",
            sb3.Op("mod",
              sb3.Op(shift_op,
                lft,
                pow2_shift_plus_offset_lft
              ),
              pow2_width
            ),
            sb3.Op("floor",
              sb3.Op(shift_op,
                rgt,
                pow2_shift_plus_offset_rgt
        )))))

    case ir.Intrinsic.PtrMask:
      ptr, mask = values
      assert isinstance(ptr, sb3.Value) and isinstance(mask, sb3.Value)
      assert isinstance(args[0].type, ir.PointerTy)
      # TODO support different width masks
      assert isinstance(args[1].type, ir.IntegerTy) and args[1].type.width == PTR_WIDTH_BITS
      assert result is not None

      res_val, ctx = binOp("and", ptr, mask, PTR_WIDTH_BITS, ctx)
      blocks.add(result.setValue(res_val))

    case _:
      raise CompException(f"Unsupported intrinsic {intrinsic}")

  return blocks, ctx

def getFnInfo(mod: ir.Module, ctx: Context) -> Context:
  """Get info about a function needed to translate it's instructions"""
  call_graph: dict[str, tuple[set[str], bool]] = {}
  return_addresses: dict[str, list[str]] = {}
  returns_to_address: dict[str, bool] = {}
  check_locations: dict[str, list[str]] = {}
  branch_alloca_size: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
  total_alloca_size: dict[str, int | None] = {}
  block_var_use: dict[str, dict[str, BlockVarUse]] = {}
  branches_to_first: dict[str, bool] = {}
  phi_assignments: defaultdict[str, defaultdict[str, defaultdict[str, list[tuple[Variable, ir.Value]]]]] = \
    defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: list())))
  fn_ptr_sig_called_by: list[set[str]] = [set() for _ in range(len(ctx.fn_ptr_sigs))]
  fn_ptr_sig_return_addrs: list[list[str]] = [[] for _ in range(len(ctx.fn_ptr_sigs))]
  makes_variadic_alloc: defaultdict[str, bool] = defaultdict(bool)

  defined_funcs: list[ir.Function] = list(filter(lambda fn: len(fn.blocks) > 0, mod.functions.values()))
  defined_func_names: list[str] = [fn.name for fn in defined_funcs]

  for fn in mod.functions.values():
    if fn.intrinsic is not None: continue
    returns_to_address[fn.name] = False
    call_graph[fn.name] = (set(), True)

  for fn in defined_funcs:
    if fn.name in ctx.fn_info:
      raise CompException(f"Function {fn.name} defined twice! You might be defining a function already defined in scratch code!")

    calls: set[str] = set() # What the function calls
    for block in fn.blocks.values():
      # Find every function the function could call
      call_id = 0
      for instr in block.instrs:
        match instr:
          case ir.Call():
            localized_call_id = localizeCallId(call_id, block.label, fn.name)

            makes_variadic_alloc[fn.name] |= instr.variadic

            if isinstance(instr.func, ir.FunctionVal):
              # Direct function call
              could_call = [instr.func.name]
              is_direct_call = True
            else:
              # Function Pointer
              signature = ir.FuncTy(return_type=instr.return_type, params=instr.params, variadic=instr.variadic)
              sig_info = getFuncPtrSignatureInfo(signature, fn.name, ctx)
              is_direct_call = False

              if sig_info is None:
                could_call = []
              else:
                signature_id, could_call = sig_info
                # If function pointer which could only be one thing, then treat as a direct call
                if len(could_call) == 1:
                  is_direct_call = True
                else:
                  fn_ptr_sig_called_by[signature_id].add(fn.name)
                  fn_ptr_sig_return_addrs[signature_id].append(localized_call_id)

            if is_direct_call and could_call[0] in defined_func_names:
              return_addresses.setdefault(could_call[0], list())
              return_addresses[could_call[0]].append(localized_call_id)

            for called_name in could_call:
              if called_name in defined_func_names:
                calls.add(called_name)

            # Note: external funcs still contribute to call ids even if unused
            call_id += 1

          case ir.Alloca():
            branch_alloca_size[fn.name][block.label] += getByteSize(instr.allocated_type,
                                                          include_padding=ctx.cfg.accurate_byte_spacing)

          case ir.Phi():
            res_var_name = instr.result.name
            res_var = Variable(res_var_name, "var", fn.name)

            to_label_name = block.label

            for val, from_label in instr.incoming:
              assert isinstance(from_label.type, ir.LabelTy)
              from_label_name = from_label.label

              phi_assignments[fn.name][from_label_name][to_label_name].append((res_var, val))

    call_graph[fn.name] = calls, False

  for fn in defined_funcs:
    block_var_use[fn.name] = getFuncBranchesVarUse(fn, phi_assignments[fn.name])

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
        stack.extend(callees) # TODO OPTI: re-use results if computed here first

    call_graph[start_func] = visited, True

  for fn in defined_funcs:
    first_label = list(fn.blocks.values())[0].label
    branches: dict[str, list[str]] = {"ret": []} # Where different branches could lead
    for block in fn.blocks.values():
      # Find every function the function could call
      # Find what each block in the function could branch to
      branches[block.label] = list(getTerminatorInstrLabels(block.instrs[-1]))

    # Branch jump tables never reset the stack
    if not ctx.cfg.use_branch_jump_table:
      fn_check_locations = sorted(util.selectCycleChecks(branches))
      could_recurse = len(fn_check_locations) > 0
      # If the branches could create a loop, we must place stack checks, so we should return to an
      # address. Furthermore, a binary search and a call is usually faster than potentially
      # hundreds of recursions backward.
      returns_to_address[fn.name] = could_recurse
      check_locations[fn.name] = fn_check_locations
      for branch in fn_check_locations:
        ctx.all_check_locations.append((fn.name, branch))

    # Any branch that may be called more than once
    repeating_branches: set[str] = util.findNodesWithCycle(branches)
    # Branches that are unavoidable
    unavoidable_branches: set[str] = util.unavoidableNodes(branches, first_label, "ret")
    # Branches that are always ran once per func call
    ran_once_branches = unavoidable_branches - repeating_branches

    known_alloc_size = True
    alloc_size = 0
    for block_label in branches.keys():
      if block_label in ran_once_branches:
        alloc_size += branch_alloca_size[fn.name][block_label]
      elif branch_alloca_size[fn.name][block_label] != 0:
        known_alloc_size = False
        break

    total_alloca_size[fn.name] = alloc_size if known_alloc_size else None

    fn_branches_to_first = False
    if len(fn.blocks) > 0:
      first_block_label = list(fn.blocks.values())[0].label
      fn_branches_to_first = any([first_block_label in branch_to for branch_to in branches.values()])
    branches_to_first[fn.name] = fn_branches_to_first

  # Propagate downstream
  for fn_name in returns_to_address:
    returns_to_address[fn_name] |= any(returns_to_address[call] for call in call_graph[fn_name][0])
  for fn in defined_funcs:
    makes_variadic_alloc[fn.name] |= any(makes_variadic_alloc[call] for call in call_graph[fn.name][0])

  for signature_id, (sig, could_call) in enumerate(ctx.fn_ptr_sigs):
    # If the function pointer signature calls any function that returns to an address,
    # it must return to an address also. All functions that can call a function signature
    # have already been accounted for because they were treated as calling all function
    # pointers with that signature directly
    sig_returns_to_address = any(returns_to_address[call] for call in could_call)
    sig_return_addrs = fn_ptr_sig_return_addrs[signature_id]
    sig_takes_ret_addr = sig_returns_to_address and len(sig_return_addrs) > 1
    sig_could_call_total = set(could_call)

    # Give each function it calls a callback return address, unless it is called directly
    if len(could_call) != 1:
      for call in could_call:
        sig_could_call_total |= call_graph[call][0]
        return_addresses.setdefault(call, list())
        return_addresses[call].append(localizeFuncPtrSigCallback(signature_id))

    sig_called_by = fn_ptr_sig_called_by[signature_id]
    sig_could_recurse = len(sig_could_call_total & sig_called_by) > 0

    ctx.fn_ptr_sig_info.append(FuncPtrSigInfo(
      signature_id, sig_could_call_total, len(sig.params), sig.variadic, sig_return_addrs,
      sig_returns_to_address, sig_takes_ret_addr, sig_could_recurse))

  for fn in defined_funcs:
    # If we know the total size a function allocates and that it doesn't call any functions
    # that rely on the stack size, we don't need to increase the stack size
    skip_stack_size_change = total_alloca_size[fn.name] is not None and \
      all(total_alloca_size[call] == 0 for call in call_graph[fn.name][0]) and \
      not makes_variadic_alloc[fn.name]

    fn_ret_addresses = return_addresses.get(fn.name, list())
    fn_returns_to_address = returns_to_address[fn.name]
    # If the function returns to an address and is called from multiple locations,
    # then it must take a return address to know where to return to
    fn_takes_ret_addr = fn_returns_to_address and len(fn_ret_addresses) > 1

    param_names = []
    param_sizes = []
    for arg in fn.params:
      param_names.append(arg.name)
      param_sizes.append(getByteSize(arg.type))

    if fn.variadic:
      param_names.append(ctx.cfg.vararg_ptr_local)
      param_sizes.append(1)
    if fn_takes_ret_addr:
      param_names.append(ctx.cfg.return_address_local)
      param_sizes.append(1)
    params = [Variable(name, "param", fn.name) for name in param_names]

    ctx.fn_info[fn.name] = FuncInfo(fn.name, ctx.next_fn_id, params, param_sizes, len(fn.params), fn.variadic,
                                    call_graph[fn.name][0], fn_ret_addresses, fn_returns_to_address,
                                    fn_takes_ret_addr, check_locations.get(fn.name, list()),
                                    branch_alloca_size[fn.name], total_alloca_size.get(fn.name, None),
                                    skip_stack_size_change, block_var_use[fn.name],
                                    branches_to_first.get(fn.name, False), phi_assignments[fn.name])
    ctx.next_fn_id += 1

  return ctx

def transFuncPtrSigs(ctx: Context) -> Context:
  addr = ctx.min_func_ptr_addr
  for signature_id, (signature, could_call) in enumerate(ctx.fn_ptr_sigs):
    # If theres only one function this signature could call, then it would be called directly instead
    if len(could_call) == 1:
      # Always 1 but to be explicit:
      addr += len(could_call)
      continue

    info = ctx.fn_ptr_sig_info[signature_id]
    sig_name = localizeFuncPtrSig(signature_id)

    arg_count = 0
    for arg in signature.params:
      arg_count += getByteSize(arg)
    arguments = [f"%{n}" for n in range(arg_count)]

    return_address = Variable(ctx.cfg.return_address_local, "param", sig_name)
    vararg_ptr = Variable(ctx.cfg.vararg_ptr_local, "param", sig_name)
    func_ptr_addr = Variable(ctx.cfg.func_ptr_parameter, "param", sig_name)
    params = [func_ptr_addr.getUnidxedRawVarName(), *deepcopy(arguments)]

    if info.is_variadic:          params.append(vararg_ptr.getUnidxedRawVarName())
    if info.takes_return_address: params.append(return_address.getUnidxedRawVarName())

    blocks = sb3.BlockList([
      sb3.ProcedureDef(sig_name, params),
    ])

    return_addr = None
    if info.returns_to_address:
      blocks.add(sb3.EditCounter("incr"))
      new_return_addr = Variable(ctx.cfg.return_address_local, "var", sig_name)
      if info.takes_return_address:
        if not info.could_recurse:
          blocks.add(new_return_addr.setValue(return_address.getValue()))
        else:
          blocks.add(sb3.EditVar("change", ctx.cfg.local_stack_size_var, sb3.Known(1)))
          blocks.add(
            storeOnStack(ctx.cfg.local_stack_var, ctx.cfg.local_stack_size_var, 0, 1, return_address.getValue()))
      return_addr = new_return_addr

    callback = localizeFuncPtrSigCallback(signature_id)

    branches = dict()
    for name in could_call:
      callee_info = ctx.fn_info[name]
      assert info.is_variadic == callee_info.is_variadic

      branch = sb3.BlockList()
      args: list[sb3.Value] = [sb3.GetParam(arg) for arg in arguments]
      if info.is_variadic:
        args.append(vararg_ptr.getValue())
      if callee_info.takes_return_address:
        callee_return_addr = callee_info.return_addresses.index(callback)
        args.append(sb3.Known(callee_return_addr))
      branch.add(sb3.ProcedureCall(name, args))

      # If we return to an address and the callee doesn't then jump to our callback ourselves
      if info.returns_to_address and not ctx.fn_info[name].returns_to_address:
        branch.add(sb3.ProcedureCall(callback, []))

      branches[addr] = branch
      addr += 1

    blocks.add(binarySearch(func_ptr_addr.getValue(), branches))

    ctx.proj.code.append(blocks)

    if info.returns_to_address:
      blocks = sb3.BlockList([
        sb3.ProcedureDef(callback, []),
        sb3.EditCounter("incr"),
      ])

      if info.could_recurse and info.takes_return_address:
        blocks.add(sb3.EditVar("change", ctx.cfg.local_stack_size_var, sb3.Known(-1)))
        return_addr_val = loadFromStack(ctx.cfg.local_stack_var, ctx.cfg.local_stack_size_var, 1, 1)
        assert isinstance(return_addr_val, sb3.Value)
      else:
        assert return_addr is not None
        return_addr_val = return_addr.getValue()

      blocks.add(transReturnAddr(return_addr_val, info, ctx))

      ctx.proj.code.append(blocks)

  return ctx

def transFuncs(mod: ir.Module, ctx: Context) -> Context:
  ctx = getFnInfo(mod, ctx)

  for func in mod.functions.values():
    # If function has no body, ignore it
    if len(func.blocks) == 0: continue

    fn_name = func.name
    info = ctx.fn_info[fn_name]

    assert not (info.returns_to_address and ctx.cfg.use_branch_jump_table)

    is_first_block = True
    total_fn_allocated = 0
    for block in func.blocks.values():
      if is_first_block:
        proc_name = fn_name
        localized_params = info.params
        localized_param_sizes = info.param_sizes
      else:
        proc_name = localizeLabel(block.label, info.name)
        localized_params = []
        localized_param_sizes = []

      # Get code to start the branch (procedure definition, etc)
      if (block.label not in info.checked_blocks) or (is_first_block and info.branches_to_first):
        starting_fn_code, ctx = getUncheckedProcedureStart(proc_name, localized_params,
                                                           localized_param_sizes, info, ctx,
                                                           is_counted=info.returns_to_address)
      else:
        next_var_use_depends = info.block_var_use[block.label].depends
        next_var_use_depends |= ctx.cfg.special_locals
        starting_fn_code, ctx = getCheckedProcedureStart(proc_name, localized_params, localized_param_sizes,
                                                         next_var_use_depends, block.label, info, ctx)

      if is_first_block and ctx.cfg.do_debug_branch_log:
        # Increment the function counter in the log
        index = sb3.Known((info.fn_id * 2) + 1)
        starting_fn_code.add(sb3.BlockList([
          sb3.EditList("replaceat", ctx.cfg.debug_branch_log_var, index, sb3.Op("add",
            sb3.Known(1), sb3.GetOfList("atindex", ctx.cfg.debug_branch_log_var, index)
        ))]))

      # Store the previous stack size if necessary
      if is_first_block and info.total_alloca_size is None and not info.skip_stack_size_change:
        starting_fn_code.add(
          localizeVar(ctx.cfg.previous_stack_size_local, False, BlockInfo(info, info.params, info.param_sizes))
          .setValue(sb3.GetVar(ctx.cfg.stack_pointer_var)))

      if is_first_block and info.branches_to_first:
        # Work out what variables might be depended on in future
        poss_depends = info.block_var_use[block.label].depends
        poss_depends |= ctx.cfg.special_locals
        starting_fn_code.add(assignParameters(info.params, info.param_sizes, poss_depends, ctx))

        first_block_proc_name = localizeLabel(block.label, info.name)

        starting_fn_code.add(sb3.ProcedureCall(first_block_proc_name, []))

        # TODO FIX: repeat code, use helper func
        ctx.proj.code.append(starting_fn_code)

        if block.label not in info.checked_blocks:
          starting_fn_code, ctx = getUncheckedProcedureStart(first_block_proc_name, [], [], info, ctx,
                                                            is_counted=info.returns_to_address)
        else:
          starting_fn_code, ctx = getCheckedProcedureStart(first_block_proc_name, [], [],
                                                           poss_depends, block.label, info, ctx)

        is_first_block = False

      # Change stack size by the amount the function/branch allocates beforehand
      # FUTURE FIX: This could technically cause issues if we normally allocate after recursing, which could lead to
      # memory being allocated to the stack when it shouldn't, but this likely won't cause any issues
      to_allocate: int = 0
      if is_first_block and info.total_alloca_size is not None:
        # We should never allocate to the stack for the whole function if we can branch the the first block because
        # that would lead to double allocation. This could be fixed but this should never happen as we can't predict
        # what we'd need to allocate for the entire function anyway
        assert (info.branches_to_first is False) or (info.total_alloca_size == 0)
        to_allocate = info.total_alloca_size
      elif info.total_alloca_size is None:
        to_allocate = info.block_alloca_size[block.label]

      if to_allocate != 0 and not info.skip_stack_size_change:
        starting_fn_code.add(sb3.EditVar("change", ctx.cfg.stack_pointer_var, sb3.Known(-to_allocate)))

      # After the first block we can no longer access the parameters in the function
      available_params, av_param_sizes = [], []

      # In the jump table method, every parameter is accessible from the first function
      # As every parameter is in the first function
      if is_first_block or ctx.cfg.use_branch_jump_table:
        available_params, av_param_sizes = info.params, info.param_sizes

      bctx = BlockInfo(info, available_params, av_param_sizes,
                       code=starting_fn_code, label=block.label,
                       allocated=total_fn_allocated)

      # Translate everything except the terminator operation
      for instr_index, instr in enumerate(block.instrs[:-1]):
        assert bctx is not None
        if isinstance(instr, ir.Call): # Call instructions handled here because they can change where code is ran
          if instr.tail_kind == ir.CallTailKind.MustTail:
            raise CompException("Tail calls not supported")

          callee_info = None
          args = instr.args
          result = None if instr.result is None else transVar(instr.result, bctx)
          result_size = None if instr.result is None else getByteSize(instr.return_type)
          following_instrs = block.instrs[instr_index + 1:]

          if not isinstance(instr.func, ir.FunctionVal):
            signature = ir.FuncTy(return_type=instr.return_type, params=instr.params, variadic=instr.variadic)
            # Don't warn again - we did this earlier in getFnInfo
            sig_info = getFuncPtrSignatureInfo(signature, bctx.fn.name, ctx, warn_no_matching=False)

            if sig_info is None:
              # TODO config to disable these errors
              bctx.code.add(scratchRuntimeError("Unmatched function signature"))
              bctx.next_call_id += 1
              continue

            signature_id, could_call = sig_info
            if len(could_call) == 1:
              # Treat as a direct call instead
              callee_info = ctx.fn_info[could_call[0]]
            else:
              callee_info = ctx.fn_ptr_sig_info[signature_id]
              # Pass function pointer to function
              args = [instr.func, *args]

          else:
            if instr.intrinsic is None and instr.func.name not in ctx.fn_info:
              raise CompException(f"Could not find function {instr.func.name}")

            callee_info = ctx.fn_info[instr.func.name] if instr.intrinsic is None else None

          if instr.intrinsic is not None:
            intrinsic, ctx = transIntrinsic(instr.intrinsic, args, result, ctx, bctx)
            bctx.code.add(intrinsic)
            bctx.next_call_id += 1

          else:
            assert callee_info is not None
            ctx, bctx = transComplexCall(info, callee_info, args, result, result_size, following_instrs, ctx, bctx)
        else:
          instr_code, ctx, bctx = transInstr(instr, ctx, bctx)
          bctx.code.add(instr_code)

      # TODO OPTI: work out where if statements can be placed etc
      terminator_code, ctx = transTerminatorInstr(block.instrs[-1], ctx, bctx)
      bctx.code.add(terminator_code)
      ctx.proj.code.append(bctx.code)

      is_first_block = False
      if info.total_alloca_size == None:
        total_fn_allocated = bctx.allocated

  ctx = transFuncPtrSigs(ctx)

  return ctx

def transEntrypointCall(ctx: Context) -> tuple[sb3.BlockList, Context]:
  entrypoint = ctx.cfg.entrypoint
  if entrypoint not in ctx.fn_info:
    raise CompException(f"Could not find entrypoint function {entrypoint}!")

  entrypoint_info = ctx.fn_info[entrypoint]
  main_params_len = sum(entrypoint_info.param_sizes) + int(entrypoint_info.takes_return_address)
  entrypoint_call = sb3.ProcedureCall(entrypoint, [sb3.Known(0)] * main_params_len)

  if not entrypoint_info.returns_to_address:
    return sb3.BlockList([entrypoint_call]), ctx

  # We need to make a jump table for everywhere the stack might need to reset
  else:
    # Stack cannot reset with branching jump table
    assert not ctx.cfg.use_branch_jump_table

    jump_table: dict[int, sb3.BlockList] = {
      EXIT_CALL_ID: sb3.BlockList([sb3.StopScript("stopthis")]),
      ENTRY_CALL_ID: sb3.BlockList([entrypoint_call]),
    }

    for id_offset, (fn_name, branch_label) in enumerate(ctx.all_check_locations):
      branch_proc_name = localizeLabel(branch_label, fn_name)
      # Branches don't have parameters
      jump_table[START_STACK_RESET_ID + id_offset] = sb3.BlockList([sb3.ProcedureCall(branch_proc_name, [])])

    jump_table_blocks = binarySearch(sb3.GetVar(ctx.cfg.jump_table_id_var), jump_table)

    jump_table_fn_name = "!jump table"
    jump_table_fn_blocks = sb3.BlockList([
      sb3.ProcedureDef(jump_table_fn_name, []),
      sb3.ControlFlow("forever", None, sb3.BlockList([
        sb3.EditCounter("clear"),
        *jump_table_blocks.blocks,
      ])),
    ])

    ctx.proj.code.append(jump_table_fn_blocks)

    return sb3.BlockList([
      sb3.EditVar("set", ctx.cfg.jump_table_id_var, sb3.Known(ENTRY_CALL_ID)),
      sb3.ProcedureCall(jump_table_fn_name, []),
    ]), ctx

def initMemory(mod: ir.Module, ctx: Context) -> tuple[sb3.BlockList, Context]:
  """
  Initialize memory and get the addresses of global variables and function pointers
  """

  # Get sizes of global variables
  # Include padding as this is accessed as part of memory
  sizes = [getByteSize(glob.type, include_padding=ctx.cfg.accurate_byte_spacing) for glob in mod.global_vars.values()]
  total_size = sum(sizes)

  # Memory layout
  null = 0 # type: ignore
  starting_global_addr = 1                               # Global memory is at the start of memory*
  starting_heap_ptr = starting_global_addr + total_size  # Heap starts after global memory
  starting_stack_ptr = ctx.cfg.memory_size               # Stack pointer will first point to the end of memory and grows backward
  starting_fn_ptr_addr = ctx.cfg.memory_size + 1         # Function pointers exists at fake addresses after the end of memory
  # *Global memory is at the start to keep pointers at low values for project.json size

  # Get global variable addresses
  ptr = starting_global_addr
  for i, glob in enumerate(mod.global_vars.values()):
    ctx.globvar_to_ptr[glob.name] = ptr
    ptr += sizes[i]

  # Get function pointer addresses
  ctx.min_func_ptr_addr = starting_fn_ptr_addr
  ctx.fn_ptr_sigs = getFuncPtrRefs(mod)

  # Initialize memory
  blocks = sb3.BlockList()
  if ctx.cfg.memory_size > SCRATCH_LIST_LIMIT:
    raise CompException("Stack is too large to fit in one list, multiple lists/hacked length not implemented yet")

  init_mem: list[sb3.Known] = []
  ptr = starting_global_addr
  for i, glob in enumerate(mod.global_vars.values()):
    if not ctx.cfg.compiler_opt:
      globvar = localizeVar(glob.name, True, None)
      blocks.add(globvar.setValue(sb3.Known(ptr)))

    if glob.init is not None:
      blocks_and_value = transValue(glob.init, ctx, None, is_global_init=True, include_padding=ctx.cfg.accurate_byte_spacing)
      unknown = len(blocks_and_value.blocks) > 0
      value = blocks_and_value.value

      if isinstance(value, IdxbleValue):
        unknown |= not all([isinstance(val, sb3.Known) for val in value.vals])
      else:
        unknown |= not isinstance(value, sb3.Known)

      if unknown: raise CompException(f"Expected static value {glob} to have a compile time known value, got {value.stringify()}")

      values: list[sb3.Known] = []
      match value:
        case sb3.Known():
          values.append(value)
        case IdxbleValue():
          values.extend(cast(list[sb3.Known], value.vals))
        case _:
          raise AssertionError("Did not expect this path")

      assert len(values) == sizes[i]
      init_mem.extend(values)

      ptr += sizes[i]

  ctx.proj.lists[ctx.cfg.init_mem_var] = init_mem

  # If the memory list is saturated (has the correct amount of items) and
  # replace at is valid on it (replace at wouldn't work on an empty list)
  list_is_saturated = sb3.BoolOp("=",
    sb3.GetListLength(ctx.cfg.mem_var),
    sb3.Known(min(ctx.cfg.memory_size, SCRATCH_LIST_LIMIT)))

  blocks.add(sb3.BlockList([
    # Reset stack pointer
    sb3.EditVar("set", ctx.cfg.stack_pointer_var, sb3.Known(starting_stack_ptr)),

    # Reset heap pointer
    # TODO: this should be part of FFI
    sb3.EditVar("set", ctx.cfg.heap_pointer_var, sb3.Known(starting_heap_ptr)),

    # Saturate memory - it needs to be filled for replace var to work
    sb3.ControlFlow("if", sb3.BoolOp("not", list_is_saturated), sb3.BlockList([
      sb3.EditList("deleteall", ctx.cfg.mem_var, None, None),
      sb3.ControlFlow("reptimes", sb3.Known(ctx.cfg.memory_size), sb3.BlockList([
        sb3.EditList("addto", ctx.cfg.mem_var, None, sb3.Known(0))
      ])),
    ])),

    # Reset the global values
    sb3.ControlFlow("for_each", var="ptr", value=sb3.Known(total_size), blocks=sb3.BlockList([
      sb3.EditList("replaceat", ctx.cfg.mem_var,
        sb3.Op("add", sb3.Known(starting_global_addr - 1), sb3.GetVar("ptr")),
        sb3.GetOfList("atindex", ctx.cfg.init_mem_var, sb3.GetVar("ptr"))),
    ])),
  ]))

  return blocks, ctx

def initLocalStack(ctx: Context) -> sb3.BlockList:
  return sb3.BlockList([
    sb3.EditVar("set", ctx.cfg.local_stack_size_var, sb3.Known(0)),
    sb3.ControlFlow("if", sb3.BoolOp("not", sb3.BoolOp("=",
          sb3.GetListLength(ctx.cfg.local_stack_var),
          sb3.Known(ctx.cfg.local_stack_size))), sb3.BlockList([
      sb3.EditList("deleteall", ctx.cfg.local_stack_var, None, None),
      sb3.ControlFlow("reptimes", sb3.Known(ctx.cfg.local_stack_size), sb3.BlockList([
        sb3.EditList("addto", ctx.cfg.local_stack_var, None, sb3.Known(0)),
      ]))
    ]))
  ])

def buildLookupTableComptime(op: Literal["and", "or", "xor"], ctx: Context) -> Context:
  name = getOpLookupTableName(op, ctx)
  lookup_size = 2 ** BINOP_LOOKUP_BITS

  lookup: list[int] = []
  match op:
    case "and": lookup = [l & r for l in range(lookup_size) for r in range(lookup_size)]
    case "or":  lookup = [l | r for l in range(lookup_size) for r in range(lookup_size)]
    case "xor": lookup = [l ^ r for l in range(lookup_size) for r in range(lookup_size)]

  # Binary op lookup tables are zero-indexed
  ctx.proj.lists[name] = [sb3.Known(v) for v in lookup[1:]]

  return ctx

def initLookupTables(ctx: Context) -> tuple[sb3.BlockList, Context]:
  blocks = sb3.BlockList()

  if ctx.cfg.gen_lut_runtime:
    if ctx.needs_and_lut or ctx.needs_or_lut or ctx.needs_xor_lut:
      # Fast runtime lookup table generator. See my project: https://scratch.mit.edu/projects/1304776208/

      # This generator requires using hex digits, so only works with byte size lookup tables
      assert BINOP_LOOKUP_BITS == 8

      # We can generate the OR table faster by using NOT the AND table in reverse (see later on)
      include_or = ctx.needs_or_lut and not ctx.needs_and_lut

      # Make a small lookup table for each nibble we compare
      gen_small_lut = lambda f: sb3.Known("".join([f"{f(a, b):0x}" for a in range(16) for b in range(16)]))

      to_generate: list[tuple[bool, str, str, sb3.Known]] = list(filter(lambda x: x[0], [
        (ctx.needs_and_lut, getOpLookupTableName("and", ctx), "&", gen_small_lut(lambda a, b: a & b)),
        (include_or,        getOpLookupTableName("or", ctx),  "|", gen_small_lut(lambda a, b: a | b)),
        (ctx.needs_xor_lut, getOpLookupTableName("xor", ctx), "^", gen_small_lut(lambda a, b: a ^ b)),
      ]))

      assert len(to_generate) >= 1

      expected_lut_size = sb3.Known(2 ** 16 - 1)

      gen_binop_lut_blocks = sb3.BlockList([
        # Fill lookup tables with zeros as we'll have to randomly access them
        sb3.ControlFlow("reptimes", expected_lut_size, sb3.BlockList([
          sb3.EditList("addto", name, None, sb3.Known(0)) for _, name, _, _ in to_generate
        ])),

        # Iterate over combinations of most significant 4 bits
        sb3.EditVar("set", "a0", sb3.Known(0)),
        sb3.ControlFlow("reptimes", sb3.Known(16), sb3.BlockList([
          sb3.EditVar("set", "b0", sb3.Known(0)),
          sb3.EditVar("set", "256b + 16a0", sb3.Op("mul", sb3.GetVar("a0"), sb3.Known(16))),
          sb3.ControlFlow("reptimes", sb3.Known(16), sb3.BlockList([
            *[
              # Index the OP'd first nibble of A and B into the small lookup table using letter of
              # and combine with an 0x prefix to allow hexadecimal coercion later
              sb3.EditVar("set", f"0x join (a0 {op_name} b0)",
                sb3.Op("join", sb3.Known("0x"),
                  sb3.Op("letter_of",
                    sumValueParts([sb3.Op("mul", sb3.GetVar("a0"), sb3.Known(16)), sb3.GetVar("b0"), sb3.Known(1)]),
                    small_table
                  )))
              for _, _, op_name, small_table in to_generate
            ],

            # Iterate over the least significant nibbles
            # Calculating 16b1 + 1 to avoid operations in the loop later
            sb3.EditVar("set", "16b1 + 1", sb3.Known(1)),
            sb3.ControlFlow("reptimes", sb3.Known(16), sb3.BlockList([
              # Use "counter" for the innermost loop because it is the fastest
              # counter represents a1. For each could also be used here but
              # it seems counter is slightly faster (at least for when two tables
              # are being generated)
              sb3.EditCounter("clear"),
              sb3.ControlFlow("reptimes", sb3.Known(16), sb3.BlockList([
                *[
                  sb3.EditList("replaceat", name,
                    sb3.Op("add", sb3.GetVar("256b + 16a0"), sb3.GetCounter()),
                    # Use scratch's hexadecimal coercion (e.g. "0x10" -> 16)
                    sb3.Op("str_to_float", sb3.Op("join",
                      sb3.GetVar(f"0x join (a0 {op_name} b0)"),
                      sb3.Op("letter_of",
                        sb3.Op("add", sb3.GetVar("16b1 + 1"), sb3.GetCounter()),
                        small_table
                      )
                    ))
                  )
                  for _, name, op_name, small_table in to_generate
                ],
                sb3.EditCounter("incr"),
              ])),
              sb3.EditVar("change", "16b1 + 1", sb3.Known(16)),
              sb3.EditVar("change", "256b + 16a0", sb3.Known(256)),
            ])),
            sb3.EditVar("change", "b0", sb3.Known(1)),
          ])),
          sb3.EditVar("change", "a0", sb3.Known(1)),
        ]))
      ])

      # Generate OR table using AND table if both are required
      # This saves a fairly significant amount of startup time
      if ctx.needs_and_lut and ctx.needs_or_lut:
        gen_binop_lut_blocks.add(
          sb3.ControlFlow("for_each", var="i", value=expected_lut_size, blocks=sb3.BlockList([
            # The OR table is equal to the reverse of the NAND table
            # Proof:
            # A | B = !(!A & !B)
            # !X = 255 - X
            # or(A, B) = 255 - and(255 - A, 255 - B)
            # or(256A + B) = 255 - and(256(255 - A) + 255 - B)
            # or(256A + B) = 255 - and(65535 - 256A - B)
            # or(i) = 255 - and(65535 - i)
            sb3.EditList("addto", getOpLookupTableName("or", ctx), None,
              sb3.Op("sub",
                sb3.Known(255),
                sb3.GetOfList("atindex", getOpLookupTableName("and", ctx),
                  sb3.Op("sub", expected_lut_size, sb3.GetVar("i"))
                )
              )
            )
          ]))
        )

      blocks.add(sb3.BlockList([
        # If we need to generate the lookup tables
        sb3.ControlFlow("if",
          sb3.BoolOp("not", sb3.BoolOp("=", sb3.GetListLength(to_generate[0][1]), expected_lut_size)),
          gen_binop_lut_blocks,
        )
      ]))
  else:
    if ctx.needs_and_lut: ctx = buildLookupTableComptime("and", ctx)
    if ctx.needs_or_lut:  ctx = buildLookupTableComptime("or", ctx)
    if ctx.needs_xor_lut: ctx = buildLookupTableComptime("xor", ctx)
  return blocks, ctx

def tableLookup(table_name: str, index_val: sb3.Known, ctx: Context) -> sb3.Known | None:
  # This will be called a lot for globals - reject early
  if table_name == ctx.cfg.mem_var: return None

  # Scratch always floors indices
  index = math.floor(sb3.scratchCastToNum(index_val))
  matches_op = lambda op: table_name == getOpLookupTableName(op, ctx)

  # Wait this is actually so clean lmao
  if binop := next(filter(lambda op: matches_op(op), ["and", "or", "xor"]), None):
    assert index >= 0 and index <= 2 ** (2 * BINOP_LOOKUP_BITS)
    lft = index >> BINOP_LOOKUP_BITS
    rgt = index & (1 << BINOP_LOOKUP_BITS) - 1
    match binop:
      case "and": return sb3.Known(lft & rgt)
      case "or":  return sb3.Known(lft | rgt)
      case "xor": return sb3.Known(lft ^ rgt)
      case _:     assert False

  if table_name == ctx.cfg.pow2_lookup_var:
    assert index >= 1
    power = index - getPow2Offset()
    return sb3.Known(2 ** power)

def getDontElide(ctx: Context) -> set[str]:
  """Returns a list of names of variables which should not be elided"""
  # Jump table ids should not be elided as in future jump ids may be depended upon after optimization in future
  res = {ctx.cfg.jump_table_id_var, ctx.cfg.return_var}
  if ctx.highest_return_size is not None:
    res |= {Variable(ctx.cfg.return_var, "special_var", None).getRawVarName(i) for i in range(ctx.highest_return_size)}
  return res

def optimize(ctx: Context) -> Context:
  ctx.proj = opt.optimize(ctx.proj,
    perf = ctx.cfg.opt_target.perf,
    all_opti = ctx.cfg.opt_passes,
    dont_remove = getDontElide(ctx),
    ignore_external_change = {ctx.cfg.stack_pointer_var},
    lookup_func = lambda n, i: tableLookup(n, i, ctx))

  return ctx

def replaceCalls(bl: sb3.BlockList, replacements: dict[str, sb3.Block]) -> sb3.BlockList:
  for i, b in enumerate(bl.blocks):
    match b:
      case sb3.ProcedureCall() if b.proc_name in replacements:
        # Should only be inling branches here
        assert len(b.arguments) == 0
        bl.blocks[i] = replacements[b.proc_name]
      case sb3.ControlFlow():
        b.blocks = replaceCalls(b.blocks, replacements)
        if b.else_blocks is not None:
          b.else_blocks = replaceCalls(b.else_blocks, replacements)
        bl.blocks[i] = b
  return bl

def postOptTransform(mod: ir.Module, ctx: Context) -> tuple[Context, bool]:
  if not ctx.cfg.use_branch_jump_table: return ctx, False

  # Transform the function calls into a jump table. This is done
  # after optimization because otherwise the optimizer would not
  # know which branch leads where (and means it can use the same
  # code when optimizing for functions)

  to_remove: list[int] = []

  # Func name vs index
  procs: dict[str, int] = {}
  for i, bl in enumerate(ctx.proj.code):
    assert len(bl) > 0
    if isinstance(bl.blocks[0], sb3.ProcedureDef):
      procs[bl.blocks[0].proc_name] = i

  for fn in mod.functions.values():
    # Ignore externally defined funcs
    if len(fn.blocks) == 0: continue

    # The start of the procedure should not have been inlined
    assert fn.name in procs

    needs_call_replacement: list[str] = []
    for i, block in enumerate(fn.blocks.values()):
      if i == 0: continue
      b_name = localizeLabel(block.label, fn.name)
      needs_call_replacement.append(b_name)

    # If the function doesn't make any branches we should
    # remove the 'stop this script' and not replace anything
    if len(needs_call_replacement) == 0:
      last = ctx.proj.code[procs[fn.name]].blocks.pop()
      assert isinstance(last, sb3.StopScript) and last.op == "stopthis"
      continue

    branch_id_var = Variable(ctx.cfg.branch_jump_table_addr_local, "var", fn.name)

    replacements: dict[str, sb3.Block] = {}
    for i, name in enumerate(needs_call_replacement):
      # TODO: add to must_store
      replacements[name] = branch_id_var.setValue(sb3.Known(i))

    for name in [fn.name, *needs_call_replacement]:
      i = procs[name]
      ctx.proj.code[i] = replaceCalls(ctx.proj.code[i], replacements)

    i = procs[fn.name]
    ctx.proj.code[i].end = False
    assert not ctx.proj.code[i].blocks[-1].isEnd()

    get_branch_id = branch_id_var.getValue()
    # TODO: this will be implemented automatically later
    if ctx.cfg.opt_target.exec.compiler_type_hints:
      get_branch_id = sb3.Op("str_to_float", get_branch_id)

    ctx.proj.code[i].add(
      sb3.ControlFlow("forever", None,
        binarySearch(
          get_branch_id,
          {
            i: sb3.BlockList(ctx.proj.code[procs[n]].blocks[1:])
            for i, n in enumerate(needs_call_replacement)
          }
        )
      )
    )

    to_remove.extend(procs[n] for n in needs_call_replacement)

  # Delete in reverse order to not affect other blocklist indices
  for i in sorted(to_remove, reverse=True):
    del ctx.proj.code[i]

  return ctx, True

def addFunc(name: str, params: list[str], contents: sb3.BlockList, ctx: Context) -> Context:
  localized_params = [Variable(param, "param", name) for param in params]
  blocks = sb3.BlockList([sb3.ProcedureDef(name, [param.getRawVarName() for param in localized_params])])
  blocks.add(contents)
  ctx.proj.code.append(blocks)
  ctx.fn_info[name] = FuncInfo(name, ctx.next_fn_id, localized_params, [1]*len(params), len(params))
  ctx.next_fn_id += 1
  return ctx

def addForeignFunctions(ctx: Context) -> Context:
  uppercase_costume_name = "uppercase"
  lowercase = "abcdefghijklmnopqrstuvwxyz"
  ctx.proj.lists[ctx.cfg.lowercase_var] = [sb3.Known(char) for char in lowercase]
  ctx.proj.addCostume(uppercase_costume_name)
  lc_costume_num = ctx.proj.addCostume(lowercase)

  ascii_lookup = []
  for x in range(1, 256): # Ignore zero; improves perf as scratch lists are 1 indexed and zero signifies end of string
    char = chr(x)
    if char.encode("unicode_escape").decode("ascii").startswith("\\") and char != "\\":
      ascii_lookup.append(sb3.Known(f"\\{x:02X}"))
    else:
      ascii_lookup.append(sb3.Known(char))
  ctx.proj.lists[ctx.cfg.ascii_lookup_var + ctx.cfg.zero_indexed_suffix] = ascii_lookup

  # Checks if an ASCII alphabet character is uppercase or not
  # The caller is responsible for setting the costume back to the original
  ctx = addFunc("!helper_is_lowercase", ["char", "alphabet_pos"], sb3.BlockList([
    sb3.EditVar("set", "original",
      sb3.GetOfList("atindex", ctx.cfg.lowercase_var, sb3.GetParam(localizeParam("alphabet_pos"))),
    ),
    sb3.EditList("replaceat", ctx.cfg.lowercase_var,
      sb3.GetParam(localizeParam("alphabet_pos")),
      sb3.GetParam(localizeParam("char"))),
    sb3.SwitchCostume(sb3.Known(uppercase_costume_name)),
    sb3.SwitchCostume(sb3.GetList(ctx.cfg.lowercase_var)),
    sb3.EditList("replaceat", ctx.cfg.lowercase_var,
      sb3.GetParam(localizeParam("alphabet_pos")),
      sb3.GetVar("original")),
    sb3.EditVar("set", ctx.cfg.return_var,
      sb3.Op("bool_to_float", sb3.BoolOp("=",
        sb3.CostumeInfo("number"),
        sb3.Known(lc_costume_num),
      ))),
  ]), ctx)

  # Converts a C string to a Scratch string.
  # Not meant to be used in C as it doesn't support Scratch strings.
  ctx = addFunc("!helper_str2scratch", ["input"], sb3.BlockList([
    sb3.EditVar("set", ctx.cfg.return_var, sb3.Known("")),
    sb3.EditVar("set", "ptr", sb3.GetParam(localizeParam("input"))),
    sb3.EditVar("set", "char", sb3.GetOfList("atindex", ctx.cfg.mem_var, sb3.GetVar("ptr"))),
    sb3.ControlFlow("until", sb3.BoolOp("=", sb3.GetVar("char"), sb3.Known(0)), sb3.BlockList([
      sb3.EditVar("set", ctx.cfg.return_var,
        sb3.Op("join",
          sb3.GetVar(ctx.cfg.return_var),
          sb3.GetOfList("atindex",
            (ctx.cfg.ascii_lookup_var + ctx.cfg.zero_indexed_suffix),
            sb3.GetVar("char")))),
      sb3.EditVar("change", "ptr", sb3.Known(1)),
      sb3.EditVar("set", "char", sb3.GetOfList("atindex", ctx.cfg.mem_var, sb3.GetVar("ptr"))),
    ])),
  ]), ctx)

  # True if the limit is high enough.
  enough_space = sb3.BoolOp("<",
    sb3.Op("length_of", sb3.GetParam(localizeParam("input"))),
    sb3.GetParam(localizeParam("count")))

  # Converts a Scratch string to a C string.
  # Not meant to be used in C as it doesn't support Scratch strings.
  ctx = addFunc("!helper_scratch2str", ["input", "str", "count"], sb3.BlockList([
    # Subtract one here so that i can be one indexed (which letter_of is)
    sb3.EditVar("set", "ptr", sb3.Op("sub", sb3.GetParam(localizeParam("str")), sb3.Known(1))),
    sb3.EditVar("set", "i", sb3.Known(1)),
    # TODO should use cost variable no idea why not setting
    sb3.EditVar("set", "cost", sb3.CostumeInfo("number")),

    # Default: full string.

    sb3.ControlFlow("if_else", enough_space, sb3.BlockList([
      # The limit is high enough.
      # Re-using the "char" variable for counting how many letters should be copied.
      # I think it makes sense, right?
      # Also, according to @Classified3D and the README, calculating the length twice is the fastest option.
      sb3.EditVar("set", "char", sb3.Op("length_of", sb3.GetParam(localizeParam("input")))),
    ]),
    # Else
    sb3.BlockList([
      # The limit is lower than the inputted string.
      # Return False and reduce the letter count.
      # Doing -1 to account for the NULL at the end.
      sb3.EditVar("set", "char", sb3.Op("sub", sb3.GetParam(localizeParam("count")), sb3.Known(1))),
    ])),

    sb3.ControlFlow("reptimes", sb3.GetVar("char"), sb3.BlockList([
      sb3.EditVar("set", "ascii",
        sb3.GetOfList("indexof",
          (ctx.cfg.ascii_lookup_var + ctx.cfg.zero_indexed_suffix),
          sb3.Op("letter_of", sb3.GetVar("i"), sb3.GetParam(localizeParam("input"))))),
      sb3.ControlFlow("if",
        sb3.BoolOp("and",
          sb3.BoolOp(">", sb3.GetVar("ascii"), sb3.Known(ord("A") - 1)),
          sb3.BoolOp("<", sb3.GetVar("ascii"), sb3.Known(ord("Z") + 1)),
        ),
        sb3.BlockList([
          # TODO set costume back to original in after helper_scratch2str finishes
          sb3.ProcedureCall("!helper_is_lowercase", [
            sb3.Op("letter_of", sb3.GetVar("i"), sb3.GetParam(localizeParam("input"))),
            sb3.Op("sub", sb3.GetVar("ascii"), sb3.Known(ord("A") - 1))
          ]),
          sb3.ControlFlow("if", sb3.BoolOp("=", sb3.GetVar(ctx.cfg.return_var), sb3.Known(1)), sb3.BlockList([
            sb3.EditVar("change", "ascii", sb3.Known(ord("a") - ord("A"))),
          ])),
        ])
      ),
      sb3.EditList("replaceat", ctx.cfg.mem_var, sb3.Op("add", sb3.GetVar("ptr"), sb3.GetVar("i")), sb3.GetVar("ascii")),
      sb3.EditVar("change", "i", sb3.Known(1)),
    ])),
    # End of string
    sb3.EditList("replaceat", ctx.cfg.mem_var, sb3.Op("add", sb3.GetVar("ptr"), sb3.GetVar("i")), sb3.Known(0)),

    sb3.SwitchCostume(sb3.GetVar("cost")),

    sb3.EditVar("set", ctx.cfg.return_var, sb3.Op("bool_to_float", enough_space)),

    # Return value is set above.
  ]), ctx)

  # Changing the volume at all forces scratch to render a
  # frame, even in a run without screen refresh procedure
  ctx = addFunc("SB3_render", [], sb3.BlockList([
    sb3.EditVolume("change", sb3.Known(0)),
  ]), ctx)

  ctx = addFunc("SB3_say_str", ["input"], sb3.BlockList([
    sb3.ProcedureCall("!helper_str2scratch", [sb3.GetParam(localizeParam("input"))]),
    sb3.Say(sb3.GetVar(ctx.cfg.return_var)),
  ]), ctx)

  ctx = addFunc("SB3_say_char", ["input"], sb3.BlockList([
    sb3.Say(sb3.GetOfList("atindex",
      (ctx.cfg.ascii_lookup_var + ctx.cfg.zero_indexed_suffix),
      sb3.GetParam(localizeParam("input")))),
  ]), ctx)

  ctx = addFunc("SB3_say_dbl", ["input"], sb3.BlockList([
    sb3.Say(sb3.GetParam(localizeParam("input"))),
  ]), ctx)

  # Wait at least duration seconds while rendering
  # Like wait seconds block, but don't stop rendering
  # like wait block in a no screen refresh procedure
  # TODO use timer blocks instead of days since 2000
  # to consider turbowarp's pausing in calculations
  ctx = addFunc("SB3_wait", ["duration"], sb3.BlockList([
    sb3.EditVar("set", "end", sb3.Op("add",
      sb3.DaysSince2000(),
      sb3.Op("div", sb3.GetParam(localizeParam("duration")), sb3.Known(24*60*60)))),
    # Always wait a frame, even for negative values, just like the wait block in scratch
    sb3.ProcedureCall("SB3_render", []),
    sb3.ControlFlow("until", sb3.BoolOp(">", sb3.DaysSince2000(), sb3.GetVar("end")), sb3.BlockList([
      sb3.ProcedureCall("SB3_render", []),
    ])),
  ]), ctx)

  # Wait at least duration seconds without rendering
  ctx = addFunc("SB3_wait_no_render", ["duration"], sb3.BlockList([
    sb3.Wait(sb3.GetParam(localizeParam("duration"))),
  ]), ctx)

  # output (str): The answer the user provided.
  # input  (str): The question to display in the text bubble.
  # count  (int): The maximum length of the string.
  ctx = addFunc("SB3_ask_str", ["output", "input", "count"], sb3.BlockList([
    sb3.ProcedureCall("!helper_str2scratch", [sb3.GetParam(localizeParam("input"))]),
    sb3.Ask(sb3.GetVar(ctx.cfg.return_var)),

    sb3.ProcedureCall("!helper_scratch2str", [
      sb3.GetAnswer(),
      sb3.GetParam(localizeParam("output")),
      sb3.GetParam(localizeParam("count"))
    ]),
    # Return value is set by scratch2str.
  ]), ctx)

  # output (str): The answer the user provided.
  # input  (str): The question to display in the text bubble.
  ctx = addFunc("SB3_ask_str_unsafe", ["output", "input"], sb3.BlockList([
    sb3.ProcedureCall("!helper_str2scratch", [sb3.GetParam(localizeParam("input"))]),
    sb3.Ask(sb3.GetVar(ctx.cfg.return_var)),

    sb3.ProcedureCall("!helper_scratch2str", [
      sb3.GetAnswer(),
      sb3.GetParam(localizeParam("output")),
      sb3.Known("Infinity"),
    ]),
    # Return value is set by scratch2str. It's always going to be 1.
  ]), ctx)

  # Same as SB3_ask_str, but it outputs a double.
  ctx = addFunc("SB3_ask_dbl", ["output", "input"], sb3.BlockList([
    sb3.ProcedureCall("!helper_str2scratch", [sb3.GetParam(localizeParam("input"))]),
    sb3.Ask(sb3.GetVar(ctx.cfg.return_var)),

    sb3.EditVar("set", "char", sb3.Op("str_to_float", sb3.GetAnswer())), # (answer + 0); casts strings to floats.
    sb3.EditList("replaceat", ctx.cfg.mem_var, sb3.GetParam(localizeParam("output")), sb3.GetVar("char")),

    # Return 1 if successful (casted value == original value), else 0
    sb3.EditVar("set", ctx.cfg.return_var, sb3.Op("bool_to_float", sb3.BoolOp("=", sb3.GetAnswer(), sb3.GetVar("char")))),
  ]), ctx)

  # Returns the days since 2000 in UTC time
  ctx = addFunc("SB3_days_since_2000", [], sb3.BlockList([
    sb3.EditVar("set", ctx.cfg.return_var, sb3.DaysSince2000()),
  ]), ctx)

  # These functions are used in libc in Scratch-Stdlib. Eventually they will be moved there
  # when a sufficient FFI API is supported

  ctx = addFunc("_exit", ["status"], sb3.BlockList([
    # TODO: this would be logged to stdout
    sb3.Ask(sb3.Op("join", sb3.Known("exit called with status "), sb3.GetParam(localizeParam("status")))),
    sb3.StopScript("stopall"),
  ]), ctx)

  ctx = addFunc("close", ["a"], sb3.BlockList([sb3.Ask(sb3.Known("close called"))]), ctx)
  ctx = addFunc("fstat", ["a", "b"], sb3.BlockList([sb3.Ask(sb3.Known("fstat called"))]), ctx)
  ctx = addFunc("isatty", ["a"], sb3.BlockList([sb3.Ask(sb3.Known("isatty called"))]), ctx)
  ctx = addFunc("lseek", ["a", "b", "c"], sb3.BlockList([sb3.Ask(sb3.Known("lseek called"))]), ctx)
  ctx = addFunc("read", ["a", "b", "c"], sb3.BlockList([sb3.Ask(sb3.Known("read called"))]), ctx)

  # Increment the heap pointer by incr. Currently this does not check for out of memory
  ctx = addFunc("sbrk", ["incr"], sb3.BlockList([
    # Return the old pointer
    sb3.EditVar("set", ctx.cfg.return_var, sb3.GetVar(ctx.cfg.heap_pointer_var)),
    # Increment the heap pointer
    sb3.EditVar("change", ctx.cfg.heap_pointer_var, sb3.GetParam(localizeParam("incr"))),
    # TODO check for out of memory, if so return -1 and set @errno
  ]), ctx)

  # TODO proper write implementation
  ctx = addFunc("write", ["file", "buf", "len"], sb3.BlockList([
    sb3.Ask(sb3.Known("write called")),
    # Note that str2scratch doesn't handle the newline (hex 0A), instead converting it to "\0A".
    sb3.ProcedureCall("!helper_str2scratch", [sb3.GetParam(localizeParam("buf"))]),
    sb3.Ask(sb3.GetVar(ctx.cfg.return_var)),
  ]), ctx)

  return ctx

def compile(llvm: str | ir.Module, cfg: Config | None = None) -> tuple[sb3.Project, DebugInfo]:
  """Compile LLVM IR to a scratch project. Returns a project and any debug info generated."""
  if cfg is None: cfg = Config()

  ctx = Context(sb3.Project(cfg.scratch_config), cfg)

  # Parse llvm
  mod: ir.Module = parser.parseAssembly(llvm) if isinstance(llvm, str) else llvm

  # Add foreign functions (run this before initMemory because it is needed for func ptr addrs)
  ctx = addForeignFunctions(ctx)

  # Run on green flag
  ctx.proj.code.append(sb3.BlockList([
    sb3.OnStartFlag(),
    # Put startup code in initialization function for run without screen refresh
    sb3.ProcedureCall("!init", []),
  ]))
  init_blocks = sb3.BlockList([sb3.ProcedureDef("!init", [])])

  # Setup memory and get global/function ptr addresses
  init_mem, ctx = initMemory(mod, ctx)
  init_blocks.add(init_mem)
  init_blocks.add(initLocalStack(ctx))

  # Reset debug info on initialization
  if cfg.do_debug_branch_log:
    init_blocks.add(sb3.EditList("deleteall", cfg.debug_branch_log_var, None, None))

    debug_branch_list_length = (ctx.next_fn_id * 2) + 10 # Adding 10 for safety + bc it doesn't really matter
    if debug_branch_list_length > SCRATCH_LIST_LIMIT:
      raise CompException("Too many functions; could not add them all to the recursive branch debug list")

    init_blocks.add(sb3.ControlFlow("reptimes", sb3.Known(debug_branch_list_length), sb3.BlockList([
      sb3.EditList("addto", cfg.debug_branch_log_var, None, sb3.Known(0))
    ])))

  # Translate functions
  ctx = transFuncs(mod, ctx)

  # Build required lookup tables used by functions
  init_lookups, ctx = initLookupTables(ctx)
  init_blocks.add(init_lookups)

  # Start program after initialization
  start_program_blocks, ctx = transEntrypointCall(ctx)
  init_blocks.add(start_program_blocks)

  # Add init code
  ctx.proj.code.append(init_blocks)

  # Optimize project
  ctx = optimize(ctx)

  # Apply any post optimization transformations
  ctx, did_transform = postOptTransform(mod, ctx)
  if did_transform: ctx = optimize(ctx)

  # Export debug info
  if cfg.do_debug_branch_log:
    branch_debug_info: list[tuple[int, str]] = []
    for info in ctx.fn_info.values():
      branch_debug_info.append((info.fn_id, info.name))
    branch_debug_info.sort(key=lambda info: info[0])

    ctx.debug_info.debug_branch_func_map = "\n".join([info[1] for info in branch_debug_info])

  return ctx.proj, ctx.debug_info
