"""LLVM -> Scratch Compiler"""

from __future__ import annotations
from dataclasses import dataclass, is_dataclass, field, fields
from collections import OrderedDict, defaultdict
from typing import Literal, Callable
from copy import deepcopy

from llvm2py import parse_assembly
from llvm2py import ir
import random
import math

from . import scratch as sb3
from . import optimizer
from . import util

VARIABLE_MAX_BITS = 48 # Maximum amount of bits to store in a fp variable. Maximum is 48 because while scratch's doubles support
                       # up to 53 bits, some operations require extra precision
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
  stack_size: int = 512 # Amount of 'bytes' on 'stack' list (one byte is VARIABLE_MAX_BITS bits), max 200,000
  label_stack_size: int = 512 # Max amount of labels on the recursion stack, max 200,000
  binop_lookup_bits: int = 8 # Amount of bits to use for AND/OR/XOR tables, creates (2**(2*n) elements per table)
  max_branch_recursion: int = 1_000_000 # Maximum amount of times a checked function can recurse before reseting
                                        # scratch's call stack via a broadcast

  debug_info: DebugInfo = field(default_factory=DebugInfo) # Info about a scratch project which can be used in
                                                           # future compilations for optimization
  do_debug_branch_log: bool = False # If the times a function recurses should be logged

  unused_var = "!unused" # Name of the scratch variable for unused values
  return_var = "!return value" # Name of the scratch variable for returing values
  stack_var = "!stack" # Name of the scratch list for the stack list
  stack_size_var = "!stack size" # Name of the scratch variable for the stack size
  local_stack_var = "!local stack" # Name of the scratch list to store variables that will be used recursively
  local_stack_size_var = "!local stack size" # Name of the scratch variable to store the label stack's size
  debug_branch_log_var = "!!debug_branch_log" # Name of the scratch list to store debug info (using underscores
                                              # to avoid spaces in exported filename for convenience)

  ascii_lookup_var = "!ascii lookup"
  binop_lookup_var = "!binop lookup"
  pow2_lookup_var = "!pow2 lookup"

  return_address_local = "return address" # Name of the local variable or parameter to the id of func the return to
  previous_stack_size_local = "prev stack size" # Name of the local variable to store the previous stack size
  special_locals = {return_address_local, previous_stack_size_local} # All special local vars

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
  # The parameters the function takes (doesn't include return address)
  params: list[Variable]
  # Everything the function might call (may include itself)
  can_call: set[str] = field(default_factory=set)
  # Any functions that call this function
  return_addresses: list[str] = field(default_factory=list)
  # If the function returns using a broadcast to an address
  returns_to_address: bool = False
  # If the function takes a return address as a parameter
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

@dataclass
class BlockInfo:
  """Info about a LLVM block"""
  fn: FuncInfo
  available_params: list[Variable] # All params that can be accessed from the function
  first_block: bool = False # Is this is the first block of the function
  code: sb3.BlockList = field(default_factory=sb3.BlockList) # The current code instructions are being added to
  label: str | None = None # Name/Label of the block
  allocated: int = 0 # Out of the amount allocated for the branch beforehand, how much has been translated into addresses
  next_call_id: int = 0 # The id to give to the nth function call

@dataclass
class BlockVarUse:
  depends: set[str] = field(default_factory=set)
  modifies: set[str] = field(default_factory=set)
  branches: set[str] = field(default_factory=set)

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
class IndexableValueAndBlocks:
  """An indexable value and any blocks that come before it needed to get that value"""
  value: IndexableValue
  blocks: sb3.BlockList = field(default_factory=sb3.BlockList)

@dataclass
class Variable:
  var_name: str
  var_type: Literal["global", "param", "var"]
  fn_name: str | None

  def getRawVarName(self) -> str:
    match self.var_type:
      case "global":
        return f"@{self.var_name}"
      case "param":
        return localizeParameter(self.var_name)
      case "var":
        assert self.fn_name is not None
        return f"%{self.fn_name}:{self.var_name}" # Localize variables per function
      case _:
        raise CompException("Unmatched")

  def getValue(self) -> sb3.Value:
    name = self.getRawVarName()
    if self.var_type == "param":
      return sb3.GetParameter(name)
    return sb3.GetVar(name)

  def setValue(self, value: sb3.Value, op: Literal["set", "change"]="set") -> sb3.Block:
    if self.var_type == "param": raise CompException(f"{self.var_name} param is read only")
    return sb3.EditVar(op, self.getRawVarName(), value)

class CompException(Exception):
  """Exception in the compiler"""
  pass

def flatAsTuple(obj):
  """Same as dataclasses.astuple but not recursively with dataclasses in dataclasses"""
  if not is_dataclass(obj):
    raise TypeError("Expected dataclass instance")
  return tuple(getattr(obj, f.name) for f in fields(obj))

def assertNoNamedTemporaries(mod: ir.Module) -> None:
  """
  Due to an issue with llvm2py, it is impossible to tell apart a named
  temporary from a global var with the same name. This function ensures
  there are no named temporaries.
  """
  for fn_name, fn in mod.funcs.items():
    temporaries: set[ir.Value | None] = set()

    for block_label, block in fn.blocks.items():
      for instr in block.instrs:
        temporaries.add(getattr(instr, "result", None))

    for param in fn.args:
      temporaries.add(param)

    for temporary in temporaries:
      if temporary is not None:
        assert isinstance(temporary.val, str)
        if not (temporary.val.startswith("%") or temporary.val == "<badref>"):
          raise CompException(f"Named temporary %{temporary.val} in {fn_name} not supported due "
                              "to llvm2py not being able to tell it apart from a global")

def getByteSize(ty: ir.Type) -> int:
  match ty:
    case ir.IntegerType():
      # Scratch's fp variables can store < 52 bits per variable accurately
      return math.ceil(ty.num_bits / VARIABLE_MAX_BITS)
    case ir.ArrayType():
      return ty.elem_count * getByteSize(ty.elem_ty)
    case ir.PtrType():
      return 1
    case _:
      raise CompException(f"Unknown Type: {ty} (py type {type(ty)})")

def decodeValue(val: ir.Value,
                ctx: Context, bctx: BlockInfo | None) -> ValueAndBlocks | IndexableValueAndBlocks:
  match val.val:
    case str(): # if a variable
      var = decodeVar(val, bctx)
      res = sb3.Known("") # this means no value - nothing in the block
      if var is not None:
        res = var.getValue()
        name = var.getRawVarName()
        if ctx.cfg.opti and name in ctx.globvar_to_ptr:
          res = ctx.globvar_to_ptr[name]
      return ValueAndBlocks(res)
    case int():
      if not isinstance(val.ty, ir.IntegerType):
        raise CompException(f"Expected {val} to be an integer, got type {type(val.ty)}")

      # TODO FIX: allow different sizes with val.ty
      if val.ty.num_bits > VARIABLE_MAX_BITS: raise CompException(f">{VARIABLE_MAX_BITS} bits not "
                                                                  f"yet supported, got {val.ty.num_bits}")

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

      return IndexableValueAndBlocks(
        IndexableValue([sb3.Known(str(int(byte))) for byte in val.val]))
    case _:
      raise CompException(f"Unknown Value: {val.val} (type: {type(val.val)})")

def decodeVar(var: ir.Value | str, bctx: BlockInfo | None) -> Variable | None:
  """Used for getting the assigned variable of an instruction"""
  if isinstance(var, ir.Value):
    if not isinstance(var.val, str):
      raise CompException(f"Expected val to be a variable, got Value: {var.val} (py type: {type(var.val)})")
    var = var.val
  if var == "<badref>":
    return None

  # This is the only way to tell apart a local value from a global one. llvm2py removes the % from the start of named temporaries so
  # it is impossible to tell apart against a global with the same name, hence the assertNoNamedTemporaries
  is_local = var.startswith("%")
  if is_local: var = var[1:]
  return localizeVar(var, not is_local, bctx)

def localizeVar(name: str, is_global: bool, bctx: BlockInfo | None) -> Variable:
  if is_global:
    var_type = "global"
    fn_name = None
  else:
    assert bctx is not None
    fn_name = bctx.fn.name
    if bctx is not None and name in [param.var_name for param in bctx.available_params]:
      var_type = "param"
    else:
      var_type = "var"

  return Variable(name, var_type, fn_name)

def localizeParameter(name: str) -> str:
  return "%" + name

def localizeLabel(label: str, fn: FuncInfo) -> str:
  return f"{fn.name}:{label}"

def localizeCallId(call_id: int, label: str, fn_name: str, recursive: bool = False) -> str:
  if not recursive:
    return f"{fn_name}:{label}:return addr {call_id}"
  return f"{fn_name}:{label}:recursive call {call_id}"

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
      pow2_lookup.append(2 ** x)
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
      if mid_val == max_poss_value: # If the value is the highest, we don't need to check below
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
                        # Sorting already taken care of
                        binarySearch(value, branches, default_branch, min_poss_value, max_poss_value, True,
                                     _lo=mid + 1, _hi=_hi),
                        binarySearch(value, branches, default_branch, min_poss_value, max_poss_value, True,
                                     _lo=_lo,     _hi=mid))])

def offsetStackSize(stack_size_var: str, offset: int) -> sb3.Value:
  ptr = sb3.GetVar(stack_size_var)
  if offset > 0:
    ptr = sb3.Op("add", ptr, sb3.Known(offset))
  elif offset < 0:
    # Subtract instead... because it looks nicer lol
    ptr = sb3.Op("sub", ptr, sb3.Known(-offset))
  return ptr

def storeOnStack(stack_var: str, stack_size_var: str, offset: int, value: sb3.Value) -> sb3.Block:
  return sb3.EditList("replaceat", stack_var, offsetStackSize(stack_size_var, offset), value)

def loadFromStack(stack_var: str, stack_size_var: str, offset: int) -> sb3.Value:
  return sb3.GetOfList("atindex", stack_var, offsetStackSize(stack_size_var, offset))

def getCallArguments(args: list[ir.Value], ctx: Context, bctx: BlockInfo) -> tuple[list[sb3.Value], sb3.BlockList]:
  arguments: list[sb3.Value] = []
  blocks = sb3.BlockList()
  for arg in args:
    value = decodeValue(arg, ctx, bctx)
    if isinstance(value, IndexableValueAndBlocks):
      raise CompException(f"Function argument cannot be an indexable value")
    blocks.add(value.blocks)
    arguments.append(value.value)
  return arguments, blocks

def assignParameters(params: list[Variable], next_var_use_depends: set[str]) -> sb3.BlockList:
  blocks = sb3.BlockList()
  for param in params:
    var = deepcopy(param)
    var.var_type = "var"
    # Don't assign anything we depend upon in future
    if var.var_name in next_var_use_depends:
      blocks.add(var.setValue(param.getValue()))
  return blocks

def getUncheckedProcedureStart(proc_name: str, params: list[Variable], fn: FuncInfo,
                               ctx: Context, is_counted: bool=False) -> tuple[sb3.BlockList, Context]:
  blocks = sb3.BlockList([sb3.ProcedureDef(proc_name, [param.getRawVarName() for param in params])])

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

def getCheckedProcedureStart(proc_name: str, params: list[Variable],
                             next_var_use_depends: set[str], fn: FuncInfo,
                             ctx: Context) -> tuple[sb3.BlockList, Context]:
  """
  Returns the blocks needed to return a branch instruction (procedure)
  that will reset the scratch's stack if reaching a max amount of permutations,
  preventing scratch from running out of memory
  """

  reset_broadcast = f"{proc_name}:reset stack"
  arguments = [name.getValue() for name in params]
  ctx.proj.code.append(sb3.BlockList([
    sb3.OnBroadcast(reset_broadcast),
    sb3.ProcedureCall(proc_name, arguments),
  ]))

  blocks, ctx = getUncheckedProcedureStart(proc_name, params, fn, ctx, is_counted=True)

  on_reset = sb3.BlockList()

  on_reset = assignParameters(params, next_var_use_depends)
  on_reset.add(sb3.BlockList([
    sb3.Broadcast(sb3.Known(reset_broadcast), False), # While broadcasts are slow, this is the fastest option in this rare case
    sb3.EditCounter("clear"),
    sb3.StopScript("stopthis")]))

  blocks.add(sb3.ControlFlow("if",
    sb3.BoolOp(">", sb3.GetCounter(), sb3.Known(ctx.cfg.max_branch_recursion)), on_reset))

  return blocks, ctx

def transSimpleCall(name: str, arguments: list[sb3.Value],
              output: Variable | None, ctx: Context) -> tuple[sb3.BlockList, sb3.BlockList]:
  """
  Translates simple function calls. Deals with passing parameters and
  return values. The first block list returned is any blocks to call
  the function, the second any needed to assign the return value to
  the output.
  """

  call_blocks = sb3.BlockList([sb3.ProcedureCall(name, arguments)])

  set_value_blocks = sb3.BlockList()
  if output is not None:
    set_value_blocks.add(output.setValue(sb3.GetVar(ctx.cfg.return_var)))

  return call_blocks, set_value_blocks

def transComplexCall(caller: FuncInfo, callee: FuncInfo,
                     args: list[ir.Value], result: Variable | None,
                     following_instrs: list[ir.Instruction],
                     ctx: Context, bctx: BlockInfo) -> tuple[Context, BlockInfo]:
  """
  Translates a function call. Deals with functions with return
  addresses and recursion. May change the function that instructions
  are being added to.
  """

  assert bctx.label is not None

  # Include the return value in variables which aren't depended on
  starting_var_use = BlockVarUse() if result is None else BlockVarUse(modifies={"%" + result.var_name})
  # All variables that might be depended on/modified after the function is called
  next_var_use = getBlockVarUse(following_instrs, caller.block_var_use, starting_var_use)

  poss_recursive = caller.name in callee.can_call
  if poss_recursive:
    # Get all variables which are used later after the recursion
    must_store: list[Variable] = []
    # Include the return value in variables which aren't depended on
    for var in next_var_use.depends:
      decoded_var = decodeVar(var, bctx)
      assert decoded_var is not None
      must_store.append(decoded_var)

    # Sort the parameters in numeric then alphabetical order for better readability
    must_store.sort(key=lambda var: (0, int(var.var_name)) if var.var_name.isdigit() else (1, var.var_name))

    if caller.total_alloca_size is None and not caller.skip_stack_size_change:
      must_store.append(localizeVar(ctx.cfg.previous_stack_size_local, False, bctx))

    if caller.takes_return_address:
      must_store.append(localizeVar(ctx.cfg.return_address_local, False, bctx))

    # If we don't need to store any parameters for later we don't need to do anything special when we recurse
    poss_recursive = len(must_store) > 0

  if callee.returns_to_address:
    return_proc_name = localizeCallId(bctx.next_call_id, bctx.label, caller.name)
    return_addr_id = callee.return_addresses.index(return_proc_name)

  if not poss_recursive: # TODO OPTI: this can also be used if possibly recusive but we don't depend on anything after
    arguments, arg_value_blocks = getCallArguments(args, ctx, bctx)

    if callee.returns_to_address:
      if callee.takes_return_address:
        # Pass return address into function
        arguments.append(sb3.Known(return_addr_id))
      # Make sure parameters can be accessed later
      bctx.code.add(assignParameters(bctx.available_params, {dependent[1:] for dependent in next_var_use.depends}
                                                            | ctx.cfg.special_locals))

    call_blocks, assign_blocks = transSimpleCall(callee.name, arguments, result, ctx)
    bctx.code.add(arg_value_blocks)
    bctx.code.add(call_blocks)

    if not callee.returns_to_address:
      bctx.code.add(assign_blocks)
    else:
      # Start new block list
      ctx.proj.code.append(bctx.code)
      bctx.first_block = False
      bctx.available_params = []

      # Add code for callback
      bctx.code, ctx = getUncheckedProcedureStart(return_proc_name, [], caller, ctx,
                                                  is_counted=callee.returns_to_address)
      bctx.code.add(assign_blocks)
  else:
    if not callee.returns_to_address:
      # Use the parameters for procedures to use scratch's stack to store any variables needed later
      recurse_proc_name = localizeCallId(bctx.next_call_id, bctx.label, caller.name, True)
      bctx.code.add(sb3.ProcedureCall(recurse_proc_name, [var.getValue() for var in must_store]))

      for i, var in enumerate(must_store):
        must_store[i].var_type = "param"

      # Start new block list
      bctx.first_block = False
      ctx.proj.code.append(bctx.code)
      # Make sure that these parameters are assigned back to variables if needed later
      bctx.available_params = must_store
      bctx.code = sb3.BlockList([sb3.ProcedureDef(recurse_proc_name, [var.getRawVarName() for var in must_store])])

      arguments, arg_value_blocks = getCallArguments(args, ctx, bctx)
      call_blocks, assign_blocks = transSimpleCall(callee.name, arguments, result, ctx)
      bctx.code.add(arg_value_blocks)
      bctx.code.add(call_blocks)
      bctx.code.add(assign_blocks)
    else:
      bctx.code.add(sb3.EditVar("change", ctx.cfg.local_stack_size_var, sb3.Known(len(must_store))))

      for i, var in enumerate(must_store):
        bctx.code.add(storeOnStack(ctx.cfg.local_stack_var, ctx.cfg.local_stack_size_var, -i, var.getValue()))
        must_store[i].var_type = "var"

      arguments, arg_value_blocks = getCallArguments(args, ctx, bctx)
      arguments.append(sb3.Known(return_addr_id))

      call_blocks, assign_blocks = transSimpleCall(callee.name, arguments, result, ctx)
      bctx.code.add(arg_value_blocks)
      bctx.code.add(call_blocks)

      ctx.proj.code.append(bctx.code)
      bctx.first_block = False
      bctx.available_params = []

      # Add code for callback
      bctx.code, ctx = getUncheckedProcedureStart(return_proc_name, [], caller, ctx,
                                                  is_counted=callee.returns_to_address)
      bctx.code.add(assign_blocks)

      for i, var in enumerate(must_store):
        bctx.code.add(var.setValue(loadFromStack(ctx.cfg.local_stack_var, ctx.cfg.local_stack_size_var, -i)))

      bctx.code.add(sb3.EditVar("change", ctx.cfg.local_stack_size_var, sb3.Known(-len(must_store))))

  bctx.next_call_id += 1

  return ctx, bctx

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

def transInstr(instr: ir.Instruction, ctx: Context, bctx: BlockInfo) -> tuple[sb3.BlockList, Context, BlockInfo]:
  blocks = sb3.BlockList()
  match instr:
    case ir.Alloca(): # Allocate space on the stack and return ptr
      var = decodeVar(instr.result, bctx)
      assert var is None or var.var_type != "param"

      blocks = sb3.BlockList()
      size = getByteSize(instr.allocated_ty)

      assert bctx.label is not None
      if bctx.fn.skip_stack_size_change:
        offset = 0
      elif bctx.fn.total_alloca_size is None:
        offset = -bctx.fn.block_alloca_size[bctx.label]
      else:
        offset = -bctx.fn.total_alloca_size
      offset += bctx.allocated + 1 # Adding one means that some addresses will be equal to the stack size, allowing better
                                   # optimization because the extra add/subtract can be ignored
      bctx.allocated += size

      if var is not None:
        blocks.add(var.setValue(offsetStackSize(ctx.cfg.stack_size_var, offset)))

    case ir.Load(): # Load a value from an address on the stack
      address = decodeValue(instr.address, ctx, bctx)
      blocks.add(address.blocks)

      if isinstance(address.value, IndexableValue):
        raise CompException(f"Address to load cannot be an indexable value in {instr}")

      var = decodeVar(instr.result, bctx)
      if var is not None:
        blocks.add(sb3.EditVar("set", var.getRawVarName(),
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
      assert res_var is None or res_var.var_type != "param"
      if res_var is not None:
        res_var_name = res_var.getRawVarName()
      res_val = None

      if isinstance(left, IndexableValue) or \
         isinstance(right, IndexableValue):
        raise CompException(f"Indexable value not supported in binop {instr}")

      match instr.opcode:
        case "add" | "sub": # Add/Sub two values
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values by using multiple vars and carrying
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          if instr.is_nsw and instr.is_nuw and ctx.cfg.opti:
            # If no wrapping behaviour is required then under/overflowing is ub so can be ignored
            res_val = sb3.Op(instr.opcode, left, right)
          else:
            res_val = sb3.Op("mod", sb3.Op(instr.opcode, left, right),
                             sb3.Known(2 ** width))

        case "mul": # Multiply two values
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")
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
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          # Division by zero is UB
          if not instr.is_exact:
            res_val = sb3.Op("floor", sb3.Op("div", left, right))
          else:
            res_val = sb3.Op("div", left, right) # Value is poison if one is not a multiple of another

        case "sdiv": # Divide one value by another (signed)
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")
          width = instr.fst_operand.ty.num_bits

          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports integers with <= {VARIABLE_MAX_BITS} bits")

          if instr.is_exact:
            signed_left, lblocks = flatAsTuple(undoTwosComplement(width, left, False, ctx))
            signed_right, rblocks = flatAsTuple(undoTwosComplement(width, right, False, ctx))
            blocks.add(lblocks)
            blocks.add(rblocks)

            res_val = twosComplement(width, sb3.Op("div", signed_left, signed_right))
          else:
            if res_var is not None:
              left, lblocks = flatAsTuple(optimizeValueUse(left, 2, ctx))
              right, rblocks = flatAsTuple(optimizeValueUse(right, 2, ctx))
              blocks.add(lblocks)
              blocks.add(rblocks)

              point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
              change = 2 ** width

              # TODO: optimise for known values

              # Undo two's complement, divide, round towards zero using floor or ceiling and calculate two's complement
              blocks.add([
                sb3.ControlFlow("if_else", sb3.BoolOp("<", left, sb3.Known(point_of_neg)), sb3.BlockList([
                  sb3.ControlFlow("if_else", sb3.BoolOp("<", right, sb3.Known(point_of_neg)), sb3.BlockList([
                    # If left + right are pos
                    sb3.EditVar("set", res_var_name, sb3.Op("floor", sb3.Op("div", left, right))),
                  ]), sb3.BlockList([
                    # If left is pos and right is neg
                    sb3.EditVar("set", res_var_name, sb3.Op("add",
                                                      sb3.Op("ceiling",
                                                        sb3.Op("div",
                                                          left,
                                                          sb3.Op("sub", right, sb3.Known(change)))),
                                                      sb3.Known(change))),
                  ]))
                ]), sb3.BlockList([
                  sb3.ControlFlow("if_else", sb3.BoolOp("<", right, sb3.Known(point_of_neg)), sb3.BlockList([
                    # If left is neg and right is pos
                    sb3.EditVar("set", res_var_name, sb3.Op("add",
                                                      sb3.Op("ceiling",
                                                        sb3.Op("div",
                                                          sb3.Op("sub", left, sb3.Known(change)),
                                                          right)),
                                                      sb3.Known(change))),
                  ]), sb3.BlockList([
                    # If left + right are neg
                    sb3.EditVar("set", res_var_name, sb3.Op("floor",
                                                      sb3.Op("div",
                                                        sb3.Op("sub", left, sb3.Known(change)),
                                                        sb3.Op("sub", right, sb3.Known(change))))),
                  ]))
                ]))
              ])
            res_val = False # We set res_var ourselves

        case "urem": # Calculate remainder (unsigned)
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports"
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          # mod 0 is UB, can ignore
          res_val = sb3.Op("mod", left, right)

        case "srem": # Calculate remainder (signed)
          # TODO OPTI: optimise for known values
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          if res_var is not None:
            # TODO: Reuse if statement to work out if a / b > 0
            left, lblocks = flatAsTuple(optimizeValueUse(left, 2, ctx))
            right_is_temp = shouldOptimiseValueUse(right, 3)
            right, rblocks = flatAsTuple(optimizeValueUse(right, 3, ctx))
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
                sb3.EditVar("set", res_var_name, sb3.Op("sub",
                                                  sb3.Op("mod",
                                                    left,
                                                    right_sub_change),
                                                  right_sub_change))]))
            else:
              # Re-use the generated temp var
              pos_neg_block = sb3.BlockList([
                sb3.EditVar("change", right.var_name, sb3.Known(-change)),
                sb3.EditVar("set", res_var_name, sb3.Op("sub",
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
                    sb3.EditVar("set", res_var_name, sb3.Op("mod", left, right)),
                  ]),
                  # If left is pos and right is neg - remainder = (l mod r) - r
                  pos_neg_block
                )
              ]), sb3.BlockList([
                sb3.ControlFlow("if_else", sb3.BoolOp("<", right, sb3.Known(point_of_neg)), sb3.BlockList([
                  # If left is neg and right is pos - remainder = (l mod r) - r
                  sb3.EditVar("set", res_var_name, sb3.Op("add",
                                                    sb3.Op("sub",
                                                      sb3.Op("mod",
                                                        sb3.Op("sub", left, sb3.Known(change)),
                                                        right),
                                                      right),
                                                    sb3.Known(change))),
                ]), sb3.BlockList([
                  # If left + right are neg
                  sb3.EditVar("set", res_var_name, sb3.Op("add",
                                                    sb3.Op("mod",
                                                      sb3.Op("sub", left, sb3.Known(change)),
                                                      sb3.Op("sub", right, sb3.Known(change))),
                                                    sb3.Known(change))),
                ]))
              ]))
            ])

          res_val = False # We set res_var ourselves

        case "shl": # Calculate left shift
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          can_shift_out = not (instr.is_nsw and instr.is_nuw)
          res_val, ctx = bitShift("left", width, left, right, ctx, can_shift_out)

        case "lshr": # Calculate right shift (unsigned)
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          can_shift_out = not instr.is_exact
          res_val, ctx = bitShift("right", width, left, right, ctx, can_shift_out)

        case "ashr": # Calculate right shift (signed)
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          if res_var is not None:
            point_of_neg = int(((2 ** width) / 2)) # Point at which a two's compilment number is negative
            change = 2 ** width

            right_mul, ctx = intPow2(right, width, ctx)

            unwrapped_pos = sb3.Op("div", left, right_mul)
            val_pos = unwrapped_pos if instr.is_exact else sb3.Op("floor", unwrapped_pos)

            unwrapped_neg = sb3.Op("div", sb3.Op("sub", left, sb3.Known(change)), right_mul)
            val_neg = sb3.Op("add",
                        unwrapped_neg if instr.is_exact else sb3.Op("ceiling", unwrapped_neg),
                        sb3.Known(change))

            blocks.add([
              sb3.ControlFlow("if_else", sb3.BoolOp("<", left, sb3.Known(point_of_neg)), sb3.BlockList([
                sb3.EditVar("set", res_var_name, val_pos),
              ]), sb3.BlockList([
                sb3.EditVar("set", res_var_name, val_neg),
              ])),
            ])

          res_val = False # We set res_var ourselves

        case "and" | "or" | "xor": # Calculate binary operation
          if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
            raise CompException(f"Instruction {instr} with opcode add only supports "
                                f"integers, got type {type(instr.fst_operand.ty)}")

          width = instr.fst_operand.ty.num_bits
          # TODO FIX: support larger values
          if width > VARIABLE_MAX_BITS:
            raise CompException(f"Instruction {instr} currently supports "
                                f"integers with <= {VARIABLE_MAX_BITS} bits")

          if instr.opcode == "or" and instr.is_disjoint:
            # If there would be no carry
            res_val = sb3.Op("add", left, right)
          else:
            # TODO OPTI: gen (11-bit) tables/use mod for known values
            if width > ctx.cfg.binop_lookup_bits:
              uses = width / ctx.cfg.binop_lookup_bits
              left, lblocks = flatAsTuple(optimizeValueUse(left, uses, ctx))
              right, rblocks = flatAsTuple(optimizeValueUse(right, uses, ctx))
              blocks.add(lblocks)
              blocks.add(rblocks)

            lookup_size = 2 ** ctx.cfg.binop_lookup_bits
            name = f"{ctx.cfg.binop_lookup_var} {instr.opcode}{ctx.cfg.zero_indexed_suffix}"

            if name not in ctx.proj.lists:
              lookup: list[int | float | str | bool] = []
              if instr.opcode == "and":
                lookup = [l & r for l in range(lookup_size) for r in range(lookup_size)]
              elif instr.opcode == "or":
                lookup = [l | r for l in range(lookup_size) for r in range(lookup_size)]
              elif instr.opcode == "xor":
                lookup = [l ^ r for l in range(lookup_size) for r in range(lookup_size)]
              else:
                raise CompException(f"Unknown opcode {instr.opcode}")
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
      assert res_var is None or res_var.var_type != "param"

      if not isinstance(instr.fst_operand.ty, ir.IntegerType): # TODO: add vector support
        raise CompException(f"Instruction {instr} with opcode add only supports "
                            f"integers, got type {type(instr.fst_operand.ty)}")
      assert isinstance(left, sb3.Value) and isinstance(right, sb3.Value)

      width = instr.fst_operand.ty.num_bits
      # TODO FIX: support larger values
      if width > VARIABLE_MAX_BITS:
        raise CompException(f"Instruction icmp currently supports "
                            f"integers with <= {VARIABLE_MAX_BITS} bits")

      match instr.cond:
        case "eq":
          res_val = sb3.BoolOp("=", left, right)
        case "ne":
          res_val = sb3.BoolOp("not", sb3.BoolOp("=", left, right))
        case "ugt":
          res_val = sb3.BoolOp(">", left, right)
        case "uge":
          if isinstance(left, sb3.Known):
            res_val = sb3.BoolOp(">", sb3.Known(sb3.scratchCastToNum(left) + 1), right)
          elif isinstance(right, sb3.Known):
            res_val = sb3.BoolOp(">", left, sb3.Known(sb3.scratchCastToNum(right) - 1))
          else:
            res_val = sb3.BoolOp("not", sb3.BoolOp("<", left, right))
        case "ult":
          res_val = sb3.BoolOp("<", left, right)
        case "ule":
          if isinstance(left, sb3.Known):
            res_val = sb3.BoolOp("<", sb3.Known(sb3.scratchCastToNum(left) - 1), right)
          elif isinstance(right, sb3.Known):
            res_val = sb3.BoolOp("<", left, sb3.Known(sb3.scratchCastToNum(right) + 1))
          else:
            res_val = sb3.BoolOp("not", sb3.BoolOp(">", left, right))
        case _:
          # TODO FIX: signed versions
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
      assert res_var is None or res_var.var_type != "param"

      if not isinstance(instr.value.ty, ir.IntegerType): # TODO: add vector support
        raise CompException(f"Instruction {instr} with opcode add only supports "
                            f"integers, got type {type(instr.value.ty)}")
      assert isinstance(value, sb3.Value)

      width = instr.value.ty.num_bits
      # TODO FIX: support larger values
      if width > VARIABLE_MAX_BITS:
        raise CompException(f"Instruction icmp currently supports "
                            f"integers with <= {VARIABLE_MAX_BITS} bits")

      res_val = None

      match instr.opcode:
        case "zext":
          res_val = value
        case _:
          raise CompException(f"Unknown instruction opcode {instr} (type Conversion)")

      if res_val is not None and res_var is not None:
        blocks.add(res_var.setValue(res_val))

    case ir.Select(): # Select between two values based on a condition
      cond = decodeValue(instr.cond, ctx, bctx)
      true_val = decodeValue(instr.true_value, ctx, bctx)
      false_val = decodeValue(instr.false_value, ctx, bctx)

      if not all(isinstance(val.ty, ir.IntegerType) for val in \
          (instr.cond, instr.true_value, instr.false_value)): # TODO: add vector support
        raise CompException(f"Instruction {instr} with opcode add only supports "
                            f"integers, got other type")
      assert isinstance(instr.cond.ty, ir.IntegerType)
      assert instr.cond.ty.num_bits == 1
      assert isinstance(cond.value, sb3.Value)
      assert isinstance(true_val.value, sb3.Value)
      assert isinstance(false_val.value, sb3.Value)

      res_var = decodeVar(instr.result, bctx)
      assert res_var is None or res_var.var_type != "param"

      if res_var is not None:
        blocks.add(cond.blocks)
        blocks.add(sb3.ControlFlow("if_else", sb3.BoolOp("=", cond.value, sb3.Known(1)), sb3.BlockList([
          *true_val.blocks.blocks,
          res_var.setValue(true_val.value)
        ]), sb3.BlockList([
          *false_val.blocks.blocks,
          res_var.setValue(false_val.value)
        ])))

    case _:
      raise CompException(f"Unsupported instruction opcode {instr} (type {type(instr)})")
  return blocks, ctx, bctx

def getTerminatorInstrLabels(instr: ir.Instruction) -> list[str]:
  """
  Returns every label a terminator instruction could branch to.
  Returns the string "ret" to indictate a return
  """
  match instr:
    case ir.Unreacheble():
      return []
    case ir.Ret():
      return ["ret"]
    case ir.Br():
      assert isinstance(instr.label_false.val, str)
      branches = [instr.label_false.val]
      if instr.label_true is not None:
        assert isinstance(instr.label_true.val, str)
        branches.append(instr.label_true.val)
      return branches
    case ir.Switch():
      assert isinstance(instr.label_default.val, str)
      branches = [instr.label_default.val]
      for _, label in instr.cases:
        assert isinstance(label.val, str)
        branches.append(label.val)
      return branches
    case ir.CallBr():
      assert isinstance(instr.fallthrough_label.val, str)
      if instr.indirect_labels is not None:
        raise CompException("Indirect (exceptional) labels not supported")
      return [instr.fallthrough_label.val]
    case _:
      raise CompException(f"Unsupported terminator instruction opcode {instr} (type {type(instr)})")

def getInstrVarUse(instr: ir.Instruction) -> tuple[set[str], set[str]]:
  """Returns what the instruction depends on and modifies"""
  depends: set[str] = set()
  modifies: set[str] = set()

  result = getattr(instr, "result", None)
  if result is not None:
    assert isinstance(result.val, str)
    if result.val.startswith("%"):
      modifies.add(result.val)

  vals: list[ir.Value | None]
  match instr:
    case ir.Unreacheble() | ir.Alloca():
      vals = []
    case ir.Ret() | ir.Conversion():
      vals = [instr.value]
    case ir.Load():
      vals = [instr.address]
    case ir.Store():
      vals = [instr.address, instr.value]
    case ir.Call():
      vals = [v for v in instr.args]
    case ir.BinOp():
      vals = [instr.fst_operand, instr.snd_operand]
    case ir.ICmp():
      vals = [instr.fst_operand, instr.snd_operand]
    case ir.Br() | ir.Switch():
      vals = [instr.cond]
    case ir.Select():
      vals = [instr.cond, instr.true_value, instr.false_value]
    case _:
      raise CompException(f"Unknown instruction {instr} (type {type(instr)})")

  for val in vals:
    if val is not None:
      match val.val:
        # TODO FIX: special handling for getElementPtr etc
        case str() | int() | bytes():
          pass
        case _:
          raise CompException(f"Unknown Value: {val.val} (type: {type(val.val)})")

      if isinstance(val.val, str) and val.val.startswith("%"):
        depends.add(val.val)

  return depends, modifies

def getBlockVarUse(instrs: list[ir.Instruction],
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
    instr_depends, instr_modifies = getInstrVarUse(instr)
    # If we modify something before using it then we don't depend on it
    res.depends |= instr_depends - res.modifies
    res.modifies |= instr_modifies

  res.branches = set(getTerminatorInstrLabels(instrs[-1])) - {"ret"}
  if block_var_use is not None:
    modified = deepcopy(res.modifies)
    for label in res.branches:
      res.depends |= block_var_use[label].depends - modified
      res.modifies |= block_var_use[label].modifies

  return res

def getFuncBranchesVarUse(func: ir.Function) -> dict[str, BlockVarUse]:
  label_var_use: dict[str, BlockVarUse] = {
    name: getBlockVarUse(block.instrs) for name, block in func.blocks.items()
  }

  memo: dict[tuple[str, frozenset[str]], BlockVarUse] = {}

  res: dict[str, BlockVarUse] = {}

  for start_label in func.blocks:
    # Worklist (current label, depends so far, modifies so far)
    stack: list[tuple[str, set[str], set[str]]] = [(start_label,
                                                    set(label_var_use[start_label].depends),
                                                    set(label_var_use[start_label].modifies))]
    # Track visited states per block to prevent infinite loops
    visited_states: dict[str, set[frozenset[str]]] = {label: set() for label in func.blocks}

    agg_depends: set[str] = set()
    agg_modifies: set[str] = set()
    agg_branches: set[str] = set()

    while stack:
      label, depends_so_far, modifies_so_far = stack.pop()
      state_key = frozenset(depends_so_far | modifies_so_far)
      if state_key in visited_states[label]:
        continue
      visited_states[label].add(state_key)

      block_use = label_var_use[label]

      # Update dependencies (only if not modified already)
      new_depends = block_use.depends - modifies_so_far
      depends_so_far = depends_so_far | new_depends
      modifies_so_far = modifies_so_far | block_use.modifies

      # Aggregate results for this start label
      agg_depends |= new_depends
      agg_modifies |= block_use.modifies
      agg_branches |= block_use.branches

      # Push successors with updated state
      for succ in block_use.branches:
        stack.append((succ, set(depends_so_far), set(modifies_so_far)))

    res[start_label] = BlockVarUse(agg_depends, agg_modifies, agg_branches)

  return res

def transTerminatorInstr(instr: ir.Instruction,
                         ctx: Context, bctx: BlockInfo) -> tuple[sb3.BlockList, Context]:
  # Work out what variables might be depended on in future
  poss_branch = set(getTerminatorInstrLabels(instr)) - {"ret"}
  poss_depends: set[str] = set()
  for branch in poss_branch:
    poss_depends |= {dependent[1:] for dependent in bctx.fn.block_var_use[branch].depends}
  poss_depends |= ctx.cfg.special_locals

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

      # Change the stack size after setting the return value because some return values might be optimized to use
      # the stack size var
      if not bctx.fn.skip_stack_size_change:
        if bctx.fn.total_alloca_size is not None:
          blocks.add(sb3.EditVar("change", ctx.cfg.stack_size_var, sb3.Known(-bctx.fn.total_alloca_size)))
        else:
          blocks.add(sb3.EditVar("set", ctx.cfg.stack_size_var, localizeVar(ctx.cfg.previous_stack_size_local,
                                                                            False, bctx).getValue()))

      if bctx.fn.returns_to_address:
        if bctx.fn.takes_return_address:
          return_address = localizeVar(ctx.cfg.return_address_local, False, bctx)
          return_to_addr_code = OrderedDict()
          for i, addr in enumerate(bctx.fn.return_addresses):
            return_to_addr_code.update({i: sb3.BlockList([sb3.ProcedureCall(addr, [])])})

          return_table = binarySearch(return_address.getValue(), return_to_addr_code, are_branches_sorted=True)
          blocks.add(return_table)
        elif len(bctx.fn.return_addresses) > 0:
          # If the function doesn't take a return address then it must always return to the same place
          assert len(bctx.fn.return_addresses) == 1
          blocks.add(sb3.BlockList([sb3.ProcedureCall(bctx.fn.return_addresses[0], [])]))
        else:
          pass # TODO FIX: if a function ever returns here the stack will unwind which may be unexpected
               # so the stop all block could be used. This happens with the main function

      # TODO FIX: deallocate on ret instruction (only if function can allocate)
      blocks.end = True

    case ir.Br(): # Jump to a label, either known or dependent on a condition
      # Allow the parameters to be accessed later
      blocks.add(assignParameters(bctx.available_params, poss_depends))

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
      # Allow the parameters to be accessed later
      blocks.add(assignParameters(bctx.available_params, poss_depends))

      assert isinstance(instr.cond.ty, ir.IntegerType)
      width = instr.cond.ty.num_bits
      if getByteSize(instr.cond.ty) > 1:
        raise CompException("Cannot currently switch with an integer more "
                           f"than {VARIABLE_MAX_BITS} bits (would take multiple vars to store)")

      val, val_blocks = flatAsTuple(decodeValue(instr.cond, ctx, bctx))
      blocks.add(val_blocks)
      if len(instr.cases) > 1:
        val, opti_val_blocks = flatAsTuple(optimizeValueUse(val, math.log2(len(instr.cases)), ctx))
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

def getFnInfo(mod: ir.Module, ctx: Context) -> Context:
  """Get info about a function needed to translate it's instructions"""
  call_graph: dict[str, tuple[set[str], bool]] = {}
  return_addresses: dict[str, list[str]] = {}
  returns_to_address: dict[str, bool] = {}
  all_check_locations: dict[str, list[str]] = {}
  branch_alloca_size: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
  total_alloca_size: dict[str, int | None] = {}
  block_var_use: dict[str, dict[str, BlockVarUse]] = {}
  branches_to_first: dict[str, bool] = {}

  for func in mod.funcs.values():
    fn_name = func.value.val
    assert isinstance(fn_name, str)

    if func.has_no_body():
      if fn_name not in ctx.fn_info:
        raise CompException(f"Externally defined function {fn_name} must have info about it")
      info = ctx.fn_info[fn_name]
      call_graph[fn_name] = info.can_call, True
      block_var_use[fn_name] = info.block_var_use

    else:
      calls: set[str] = set() # What the function calls
      for block in func.blocks.values():
        # Find every function the function could call
        block_label = block.value.val
        assert isinstance(block_label, str)
        call_id = 0
        for instr in block.instrs:
          match instr:
            case ir.Call() | ir.CallBr(): # TODO FIX: tail recursion treated differently?
              called_name = instr.callee.val
              assert isinstance(called_name, str)
              calls.add(called_name)
              return_addresses.setdefault(called_name, list())
              return_addresses[called_name].append(localizeCallId(call_id, block_label, fn_name))
              call_id += 1
            case ir.Alloca():
              branch_alloca_size[fn_name][block_label] += getByteSize(instr.allocated_ty)
      call_graph[fn_name] = calls, False

      block_var_use[fn_name] = getFuncBranchesVarUse(func)

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

  for func in mod.funcs.values():
    fn_name = func.value.val
    assert isinstance(fn_name, str)

    if func.has_no_body():
      info = ctx.fn_info[fn_name]
      returns_to_address[fn_name] = info.returns_to_address
      all_check_locations[fn_name] = info.checked_blocks
      total_alloca_size[fn_name] = info.total_alloca_size
    else:
      first_label = list(func.blocks.values())[0].value.val
      assert isinstance(first_label, str)
      branches: dict[str, list[str]] = {"ret": []} # Where different branches could lead
      for block in func.blocks.values():
        # Find every function the function could call
        block_label = block.value.val
        assert isinstance(block_label, str)

        # Find what each block in the function could branch to
        branches[block_label] = getTerminatorInstrLabels(block.instrs[-1])

      cycles = util.find_all_cycles(branches)
      fn_check_locations = util.select_cycle_checks(cycles)
      could_recurse = len(fn_check_locations) > 0
      # If the branches could create a loop, we must place stack checks, so we should return to an
      # address. Furthermore, a binary search and a call is usually faster than potentially
      # hundreds of recursions backward.
      returns_to_address[fn_name] = could_recurse
      all_check_locations[fn_name] = fn_check_locations

      # Any branch that may be called more than once
      repeating_branches: set[str] = set().union(*[set(branch) for branch in cycles])
      # Branches that are unavoidable
      unavoidable_branches: set[str] = util.unavoidable_nodes(branches, first_label, "ret")
      # Branches that are always ran once per func call
      ran_once_branches = unavoidable_branches - repeating_branches

      known_alloc_size = True
      alloc_size = 0
      for block_label in branches.keys():
        if block_label in ran_once_branches:
          alloc_size += branch_alloca_size[fn_name][block_label]
        elif branch_alloca_size[fn_name][block_label] != 0:
          known_alloc_size = False
          break

      total_alloca_size[fn_name] = alloc_size if known_alloc_size else None

      # FUTURE OPTI: could check to see if in a non-recursive environment, there are enough
      # branches that it is faster to recurse backward. However this happens rarely, it is
      # difficult to gauge if it will actually have a net positive impact (will force callers
      # to return to an address, the longest path may only be taken some of the times).

      fn_branches_to_first = False
      if len(func.blocks) > 0:
        first_block_label = list(func.blocks.values())[0].value.val
        assert isinstance(first_block_label, str)
        fn_branches_to_first = any([first_block_label in branch_to for branch_to in branches.values()])
      branches_to_first[fn_name] = fn_branches_to_first

  for fn_name in returns_to_address:
    # If the function returns to an address, then callers must also return to an address
    returns_to_address[fn_name] |= any(returns_to_address[call] for call in call_graph[fn_name][0])

  for func in mod.funcs.values():
    fn_name = func.value.val
    assert isinstance(fn_name, str)

    # If we know the total size a function allocates and that it doesn't call any functions
    # that rely on the stack size, we don't need to increase the stack size
    skip_stack_size_change = total_alloca_size[fn_name] is not None and \
      all(total_alloca_size[call] == 0 for call in call_graph[fn_name][0])

    fn_ret_addresses = return_addresses.get(fn_name, list())
    fn_returns_to_address = returns_to_address[fn_name]
    # If the function returns to an address and is called from multiple locations,
    # then it must take a return address to know where to return to
    fn_takes_ret_addr = fn_returns_to_address and len(fn_ret_addresses) > 1

    param_names = []
    for arg in func.args:
      assert isinstance(arg.val, str)
      assert arg.val.startswith("%")
      param_names.append(arg.val[1:])

    if fn_takes_ret_addr: param_names.append(ctx.cfg.return_address_local)
    params = [Variable(name, "param", fn_name) for name in param_names]

    ctx.fn_info[fn_name] = FuncInfo(fn_name, ctx.next_fn_id, params, call_graph[fn_name][0],
                                    fn_ret_addresses, fn_returns_to_address, fn_takes_ret_addr,
                                    all_check_locations.get(fn_name, list()),
                                    branch_alloca_size[fn_name], total_alloca_size.get(fn_name, None),
                                    skip_stack_size_change, block_var_use[fn_name],
                                    branches_to_first.get(fn_name, False))
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

    is_first_block = True
    total_fn_allocated = 0
    for block in func.blocks.values():
      call_id = 0
      block_label = block.value.val
      assert isinstance(block_label, str)

      if is_first_block:
        proc_name = fn_name
        localized_params = info.params
      else:
        proc_name = localizeLabel(block_label, info)
        localized_params = []

      # Get code to start the branch (procedure definition, etc)
      if (block_label not in info.checked_blocks) or (is_first_block and info.branches_to_first):
        starting_fn_code, ctx = getUncheckedProcedureStart(proc_name, localized_params, info, ctx,
                                                           is_counted=info.returns_to_address)
      else:
        next_var_use_depends = {dependent[1:] for dependent in info.block_var_use[block_label].depends}
        next_var_use_depends |= ctx.cfg.special_locals
        starting_fn_code, ctx = getCheckedProcedureStart(proc_name, localized_params,
                                                         next_var_use_depends, info, ctx)

      if is_first_block and ctx.cfg.do_debug_branch_log:
        # Increment the function counter in the log
        index = sb3.Known((info.fn_id * 2) + 1)
        starting_fn_code.add(sb3.BlockList([
          sb3.EditList("replaceat", ctx.cfg.debug_branch_log_var, index, sb3.Op("add",
            sb3.Known(1), sb3.GetOfList("atindex", ctx.cfg.debug_branch_log_var, index)
        ))]))

      # Store the previous stack size if necessary
      if is_first_block and info.total_alloca_size is None and not info.skip_stack_size_change:
        starting_fn_code.add(localizeVar(ctx.cfg.previous_stack_size_local, False, BlockInfo(info, info.params))
          .setValue(sb3.GetVar(ctx.cfg.stack_size_var)))

      if is_first_block and info.branches_to_first:
        # Work out what variables might be depended on in future
        poss_depends = {dependent[1:] for dependent in info.block_var_use[block_label].depends}
        poss_depends |= ctx.cfg.special_locals
        starting_fn_code.add(assignParameters(info.params, poss_depends))

        first_block_proc_name = localizeLabel(block_label, info)

        starting_fn_code.add(sb3.ProcedureCall(first_block_proc_name, []))

        # TODO FIX: repeat code, use helper func
        ctx.proj.code.append(starting_fn_code)

        if block_label not in info.checked_blocks:
          starting_fn_code, ctx = getUncheckedProcedureStart(first_block_proc_name, [], info, ctx,
                                                            is_counted=info.returns_to_address)
        else:
          starting_fn_code, ctx = getCheckedProcedureStart(first_block_proc_name, [],
                                                           poss_depends, info, ctx)

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
        to_allocate = info.block_alloca_size[block_label]

      if to_allocate != 0 and not info.skip_stack_size_change:
        starting_fn_code.add(sb3.EditVar("change", ctx.cfg.stack_size_var, sb3.Known(to_allocate)))

      # After the first block we can no longer access the parameters in the function
      available_params: list[Variable] = info.params if is_first_block else []
      bctx = BlockInfo(info, available_params, first_block=is_first_block,
                       code=starting_fn_code, label=block_label,
                       allocated=total_fn_allocated)

      # Translate everything except the terminator operation
      for instr_index, instr in enumerate(block.instrs[:-1]):
        assert bctx is not None
        if isinstance(instr, ir.Call): # Call instructions handled here because they can change where code is ran
          callee_name = instr.callee.val
          assert isinstance(callee_name, str)
          callee_info = ctx.fn_info[callee_name]
          args = instr.args
          result = decodeVar(instr.result, bctx)
          following_instrs = block.instrs[instr_index + 1:]

          ctx, bctx = transComplexCall(info, callee_info, args, result, following_instrs, ctx, bctx)
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

  return ctx

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

    globvar_name = globvar.getRawVarName()
    ctx.globvar_to_ptr[globvar_name] = sb3.Known(ptr)
    if not ctx.cfg.opti:
      blocks.add(sb3.EditVar("set", globvar_name, sb3.Known(ptr)))
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

def initLocalStack(ctx: Context) -> sb3.BlockList:
  return sb3.BlockList([
    sb3.EditVar("set", ctx.cfg.local_stack_size_var, sb3.Known(0)),
    sb3.EditList("deleteall", ctx.cfg.local_stack_var, None, None),
    sb3.ControlFlow("reptimes", sb3.Known(ctx.cfg.label_stack_size), sb3.BlockList([
      sb3.EditList("addto", ctx.cfg.local_stack_var, None, sb3.Known(0)),
    ]))
  ])

def addFunc(name: str, params: list[str], total_alloca_size: int, contents: sb3.BlockList, ctx: Context) -> Context:
  """
  total_alloca_size: int of how much the function allocates to the stack.
  0 if it doesn't, None if a not fixed amount.
  """
  localized_params = [Variable(param, "param", name) for param in params]
  blocks = sb3.BlockList([sb3.ProcedureDef(name, [param.getRawVarName() for param in localized_params])])
  blocks.add(contents)
  ctx.proj.code.append(blocks)
  ctx.fn_info[name] = FuncInfo(name, ctx.next_fn_id, localized_params)
  ctx.next_fn_id += 1
  return ctx

def compile(llvm: str | ir.Module, cfg: Config | None = None) -> tuple[sb3.Project, DebugInfo]:
  """Compile LLVM IR to a scratch project. Returns a project and any debug info generated."""
  if cfg is None: cfg = Config()
  debug_info = cfg.debug_info
  scfg = sb3.ScratchConfig(
    invis_blocks=cfg.invis_blocks)

  ctx = Context(sb3.Project(scfg), cfg)

  # Parse llvm
  mod: ir.Module = parse_assembly(llvm) if isinstance(llvm, str) else llvm

  # Due to an issue with llvm2py, it is impossible to tell apart a named temporary from a global var with the same name
  assertNoNamedTemporaries(mod)

  # Starting code
  initblocks = sb3.BlockList([sb3.OnStartFlag()])

  # Reset call stack
  initblocks.add(sb3.EditCounter("clear"))

  # Setup stack
  globblocks, ctx = transGlobals(mod, ctx)
  initblocks.add(globblocks)
  initblocks.add(initLocalStack(ctx))

  # Add foreign functions
  ascii_lookup = []
  for x in range(1, 256): # Ignore zero; improves perf as scratch lists are 1 indexed and zero signifies end of string
    char = chr(x)
    if char.encode("unicode_escape").decode("ascii").startswith("\\") and char != "\\":
      ascii_lookup.append(f"\\{x:02X}")
    else:
      ascii_lookup.append(char)
  ctx.proj.lists[cfg.ascii_lookup_var + cfg.zero_indexed_suffix] = ascii_lookup

  ctx = addFunc("puts", ["input"], 0, sb3.BlockList([
    sb3.EditVar("set", "buffer", sb3.Known("")),
    sb3.EditVar("set", "ptr", sb3.GetParameter(localizeParameter("input"))),
    sb3.EditVar("set", "char", sb3.GetOfList("atindex", cfg.stack_var, sb3.GetVar("ptr"))),
    sb3.ControlFlow("until", sb3.BoolOp("=", sb3.GetVar("char"), sb3.Known(0)), sb3.BlockList([
      sb3.EditVar("set", "buffer",
        sb3.Op("join",
          sb3.GetVar("buffer"),
          sb3.GetOfList("atindex",
            (cfg.ascii_lookup_var + cfg.zero_indexed_suffix),
            sb3.GetVar("char")))),
      sb3.EditVar("change", "ptr", sb3.Known(1)),
      sb3.EditVar("set", "char", sb3.GetOfList("atindex", cfg.stack_var, sb3.GetVar("ptr"))),
    ])),
    sb3.Say(sb3.GetVar("buffer")),
    sb3.EditVar("set", cfg.return_var, sb3.Known(0)),
  ]), ctx)

  ctx = addFunc("putchar", ["input"], 0, sb3.BlockList([
    sb3.Say(sb3.GetOfList("atindex",
      (cfg.ascii_lookup_var + cfg.zero_indexed_suffix),
      sb3.GetParameter(localizeParameter("input")))),
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
  main_params_len = len(ctx.fn_info["main"].params) + int(ctx.fn_info["main"].takes_return_address)
  initblocks.add(sb3.ProcedureCall("main", [sb3.Known("")] * main_params_len))

  # Add init code
  ctx.proj.code.append(initblocks)

  # Optimise scratch project
  if cfg.opti: ctx.proj = optimizer.optimize(ctx.proj, ignore_external_change={ctx.cfg.stack_size_var})

  return ctx.proj, debug_info
