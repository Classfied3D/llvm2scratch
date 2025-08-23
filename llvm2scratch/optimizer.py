"""Scratch project post-optimiser for the LLVM -> Scratch compiler"""

from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from typing import Literal
from copy import deepcopy
from enum import Enum

import math

from . import scratch as sb3

# Prevent an infinite loop of optimisations
MAX_OPTIMIZATIONS = 50

# Time (s) per 200,000 iterations
SET_VAR_COST = 7.550
GET_VAR_COST = 1.538
GET_PARAM_COST = 1.178
MATHOP_COST = 0.765
MOD_COST = 0.715
RAND_COST = 2.473
NOT_COST = 0.725
AND_OR_COST = 0.864
COMPARISON_COST = 0.929
MATH_FUNC_COST = 1.607
JOIN_COST = 1.091
LETTER_OF_COST = 0.737
LENGTH_OF_COST = 0.483
CONTAINS_STR_COST = 1.272
BOOL_TO_INT_COST = 0.304 # Bool to int use round on a boolean value, which is cheaper than a round on a fp value
ROUND_COST = 1.250
GET_ITEM_COST = 1.679
ITEM_NUM_COST = 4.920 # Unreliable benchmark
GET_COUNTER_COST = 0.190
KNOWN_COST = 0 # Probably not zero, but is included in the cost of the block that uses it

class OptimizerException(Exception):
  """Exception in the optimizer"""
  pass

class Optimisation(str, Enum):
  ASSIGNMENT_ELISION = "Assignment Elision"
  KNOWN_VALUE_PROPAGATION = "Known Value Proagation"

ALL_OPTIMISATIONS = set(Optimisation)

@dataclass
class BlockListInfo:
  might_modify: set[str] # What the blocklist might modify
  always_modify: set[str] # What the blocklist definitely modifies
  dependent: set[str] # What variables the blocklist might depend on
  might_call: set[str] # What the function might call
  always_call: set[str] # What the function always calls
  use_counts: Counter[str] = field(default_factory=Counter)

def getInputs(block: sb3.Block) -> list[sb3.Value]:
  """Gets all inputs a block takes. Does not check contents of if blocks etc, this must be done manually"""
  match block:
    case sb3.Say() | sb3.EditVar() | sb3.ControlFlow() | sb3.Broadcast():
      return [block.value]
    case sb3.EditList():
      inputs = []
      if block.item is not None: inputs.append(block.item)
      if block.index is not None: inputs.append(block.index)
      return inputs
    case sb3.ProcedureCall():
      return block.arguments
    case _:
      return []

def setInputs(block: sb3.Block, inputs: list[sb3.Value]) -> sb3.Block:
  """Sets the inputs a block takes. Uses the same order as getInputs"""
  match block:
    case sb3.Say() | sb3.EditVar() | sb3.ControlFlow() | sb3.Broadcast():
      assert len(inputs) == 1
      block.value = inputs[0]
    case sb3.EditList():
      if block.item is not None: block.item = inputs.pop(0)
      if block.index is not None: block.index = inputs.pop(0)
    case sb3.ProcedureCall():
      assert len(inputs) == len(block.arguments)
      block.arguments = inputs
    case _:
      assert len(inputs) == 0
  return block

def simplifyValue(value: sb3.Value) -> tuple[sb3.Value, bool]:
  """Optimise a value by using context"""
  did_opti_total = False
  match value:
    case sb3.BoolOp():
      value.left, did_opti_1 = simplifyValue(value.left)
      did_opti_2 = False
      if value.right is not None:
        value.right, did_opti_2 = simplifyValue(value.right)
      did_opti_total |= did_opti_1 or did_opti_2

      if value.op == "not":
        if isinstance(value.left, sb3.KnownBool):
          did_opti_total = True
          value = sb3.KnownBool(not sb3.scratchCastToBool(value.left))

      elif isinstance(value.left, sb3.Known) and isinstance(value.right, sb3.Known):
        if value.op in ["<", ">", "="]:
          did_opti_total = True
          left = sb3.scratchCastToNum(value.left)
          right = sb3.scratchCastToNum(value.right)
          match value.op:
            case "<":
              value = sb3.KnownBool(left < right)
            case ">":
              value = sb3.KnownBool(left > right)
            case "=":
              value = sb3.KnownBool(left == right)

        elif value.op in ["and", "or"]:
          did_opti_total = True
          left = sb3.scratchCastToBool(value.left)
          right = sb3.scratchCastToBool(value.right)
          if value.op == "and":
            value = sb3.KnownBool(left and right)
          else:
            value = sb3.KnownBool(left or right)

    case sb3.Op():
      value.left, did_opti_1 = simplifyValue(value.left)
      did_opti_2 = False
      if value.right is not None:
        value.right, did_opti_2 = simplifyValue(value.right)
      did_opti_total |= did_opti_1 or did_opti_2

      if isinstance(value.left, sb3.Known) and (isinstance(value.right, sb3.Known) or value.right is None):
        left = sb3.scratchCastToNum(value.left)
        if value.right is not None: right = sb3.scratchCastToNum(value.right)
        did_opti_total = True
        match value.op:
          case "add":
            value = sb3.Known(left + right)
          case "sub":
            value = sb3.Known(left - right)
          case "mul":
            value = sb3.Known(left * right)
          case "div":
            value = sb3.Known(left / right)
          case "mod":
            value = sb3.Known(left % right)
          case "abs":
            value = sb3.Known(abs(left))
          case "floor":
            value = sb3.Known(math.floor(left))
          case "ceiling":
            value = sb3.Known(math.ceil(left))
          case _:
            did_opti_total = False
      elif (isinstance(value.left, sb3.Known) or isinstance(value.right, sb3.Known)) and value.right is not None:
        left_known = isinstance(value.left, sb3.Known)
        unknown = value.right
        if left_known:
          assert isinstance(value.left, sb3.Known) # why do I have to assert this lol
          known = sb3.scratchCastToNum(value.left)
        else:
          assert isinstance(value.right, sb3.Known)
          known = sb3.scratchCastToNum(value.right)
          unknown = value.left
        match value.op:
          case "add" | "sub":
            if known == 0: value = unknown
          case "mul":
            if known == 0: value = sb3.Known(0)
    
    case sb3.GetOfList():
      value.value, did_opti = simplifyValue(value.value)
      did_opti_total |= did_opti
      # TODO OPTI: known values of lists can be looked up (assuming the list stays constant)

    case _:
      did_opti_total |= False
  return value, did_opti_total

def knownValuePropagationBlock(blocklist: sb3.BlockList) -> tuple[sb3.BlockList, bool]:
  """Optimise a code block by evaluating known values or general optimisation"""
  did_opti_total = False
  new_blocklist = sb3.BlockList()
  for block in blocklist.blocks:
    # First, optimise the values in the blocks
    did_opti = True
    while did_opti:
      did_opti = False
      inputs = getInputs(block)
      for i, value in enumerate(inputs):
        inputs[i], did_opti_value = simplifyValue(value)
        did_opti |= did_opti_value
      block = setInputs(block, inputs)
      did_opti_total |= did_opti

    # Then, repeat for any sub block lists
    if isinstance(block, sb3.ControlFlow):
      block.blocks, did_opti_1 = knownValuePropagationBlock(block.blocks)
      did_opti_2 = False
      if block.else_blocks is not None:
        block.else_blocks, did_opti_2 = knownValuePropagationBlock(block.else_blocks)
    
      did_opti_total |= did_opti_1 or did_opti_2

    # Finally, optimise the blocks themselves depending on the value
    add_block = True
    did_opti = False
    match block:
      case sb3.ControlFlow():
        if isinstance(block.value, sb3.Known):
          match block.op:
            case "if" | "if_else":
              did_opti = True
              add_block = False
              if block.value.known:
                new_blocklist.add(block.blocks)
              elif block.else_blocks is not None:
                new_blocklist.add(block.else_blocks)
            case "until":
              did_opti = True
              add_block = not block.value.known # TODO OPTI: otherwise use forever
            case "while":
              did_opti = True
              add_block = block.value.known
        elif isinstance(block.value, sb3.BoolOp) and block.value.op == "not":
          match block.op:
            case "if_else":
              did_opti = True
              assert block.else_blocks is not None
              tmp = block.blocks
              block.blocks = block.else_blocks
              block.else_blocks = tmp
              block.value = block.value.left
            case "until" | "while":
              did_opti = True
              block.op = "until" if block.op == "while" else "while"
              block.value = block.value.left
            
    did_opti_total |= did_opti

    if add_block: new_blocklist.add(block)

  return new_blocklist, did_opti_total

def knownValuePropagation(proj: sb3.Project) -> tuple[sb3.Project, bool]:
  new_code = []
  did_total_opti = False
  for blocklist in proj.code:
    did_opti = True
    while did_opti:
      blocklist, did_opti = knownValuePropagationBlock(blocklist)
      did_total_opti |= did_opti
    new_code.append(blocklist)
  proj.code = new_code
  return proj, did_total_opti

def getValueVarUse(value: sb3.Value) -> tuple[set[str], Counter[str]]:
  """Returns what variable a value depends upon and how many times a var is used"""
  match value:
    case sb3.Known() | sb3.GetParameter():
      return set(), Counter()
    case sb3.GetVar():
      name = "var:" + value.var_name
      return {name}, Counter({name: 1})
    case sb3.GetCounter():
      return {"counter:"}, Counter()
    case sb3.Op() | sb3.BoolOp():
      depends, counts = getValueVarUse(value.left)
      if value.right is not None:
        new_depends, new_counts = getValueVarUse(value.right)
        depends.update(new_depends)
        counts += new_counts
      return depends, counts
    case sb3.GetOfList():
      use, counts = getValueVarUse(value.value)
      return {"list:" + value.list_name} | use, counts
    case _:
      raise OptimizerException(f"Unknown value type {type(value)}")

def getBlockListVarUse(blocklist: sb3.BlockList, func_info: dict[str, BlockListInfo] | None=None) -> BlockListInfo:
  """Returns what a block list sometimes modifies, always modifies and depends on"""
  info = BlockListInfo(set(), set(), set(), set(), set())

  for block in blocklist.blocks:
    # Work out what any values used depend on
    all_value_dependent: set[str] = set()
    for value in getInputs(block):
      value_dependent, counts = getValueVarUse(value)
      all_value_dependent |= value_dependent
      if isinstance(block, sb3.ControlFlow) and block.op in ["until", "while"]:
        # Might be depended on an unlimited amount of times
        info.use_counts += Counter({key: float("inf") for key in counts})
      else:
        info.use_counts += counts
    info.dependent |= all_value_dependent - info.always_modify

    match block:
      case sb3.EditVar():
        name = "var:" + block.var_name
        if block.op == "change" and name not in info.always_modify:
          info.dependent.add(name)
        info.might_modify.add(name)
        info.always_modify.add(name)
      case sb3.EditList():
        name = "list:" + block.list_name
        # NOTE: dependents might be removed if in always modify even though this doesn't work with lists.
        # this is fine because we won't try and elide lists
        info.dependent.add(name)
        info.might_modify.add(name)
        info.always_modify.add(name)
      case sb3.EditCounter():
        name = "counter:"
        if block.op == "incr":
          info.dependent.add(name)
        info.might_modify.add(name)
        info.always_modify.add(name)
      case sb3.ProcedureCall() | sb3.Broadcast():
        if isinstance(block , sb3.ProcedureCall):
          name = "func:" + block.proc_name
        else:
          if not isinstance(block.value, sb3.Known):
            raise OptimizerException("Broadcasted address expected to be constant/known")
          name = "broadcast:" + sb3.scratchCastToStr(block.value)
        info.might_call.add(name)
        info.always_call.add(name)

        if func_info is not None:
          callee_info = func_info[name]
          info.dependent |= callee_info.dependent - info.always_modify
          info.might_modify |= callee_info.might_modify
          info.always_modify |= callee_info.always_modify
          info.use_counts += callee_info.use_counts
      case sb3.ControlFlow():
        match block.op:
          case "if" | "reptimes" | "until" | "while":
            b_info = getBlockListVarUse(block.blocks, func_info)
            
            info.dependent    |= b_info.dependent - info.always_modify
            info.might_modify |= b_info.might_modify
            info.might_call   |= b_info.might_call
            info.always_call  |= b_info.always_call

            if block.op == "if":
              # Assume called half the times
              info.use_counts += Counter({k: v / 2 for k, v in b_info.use_counts.items()})
            else:
              # Might repeat an unlimited amount of times
              info.use_counts += Counter({key: float("inf") for key in b_info.use_counts})
          case "if_else":
            assert block.else_blocks is not None
            b1_info = getBlockListVarUse(block.blocks, func_info)
            b2_info = getBlockListVarUse(block.else_blocks, func_info)

            info.dependent     |= (b1_info.dependent | b2_info.dependent) - info.always_modify
            info.might_modify  |= b1_info.might_modify | b2_info.might_modify
            info.always_modify |= b1_info.always_modify & b2_info.always_modify
            info.might_call    |= b1_info.might_call | b2_info.might_call
            info.always_call   |= b1_info.always_call & b2_info.always_call

            # Work out the average use counts between the two branches
            b1_counts, b2_counts = b1_info.use_counts, b2_info.use_counts
            all_keys = set(b1_counts) | set(b2_counts)
            info.use_counts += Counter({key: (b1_counts.get(key, 0) + b2_counts.get(key, 0)) / 2 for key in all_keys})
  
  assert info.might_modify.issuperset(info.always_modify)
  assert info.might_call.issuperset(info.always_call)
  return info

def getValueCost(value: sb3.Value) -> float:
  match value:
    case sb3.Known():
      cost = KNOWN_COST
    case sb3.Op():
      if value.op in ["add", "sub", "mul", "div"]: cost = MATHOP_COST
      elif value.op == "mod":          cost = MOD_COST
      elif value.op == "length_of":    cost = LENGTH_OF_COST
      elif value.op == "letter_n_of":  cost = LETTER_OF_COST
      elif value.op == "join":         cost = JOIN_COST
      elif value.op == "rand_between": cost = RAND_COST
      elif value.op == "round":        cost = ROUND_COST
      else:                            cost = MATH_FUNC_COST
      # FUTURE Bool To Int cost
    case sb3.BoolOp():
      if value.op in ["=", "<", ">"]:  cost = COMPARISON_COST
      elif value.op in ["and", "or"]:  cost = AND_OR_COST
      elif value.op == "not":          cost = NOT_COST
      elif value.op == "contains":     cost = CONTAINS_STR_COST
      else:
        raise OptimizerException(f"Unknown BoolOp opcode, {value.op}")
    case sb3.GetCounter():             cost = GET_COUNTER_COST
    case sb3.GetVar():                 cost = GET_VAR_COST
    case sb3.GetParameter():           cost = GET_PARAM_COST
    case sb3.GetOfList():
      cost = GET_ITEM_COST if value.op == "atindex" else ITEM_NUM_COST
    case _:
      raise OptimizerException(f"Unknown value, {type(value)}")
  
  match value:
    case sb3.Op() | sb3.BoolOp():
      cost += getValueCost(value.left)
      if value.right is not None:
        cost += getValueCost(value.right)
    case sb3.GetOfList():
      cost += getValueCost(value.value)

  return cost

def assignmentElisionValue(value: sb3.Value, to_elide: dict[str, sb3.Value]) -> tuple[sb3.Value, bool]:
  match value:
    case sb3.Known() | sb3.GetParameter() | sb3.GetCounter():
      return value, False
    case sb3.GetVar():
      name = "var:" + value.var_name
      if name in to_elide:
        return to_elide[name], True
      return value, False
    case sb3.Op() | sb3.BoolOp():
      value.left, did_opti = assignmentElisionValue(value.left, to_elide)
      if value.right is not None:
        value.right, did_opti_right = assignmentElisionValue(value.right, to_elide)
        did_opti |= did_opti_right
      return value, did_opti
    case sb3.GetOfList():
      value.value, did_opti = assignmentElisionValue(value.value, to_elide)
      return value, did_opti
    case _:
      raise OptimizerException(f"Unknown value type {type(value)}")

def assignmentElisionBlock(blocklist: sb3.BlockList, to_elide: dict[str, sb3.Value]) -> tuple[sb3.BlockList, bool]:
  did_opti = False
  new_blocklist = sb3.BlockList()
  for block in blocklist.blocks:
    # Elide assignments
    if not (isinstance(block, sb3.EditVar) and "var:" + block.var_name in to_elide):
      # Elide values
      inputs = getInputs(block)
      for i, value in enumerate(inputs):
        inputs[i], did_opti_value = assignmentElisionValue(value, to_elide)
        did_opti |= did_opti_value
      block = setInputs(block, inputs)

      if isinstance(block, sb3.ControlFlow):
        block.blocks, did_opti_block = assignmentElisionBlock(block.blocks, to_elide)
        did_opti |= did_opti_block
        if block.else_blocks is not None:
          block.else_blocks, did_opti_block = assignmentElisionBlock(block.else_blocks, to_elide)
          did_opti |= did_opti_block

      new_blocklist.add(block)
    else:
      did_opti = True

  return new_blocklist, did_opti

def assignmentElision(proj: sb3.Project) -> tuple[sb3.Project, bool]:
  """Optimise a code block by removing variable assignments which only lead to one read"""
  did_total_opti = False

  fn_blocks: dict[str, sb3.BlockList] = {}
  fn_info: dict[str, BlockListInfo] = {}
  for blocklist in proj.code:
    if len(blocklist.blocks) > 0:
      first_block = blocklist.blocks[0]
      match first_block:
        case sb3.ProcedureDef():
          name = "func:" + first_block.name
        case sb3.OnBroadcast():
          name = "broadcast:" + first_block.name
        case sb3.OnStartFlag():
          name = "start:"
          if name in fn_blocks:
            raise OptimizerException("Multiple starting blocks")
        case _:
          raise OptimizerException(f"Unknown starting block, {type(first_block)}")
        
      fn_blocks[name] = blocklist
      fn_info[name] = getBlockListVarUse(blocklist)

  cert_callers = defaultdict(set)
  poss_callers = defaultdict(set)
  for caller, callees in fn_info.items():
    for callee in callees.might_call:
      poss_callers[callee].add(caller)
    for callee in callees.always_call:
      cert_callers[callee].add(caller)
  
  worklist = deque(fn_info.keys())
  recu_fn_info: dict[str, BlockListInfo] = deepcopy(fn_info)

  while worklist:
    name = worklist.popleft()
    info = recu_fn_info[name]

    for callee in info.always_call:
      info.always_modify |= recu_fn_info[callee].always_modify

    for callee in info.might_call:
      callee_info = recu_fn_info[callee]
      info.might_modify |= callee_info.might_modify | callee_info.always_modify
      info.dependent |= callee_info.dependent
      info.use_counts += callee_info.use_counts

    if info != recu_fn_info[name]:
      recu_fn_info[name] = info
      for caller in cert_callers[name]:
        worklist.append(caller)
      for caller in poss_callers[name]:
        worklist.append(caller)

  for name, blocklist in list(fn_blocks.items()):
    # Find every variable that isn't used elsewhere
    cannot_elide: set[str] = set()
    for other_name, other_info in fn_info.items():
      # FUTURE OPTI: across function value elison - would require working out where each function returns
      if other_name != name:
        cannot_elide |= other_info.might_modify | other_info.always_call | other_info.dependent

    # Each variable that could be elided and its dependents
    to_elide: dict[str, tuple[set[str], sb3.Value]] = {}
    # A variable's dependents might have changed but it can still be elided if not read after this
    changed_but_unread: dict[str, tuple[set[str], sb3.Value]] = {}

    var_use_counts: Counter[str] = Counter()

    # Figure out what can be elided
    for block in blocklist.blocks:
      use = getBlockListVarUse(sb3.BlockList([block]), recu_fn_info)
      var_use_counts += use.use_counts

      to_remove = []
      for var_name in changed_but_unread.keys():
        if var_name in use.dependent:
          to_remove.append(var_name)
      for var_name in to_remove:
        del changed_but_unread[var_name]

      to_remove = []
      for var_name, (var_dependents, var_value) in to_elide.items():
        if var_name in use.might_modify:
          to_remove.append(var_name)
        elif bool(use.might_modify & var_dependents):
          # If any of the variables are changed
          to_remove.append(var_name)
          changed_but_unread[var_name] = var_dependents, var_value
      for var_name in to_remove:
        del to_elide[var_name]
      
      if not isinstance(block, sb3.EditVar):
        cannot_elide |= use.might_modify
      else:
        var_name = "var:" + block.var_name
        depends_on, _ = getValueVarUse(block.value)
        if block.op == "set" and var_name not in (depends_on | cannot_elide):
          to_elide[var_name] = depends_on, block.value
        # Don't elide if overwritten
        cannot_elide.add(var_name)

    to_elide.update(changed_but_unread)

    # Calculate if it is actually faster to elide
    slower_to_elide: set[str] = set()
    for var_name, (_, value) in to_elide.items():
      times_used = var_use_counts.get(var_name, 0)
      calc_cost = getValueCost(value)
      elision_cost = calc_cost * times_used
      no_elision_cost = SET_VAR_COST + calc_cost + (GET_VAR_COST * times_used)
      if no_elision_cost < elision_cost:
        slower_to_elide.add(var_name)
    for var_name in slower_to_elide:
      del to_elide[var_name]

    # Apply elisions to elision values
    total_was_subelision_made = True
    while total_was_subelision_made:
      total_was_subelision_made = False
      for var_name, (depends, value) in list(to_elide.items()):
        # We don't care if an optimisation was made if it was never used - used elided values set
        # did_opti to true anyway
        new_value, was_subelision_made = assignmentElisionValue(value, {k: v for k, (_, v) in to_elide.items()})
        to_elide[var_name] = depends, new_value
        total_was_subelision_made |= was_subelision_made

    # Perform the elision
    fn_blocks[name], did_opti = assignmentElisionBlock(blocklist, {k: v for k, (_, v) in to_elide.items()})
    did_total_opti |= did_opti
  
  proj.code = list(fn_blocks.values())

  return proj, did_total_opti

def optimize(proj: sb3.Project, all_opti: set[Optimisation] | None = None) -> sb3.Project:
  if all_opti is None:
    all_opti = ALL_OPTIMISATIONS
  
  times_optimized = 0
  opti_to_perform = all_opti
  while len(opti_to_perform) > 0 and times_optimized < MAX_OPTIMIZATIONS:
    current_opti = all_opti.pop()
    match current_opti:
      case Optimisation.ASSIGNMENT_ELISION:
        proj, did_opti = assignmentElision(proj)
      case Optimisation.KNOWN_VALUE_PROPAGATION:
        proj, did_opti = knownValuePropagation(proj)
      case _:
        raise OptimizerException(f"Unknown Optimisation {current_opti}")
    if did_opti: opti_to_perform = all_opti - {current_opti}
    times_optimized += 1

  return proj