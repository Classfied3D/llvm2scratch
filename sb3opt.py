"""Scratch project optimiser for the LLVM -> Scratch compiler"""

import math

import sb3

def scratchCastToNum(value: sb3.Known) -> float:
  """Performs the same casting to number as scratch"""
  raw = value.known
  if not isinstance(raw, float):
    try:
      raw = float(raw)
    except ValueError:
      raw = math.nan

  return 0 if math.isnan(raw) else raw

def scratchCastToBool(value: sb3.Known) -> bool:
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
          value = sb3.KnownBool(not scratchCastToBool(value.left))

      elif isinstance(value.left, sb3.Known) and isinstance(value.right, sb3.Known):
        if value.op in ["<", ">", "="]:
          did_opti_total = True
          left = scratchCastToNum(value.left)
          right = scratchCastToNum(value.right)
          match value.op:
            case "<":
              value = sb3.KnownBool(left < right)
            case ">":
              value = sb3.KnownBool(left > right)
            case "=":
              value = sb3.KnownBool(left == right)

        elif value.op in ["and", "or"]:
          did_opti_total = True
          left = scratchCastToBool(value.left)
          right = scratchCastToBool(value.right)
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
        left = scratchCastToNum(value.left)
        if value.right is not None: right = scratchCastToNum(value.right)
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
          known = scratchCastToNum(value.left)
        else:
          assert isinstance(value.right, sb3.Known)
          known = scratchCastToNum(value.right)
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

def knownValuePropagation(blocklist: sb3.BlockList) -> tuple[sb3.BlockList, bool]:
  """Optimise a code block by evaluating with known values or applying specific contexts"""
  did_opti_total = False
  new_blocklist = sb3.BlockList()
  for block in blocklist.blocks:
    # First, optimise the values in the blocks
    did_opti = True
    while did_opti:
      did_opti = False
      match block:
        case sb3.Say() | sb3.EditVar() | sb3.ControlFlow():
          block.value, did_opti = simplifyValue(block.value)
        case sb3.EditList():
          did_opti_1 = did_opti_2 = False
          if block.item is not None: block.item, did_opti_1 = simplifyValue(block.item)
          if block.index is not None: block.index, did_opti_2 = simplifyValue(block.index)
          did_opti = did_opti_1 or did_opti_2
        case sb3.ProcedureCall():
          for i, argument in enumerate(block.arguments):
            opti_arg, did_opti_arg = simplifyValue(argument)
            block.arguments[i] = opti_arg
            did_opti |= did_opti_arg
      did_opti_total |= did_opti

    # Then, repeat for any sub block lists
    if isinstance(block, sb3.ControlFlow):
      block.blocks, did_opti_1 = knownValuePropagation(block.blocks)
      did_opti_2 = False
      if block.else_blocks is not None:
        block.else_blocks, did_opti_2 = knownValuePropagation(block.else_blocks)
    
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

def optimise(proj: sb3.Project) -> sb3.Project:
  new_code = []
  for blocklist in proj.code:
    did_opti = True
    while did_opti:
      blocklist, did_opti = knownValuePropagation(blocklist)
    new_code.append(blocklist)
  proj.code = new_code
  return proj

def main():
  proj = sb3.Project(sb3.ScratchConfig())
  proj.code.append(sb3.BlockList([
    sb3.ProcedureDef("main", ["%1", "%2"]),
    sb3.ControlFlow("if", sb3.BoolOp("=", sb3.Known(30), sb3.Known(0)), sb3.BlockList([
      sb3.EditVar("set", "hello2", sb3.Known(10)),
    ]))
  ]))

  proj = optimise(proj)

  sb3.exportSpriteFile(proj.getCtx(), "out.sprite3")

if __name__ == "__main__":
  main()