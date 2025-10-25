from __future__ import annotations
from llvmlite import binding as llvm
from dataclasses import dataclass
from typing import cast
from .parser_util import valueFromInitializerText, extractFirstType, extractTypedValue, stripReturnAttrs
from .ir import *

def decodeTypeStr(type_str: str, mod: llvm.ModuleRef) -> Type:
  if type_str == "void":
    return VoidTy()
  ir = "\n".join(str(struct) for struct in mod.struct_types) \
    + f"""
  declare void @dummy({type_str})
  """
  mod = llvm.parse_assembly(ir)
  func = next(mod.functions)
  arg = next(func.arguments)
  return decodeType(arg.type)

def getResultLocalVar(instr: llvm.ValueRef) -> ResultLocalVar | None:
  if str(instr).strip().startswith("%"):
    return ResultLocalVar(str(instr).split("=")[0].strip()[1:])
  return None

def getGEPConstExprStr(gep_str: str, mod: llvm.ModuleRef) -> GetElementPtr:
  rest = gep_str

  rest = rest.removeprefix("getelementptr").strip()
  keywords = ["inbounds", "inrange", "nusw", "nsw"]
  has_keyword = True
  while has_keyword:
    has_keyword = False
    for kw in keywords:
      if rest.startswith(kw):
        if kw != "inrange":
          rest = rest.removeprefix(kw + " ")
        else:
          kw = kw.split(")", 1)[-1]
        has_keyword = True

  rest = rest.strip()

  # Remove brackets that are present in only getelementptr
  # constant expressions for some reason
  if rest.startswith("("):
    rest = rest[1:-1]

  gep_str = "getelementptr " + rest

  ir = "\n".join(str(struct) for struct in mod.struct_types) + \
        "\n".join(str(glob) for glob in mod.global_variables) + \
  f"""
  define void @dummy() {{
    {gep_str}
    ret void
  }}
  """

  mod = llvm.parse_assembly(ir)
  func = next(mod.functions)
  block = next(func.blocks)
  instr = next(block.instructions)
  parsed_instr = decodeInstr(instr, mod)
  assert isinstance(parsed_instr, GetElementPtr)

  return parsed_instr

def decodeType(type: llvm.TypeRef) -> Type:
  match type.type_kind:
    case llvm.TypeKind.void:
      return VoidTy()

    case llvm.TypeKind.integer:
      return IntegerTy(type.type_width)

    case llvm.TypeKind.half:
      return HalfTy()
    case llvm.TypeKind.float:
      return FloatTy()
    case llvm.TypeKind.double:
      return DoubleTy()
    case llvm.TypeKind.fp128:
      return Fp128Ty()

    case llvm.TypeKind.pointer:
      return PointerTy()

    case llvm.TypeKind.vector:
      inner_type = decodeType(next(type.elements))
      assert isinstance(inner_type, VecTargetTy)
      return VecTy(inner_type, type.element_count)

    case llvm.TypeKind.label:
      return LabelTy()

    case llvm.TypeKind.array:
      inner_type = decodeType(next(type.elements))
      assert isinstance(inner_type, AggTargetTy)
      return ArrayTy(inner_type, type.element_count)

    case llvm.TypeKind.struct:
      name = type.name
      members = [decodeType(ty) for ty in type.elements]
      assert all(isinstance(mem, AggTargetTy) for mem in members)
      members = cast(list[AggTargetTy], members)
      return StructTy(name, members)

    case _:
      raise ValueError(f"Unknown type: {type.type_kind.name}")

def decodeValue(value: llvm.ValueRef, mod: llvm.ModuleRef) -> Value:
  type = decodeType(value.type)

  match value.value_kind:
    case llvm.ValueKind.argument:
      # A function argument
      name = str(value).split(" ")[-1][1:]
      return ArgumentVal(type, name)

    case llvm.ValueKind.function:
      # A function reference
      rest = str(value)
      while not (rest.startswith("declare ") or rest.startswith("define ")):
        rest = rest.split("\n", 1)[-1]
      rest = rest.removeprefix("declare ").removeprefix("define ").strip()
      rest = stripReturnAttrs(rest).strip()
      func_ret_type_str, _ = extractFirstType(rest)
      func_ret_type = decodeTypeStr(func_ret_type_str, mod)

      name = value.name
      intrinsic = decodeIntrinsic(name)

      return FunctionVal(type, name, func_ret_type, intrinsic)

    case llvm.ValueKind.instruction:
      # An instruction of which the SSA value is set from
      res = getResultLocalVar(value)
      assert res is not None
      return LocalVarVal(type, res.name)

    case llvm.ValueKind.global_variable:
      # A global variable reference
      return GlobalVarVal(type, value.name)

    case llvm.ValueKind.undef_value:
      # An undef value
      return UndefVal(type)

    case llvm.ValueKind.constant_int:
      # An constant integer (e.g. 5)
      val = value.get_constant_value()
      assert isinstance(val, int)
      assert isinstance(type, IntegerTy)
      return KnownIntVal(type, val, type.width)

    case llvm.ValueKind.constant_fp:
      val = value.get_constant_value(round_fp=True)
      assert isinstance(val, float)
      assert isinstance(type, (HalfTy, FloatTy, DoubleTy, Fp128Ty))
      return KnownFloatVal(type, val)

    case llvm.ValueKind.basic_block:
      # A basic block label
      label_name = str(value).split(":")[0].strip()
      if any(x in label_name for x in "\n =%\t"):
        label_name = "0"
      return LabelVal(type, label_name)

    case llvm.ValueKind.constant_data_vector:
      # A constant vector (e.g. <4 x i32> <i32 1, i32 2, i32 3, i32 4>)
      return valueFromInitializerText(str(value), type, lambda string: getGEPConstExprStr(string, mod))

    case llvm.ValueKind.constant_expr:
      # A constant expression (e.g. getelementptr)
      rest = str(value)
      expr_ty_str, rest = extractFirstType(rest)
      expr_ty = decodeTypeStr(expr_ty_str, mod)

      if rest.startswith("getelementptr"):
        assert isinstance(expr_ty, PointerTy)
        return ConstExprVal(expr_ty, getGEPConstExprStr(rest, mod))
      else:
        raise ValueError(f"Unknown constant expression: {rest}")

    case _:
      raise ValueError(f"Unknown value type: {value.value_kind.name}")

def decodeLabel(value: llvm.ValueRef, mod: llvm.ModuleRef) -> LabelVal:
  res = decodeValue(value, mod)
  assert isinstance(res, LabelVal)
  return res

def decodeIntrinsic(name: str) -> Intrinsic | None:
  if not name.startswith("llvm."):
    return None
  name = name.removeprefix("llvm.")

  if name.startswith("memcpy"):
    if name.startswith("memcpy.inline"):
      return Intrinsic.MemCpyInline
    return Intrinsic.MemCpy
  elif name.startswith("memmove"):
    return Intrinsic.MemMove
  elif name.startswith("lifetime.start"):
    return Intrinsic.LifetimeStart
  elif name.startswith("lifetime.end"):
    return Intrinsic.LifetimeEnd
  raise ValueError(f"Unknown intrinsic {name}")

def decodeInstr(instr: llvm.ValueRef, mod: llvm.ModuleRef) -> Instr:
  result = getResultLocalVar(instr)
  raw_instr_no_res = str(instr).strip()
  if result is not None:
    raw_instr_no_res = raw_instr_no_res.split("=", 1)[1].strip()

  match instr.opcode:
    case "ret":
      if len(list(instr.operands)) > 0:
        value = decodeValue(next(instr.operands), mod)
        return Ret(value)
      return Ret(None)

    case "br":
      if len(list(instr.operands)) > 1:
        cond, branch_false, branch_true, *_ = instr.operands
        return CondBr(decodeValue(cond, mod), decodeLabel(branch_true, mod), decodeLabel(branch_false, mod))
      return UncondBr(decodeLabel(next(instr.operands), mod))

    case "switch":
      value, default_label, *rest = instr.operands
      assert len(rest) % 2 == 0

      branch_table: list[tuple[KnownIntVal, LabelVal]] = []
      for i in range(0, len(rest), 2):
        case_val, label = decodeValue(rest[i], mod), decodeLabel(rest[i+1], mod)
        assert isinstance(case_val, KnownIntVal)
        branch_table.append((case_val, label))

      return Switch(decodeValue(value, mod), decodeLabel(default_label, mod), branch_table)

    case "unreachable":
      return Unreachable()

    case "fneg":
      assert result is not None
      operand, *_ = instr.operands
      return UnaryOp(result, UnaryOpcode.FNeg, decodeValue(operand, mod))

    case "add" | "fadd" | "sub" | "fsub" | "mul" | "fmul" | "udiv" | "sdiv" | "fdiv" | \
         "urem" | "srem" | "frem" | "shl" | "lshr" | "ashr" | "and" | "or" | "xor":
      assert result is not None
      opcode = BinaryOpcode(instr.opcode)
      left, right, *_ = instr.operands

      flags = {"nuw": False, "nsw": False, "exact": False, "disjoint": False}
      flags_string = str(raw_instr_no_res).split(opcode.value + " ", 1)[1]

      while True:
        for flag in flags:
          if flags_string.startswith(flag + " "):
            flags[flag] = True
            flags_string = flags_string[len(flag) + 1:]
            break
        else:
          break

      return BinaryOp(
        result, opcode,
        decodeValue(left, mod), decodeValue(right, mod),
        flags["nuw"], flags["nsw"], flags["exact"], flags["disjoint"])

    case "extractelement":
      assert result is not None
      vec, index, *_ = instr.operands
      return ExtractElement(result, decodeValue(vec, mod), decodeValue(index, mod))

    case "insertelement":
      vec, item, index, *_ = instr.operands
      return InsertElement(decodeValue(vec, mod), decodeValue(item, mod), decodeValue(index, mod))

    case "shufflevector":
      assert result is not None
      vec1, vec2, *_ = instr.operands

      rest = raw_instr_no_res.split("shufflevector ", 1)[-1].strip()
      _, _, rest = extractTypedValue(rest)
      _, _, rest = extractTypedValue(rest)
      mask_ty, mask_val, _ = extractTypedValue(rest)

      # Workaround for llvmlite not exposing the mask values directly (why lol)
      ir = f"""
      define void @dummy() {{
      entry:
        %c = extractelement {mask_ty} {mask_val}, i32 0
        ret void
      }}
      """
      mod = llvm.parse_assembly(ir)
      func = next(mod.functions)
      block = next(func.blocks)
      instr = next(block.instructions)
      mask = next(instr.operands)

      return ShuffleVector(result, decodeValue(vec1, mod), decodeValue(vec2, mod), decodeValue(mask, mod))

    case "alloca":
      assert result is not None

      num_elements = decodeValue(next(instr.operands), mod)

      rest = raw_instr_no_res.split("alloca ", 1)[1].strip().removeprefix("inalloca ")
      allocated_type_str, _ = extractFirstType(rest)
      allocated_type = decodeTypeStr(allocated_type_str, mod)

      return Alloca(result, allocated_type, num_elements)

    case "load":
      assert result is not None
      value, *_ = instr.operands
      return Load(result, decodeValue(value, mod))

    case "store":
      value, addr, *_ = instr.operands
      return Store(decodeValue(value, mod), decodeValue(addr, mod))

    case "getelementptr":
      assert result is not None
      base_ptr, *indices = instr.operands

      index_values = [decodeValue(idx, mod) for idx in indices]

      rest = raw_instr_no_res.split("getelementptr ", 1)[1].strip()
      keywords = ["inbounds", "inrange", "nusw", "nsw"]
      has_keyword = True
      while has_keyword:
        has_keyword = False
        for kw in keywords:
          if rest.startswith(kw):
            if kw != "inrange":
              rest = rest.removeprefix(kw + " ")
            else:
              kw = kw.split(")", 1)[-1]
            has_keyword = True

      ptr_type_str, _ = extractFirstType(rest)
      ptr_type = decodeTypeStr(ptr_type_str, mod)

      return GetElementPtr(result, ptr_type, decodeValue(base_ptr, mod), index_values)

    case "trunc" | "zext" | "sext" | "fptrunc" | "fpext" | "fptoui" | "fptosi" | \
         "uitofp" | "sitofp" | "ptrtoint" | "ptrtoaddr" | "inttoptr" | "bitcast" | \
         "addrspacecast":
      assert result is not None
      opcode = ConvOpcode(instr.opcode)
      value, *_ = instr.operands

      conv_type_str, _ = extractFirstType(raw_instr_no_res.split(" to ", 1)[-1].strip())
      conv_type = decodeTypeStr(conv_type_str, mod)

      flags = {"nuw": False, "nsw": False}
      if instr.opcode == "trunc":
        flags_string = str(raw_instr_no_res).split(opcode.value + " ", 1)[1]

        while True:
          for flag in flags:
            if flags_string.startswith(flag + " "):
              flags[flag] = True
              flags_string = flags_string[len(flag) + 1:]
              break
          else:
            break

      return Conversion(result, opcode, decodeValue(value, mod), conv_type, flags["nuw"], flags["nsw"])

    case "icmp":
      assert result is not None
      left, right, *_ = instr.operands
      rest = str(raw_instr_no_res).split("icmp ", 1)[-1].strip()

      samesign = rest.startswith("samesign ")
      rest = rest.removeprefix("samesign ")

      cond = ICmpCond(rest.split(" ", 1)[0])
      return ICmp(result, cond, decodeValue(left, mod), decodeValue(right, mod), samesign)

    case "fcmp":
      assert result is not None
      left, right, *_ = instr.operands
      rest = str(raw_instr_no_res).split("fcmp ", 1)[-1].strip()

      # Skip over fast math flags
      cond_found = False
      while not cond_found:
        cond_str, rest = rest.split(" ", 1)
        if cond_str in FCmpCond:
          cond_found = True
          cond = FCmpCond(cond_str)

      return FCmp(result, cond, decodeValue(left, mod), decodeValue(right, mod))

    case "phi":
      assert result is not None

      incoming: list[tuple[Value, LabelVal]] = []
      for val, label in zip(instr.operands, instr.incoming_blocks):
        incoming.append((decodeValue(val, mod), decodeLabel(label, mod)))
      return Phi(result, incoming)

    case "select":
      assert result is not None
      cond, true_val, false_val, *_ = instr.operands
      return Select(result, decodeValue(cond, mod), decodeValue(true_val, mod), decodeValue(false_val, mod))

    case "call":
      *args, callee = instr.operands
      func_val = decodeValue(callee, mod)
      assert isinstance(func_val, FunctionVal)
      arg_vals = [decodeValue(arg, mod) for arg in args]

      tail_kind = CallTailKind.NoTail
      if raw_instr_no_res.startswith("tail "):
        tail_kind = CallTailKind.Tail
      elif raw_instr_no_res.startswith("musttail "):
        tail_kind = CallTailKind.MustTail

      return Call(result, func_val, arg_vals, tail_kind)

    case "freeze":
      assert result is not None
      value, *_ = instr.operands
      return Freeze(result, decodeValue(value, mod))

    case _:
      raise ValueError(f"Opcode {instr.opcode} not implemented")

def decodeModule(mod: llvm.ModuleRef) -> Module:
  glob_vars: dict[str, GlobalVar] = {}
  for glob in mod.global_variables:
    glob_name = glob.name

    rest = str(glob).strip()
    if "constant" in rest:
      glob_constant = True
      rest = rest.split("constant", 1)[-1].strip()
    else:
      glob_constant = False
      rest = rest.split("global", 1)[-1].strip()
    type_str, rest = extractFirstType(rest)
    type = decodeTypeStr(type_str, mod)
    rest = rest.strip()
    if len(rest) == 0 or rest.startswith(","):
      glob_init = None
    else:
      glob_init = valueFromInitializerText(rest.rsplit(",", 1)[0].strip(), type, lambda string: getGEPConstExprStr(string, mod))
      assert isinstance(glob_init, (KnownVal, GlobalVarVal, ConstExprVal))

    glob_vars.update({glob_name: GlobalVar(glob_name, type, glob_constant, glob_init)})

  functions: dict[str, Function] = {}

  for func in mod.functions:
    fn_name = func.name
    fn_return_type = decodeValue(func, mod)
    assert isinstance(fn_return_type, FunctionVal)
    fn_ret_type = fn_return_type.return_type

    fn_args: list[ArgumentVal] = []
    for arg in func.arguments:
      arg_val = decodeValue(arg, mod)
      assert isinstance(arg_val, ArgumentVal)
      fn_args.append(arg_val)

    fn_blocks: dict[str, Block] = {}

    for block in func.blocks:
      block_val = decodeValue(block, mod)
      assert isinstance(block_val, LabelVal)
      block_name = block_val.label

      instructions: list[Instr] = []
      for inst in block.instructions:
        instructions.append(decodeInstr(inst, mod))
      fn_blocks.update({block_name: Block(block_name, instructions)})

    intrinsic = decodeIntrinsic(fn_name)

    functions.update({fn_name: Function(fn_name, fn_ret_type, fn_args, intrinsic, fn_blocks)})

  return Module(mod.name, functions, glob_vars)

def parseAssembly(llvm_ir: str, verify_ir: bool=False) -> Module:
  mod_ref = llvm.parse_assembly(llvm_ir)
  if verify_ir:
    mod_ref.verify()

  return decodeModule(mod_ref)
