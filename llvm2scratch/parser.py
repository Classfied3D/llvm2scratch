from __future__ import annotations
from llvmlite import binding as llvm
from typing import cast

from .parser_util import *
from .ir import *

def getResultLocalVar(instr: llvm.ValueRef) -> ResultLocalVar | None:
  stringified = str(instr).strip()
  if stringified.startswith("%"):
    return ResultLocalVar(stringified.split("=")[0][1:].strip())
  return None

def decodeType(type: llvm.TypeRef, structs: dict[str, StructTy], func_names: list[str]) -> Type:
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
      ty, rest = parseTypeTokens(parseUntilEnd(str(type)), structs)
      assert len(rest) == 0
      return ty

    case llvm.TypeKind.vector:
      inner_type = decodeType(next(type.elements), structs, func_names)
      assert isinstance(inner_type, VecTargetTy)
      return VecTy(inner_type, type.element_count)

    case llvm.TypeKind.label:
      return LabelTy()

    case llvm.TypeKind.array:
      inner_type = decodeType(next(type.elements), structs, func_names)
      assert isinstance(inner_type, AggTargetTy)
      return ArrayTy(inner_type, type.element_count)

    case llvm.TypeKind.struct:
      members = [decodeType(ty, structs, func_names) for ty in type.elements]
      assert all(isinstance(mem, AggTargetTy) for mem in members)
      members = cast(list[AggTargetTy], members)
      is_packed = str(type).replace(" ", "").startswith("<{")
      return StructTy(is_packed, members)

    case llvm.TypeKind.metadata:
      return MetadataTy()

    case _:
      raise ValueError(f"Unknown type: {type.type_kind.name}")

def decodeValue(value: llvm.ValueRef, structs: dict[str, StructTy], func_names: list[str]) -> Value:
  type = decodeType(value.type, structs, func_names)

  match value.value_kind:
    case llvm.ValueKind.argument:
      # A function argument
      name = str(value).split(" ")[-1][1:]
      return ArgumentVal(type, name)

    case llvm.ValueKind.function:
      # A function reference
      return FunctionVal(type, value.name)

    case llvm.ValueKind.instruction:
      # An instruction of which the SSA value is set from
      res = getResultLocalVar(value)
      assert res is not None
      return LocalVarVal(type, res.name)

    case llvm.ValueKind.global_variable:
      # A global variable reference
      return GlobalPtrVal(type, value.name)

    case llvm.ValueKind.undef_value | llvm.ValueKind.poison_value:
      # Treat both as undef
      return UndefVal(type)

    case llvm.ValueKind.constant_int:
      # An constant integer (e.g. 5)
      val = value.get_constant_value()
      assert isinstance(val, int)
      assert isinstance(type, IntegerTy)
      return KnownIntVal(type, val, type.width)

    case llvm.ValueKind.constant_fp:
      # Known fp constant
      val = value.get_constant_value(round_fp=True)
      assert isinstance(val, float)
      assert isinstance(type, (HalfTy, FloatTy, DoubleTy, Fp128Ty))
      return KnownFloatVal(type, val)

    case llvm.ValueKind.constant_pointer_null:
      # Null pointer constant
      return NullPtrVal(type)

    case llvm.ValueKind.basic_block:
      # A basic block label
      label_name = str(value).split(":")[0].strip()
      if any(x in label_name for x in "\n =%\t"):
        label_name = "0"
      return LabelVal(type, label_name)

    case llvm.ValueKind.metadata_as_value:
      # Metadata as value, used in some instrinsics
      return MetadataVal(type)

    case llvm.ValueKind.constant_expr | llvm.ValueKind.constant_data_vector:
      # A constant vector (e.g. <4 x i32> <i32 1, i32 2, i32 3, i32 4>)
      # A constant expression (e.g. getelementptr)
      val, rest = parseTypeConstantTokens(parseUntilEnd(str(value)), structs, func_names)
      assert len(rest) == 0
      return val

    case _:
      raise ValueError(f"Unknown value type: {value.value_kind.name}")

def decodeLabel(value: llvm.ValueRef, structs: dict[str, StructTy], func_names: list[str]) -> LabelVal:
  res = decodeValue(value, structs, func_names)
  assert isinstance(res, LabelVal)
  return res

def decodeIntrinsic(name: str) -> Intrinsic | None:
  if not name.startswith("llvm."):
    return None
  matches = [item for item in Intrinsic if name.startswith(item.value)]
  if not matches:
    raise ValueError(f"Unknown intrinsic {name}")

  # Choose the longest one, i.e if llvm.memcopy and llvm.memcopy.inline were selected, the latter would be chosen
  return max(matches, key=lambda x: len(x.value))

def decodeInstr(instr: llvm.ValueRef, structs: dict[str, StructTy], func_names: list[str]) -> Instr:

  result = getResultLocalVar(instr)
  raw_instr_no_res = str(instr).strip()
  if result is not None:
    raw_instr_no_res = raw_instr_no_res.split("=", 1)[1].strip()

  match instr.opcode:
    case "ret":
      if len(list(instr.operands)) > 0:
        value = decodeValue(next(instr.operands), structs, func_names)
        return Ret(value)
      return Ret(None)

    case "br":
      if len(list(instr.operands)) > 1:
        cond, branch_false, branch_true, *_ = instr.operands
        return CondBr(decodeValue(cond, structs, func_names), decodeLabel(branch_true, structs, func_names), decodeLabel(branch_false, structs, func_names))
      return UncondBr(decodeLabel(next(instr.operands), structs, func_names))

    case "switch":
      value, default_label, *rest = instr.operands
      assert len(rest) % 2 == 0

      branch_table: list[tuple[KnownIntVal, LabelVal]] = []
      for i in range(0, len(rest), 2):
        case_val, label = decodeValue(rest[i], structs, func_names), decodeLabel(rest[i+1], structs, func_names)
        assert isinstance(case_val, KnownIntVal)
        branch_table.append((case_val, label))

      return Switch(decodeValue(value, structs, func_names), decodeLabel(default_label, structs, func_names), branch_table)

    case "unreachable":
      return Unreachable()

    case "fneg":
      assert result is not None
      operand, *_ = instr.operands
      return UnaryOp(result, UnaryOpcode.FNeg, decodeValue(operand, structs, func_names))

    case "add" | "fadd" | "sub" | "fsub" | "mul" | "fmul" | "udiv" | "sdiv" | "fdiv" | \
         "urem" | "srem" | "frem" | "shl" | "lshr" | "ashr" | "and" | "or" | "xor":
      assert result is not None
      opcode = BinaryOpcode(instr.opcode)
      left, right, *_ = instr.operands

      flags = {"nuw": False, "nsw": False, "exact": False, "disjoint": False}
      flags_string = raw_instr_no_res.split(opcode.value + " ", 1)[1]

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
        decodeValue(left, structs, func_names), decodeValue(right, structs, func_names),
        flags["nuw"], flags["nsw"], flags["exact"], flags["disjoint"])

    case "extractelement":
      assert result is not None
      vec, index, *_ = instr.operands
      return ExtractElement(result, decodeValue(vec, structs, func_names), decodeValue(index, structs, func_names))

    case "insertelement":
      vec, item, index, *_ = instr.operands
      return InsertElement(decodeValue(vec, structs, func_names), decodeValue(item, structs, func_names), decodeValue(index, structs, func_names))

    case "shufflevector":
      assert result is not None
      vec1, vec2, *_ = instr.operands

      rest = raw_instr_no_res.split("shufflevector ", 1)[-1].strip()
      tokens_list = parseCommaSeperated(rest)[2]
      mask_val, rest = parseTypeConstantTokens(tokens_list, structs, func_names)
      assert len(rest) == 0

      return ShuffleVector(result, decodeValue(vec1, structs, func_names), decodeValue(vec2, structs, func_names), mask_val)

    case "insertvalue":
      agg, element, *indices = [decodeValue(val, structs, func_names) for val in instr.operands]

      return InsertValue(agg, element, indices)

    case "extractvalue":
      assert result is not None
      agg, *indices = [decodeValue(val, structs, func_names) for val in instr.operands]

      return ExtractValue(result, agg, indices)

    case "alloca":
      assert result is not None

      num_elements = decodeValue(next(instr.operands), structs, func_names)

      rest = raw_instr_no_res.split("alloca ", 1)[1].strip().removeprefix("inalloca ")
      allocated_type, _ = parseTypeTokens(parseUntilComma(rest), structs)

      return Alloca(result, allocated_type, num_elements)

    case "load":
      assert result is not None
      rest = raw_instr_no_res.split("load ", 1)[1].strip() \
        .removeprefix("atomic ").removeprefix("volatile ")

      loaded_type, _ = parseTypeTokens(parseUntilComma(rest), structs)

      value, *_ = instr.operands
      return Load(result, loaded_type, decodeValue(value, structs, func_names))

    case "store":
      value, addr, *_ = instr.operands
      return Store(decodeValue(value, structs, func_names), decodeValue(addr, structs, func_names))

    case "getelementptr":
      assert result is not None
      base_ptr, *indices = instr.operands

      index_values = [decodeValue(idx, structs, func_names) for idx in indices]

      rest = raw_instr_no_res.split("getelementptr ", 1)[1].strip()
      all_keywords = {"inbounds", "inrange", "nusw", "nuw"}
      keywords: set[str] = set()
      has_keyword = True
      while has_keyword:
        has_keyword = False
        for kw in all_keywords:
          if rest.startswith(kw):
            keywords.add(kw)
            if kw != "inrange":
              rest = rest.removeprefix(kw + " ")
            else:
              kw = kw.split(")", 1)[-1]
            has_keyword = True

      ptr_type, _ = parseTypeTokens(parseUntilComma(rest), structs)

      return GetElementPtr(
        result,
        ptr_type,
        decodeValue(base_ptr, structs, func_names),
        index_values,
        "inbounds" in keywords,
        "nusw" in keywords,
        "nuw" in keywords,
      )

    case "trunc" | "zext" | "sext" | "fptrunc" | "fpext" | "fptoui" | "fptosi" | \
         "uitofp" | "sitofp" | "ptrtoint" | "ptrtoaddr" | "inttoptr" | "bitcast" | \
         "addrspacecast":
      assert result is not None
      opcode = ConvOpcode(instr.opcode)
      value, *_ = instr.operands

      conv_type_str = raw_instr_no_res.split(" to ", 1)[-1].strip()
      conv_type, _ = parseTypeTokens(parseUntilComma(conv_type_str), structs)

      flags = {"nuw": False, "nsw": False}
      if instr.opcode == "trunc":
        flags_string = raw_instr_no_res.split(opcode.value + " ", 1)[1]

        while True:
          for flag in flags:
            if flags_string.startswith(flag + " "):
              flags[flag] = True
              flags_string = flags_string[len(flag) + 1:]
              break
          else:
            break

      return Conversion(result, opcode, decodeValue(value, structs, func_names), conv_type, flags["nuw"], flags["nsw"])

    case "icmp":
      assert result is not None
      left, right, *_ = instr.operands
      rest = raw_instr_no_res.split("icmp ", 1)[-1].strip()

      samesign = rest.startswith("samesign ")
      rest = rest.removeprefix("samesign ")

      cond = ICmpCond(rest.split(" ", 1)[0])
      return ICmp(result, cond, decodeValue(left, structs, func_names), decodeValue(right, structs, func_names), samesign)

    case "fcmp":
      assert result is not None
      left, right, *_ = instr.operands
      rest = raw_instr_no_res.split("fcmp ", 1)[-1].strip()

      # Skip over fast math flags
      cond_found = False
      cond = None
      while not cond_found:
        cond_str, rest = rest.split(" ", 1)
        if cond_str in FCmpCond:
          cond_found = True
          cond = FCmpCond(cond_str)
      assert cond is not None

      return FCmp(result, cond, decodeValue(left, structs, func_names), decodeValue(right, structs, func_names))

    case "phi":
      assert result is not None

      incoming: list[tuple[Value, LabelVal]] = []
      for val, label in zip(instr.operands, instr.incoming_blocks):
        incoming.append((decodeValue(val, structs, func_names), decodeLabel(label, structs, func_names)))
      return Phi(result, incoming)

    case "select":
      assert result is not None
      cond, true_val, false_val, *_ = instr.operands
      return Select(result, decodeValue(cond, structs, func_names), decodeValue(true_val, structs, func_names), decodeValue(false_val, structs, func_names))

    case "call":
      *args, callee = instr.operands
      func_val = decodeValue(callee, structs, func_names)
      arg_vals = [decodeValue(arg, structs, func_names) for arg in args]

      tail_kind = CallTailKind.NoTail
      if raw_instr_no_res.startswith("tail "):
        tail_kind = CallTailKind.Tail
      elif raw_instr_no_res.startswith("musttail "):
        tail_kind = CallTailKind.MustTail

      tokens = parseUntilEnd(raw_instr_no_res)

      i = 0
      while tokens[i] in PRE_RET_CALL_ATTRS or tokens[i].isdigit() or \
          (tokens[i].startswith("(") and tokens[i].endswith(")")):
        i += 1

      # Intrinsics cannot be referenced directly
      intrinsic = decodeIntrinsic(func_val.name) if isinstance(func_val, FunctionVal) else None

      fn_type, _ = parseTypeTokens(tokens[i:], structs)
      if isinstance(fn_type, FuncTy):
        return_type = fn_type.return_type
        variadic = fn_type.variadic
      else:
        return_type = fn_type
        variadic = False

      return Call(result, func_val, return_type, arg_vals, variadic, tail_kind, intrinsic)

    case "freeze":
      assert result is not None
      value, *_ = instr.operands
      return Freeze(result, decodeValue(value, structs, func_names))

    case _:
      raise ValueError(f"Opcode {instr.opcode} not implemented")

def decodeModule(mod: llvm.ModuleRef) -> Module:
  func_names = list()
  for func in mod.functions:
    func_names.append(func.name)

  structs: dict[str, StructTy] = {}
  for struct in mod.struct_types:
    # nb Decode type doesn't rely on structs for StructTy yet, but if it did we would need to treat
    # structs referencing structs correctly
    ty = decodeType(struct, {}, func_names)
    assert isinstance(ty, StructTy)
    structs.update({struct.name: ty})

  glob_vars: dict[str, GlobalVar] = {}
  for glob in mod.global_variables:
    glob_name = glob.name

    tokens = parseUntilEnd(str(glob))
    start = 0
    while tokens[start] not in ["constant", "global"]:
      start += 1
    is_glob_constant = tokens[start] == "constant"
    start += 1

    end = start
    while end < len(tokens) - 1 and not tokens[end].endswith(","):
      end += 1

    if tokens[end] == ",":
      end -= 1
    else:
      tokens[end] = tokens[end].removesuffix(",")

    glob_init, rest = parseTypeConstantTokens(tokens[start:end+1], structs, func_names)
    assert len(rest) == 0
    assert isinstance(glob_init, KnownVal)

    glob_vars.update({glob_name: GlobalVar(glob_name, glob_init.type, is_glob_constant, glob_init)})

  functions: dict[str, Function] = {}

  for func in mod.functions:
    fn_name: str = func.name

    fn_rest = str(func)
    while not (fn_rest.startswith("declare ") or fn_rest.startswith("define ")):
      assert "\n" in fn_rest
      fn_rest = fn_rest.split("\n", 1)[-1]
    fn_rest = fn_rest.removeprefix("declare ").removeprefix("define ").strip()
    tokens = parseUntilEnd(fn_rest.split("\n")[0], ignore_unterminated=True)

    i = 0
    while tokens[i] in PRE_RET_FUNC_ATTRS or tokens[i].isdigit() or \
        (tokens[i].startswith("(") and tokens[i].endswith(")")):
      i += 1

    fn_ret_type, rest = parseTypeTokens(tokens[i:], structs)

    # Arguments in token after func name: <ResultType> @<FunctionName> ([argument list])
    # Then strip brackets
    argument_tokens = parseCommaSeperated(rest[1][1:-1])
    fn_variadic = bool(argument_tokens and argument_tokens[-1] == ["..."])

    fn_params: list[ArgumentVal] = []
    for arg in func.arguments:
      arg_val = decodeValue(arg, structs, func_names)
      assert isinstance(arg_val, ArgumentVal)
      fn_params.append(arg_val)

    fn_blocks: dict[str, Block] = {}

    for block in func.blocks:
      block_val = decodeValue(block, structs, func_names)
      assert isinstance(block_val, LabelVal)
      block_name = block_val.label

      instructions: list[Instr] = []
      for instr in block.instructions:
        instructions.append(decodeInstr(instr, structs, func_names))
      fn_blocks.update({block_name: Block(block_name, instructions)})

    intrinsic = decodeIntrinsic(fn_name)

    functions[fn_name] = Function(fn_name, fn_ret_type, fn_params, fn_variadic, intrinsic, fn_blocks)

  return Module(mod.name, functions, glob_vars)

def parseAssembly(llvm_ir: str, verify_ir: bool=False) -> Module:
  mod_ref = llvm.parse_assembly(llvm_ir)
  if verify_ir:
    mod_ref.verify()

  return decodeModule(mod_ref)
