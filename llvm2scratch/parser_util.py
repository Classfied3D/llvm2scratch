from typing import Callable, cast
import struct
import copy
import re

from . ir import *

# Parameter Attributes
PARAM_ATTRS = [
  "zeroext", "signext", "noext", "inreg",
  "byval", "byref", "preallocated", "inalloca",
  "sret", "elementtype", "align", "noalias",
  "captures", "nofree", "nest", "returned",
  "nonnull", "dereferenceable",
  "dereferenceable_or_null", "swiftself",
  "swiftasync", "swifterror", "immarg",
  "noundef", "nofpclass", "alignstack",
  "allocalign", "allocptr", "readnone",
  "readonly", "writeonly", "writeable",
  "initializes", "dead_on_unwind",
  "dead_on_return", "range"
]

# Calling Conventions
CALL_CONV = [
  "ccc", "fastcc", "coldcc", "ghccc", "anyregcc",
  "preserve_mostcc", "preserve_allcc", "preserve_nonecc",
  "cxx_fast_tlscc", "tailcc", "swiftcc",
  "swifttailcc", "cfguard_checkcc", "cc",
]

# Note some of these attrs may contain a number or something surrounded by brackets after
PRE_RET_FUNC_ATTRS = [
  # Linkage Types
  "private", "internal", "available_externally", "linkonce",
  "weak", "common", "appending", "extern_weak",
  "linkonce_odr", "weak_odr", "external",

  # Runtime Preemption Specifiers
  "dso_preemptable", "dso_local",

  # Visiblity Style
  "default", "hidden", "protected",

  # DLL Storage Class
  "dllimport", "dllexport", "localdynamic",

  # Calling Convention
  *CALL_CONV,

  # Parameter Attributes for return type
  *PARAM_ATTRS
]

PRE_RET_CALL_ATTRS = [
  # Tail call optimization markers
  "tail", "musttail", "notail",

  # The call instruction itself
  "call",

  # Fast Math Flags
  "nnan", "ninf", "nsz", "arcp", "contract",
  "afn", "reassoc", "fast",

  # Calling Convention
  *CALL_CONV,

  # Parameter Attributes for return type
  *PARAM_ATTRS,

  # Addrspace of called function
  "addrspace",
]

# Returns (decoded, rest, parsed_len), where parsed_len is the length
# of syntax decoded (including quotes)
def parseQuoted(rest: str) -> tuple[str, str, int]:
  assert rest.startswith('"')
  decoded = ""
  i = 1
  escaped = False
  escaped_hex = ""

  while rest[i] != '"':
    is_backslash = rest[i] == "\\"

    if escaped:
      if is_backslash:
        escaped = False
        decoded += "\\"
      else:
        assert rest[i].lower() in "0123456789abcdef"
        escaped_hex += rest[i].lower()
        if len(escaped_hex) >= 2:
          decoded += chr(int(escaped_hex, 16))
          escaped_hex = ""
          escaped = False

    elif is_backslash:
      escaped = True
    else:
      decoded += rest[i]
    i += 1

  i += 1
  assert not escaped

  return decoded, rest[i:], i

# Follow strings/brackets until reaching token where match(token) is true (spaces are removed from token)
# A token occurs when a bracket is closed, a quote is closed or a space
# Returns (found, tokens, parsed, match, rest)
def parseUntil(rest: str, match: Callable[[str], bool], ignore_unterminated=False) -> tuple[bool, list[str], str, str, str]:
  brackets = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
  }

  i = start_i = 0
  token = ""
  tokens = []
  bracket_stack = []

  while i == 0 or not match(token):
    token = ""
    start_i = i

    if i > len(rest) - 1:
      return False, tokens, rest, "", ""

    if rest[i] in brackets.keys():
      bracket_stack.append(brackets[rest[i]])
      while len(bracket_stack) > 0:
        i += 1
        if i > len(rest) - 1:
          if ignore_unterminated:
            break
          else:
            raise ValueError("Could not find closing bracket, exceeded length of rest")

        if rest[i] == bracket_stack[-1]:
          bracket_stack.pop()
        elif rest[i] in brackets.values():
          raise ValueError(f"Got closing bracket without opening one: {rest[i]} in {rest}")
        elif rest[i] in brackets.keys():
          bracket_stack.append(brackets[rest[i]])
        elif rest[i] == '"':
          _, _, parsed_len = parseQuoted(rest[i:])
          i += parsed_len - 1
      token = rest[start_i:i+1].strip()

    elif rest[i] in brackets.values():
      raise ValueError(f"Got closing bracket without opening one: {rest[i]} in {rest}")

    elif rest[i] == '"':
      _, _, parsed_len = parseQuoted(rest[i:])
      i += parsed_len - 1
      token = rest[start_i:i+1].strip()

    else:
      while i < len(rest) and rest[i] != " " and rest[i] not in brackets.keys() and rest[i] != '"':
        i += 1

      token = rest[start_i:i].strip()

      # We included the opening bracket of the next token, give it back
      if i < len(rest) and (rest[i] in brackets.keys() or rest[i] == '"'):
        i -= 1

    if token != "":
      tokens.append(token)
    i += 1

  # Don't include last i += 1, unless we didn't parse anything
  if i > 0:
    i -= 1

  return True, tokens, rest[:start_i], rest[start_i:i], rest[i:]

def parseUntilEnd(rest: str, ignore_unterminated=False) -> list[str]:
  _, tokens, _, _, _ = parseUntil(rest, lambda _: False, ignore_unterminated)
  return tokens

def parseUntilComma(rest: str) -> list[str]:
  _, tokens, _, _, _ = parseUntil(rest, lambda x: x.endswith(","))
  # Remove comma from end token, unless the end token was just a commas
  if tokens[-1] == ",":
    del tokens[-1]
  else:
    tokens[-1] = tokens[-1].removesuffix(",")
  return tokens

def parseCommaSeperated(rest: str) -> list[list[str]]:
  if rest.strip() == "": return []

  tokens_list: list[list[str]] = []
  found = True

  while found:
    found, tokens, _, _, rest = parseUntil(rest, lambda x: x.endswith(","))
    assert len(tokens) > 0
    # Remove comma from end token, unless the end token was just a comma
    if tokens[-1] == ",":
      del tokens[-1]
    else:
      tokens[-1] = tokens[-1].removesuffix(",")
    tokens_list.append(tokens)

  assert rest.strip() == ""

  return tokens_list

def parseTypeToken(type: str, structs: dict[str, StructTy], is_struct: bool=False) -> Type:
  if (type.startswith("{") and type.endswith("}")) or \
      (type.startswith("<{") and type.endswith("}>")) or is_struct: # If previous token was "type"
    if type == "opaque":
      raise ValueError("Opaque structures not yet supported") # TODO

    is_packed = type.startswith("<{")
    if is_packed:
      assert type.endswith("}>")
      rest = type[2:-2].strip()
    else:
      assert type.startswith("{")
      assert type.endswith("}")
      rest = type[1:-1].strip()

    tokens_list = parseCommaSeperated(rest)
    members: list[AggTargetTy] = []
    for tokens in tokens_list:
      member, rest = parseTypeTokens(tokens, structs)
      assert len(rest) == 0
      assert isinstance(member, AggTargetTy)
      members.append(member)

    return StructTy(is_packed, members)

  elif type == "void": return VoidTy()

  elif type.startswith("i") and type[1:].isdigit():
    return IntegerTy(int(type[1:]))

  elif type == "half":   return HalfTy()
  elif type == "float":  return FloatTy()
  elif type == "double": return DoubleTy()
  elif type == "fp128":  return Fp128Ty()
  elif type in ["bfloat", "x86_fp80", "ppc_fp128"]:
    raise ValueError(f"Unsupported FP type: {type}")

  elif type in ["x86_amx", "x86_mmx"]:
    raise ValueError(f"Unsupported type: {type}")

  elif type == "ptr":
    return PointerTy(addrspace=0)

  elif type.startswith("<") and type.endswith(">") and " x " in type:
    parts = [x.strip() for x in type[1:-1].split(" x ", 1)]

    if parts[0].isdigit():
      size = int(parts[0])
      inner, rest = parseTypeTokens(parseUntilEnd(parts[1]), structs)
      assert len(rest) == 0
      assert isinstance(inner, VecTargetTy)
      return VecTy(inner, size)

    else:
      assert parts[0] == "vscale"
      raise ValueError(f"Scalable vectors not supported: {type}")

  elif type == "label":    return LabelTy()
  elif type == "token":    raise ValueError(f"Token type not supported yet")
  elif type == "metadata": return MetadataTy()

  elif type.startswith("[") and type.endswith("]") and " x " in type:
    parts = [x.strip() for x in type[1:-1].split(" x ", 1)]
    size = int(parts[0])
    inner, rest = parseTypeTokens(parseUntilEnd(parts[1]), structs)
    assert len(rest) == 0
    assert isinstance(inner, AggTargetTy)
    return ArrayTy(inner, size)

  elif type.startswith("%"):
    name = type[1:]
    return structs[name]

  else:
    raise ValueError(f"Unsupported type: {type}")

def parseTypeTokens(tokens: list[str], structs: dict[str, StructTy]) -> tuple[Type, list[str]]:
  is_struct = False
  is_addrspace = False
  current_type: Type | None = None

  for i, token in enumerate(tokens):
    if token.startswith("(") and token.endswith(")"):
      assert not is_struct
      if is_addrspace:
        # Addrspace identifier
        contents = token[1:-1].strip()
        if contents.startswith('"'):
          space, rest, _ = parseQuoted(contents)
          assert rest.strip() == ""
        else:
          assert contents.isdigit()
          space = int(contents)

        assert isinstance(current_type, PointerTy)
        current_type.addrspace = space
        is_addrspace = False
      else:
        # Function pointer
        assert current_type is not None

        args: list[Type] = []
        variadic = False
        tokens_list = parseCommaSeperated(token[1:-1])
        for i, tokens in enumerate(tokens_list):
          if "..." in tokens:
            assert len(tokens) == 1
            assert i == len(tokens_list) - 1
            variadic = True
            continue
          parsed, rest = parseTypeTokens(tokens, structs)
          assert len(rest) == 0
          args.append(parsed)

        current_type = FuncTy(current_type, args, variadic)

    elif token == "type":
      assert is_addrspace is False
      is_struct = True

    elif token == "addrspace":
      assert is_struct is False
      assert is_addrspace is False
      assert isinstance(current_type, PointerTy)
      assert current_type.addrspace == 0
      is_addrspace = True

    elif current_type is None:
      assert is_addrspace is False
      current_type = parseTypeToken(token, structs, is_struct)
      is_struct = False

    else:
      # Reached end of tokens
      assert is_struct is False
      assert is_addrspace is False
      return current_type, tokens[i:]

  assert current_type is not None
  return current_type, []

def undoTwosComplement(val: int, width: int) -> int:
  if (val & (1 << (width - 1))) != 0:
    val -= (1 << width)
  return val

def applyTwosComplement(val: int, width: int) -> int:
  if val < 0:
    val += (1 << width)
  return val

def parseIEEEFloat(s: str) -> float:
  bits = int(s[2:], 16)
  assert bits < 2**64
  return struct.unpack(">d", struct.pack(">Q", bits))[0]

def getZeroInitVal(ty: Type) -> Value:
  return parseConstantToken(ty, "zeroinitializer", {}, [])

def parseBracketedListToken(token: str, size: int, opening: str, closing: str, structs: dict[str, StructTy], func_names: list[str]) -> list[Value]:
  if not (token.startswith(opening) and token.endswith(closing)):
    raise ValueError(f"Invalid constant: {token}")

  member_tokens = parseCommaSeperated(token[len(opening):-len(closing)])
  assert len(member_tokens) == size

  values = []
  for mem_tokens in member_tokens:
    mem, rest = parseTypeConstantTokens(mem_tokens, structs, func_names)
    assert len(rest) == 0
    values.append(mem)

  return values

def parseConstantToken(ty: Type, token: str, structs: dict[str, StructTy], func_names: list[str], is_char_arr: bool=False, is_splat: bool=False) -> Value:
  if token in ["undef", "poison"]:
    return UndefVal(ty)

  match ty:
    case IntegerTy():
      if token == "true":
        assert ty.width == 1
        value = 1
      elif token == "false":
        assert ty.width == 1
        value = 0

      elif token.isdigit():
        value = int(token)

      elif token.startswith("-") and token[1:].isdigit():
        value = applyTwosComplement(int(token), ty.width)

      elif token.startswith("u0x") or token.startswith("s0x"):
        assert all(x.lower() in "0123456789abcdef" for x in token[3:])
        value = int(token[3:], 16)
        if token.startswith("s0x"):
          value_width = value.bit_length()
          value = applyTwosComplement(
            undoTwosComplement(value, value_width),
            ty.width)

      elif token == "zeroinitializer":
        value = 0

      else:
        raise ValueError(f"Invalid integer constant: {token}")

      # Must be two's complement form
      assert value >= 0
      assert value < 2**ty.width

      return KnownIntVal(ty, value, ty.width)

    case FloatingPointTy():
      # n.b. Precision can be lost - we don't care because this doesn't matter in scratch since
      # it will always use 64-bit fp values anyway
      if "." in token:
        if "e" not in token:
          # Decimal notation
          value = float(token)
        else:
          # Exponential notation
          parts = token.split("e")
          assert len(parts) == 2
          assert parts[1][0] in ["+", "-"]
          coeff = float(parts[0])
          exp = int(parts[1][1:]) * (1 if parts[1][0] == "+" else -1)
          value = coeff * 10**exp

      elif token.startswith("0x"):
        if isinstance(ty, Fp128Ty):
          raise ValueError("0xL 128-bit fp format not yet supported")
        value = parseIEEEFloat(token)

      elif token == "zeroinitializer":
        value = 0.0

      else:
        raise ValueError(f"Invalid fp constant: {token}")

      return KnownFloatVal(ty, value)

    case PointerTy():
      if token in ["null", "zeroinitializer"]:
        return NullPtrVal(ty)

      elif token.startswith("@") or token.startswith("%"):
        name = token[1:]
        if name in func_names:
          return FunctionVal(ty, name)
        else:
          return GlobalPtrVal(ty, name)

      else:
        raise ValueError(f"Invalid pointer constant: {token}")

    case StructTy():
      if token == "zeroinitializer":
        mems = cast(list[KnownAggTargetVal], [getZeroInitVal(mem) for mem in ty.members])
        return KnownStructVal(ty, mems)
      else:
        struct_mems = []
        opening, closing = ("<{", "}>") if ty.is_packed else ("{", "}")
        mems = parseBracketedListToken(token, len(ty.members), opening, closing, structs, func_names)
        for i, mem in enumerate(mems):
          assert isinstance(mem, KnownAggTargetVal)
          assert mem.type == ty.members[i]
          struct_mems.append(mem)
        return KnownStructVal(ty, struct_mems)

    case ArrayTy():
      if token == "zeroinitializer":
        arr_vals = cast(list[KnownAggTargetVal], [getZeroInitVal(ty.inner) for _ in range(ty.size)])
        return KnownArrVal(ty, arr_vals)

      elif not is_char_arr:
        arr_mems = []
        mems = parseBracketedListToken(token, ty.size, "[", "]", structs, func_names)
        for mem in mems:
          assert isinstance(mem, KnownAggTargetVal)
          assert mem.type == ty.inner
          arr_mems.append(mem)
        return KnownArrVal(ty, arr_mems)

      else:
        assert ty.inner == IntegerTy(8)
        decoded, rest, _ = parseQuoted(token)
        assert len(rest) == 0
        assert len(decoded) == ty.size

        values: list[KnownAggTargetVal] = []
        for char in decoded:
          values.append(KnownIntVal(IntegerTy(8), ord(char), 8))

        return KnownArrVal(ty, values)

    case VecTy():
      if token == "zeroinitializer":
        vec_vals = cast(list[KnownVecTargetVal], [getZeroInitVal(ty.inner) for _ in range(ty.size)])
        return KnownVecVal(ty, vec_vals)

      elif not is_splat:
        vec_mems = []
        mems = parseBracketedListToken(token, ty.size, "<", ">", structs, func_names)
        for mem in mems:
          assert isinstance(mem, KnownVecTargetVal)
          assert mem.type == ty.inner
          vec_mems.append(mem)
        return KnownVecVal(ty, vec_mems)

      else:
        if not (token.startswith("(") and token.endswith(")")):
          raise ValueError(f"Invalid splat vec constant: {token}")
        value, rest = parseTypeConstantTokens(parseUntilEnd(token[1:-1]), structs, func_names)
        assert len(rest) == 0
        assert isinstance(value, KnownVecTargetVal)
        assert value.type == ty.inner
        return KnownVecVal(ty, [copy.deepcopy(value) for _ in range(ty.size)])

    case _:
      raise ValueError(f"Invalid constant - type {ty} not supported: {token}")

def getConstExprBracketValues(brackets: str, count: int, structs: dict[str, StructTy], func_names: list[str]) -> list[Value]:
  assert brackets.startswith("(") and brackets.endswith(")")
  tokens_list = parseCommaSeperated(brackets[1:-1])
  values = []
  for tokens in tokens_list:
    parsed, rest = parseTypeConstantTokens(tokens, structs, func_names)
    assert len(rest) == 0
    values.append(parsed)
  assert len(values) == count
  return values

# Parse a type and constant i.e. i32 712
def parseTypeConstantTokens(tokens: list[str], structs: dict[str, StructTy], func_names: list[str]) -> tuple[Value, list[str]]:
  ty, tokens = parseTypeTokens(tokens, structs)
  is_char_arr = is_splat = False

  if tokens[0] in ["blockaddress", "dso_local_equivalent", "no_cfi", "ptrauth", "asm"]:
    raise ValueError(f"Unsupported value type {tokens[0]}")

  # Constant expressions
  elif tokens[0] in ["trunc", "ptrtoint", "ptrtoaddr", "inttoptr", "bitcast", "addrspacecast"]:
    assert tokens[1].startswith("(") and tokens[1].endswith(")")
    contents = tokens[1][1:-1]

    found, casted_tokens, _, _, rest = parseUntil(contents, lambda x: x == "to")
    assert found

    opcode = ConvOpcode(tokens[0])
    # casted_tokens[:-1] to remove "to"
    casted, rest_casted_tokens = parseTypeConstantTokens(casted_tokens[:-1], structs, func_names)
    assert len(rest_casted_tokens) == 0
    casted_type, rest = parseTypeTokens(parseUntilEnd(rest), structs)
    assert len(rest) == 0
    assert casted_type == ty

    return ConstExprVal(ty,
      Conversion(ResultLocalVar(""), opcode, casted, casted_type, False, False)), tokens[2:]

  elif tokens[0] == "getelementptr":
    i = 1
    while not tokens[i].startswith("("):
      if tokens[i] == "inrange":
        i += 1 # Skip over inrange brackets
      i += 1
    assert tokens[i][-1] == ")"

    parts = parseCommaSeperated(tokens[i][1:-1])
    assert len(parts) >= 3

    base_ptr_type, rest = parseTypeTokens(parts[0], structs)
    assert len(rest) == 0

    parsed_parts = []
    for part in parts[1:]:
      parsed, rest = parseTypeConstantTokens(part, structs, func_names)
      assert len(rest) == 0
      parsed_parts.append(parsed)

    base_ptr = parsed_parts[0]
    assert isinstance(base_ptr.type, PointerTy)

    return ConstExprVal(ty,
      GetElementPtr(
        ResultLocalVar(""),
        base_ptr_type,
        base_ptr,
        parsed_parts[1:],
        False, False, False, # Shouldn't matter to the compiler for a known offset
                             # TODO constant GEP keywords
      )
    ), tokens[i+1:]

  elif tokens[0] == "extractelement":
    agg, idx = getConstExprBracketValues(tokens[1], 2, structs, func_names)
    return ConstExprVal(ty, ExtractElement(ResultLocalVar(""), agg, idx)), tokens[2:]

  elif tokens[0] == "insertelement":
    agg, el, idx = getConstExprBracketValues(tokens[1], 3, structs, func_names)
    return ConstExprVal(ty, InsertElement(agg, el, idx)), tokens[2:]

  elif tokens[0] == "shufflevector":
    fst_vector, snd_vector, mask_vector = getConstExprBracketValues(tokens[1], 3, structs, func_names)
    return ConstExprVal(ty, ShuffleVector(ResultLocalVar(""), fst_vector, snd_vector, mask_vector)), tokens[2:]

  elif tokens[0] in ["add", "sub", "mul", "shl", "xor"]:
    lhs, rhs = getConstExprBracketValues(tokens[1], 2, structs, func_names)
    opcode = BinaryOpcode(tokens[0])
    return ConstExprVal(ty, BinaryOp(ResultLocalVar(""), opcode, lhs, rhs, False, False, False, False)), tokens[2:]

  else:
    # Character array
    if isinstance(ty, ArrayTy) and tokens[0] == "c":
      assert tokens[1].startswith('"')
      tokens = tokens[1:]
      is_char_arr = True

    # Splat vec
    elif isinstance(ty, VecTy) and tokens[0] == "splat":
      assert tokens[1].startswith("(")
      tokens = tokens[1:]
      is_splat = True

    return parseConstantToken(ty, tokens[0], structs, func_names, is_char_arr, is_splat), tokens[1:]
