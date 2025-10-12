from typing import Any
import re

from . ir import *

# Regexes used
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
_INT_IN_TYPE_RE = re.compile(r"\b[iI]\d+\s+(-?\d+)\b")
_CSTRING_RE = re.compile(r"c\"((?:[^\"\\]|\\.)*)\"")
_INITLINE_RE = re.compile(r"=\s*(?:global|constant)\s*(.+)$")

_RETURN_ATTRS = [
  "zeroext", "signext", "noext", "inreg",
  "sret", "byval", "byref", "preallocated",
  "inalloca", "elementtype", "alignstack",
  "allocalign", "allocptr", "returned",
  "nonnull", "dereferenceable", "dereferenceable_or_null",
  "nofpclass", "range", "align", "captures",
  "nofree", "nest", "swiftself", "swiftasync",
  "swifterror", "immarg", "noundef", "readnone",
  "readonly", "writeonly", "writable",
  "initializes", "dead_on_unwind", "dead_on_return"
]

# Build regex that matches any of them, optionally with (<...>) or <n> arguments
_ATTR_RE = re.compile(
  r"^(?:" + "|".join(map(re.escape, _RETURN_ATTRS)) + r")(\s*\([^)]*\))?\s+"
)

# ---------- Helpers ----------

def findMatchingBracket(s, start) -> int:
  """Given s[start] is one of '<[{', find index of matching closing bracket.
     Returns index of closing bracket (inclusive). Raises ValueError if not found."""
  pairs = {"<": ">", "[": "]", "{": "}"}
  open_ch = s[start]
  if open_ch not in pairs:
    raise ValueError("Not an opening bracket at start")
  close_ch = pairs[open_ch]
  depth = 0
  i = start
  n = len(s)
  while i < n:
    c = s[i]
    if c == open_ch:
      depth += 1
    elif c == close_ch:
      depth -= 1
      if depth == 0:
        return i
    elif c == '"' and open_ch != '"':
      # skip quoted strings (only relevant for c"...")
      # walk until matching unescaped quote
      j = i + 1
      while j < n:
        if s[j] == '"' and s[j-1] != "\\":
          i = j  # will be incremented at end of loop
          break
        j += 1
    i += 1
  raise ValueError("No matching closing bracket for {} at pos {}".format(open_ch, start))

def stripReturnAttrs(rest: str) -> str:
  s = rest.strip()
  while True:
    m = _ATTR_RE.match(s)
    if not m:
      break
    s = s[m.end():].lstrip()
  return s

def extractBracketContent(s, start) -> tuple[str, int]:
  """Return (content_string, end_index_plus_one).
     start must point at an opening bracket char (<, [, {)."""
  end = findMatchingBracket(s, start)
  return s[start+1:end], end+1

def splitTopLevelCommas(s):
  """
  Split s on commas that are at top-level (not nested inside < >, [ ], { } or quotes).
  Returns list of substrings (stripped).
  """
  parts = []
  buf = []
  depth = 0
  i = 0
  n = len(s)
  while i < n:
    c = s[i]
    if c in "<[{":
      # consume full bracket expression
      j = i
      end = findMatchingBracket(s, i)
      buf.append(s[i:end+1])
      i = end + 1
      continue
    if c == '"':
      # copy quoted string including escapes
      j = i + 1
      buf.append(c)
      while j < n:
        buf.append(s[j])
        if s[j] == '"' and s[j-1] != "\\":
          break
        j += 1
      i = j + 1
      continue
    if c == "," and depth == 0:
      parts.append("".join(buf).strip())
      buf = []
      i += 1
      continue
    # track parentheses depth for safety (though we handled the main bracket types)
    if c in ">]})":
      # shouldn't happen if bracket handling works, but keep char
      buf.append(c)
    else:
      buf.append(c)
    i += 1
  if buf:
    parts.append("".join(buf).strip())
  # remove empty parts (possible from extra commas)
  return [p for p in parts if p != ""]


def _decodeLLVMCStringLiteral(inner) -> list[int]:
  """
  Decode the contents found inside c"..." in LLVM IR.
  Handles simple escapes and octal escapes like \00.
  Returns a list of integer byte values (0..255) instead of bytes.
  """
  # convert octal escapes \NNN to \xHH sequences
  def hex_to_x(m):
    return "\\x" + m.group(1)
  s = re.sub(r"\\([0-F]{2})", hex_to_x, inner)
  # Now interpret standard backslash escapes and \xHH
  # We'll decode via 'unicode_escape' then map to bytes (latin-1) and convert to ints.
  try:
    decoded_bytes = s.encode("utf-8").decode("unicode_escape").encode("latin-1")
  except Exception:
    # fallback: naively replace \xHH occurrences
    out = bytearray()
    i = 0
    while i < len(s):
      if s[i] == "\\" and i+1 < len(s) and s[i+1] == "x":
        hexv = s[i+2:i+4]
        out.append(int(hexv, 16))
        i += 4
      else:
        out.append(ord(s[i]))
        i += 1
    decoded_bytes = bytes(out)
  return list(decoded_bytes)


# ---------- Core parsing ----------

def parseScalarToken(tok) -> int | float | None | str | list[int]:
  """
  Parse a scalar token like:
    - "float 1.000000e+00" -> float
    - "1.0000e+00" -> float
    - "i32 42" -> int
    - "42" -> int
    - "undef" -> None (keeps as None)
    - "null" -> None
  """
  t = tok.strip()
  if not t:
    return t
  # c"..." handled elsewhere
  if t == "undef" or t == "zeroinitializer" or t == "null":
    return None
  # match c"..." pattern
  m = _CSTRING_RE.search(t)
  if m:
    # return list of ints for the string literal
    return _decodeLLVMCStringLiteral(m.group(1))
  # float with a leading "float" keyword
  if t.startswith("float "):
    num = t[len("float "):].strip()
    try:
      return float(num)
    except Exception:
      pass
  # integer with leading type "i32 42" etc
  m = _INT_IN_TYPE_RE.match(t)
  if m:
    return int(m.group(1))
  # try float literal
  m = _FLOAT_RE.search(t)
  if m:
    toknum = m.group(0)
    # decide int vs float
    if re.match(r"^[+-]?\d+$", toknum):
      return int(toknum)
    else:
      return float(toknum)
  # fallback: return raw token
  return t


def parseInitializer(init_text) -> float | int | list[int] | list[Any] | str | None:
  """
  Parse the initializer string and return Python structure:
    - list[int] for c"..." style constant-data arrays
    - list for aggregates
    - scalar for single scalar
    - string fallback otherwise
  """
  s = init_text.strip()

  # 1) c"..." immediate -> list[int]
  m = _CSTRING_RE.search(s)
  if m and s.strip().startswith("c\""):
    return _decodeLLVMCStringLiteral(m.group(1))

  # 2) If begins with a bracket, it may be a type specifier followed by data
  if s and s[0] in "<[{":
    # extract first bracketed group
    try:
      first_content, pos_after_first = extractBracketContent(s, 0)
    except ValueError:
      # can't parse bracket; fallback to scalar parse
      return parseScalarToken(s)

    # decide if the first group is a type-spec (contains 'x' and a type)
    if re.search(r"\b\d+\s*x\b", first_content) or re.search(r"\bx\b", first_content):
      # likely a type spec; skip whitespace and check for a following bracket/data
      rest = s[pos_after_first:].lstrip()
      if rest and rest[0] in "<[{":
        # parse the next bracketed part as the real data
        data_content, pos_after_data = extractBracketContent(rest, 0)
        elems = splitTopLevelCommas(data_content)
        return [parseInitializer(elem) for elem in elems]
      # NEW: if rest contains a c"..." constant-data string, decode it
      m_c = _CSTRING_RE.search(rest)
      if m_c:
        return _decodeLLVMCStringLiteral(m_c.group(1))
      else:
        # no separate data bracket and no c"..." — maybe the first bracket included elements
        elems = splitTopLevelCommas(first_content)
        return [parseInitializer(elem) for elem in elems]
    else:
      # first bracket likely contains actual data elements
      elems = splitTopLevelCommas(first_content)
      return [parseInitializer(elem) for elem in elems]

  # 3) If it looks like an array/vector printed without an outer bracket (rare),
  #    split top-level commas directly.
  top_elems = splitTopLevelCommas(s)
  if len(top_elems) > 1:
    return [parseInitializer(elem) for elem in top_elems]

  # 4) Else try scalar
  return parseScalarToken(s)

# ---------- Convenience wrapper ----------

def decodeGlobalInitializerText(global_str):
  """Given str(gv) or initializer string, extract initializer fragment and parse."""
  m = _INITLINE_RE.search(global_str.strip())
  if m:
    init_text = m.group(1).strip()
    return parseInitializer(init_text)
  else:
    # If not a full global line, assume the whole string is the initializer itself
    return parseInitializer(global_str)

def valueFromParsed(parsed: Any, typ: Type) -> Value:
  """
  Build a Value (KnownIntVal / KnownFloatVal / KnownVecVal / KnownArrVal, etc)
  from a parsed initializer (the result of parse_initializer) and a Type instance.

  - parsed: int | float | list (nested)
  - typ: an instance of your Type classes (IntegerTy, FloatTy, VecTy, ArrayTy, ...)
  """

  if isinstance(typ, IntegerTy):
    if isinstance(parsed, int):
      return KnownIntVal(typ, parsed, typ.width)

    if isinstance(parsed, float) and parsed.is_integer():
      return KnownIntVal(typ, int(parsed), typ.width)
    raise ValueError(f"Type IntegerTy expected int, got {type(parsed)!r}: {parsed}")

  elif isinstance(typ, FloatingPointTy):
    if isinstance(parsed, (int, float)):
      return KnownFloatVal(typ, float(parsed))
    raise ValueError(f"FloatingPointTy expected numeric, got {type(parsed)!r}: {parsed}")

  elif isinstance(typ, VecTy):
    if not isinstance(parsed, list):
      raise ValueError(f"VecTy expected list of elements, got {type(parsed)!r}: {parsed}")
    if len(parsed) != typ.size:
      raise ValueError(f"VecTy size mismatch: expected {typ.size}, got {len(parsed)}")
    vec_values: list[KnownVecTargetVal] = []

    for elem in parsed:
      val = valueFromParsed(elem, typ.inner)
      if not isinstance(val, KnownVecTargetVal):
        raise ValueError(f"Vector element produced non-vec-target value: {val}")
      vec_values.append(val)
    return KnownVecVal(typ, vec_values)

  elif isinstance(typ, ArrayTy):
    if not isinstance(parsed, list):
      raise ValueError(f"ArrayTy expected list of elements, got {type(parsed)!r}: {parsed}")
    if len(parsed) != typ.size:
      raise ValueError(f"ArrayTy size mismatch: expected {typ.size}, got {len(parsed)}")
    arr_values: list[KnownArrTargetVal] = []

    for elem in parsed:
      val = valueFromParsed(elem, typ.inner)
      if not isinstance(val, KnownArrTargetVal):
        raise ValueError(f"Array element produced non-arr-target value: {val}")
      arr_values.append(val)

    return KnownArrVal(typ, arr_values)

  elif isinstance(typ, PointerTy):
    if parsed is None:
      raise NotImplementedError("Null pointer initializers not implemented yet")
    # If parsed is a string (e.g., "@globalname") we could return GlobalVarVal
    if isinstance(parsed, str) and parsed.startswith("@"):
      # name without @
      return GlobalVarVal(typ, parsed[1:])
    # otherwise fallback: raise
    raise ValueError(f"PointerTy initializer unrecognized: {parsed!r}")

  # Label type (e.g., basic block label) — parsed must be string
  elif isinstance(typ, LabelTy):
    if isinstance(parsed, str):
      return LabelVal(typ, parsed)
    raise ValueError(f"LabelTy initializer expected string label, got {parsed!r}")

  # Void or unknown
  elif isinstance(typ, VoidTy):
    raise ValueError("VoidTy cannot have an initializer value")

  # Fallback for unknown types
  raise NotImplementedError(f"value_from_parsed not implemented for type {type(typ).__name__}")

def valueFromInitializerText(init_text: str, typ: Type) -> Value:
  """
  Convenience wrapper: parse initializer text (via parse_initializer),
  then convert into a Value using value_from_parsed.
  """
  # parse_initializer must be defined in the module (from your earlier parser)
  parsed = parseInitializer(init_text)
  return valueFromParsed(parsed, typ)

def extractFirstType(s: str) -> tuple[str, str]:
  """
  Given an LLVM IR instruction fragment, return (type_str, remainder).
  Handles inline structs, arrays, vectors, pointers, and named types.

  Example:
    " { i32, i8 }, align 8" -> ("{ i32, i8 }", "align 8")
    " i32, align 4" -> ("i32", "align 4")
    " [10 x { i32, float }], something" -> ("[10 x { i32, float }]", "something")
  """
  s = s.strip()
  if not s:
    return "", ""

  # If it starts with a bracketed type ({, [, <), consume full group
  if s[0] in "<[{":
    end = findMatchingBracket(s, 0)
    type_str = s[:end+1].strip()
    remainder = s[end+1:].lstrip(" ,")
    return type_str, remainder

  # Otherwise, it's a word-like type (e.g. i32, %MyStruct, float, i8*)
  i = 0
  n = len(s)
  while i < n:
    c = s[i]
    if c in ", ":
      break
    i += 1
  type_str = s[:i].strip()
  remainder = s[i:].lstrip(" ,")
  return type_str, remainder

def extractTypedValue(s: str) -> tuple[str, str, str]:
  """
  Extracts a 'type value' pair.
  Returns (type_str, value_str, remainder).
  """
  type_str, rest = extractFirstType(s)
  rest = rest.lstrip()

  if rest and rest[0] in "<[{":
    # constant literal in brackets
    end = findMatchingBracket(rest, 0)
    value_str = rest[:end+1].strip()
    remainder = rest[end+1:].lstrip(" ,")
    return type_str, value_str, remainder
  else:
    # single scalar or identifier
    m = re.match(r"([^,]+)", rest)
    if not m:
      return type_str, "", rest
    value_str = m.group(1).strip()
    remainder = rest[m.end():].lstrip(" ,")
    return type_str, value_str, remainder
