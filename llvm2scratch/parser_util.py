from llvm2scratch.ir import KnownAggTargetVal
from llvm2scratch.ir import KnownIntVal
from typing import Any, Optional, cast
import re

from . ir import *

FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
TYPE_NAME_RE = re.compile(r'%[A-Za-z0-9._]+')
INT_IN_TYPE_RE = re.compile(r"\b[iI]\d+\s+(-?\d+)\b")
CSTRING_RE = re.compile(r'c"((?:\\.|[^"])*)"')
INITLINE_RE = re.compile(r"=\s*(?:global|constant)\s*(.+)$")

RETURN_ATTRS = [
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
ATTR_RE = re.compile(
  r"^(?:" + "|".join(map(re.escape, RETURN_ATTRS)) + r")(\s*\([^)]*\))?\s+"
)

def extractBracketContent(s: str, start: int) -> tuple[str,int]:
  openCh = s[start]
  pairs = {'<':'>','[':']','{':'}','(':')'}
  if openCh not in pairs:
    raise ValueError("not a bracket at start")
  closeCh = pairs[openCh]
  i = start + 1
  depth = 1
  contentStart = i
  L = len(s)
  while i < L and depth > 0:
    c = s[i]
    if c == '"':
      j = i + 1
      while j < L:
        if s[j] == '"' and s[j-1] != '\\':
          i = j
          break
        j += 1
    elif c == openCh:
      depth += 1
    elif c == closeCh:
      depth -= 1
      if depth == 0:
        return s[contentStart:i], i+1
    i += 1
  raise ValueError("unmatched bracket")

def splitTopLevelCommas(s: str) -> list[str]:
  elems = []
  i = 0
  start = 0
  stack = []
  l = len(s)
  pairs = {'<':'>','{':'}','[':']','(':')'}
  opens = set(pairs.keys()); closes = set(pairs.values())
  in_quote = False
  while i < l:
    c = s[i]
    if c == '"' and (i == 0 or s[i-1] != '\\'):
      in_quote = not in_quote
      i += 1
      continue
    if in_quote:
      i += 1
      continue
    if c in opens:
      stack.append(c)
    elif c in closes:
      if stack:
        top = stack[-1]
        if pairs.get(top) == c:
          stack.pop()
        else:
          stack.pop()
    elif c == ',' and not stack:
      elems.append(s[start:i].strip())
      start = i+1
    i += 1
  last = s[start:].strip()
  if last:
    elems.append(last)
  return elems

def isTypeToken(tok: str) -> bool:
  tok = tok.strip()
  if re.fullmatch(r'\[\s*\d+\s*x\b[^\]]+\]', tok):
    return True
  if re.fullmatch(r'(?:i\d+|ptr|float|double|void|int|char|i8|i16|i32|i64|i128)', tok):
    return True
  if TYPE_NAME_RE.fullmatch(tok):
    return True
  return False

def isTypeOnly(content: str) -> bool:
  if not content:
    return False
  if re.search(r'c"|zeroinitializer|@|null|getelementptr\b', content):
    return False
  if re.match(r'^\s*\d+\s*x\b', content):
    return True
  pieces = splitTopLevelCommas(content)
  if not pieces:
    return False
  for p in pieces:
    p = p.strip()
    if isTypeToken(p):
      continue
    m = re.match(r'^(%[A-Za-z0-9._]+)\s*(\{.*\})$', p, re.S)
    if m:
      inner = m.group(2)
      if inner.startswith('{') and inner.endswith('}') and isTypeOnly(inner[1:-1].strip()):
        continue
    if p.startswith('{') and p.endswith('}'):
      inner = p[1:-1].strip()
      if isTypeOnly(inner):
        continue
      return False
    if p.startswith('<{') and p.endswith('}>'):
      inner = p[2:-2].strip()
      if isTypeOnly(inner):
        continue
      return False
    return False
  return True

def decodeLLVMCStringLiteral(inner: str) -> list[int]:
  def hex_to_x(m): return "\\x"+m.group(1)
  s = re.sub(r"\\([0-F]{2})", hex_to_x, inner)
  try:
    decoded_bytes = s.encode("utf-8").decode("unicode_escape").encode("latin-1")
  except Exception:
    out=bytearray(); i=0; L=len(s)
    while i < L:
      if s[i]=='\\' and i+1 < L and s[i+1]=='x':
        out.append(int(s[i+2:i+4],16)); i+=4
      else:
        out.append(ord(s[i])); i+=1
    decoded_bytes = bytes(out)
  return list(decoded_bytes)

def stripLeadingTypes(s: str) -> str:
  s = s.strip()
  if not s: return s
  if s[0] in '<[{(':
    return s
  if s.startswith('c"') or s.startswith('@') or s.startswith('%') or s.startswith('getelementptr') or s.startswith('null') or s[0] == '-' or s[0].isdigit():
    return s
  parts = s.split(None, 1)
  return parts[1] if len(parts) > 1 else parts[0]

def readAdjacentValueAfterType(elem: str, decode_gep_str_func) -> Optional[Any]:
  elem = elem.strip()

  if elem.startswith("getelementptr"):
    return parseScalarToken(elem, decode_gep_str_func)

  m = re.match(r'^\s*\[\s*(\d+)\s*x\s*[^\]]+\]\s*(.+)$', elem, re.S)
  if m:
    rest = m.group(2).strip()
    if ',' in rest and splitTopLevelCommas(rest)[0] != rest:
      return None
    if rest.startswith('c"'):
      mm = CSTRING_RE.match(rest)
      if mm:
        return decodeLLVMCStringLiteral(mm.group(1))
    if rest.startswith('zeroinitializer'):
      return "zeroinitializer"
    if rest and rest[0] in '<[{(':
      return parseInitializer(rest, decode_gep_str_func)
    if rest.startswith('@') or rest.startswith('%'):
      return rest
    num_m = re.match(r'^-?0x[0-9a-fA-F]+|^-?\d+', rest)
    if num_m:
      tok = num_m.group(0)
      try:
        if tok.lower().startswith('0x'):
          return int(tok, 16)
        return int(tok, 10)
      except:
        return rest
    return rest

  if elem and elem[0] in '<{(':
    try:
      grp, after = extractBracketContent(elem, 0)
    except ValueError:
      return None
    rest = elem[after:].strip()
    if ',' in rest and splitTopLevelCommas(rest)[0] != rest:
      return None
    if rest.startswith('c"'):
      mm = CSTRING_RE.match(rest)
      if mm:
        return decodeLLVMCStringLiteral(mm.group(1))
    if rest.startswith('zeroinitializer'):
      return "zeroinitializer"
    if rest and rest[0] in '<[{(':
      return parseInitializer(rest, decode_gep_str_func)
    if rest.startswith('@') or rest.startswith('%'):
      return rest
    num_m = re.match(r'^-?0x[0-9a-fA-F]+|^-?\d+', rest)
    if num_m:
      tok = num_m.group(0)
      try:
        if tok.lower().startswith('0x'):
          return int(tok, 16)
        return int(tok, 10)
      except:
        return rest
    return None
  return None

def findDataBracket(s: str, start: int = 0) -> tuple[str,int]:
  pos = start; L=len(s)
  while pos < L and s[pos].isspace(): pos+=1
  if pos >= L or s[pos] not in '<[{': raise ValueError("expected bracket at start")
  while pos < L and s[pos] in '<[{':
    content, after = extractBracketContent(s, pos)
    if re.search(r'c"|zeroinitializer|@|null|getelementptr\b', content) or not isTypeOnly(content):
      return content, after
    k=0; groups=[]
    while k < len(content):
      while k < len(content) and content[k].isspace(): k+=1
      if k >= len(content): break
      if content[k] in '<[{':
        inner, inner_after = extractBracketContent(content, k)
        groups.append((inner, k, inner_after))
        k = inner_after; continue
      k += 1
    for inner, _, inner_after in groups:
      if re.search(r'c"|zeroinitializer|@|null|getelementptr\b', inner) or not isTypeOnly(inner):
        abs_after = pos + inner_after
        return inner, abs_after
    pos = after
    while pos < L and s[pos].isspace(): pos+=1
    if pos >= L or s[pos] not in '<[{': return content, after
  return content, after

def parseScalarToken(s: str, decode_gep_str_func) -> Any:
  s = s.strip()

  if s.startswith("getelementptr"):
    return decode_gep_str_func(s)

  s = s.strip()
  if not s: return None
  m = CSTRING_RE.match(s)
  if m and s.startswith('c"'): return decodeLLVMCStringLiteral(m.group(1))
  adj = readAdjacentValueAfterType(s, decode_gep_str_func)
  if adj is not None: return adj
  s2 = stripLeadingTypes(s).strip()
  if not s2: return None
  if s2 == 'null': return None
  if s2.startswith('@') or s2.startswith('%'): return s2
  num_m = re.match(r'^-?0x[0-9a-fA-F]+|^-?\d+', s2)
  if num_m:
    tok = num_m.group(0)
    try:
      if tok.lower().startswith('0x'): return int(tok,16)
      return int(tok,10)
    except: pass
  if s2.startswith('getelementptr'):
    idx = s2.find('(')
    if idx != -1:
      depth = 0; i = idx; L = len(s2)
      while i < L:
        if s2[i] == '(':
          depth += 1
        elif s2[i] == ')':
          depth -= 1
          if depth == 0:
            return s2[:i+1].strip()
        i += 1
    return s2
  if s2 and s2[0] in '<[{(':
    return parseInitializer(s2, decode_gep_str_func)
  return s2

def parseInitializer(init_text: str, decode_gep_str_func) -> Any:
  s = init_text.strip()
  if not s:
    return None

  # direct getelementptr -> delegate to scalar handler / decode function
  if s.startswith("getelementptr"):
    return parseScalarToken(s, decode_gep_str_func)

  # direct c-string
  m = CSTRING_RE.match(s)
  if m and s.startswith('c"'):
    return decodeLLVMCStringLiteral(m.group(1))

  # bracketed forms
  if s[0] in '<[{(':
    try:
      first_content, pos_after_first = extractBracketContent(s, 0)
    except ValueError:
      return parseScalarToken(s, decode_gep_str_func)

    # If the first bracket looks like a pure type spec and there's an adjacent
    # value (e.g. "[N x i8] c'...'" or "{ i8 } c'...'"), parse that adjacent value
    rest = s[pos_after_first:].lstrip()
    if isTypeOnly(first_content) and rest:
      if rest.startswith('c"'):
        mm = CSTRING_RE.match(rest)
        if mm:
          return decodeLLVMCStringLiteral(mm.group(1))
      if rest.startswith('zeroinitializer'):
        return "zeroinitializer"
      if rest.startswith('getelementptr'):
        return parseScalarToken(rest, decode_gep_str_func)
      if rest and rest[0] in '<[{(':
        return parseInitializer(rest, decode_gep_str_func)
      if rest.startswith('@') or rest.startswith('%'):
        return rest
      num_m = re.match(r'^-?0x[0-9a-fA-F]+|^-?\d+', rest)
      if num_m:
        tok = num_m.group(0)
        try:
          if tok.lower().startswith('0x'):
            return int(tok, 16)
          return int(tok, 10)
        except:
          pass

    # At this point we are dealing with a real bracketed aggregate.
    # Remember the opening bracket to preserve semantics later.
    open_ch = s[0]

    try:
      outer_content, pos_after = extractBracketContent(s, 0)
    except ValueError:
      return parseScalarToken(s, decode_gep_str_func)

    # findDataBracket returns the bracket that actually contains data (skipping type-only wrappers)
    try:
      data_content, data_after = findDataBracket(s, 0)
    except Exception:
      return parseScalarToken(s, decode_gep_str_func)

    top_elems = splitTopLevelCommas(outer_content)   # elements as written in outer bracket
    data_elems = splitTopLevelCommas(data_content)   # the actual data elements we will parse

    parsed: list[Any] = []
    for de in data_elems:
      de = de.strip()
      # If element is a named aggregate "%name { ... }", parse the inner braces
      m_named = re.match(r'^(%[A-Za-z0-9._]+)\s*(\{.*\})$', de, re.S)
      if m_named:
        inner = m_named.group(2)
        parsed.append(parseInitializer(inner, decode_gep_str_func))
        continue

      # If it's an adjacency like "[N x T] c'...'" or "{ T } value", read it
      adj = readAdjacentValueAfterType(de, decode_gep_str_func)
      if adj is not None:
        parsed.append(adj)
        continue

      # Otherwise strip a leading type token and parse recursively
      cleaned = stripLeadingTypes(de)
      parsed.append(parseInitializer(cleaned, decode_gep_str_func))

    #
    # Deterministic wrapping rules
    #

    # 1) If outer bracket is array-like ('<' or '['):
    #    - if outer had multiple top-level elements -> return parsed as-is
    #    - if outer had exactly one top-level element:
    #        * if parsed is a single byte-array (list[int]) produced from a c"..." adjacency,
    #          return that byte-array directly (user wanted plain byte arrays for c"...")
    #        * if parsed is a single aggregate (i.e. parsed == [[...]]), keep that aggregate
    #          (return parsed) so we represent "array containing one aggregate" as [[...]]
    #        * otherwise return [parsed] to represent an array with one element
    if open_ch in '[<':
      if len(top_elems) == 1:
        if len(parsed) == 1 and isinstance(parsed[0], list) and all(isinstance(x, int) for x in parsed[0]):
          # special-case: [ { i8 } c"..." ] -> c"..." -> list[int]
          return parsed[0]
        if len(parsed) == 1 and isinstance(parsed[0], list):
          # already a single aggregate inside (e.g. parsed == [[...]]), return as-is
          return parsed
        # otherwise return an array containing the parsed element(s)
        return [parsed]
      else:
        return parsed

    # 2) If outer bracket is an aggregate ('{' or '('):
    #    - aggregates should preserve one-level aggregate semantics.
    #    - If parsed has exactly one inner aggregate (parsed == [[...]]), return
    #      a wrapper list so the outer aggregate is preserved: [[...]]
    #    - Otherwise return parsed (list of field values)
    if open_ch in '{(':
      if len(parsed) == 1 and isinstance(parsed[0], list):
        return [parsed[0]]
      return parsed

    # fallback
    return parsed

  # Non-bracketed: maybe multiple top-level comma-separated parts
  parts = splitTopLevelCommas(s)
  if len(parts) > 1:
    return [parseScalarToken(stripLeadingTypes(p), decode_gep_str_func) for p in parts]

  return parseScalarToken(s, decode_gep_str_func)

def stripReturnAttrs(rest: str) -> str:
  s = rest.strip()
  while True:
    m = ATTR_RE.match(s)
    if not m:
      break
    s = s[m.end():].lstrip()
  return s

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

def valueFromParsed(parsed: Any, typ: Type) -> Value:
  """
  Build a Value (KnownIntVal / KnownFloatVal / KnownVecVal / KnownArrVal, etc)
  from a parsed initializer (the result of parse_initializer) and a Type instance.

  - parsed: int | float | list (nested)
  - typ: an instance of your Type classes (IntegerTy, FloatTy, VecTy, ArrayTy, ...)
  """

  if isinstance(typ, IntegerTy):
    if parsed == "zeroinitializer":
      return KnownIntVal(typ, 0, typ.width)

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
    if parsed == "zeroinitializer":
      values = [valueFromParsed("zeroinitializer", typ.inner) for _ in range(typ.size)]
      assert all([isinstance(value, KnownAggTargetVal) for value in values])
      return KnownArrVal(typ, [cast(KnownAggTargetVal, value) for value in values])

    if not isinstance(parsed, list):
      raise ValueError(f"ArrayTy expected list of elements, got {type(parsed)!r}: {parsed}")
    if len(parsed) != typ.size:
      raise ValueError(f"ArrayTy size mismatch: expected {typ.size}, got {len(parsed)}")
    arr_values: list[KnownAggTargetVal] = []

    for elem in parsed:
      val = valueFromParsed(elem, typ.inner)
      if not isinstance(val, KnownAggTargetVal):
        raise ValueError(f"Array element produced non-arr-target value: {val}")
      arr_values.append(val)

    return KnownArrVal(typ, arr_values)

  elif isinstance(typ, PointerTy):
    if parsed is None or parsed == "zeroinitializer":
      return NullPtrVal(typ)

    # If parsed is a string (e.g., "@globalname") we could return GlobalVarVal
    if isinstance(parsed, str) and parsed.startswith("@"):
      # name without @
      return GlobalVarVal(typ, parsed[1:])

    if isinstance(parsed, GetElementPtr):
      # GEP value
      return ConstExprVal(typ, parsed)

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

  elif isinstance(typ, StructTy):
    if parsed == "zeroinitializer":
      values = [valueFromParsed("zeroinitializer", mem) for mem in typ.members]
      assert all([isinstance(value, KnownAggTargetVal) for value in values])
      return KnownStructVal(typ, [cast(KnownAggTargetVal, value) for value in values])

    if not isinstance(parsed, list):
      raise ValueError(f"StructTy expected list of elements, got {type(parsed)!r}: {parsed}")
    if typ.is_packed:
      # the <{ ... }> brackets are incorrectly treated as two brackets
      assert len(parsed) == 1
      parsed = parsed[0]
    if len(parsed) != len(typ.members):
      raise ValueError(f"StructTy member mismatch: expected {len(typ.members)} got {len(parsed)}")

    struct_values: list[KnownAggTargetVal] = []

    for i, elem in enumerate(parsed):
      val = valueFromParsed(elem, typ.members[i])
      if not isinstance(val, KnownAggTargetVal):
        raise ValueError(f"Struct element produced non-agg-target value: {val}")
      struct_values.append(val)

    return KnownStructVal(typ, struct_values)

  # Fallback for unknown types
  raise NotImplementedError(f"value_from_parsed not implemented for type {type(typ).__name__}")

def valueFromInitializerText(init_text: str, typ: Type, decode_gep_str_func) -> Value:
  """
  Convenience wrapper: parse initializer text (via parse_initializer),
  then convert into a Value using value_from_parsed.
  """
  parsed = parseInitializer(init_text, decode_gep_str_func)
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
