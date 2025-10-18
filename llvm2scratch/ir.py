from __future__ import annotations
from llvmlite import binding as llvm
from dataclasses import dataclass
from enum import Enum

class UnaryOpcode(Enum):
  FNeg = "fneg"

class BinaryOpcode(Enum):
  Add = "add"
  FAdd = "fadd"
  Sub = "sub"
  FSub = "fsub"
  Mul = "mul"
  FMul = "fmul"
  UDiv = "udiv"
  SDiv = "sdiv"
  FDiv = "fdiv"
  URem = "urem"
  SRem = "srem"
  FRem = "frem"
  Shl = "shl"
  LShr = "lshr"
  AShr = "ashr"
  And = "and"
  Or = "or"
  Xor = "xor"

class ConvOpcode(Enum):
  Trunc = "trunc"
  ZExt = "zext"
  SExt = "sext"
  FPTrunc = "fptrunc"
  FPExt = "fpext"
  FPToUI = "fptoui"
  FPToSI = "fptosi"
  UIToFP = "uitofp"
  SIToFP = "sitofp"
  PtrToInt = "ptrtoint"
  PtrToAddr = "ptrtoaddr"
  IntToPtr = "inttoptr"
  BitCast = "bitcast"
  AddrSpaceCast = "addrspacecast"

class ICmpCond(Enum):
  Eq = "eq"
  Ne = "ne"
  Ugt = "ugt"
  Uge = "uge"
  Ult = "ult"
  Ule = "ule"
  Sgt = "sgt"
  Sge = "sge"
  Slt = "slt"
  Sle = "sle"

class FCmpCond(Enum):
  FalseCond = "false"
  Oeq = "oeq"
  Ogt = "ogt"
  Oge = "oge"
  Olt = "olt"
  Ole = "ole"
  One = "one"
  Ord = "ord"
  Ueq = "ueq"
  Ugt = "ugt"
  Uge = "uge"
  Ult = "ult"
  Ule = "ule"
  Une = "une"
  Uno = "uno"
  TrueCond = "true"

class CallTailKind(Enum):
  NoTail = "notail"
  Tail = "tail"
  MustTail = "musttail"

class Intrinsic(Enum):
  MemCpy = "llvm.memcpy"
  MemCpyInline = "llvm.memcpy.inline"
  MemMove = "llvm.memmove"
  LifetimeStart = "llvm.lifetime.start"
  LifetimeEnd = "llvm.lifetime.end"

@dataclass
class ResultLocalVar:
  name: str

class Type:
  pass

class AggTargetTy(Type):
  pass

class VecTargetTy(Type):
  pass

class VoidTy(Type):
  pass

@dataclass
class IntegerTy(VecTargetTy, AggTargetTy):
  width: int

class FloatingPointTy(VecTargetTy, AggTargetTy):
  pass

class HalfTy(FloatingPointTy):
  pass

class FloatTy(FloatingPointTy):
  pass

class DoubleTy(FloatingPointTy):
  pass

class Fp128Ty(FloatingPointTy):
  pass

class PointerTy(VecTargetTy):
  pass

@dataclass
class VecTy(AggTargetTy):
  inner: VecTargetTy
  size: int

class LabelTy(Type):
  pass

@dataclass
class ArrayTy(AggTargetTy):
  inner: AggTargetTy
  size: int

@dataclass
class StructTy(AggTargetTy):
  name: str
  members: list[AggTargetTy]

@dataclass
class Value:
  type: Type

class KnownVal(Value):
  pass

class KnownArrTargetVal(Value):
  pass

class KnownVecTargetVal(Value):
  pass

@dataclass
class ArgumentVal(Value):
  name: str

@dataclass
class FunctionVal(Value):
  name: str
  return_type: Type
  intrinsic: Intrinsic | None

@dataclass
class LocalVarVal(Value):
  name: str

@dataclass
class GlobalVarVal(Value):
  name: str

@dataclass
class UndefVal(KnownVal, KnownVecTargetVal, KnownArrTargetVal):
  pass

@dataclass
class KnownIntVal(KnownVal, KnownVecTargetVal, KnownArrTargetVal):
  value: int
  width: int

@dataclass
class KnownFloatVal(KnownVal, KnownVecTargetVal, KnownArrTargetVal):
  value: float

@dataclass
class KnownVecVal(KnownVal, KnownArrTargetVal):
  values: list[KnownVecTargetVal]

@dataclass
class KnownArrVal(KnownVal, KnownArrTargetVal):
  values: list[KnownArrTargetVal]

@dataclass
class LabelVal(Value):
  label: str

@dataclass
class ConstExprVal(KnownVal):
  expr: GetElementPtr

class Instr:
  pass

@dataclass
class HasResult:
  result: ResultLocalVar

@dataclass
class MaybeHasResult:
  result: ResultLocalVar | None

@dataclass
class Ret(Instr):
  value: Value | None

class Br(Instr):
  pass

@dataclass
class UncondBr(Br):
  branch: LabelVal

@dataclass
class CondBr(Br):
  cond: Value
  branch_true: LabelVal
  branch_false: LabelVal

@dataclass
class Switch(Instr):
  cond: Value
  branch_default: LabelVal
  branch_table: list[tuple[KnownIntVal, LabelVal]]

class Unreachable(Instr):
  pass

@dataclass
class UnaryOp(Instr, HasResult):
  opcode: UnaryOpcode
  operand: Value

@dataclass
class BinaryOp(Instr, HasResult):
  opcode: BinaryOpcode
  left: Value
  right: Value
  is_nuw: bool
  is_nsw: bool
  is_exact: bool
  is_disjoint: bool

@dataclass
class ExtractElement(Instr, HasResult):
  vector: Value
  index: Value

@dataclass
class InsertElement(Instr):
  vector: Value
  item: Value
  index: Value

@dataclass
class ShuffleVector(Instr, HasResult):
  fst_vector: Value
  snd_vector: Value
  mask_vector: Value

@dataclass
class Alloca(Instr, HasResult):
  allocated_type: Type
  num_elements: Value

@dataclass
class Load(Instr, HasResult):
  address: Value

@dataclass
class Store(Instr):
  value: Value
  address: Value

@dataclass
class GetElementPtr(Instr, HasResult):
  base_ptr_type: Type
  base_ptr: Value
  indices: list[Value]

@dataclass
class Conversion(Instr, HasResult):
  opcode: ConvOpcode
  value: Value
  res_type: Type
  is_nuw: bool
  is_nsw: bool

@dataclass
class ICmp(Instr, HasResult):
  cond: ICmpCond
  left: Value
  right: Value
  is_samesign: bool

@dataclass
class FCmp(Instr, HasResult):
  cond: FCmpCond
  left: Value
  right: Value

@dataclass
class Phi(Instr, HasResult):
  incoming: list[tuple[Value, LabelVal]]

@dataclass
class Select(Instr, HasResult):
  cond: Value
  true_value: Value
  false_value: Value

@dataclass
class Call(Instr, MaybeHasResult):
  func: FunctionVal
  args: list[Value]
  tail_kind: CallTailKind

@dataclass
class Block():
  label: str
  instrs: list[Instr]

@dataclass
class Function():
  name: str
  return_type: Type
  args: list[ArgumentVal]
  intrinsic: Intrinsic | None
  blocks: dict[str, Block]

@dataclass
class GlobalVar():
  name: str
  type: Type
  is_constant: bool
  init: KnownVal | GlobalVarVal | ConstExprVal | None

@dataclass
class Module():
  name: str
  functions: dict[str, Function]
  global_vars: dict[str, GlobalVar]
