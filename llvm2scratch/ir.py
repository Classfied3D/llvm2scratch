from __future__ import annotations
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
  VaStart = "llvm.va_start"
  VaEnd = "llvm.va_end"
  VaCopy = "llvm.va_copy"
  Abs = "llvm.abs"
  SMax = "llvm.smax"
  SMin = "llvm.smin"
  UMax = "llvm.umax"
  UMin = "llvm.umin"
  MemCpy = "llvm.memcpy"
  MemCpyInline = "llvm.memcpy.inline"
  MemMove = "llvm.memmove"
  FAbs = "llvm.fabs"
  FShl = "llvm.fshl"
  FShr = "llvm.fshr"
  UAddWithOverflow = "llvm.uadd.with.overflow"
  USubWithOverflow = "llvm.usub.with.overflow"
  UMulWithOverflow = "llvm.umul.with.overflow"
  FMulAdd = "llvm.fmuladd"
  LifetimeStart = "llvm.lifetime.start"
  LifetimeEnd = "llvm.lifetime.end"
  NoAliasScopeDecl = "llvm.experimental.noalias.scope.decl"
  Expect = "llvm.expect"
  ExpectWithProbability = "llvm.expect.with.probability"
  Assume = "llvm.assume"
  PtrMask = "llvm.ptrmask"

@dataclass
class ResultLocalVar:
  name: str

@dataclass
class Type:
  pass

@dataclass
class AggTargetTy(Type):
  pass

@dataclass
class VecTargetTy(Type):
  pass

@dataclass
class VoidTy(Type):
  pass

@dataclass
class FuncTy(Type):
  return_type: Type
  params: list[Type]
  variadic: bool

@dataclass
class IntegerTy(VecTargetTy, AggTargetTy):
  width: int

@dataclass
class FloatingPointTy(VecTargetTy, AggTargetTy):
  pass

@dataclass
class HalfTy(FloatingPointTy):
  pass

@dataclass
class FloatTy(FloatingPointTy):
  pass

@dataclass
class DoubleTy(FloatingPointTy):
  pass

@dataclass
class Fp128Ty(FloatingPointTy):
  pass

@dataclass
class PointerTy(VecTargetTy, AggTargetTy):
  addrspace: str | int

@dataclass
class MetadataTy(Type):
  pass

@dataclass
class VecTy(AggTargetTy):
  inner: VecTargetTy
  size: int

@dataclass
class LabelTy(Type):
  pass

@dataclass
class ArrayTy(AggTargetTy):
  inner: AggTargetTy
  size: int

@dataclass
class StructTy(AggTargetTy):
  is_packed: bool
  members: list[AggTargetTy]

@dataclass
class Value:
  type: Type

@dataclass
class KnownVal(Value):
  pass

@dataclass
class KnownAggTargetVal(Value):
  pass

@dataclass
class KnownVecTargetVal(Value):
  pass

@dataclass
class ArgumentVal(Value):
  name: str

@dataclass
class FunctionVal(KnownVal, KnownAggTargetVal):
  name: str

@dataclass
class LocalVarVal(Value):
  name: str

@dataclass
class GlobalPtrVal(KnownVal, KnownAggTargetVal):
  name: str

@dataclass
class NullPtrVal(KnownVal, KnownAggTargetVal):
  pass

@dataclass
class UndefVal(KnownVal, KnownVecTargetVal, KnownAggTargetVal):
  pass

@dataclass
class KnownIntVal(KnownVal, KnownVecTargetVal, KnownAggTargetVal):
  value: int
  width: int

@dataclass
class KnownFloatVal(KnownVal, KnownVecTargetVal, KnownAggTargetVal):
  value: float

@dataclass
class KnownVecVal(KnownVal, KnownAggTargetVal):
  values: list[KnownVecTargetVal]

@dataclass
class KnownArrVal(KnownVal, KnownAggTargetVal):
  values: list[KnownAggTargetVal]

@dataclass
class KnownStructVal(KnownVal, KnownAggTargetVal):
  values: list[KnownAggTargetVal]

@dataclass
class LabelVal(KnownVal):
  label: str

@dataclass
class ConstExprVal(KnownVal, KnownAggTargetVal):
  expr: Conversion | GetElementPtr | ExtractElement | InsertElement | ShuffleVector | BinaryOp

@dataclass
class MetadataVal(KnownVal):
  pass

@dataclass
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

@dataclass
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

@dataclass
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
  agg: Value
  index: Value

@dataclass
class InsertElement(Instr):
  agg: Value
  item: Value
  index: Value

@dataclass
class ShuffleVector(Instr, HasResult):
  fst_vector: Value
  snd_vector: Value
  mask_vector: Value

@dataclass
class ExtractValue(Instr, HasResult):
  agg: Value
  indices: list[Value]

@dataclass
class InsertValue(Instr):
  agg: Value
  element: Value
  indices: list[Value]

@dataclass
class Alloca(Instr, HasResult):
  allocated_type: Type
  num_elements: Value

@dataclass
class Load(Instr, HasResult):
  loaded_type: Type
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
  is_inbounds: bool
  is_nusw: bool
  is_nuw: bool

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
class Freeze(Instr, HasResult):
  value: Value

@dataclass
class Call(Instr, MaybeHasResult):
  func: Value
  return_type: Type
  args: list[Value]
  params: list[Type]
  variadic: bool
  tail_kind: CallTailKind
  intrinsic: Intrinsic | None

@dataclass
class VaArg(Instr, HasResult):
  arglist: Value
  argty: Type

@dataclass
class Block():
  label: str
  instrs: list[Instr]

@dataclass
class Function():
  name: str
  return_type: Type
  params: list[ArgumentVal]
  variadic: bool
  intrinsic: Intrinsic | None
  blocks: dict[str, Block]

@dataclass
class GlobalVar():
  name: str
  type: Type
  is_constant: bool
  init: KnownVal

@dataclass
class Module():
  name: str
  functions: dict[str, Function]
  global_vars: dict[str, GlobalVar]
