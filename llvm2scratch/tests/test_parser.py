import unittest
from llvm2scratch.ir import *
from llvm2scratch.parser import *


class TokenParse(unittest.TestCase):
  def testQuoted(self):
    self.assertEqual(parseQuoted('"hello world" etc etc'), ("hello world", " etc etc", 13))

  def testQuotedBackslash(self):
    self.assertEqual(parseQuoted(r'"\\"lol'), ("\\", "lol", 4))

  def testQuotedByte(self):
    self.assertEqual(parseQuoted(r'"\22\00\\""another string"'), ('"\x00\\', '"another string"', 10))

  def testNormalToken(self):
    self.assertEqual(
      parseUntil("hi, a, lol", lambda t: t.endswith(",")),
      (True, ["hi,"], "", "hi,", " a, lol"))

  def testCloseBracketToken(self):
    self.assertEqual(
      parseUntil('[<>, "hello"] ("hi"), hi', lambda t: t == ","),
      (True, ['[<>, "hello"]', '("hi")', ","], '[<>, "hello"] ("hi")', ",", " hi"))

  def testEndQuoteToken(self):
    self.assertEqual(
      parseUntil('"hello world"hi lol', lambda t: t == "hi"),
      (True, ['"hello world"', 'hi'], '"hello world"', 'hi', ' lol'))

  def testNotFoundToken(self):
    self.assertEqual(
      parseUntil('[ { } ]hi ""hello', lambda _: False),
      (False, ["[ { } ]", "hi", '""', "hello"], '[ { } ]hi ""hello', "", ""))

  def testUnmatchingBracketToken(self):
    with self.assertRaises(ValueError) as _:
      _ = parseUntil("[ { }} ] hi hello", lambda _: False)

  def testInterruptedNormalToken(self):
    self.assertEqual(parseUntilEnd("hi(lol())"), ["hi", "(lol())"])

  def testQuoteInterruptedNormalToken(self):
    self.assertEqual(parseUntilEnd('hi, c"hello world \00"etc hi'), ["hi,", "c", '"hello world \00"', "etc", "hi"])


class TypeParse(unittest.TestCase):
  def testInt(self):
    self.assertEqual(parseTypeToken("i1872871", {}), IntegerTy(width=1872871))

  def testIntFail(self):
    with self.assertRaises(ValueError) as context:
      _ = parseTypeToken("i3.14", {})
    self.assertIn("unsupported type", str(context.exception).lower())

  def testFloat(self):
    self.assertEqual(parseTypeToken("fp128", {}), Fp128Ty())

  def testVec(self):
    self.assertEqual(parseTypeToken("<2 x ptr>", {}), VecTy(PointerTy(0), size=2))

  def testArr(self):
    self.assertEqual(
      parseTypeToken("[3 x [2 x <2 x i3>]]", {}),
      ArrayTy(ArrayTy(VecTy(IntegerTy(width=3), size=2), size=2), size=3))

  def testStruct(self):
    self.assertEqual(
      parseTypeTokens(["type", "{ i32, float, <2 x i3> }"], {}),
      (StructTy(is_packed=False, members=[IntegerTy(width=32), FloatTy(), VecTy(IntegerTy(width=3), size=2)]), []))

  def testLiteralStruct(self):
    self.assertEqual(
      parseTypeTokens(parseUntilEnd("[2 x {i32, i32}]"), {}),
      (ArrayTy(StructTy(is_packed=False, members=[IntegerTy(width=32), IntegerTy(width=32)]), size=2), []))

  def testNamedStruct(self):
    self.assertEqual(
      parseTypeConstantTokens(parseUntilEnd("%hehe { i16 1000 }"), {"hehe": StructTy(False, [IntegerTy(16)])}, []),
      (KnownStructVal(StructTy(False, [IntegerTy(16)]), [KnownIntVal(IntegerTy(16), 1000, 16)]), []))

  def testPtrAddrspace(self):
    self.assertEqual(
      parseTypeTokens(["ptr", "addrspace", "(12)"], {}), (PointerTy(addrspace=12), []))

  def testPackedStruct(self):
    self.assertEqual(
      parseTypeToken("<{ i1, [2 x type { i32, i1 } ] }>", {}, True),
      StructTy(is_packed=True, members=[
        IntegerTy(width=1),
        ArrayTy(StructTy(is_packed=False, members=[IntegerTy(width=32), IntegerTy(width=1)]), size=2)]))

  def testFuncPtr(self):
    self.assertEqual(
      parseTypeTokens(["ptr", "addrspace", '("G")', "(i32, i64)"], {}),
      (FuncTy(return_type=PointerTy("G"), params=[IntegerTy(width=32), IntegerTy(width=64)]), []))

  def testNestedFuncPtr(self):
    self.assertEqual(
      parseTypeTokens(["ptr", "(i32, i64)", "(i32, i64)"], {}),
        (FuncTy(return_type=FuncTy(return_type=PointerTy(0), params=[IntegerTy(width=32), IntegerTy(width=64)]), params=[IntegerTy(width=32), IntegerTy(width=64)]), []))


class ConstantParse(unittest.TestCase):
  def testParseInt(self):
    ty = IntegerTy(8)
    self.assertEqual(parseConstantToken(ty, "120", {}, []), KnownIntVal(ty, 120, 8))

  def testParseNegInt(self):
    ty = IntegerTy(8)
    self.assertEqual(parseConstantToken(ty, "-2", {}, []), KnownIntVal(ty, 254, 8)) # 254 is equivelent to -2

  def testParseBool(self):
    ty = IntegerTy(1)
    self.assertEqual(parseConstantToken(ty, "true", {}, []), KnownIntVal(ty, 1, 1))

  def testParseHex(self):
    ty = IntegerTy(16)
    self.assertEqual(parseConstantToken(ty, "u0xFF", {}, []), KnownIntVal(ty, 255, 16))

  def testParseSignedHex(self):
    ty = IntegerTy(8)
    self.assertEqual(parseConstantToken(ty, "s0x01", {}, []), KnownIntVal(ty, 255, 8)) # 255 is equivelent to -1

  def testTooLargeInt(self):
    with self.assertRaises(AssertionError) as _:
      _ = parseConstantToken(IntegerTy(8), "u0x100", {}, [])

  def testFloat(self):
    parsed = parseConstantToken(DoubleTy(), "-3.1415", {}, [])
    assert isinstance(parsed, KnownFloatVal)
    self.assertAlmostEqual(parsed.value, -3.1415)

  def testFloatExpPos(self):
    parsed = parseConstantToken(DoubleTy(), "-3.1415e+5", {}, [])
    assert isinstance(parsed, KnownFloatVal)
    self.assertAlmostEqual(parsed.value, -3.1415 * 10**5)

  def testFloatExpNeg(self):
    parsed = parseConstantToken(DoubleTy(), "-3.1415e-5", {}, [])
    assert isinstance(parsed, KnownFloatVal)
    self.assertAlmostEqual(parsed.value, -3.1415 * 10**-5)

  def testFloatHex(self):
    parsed = parseConstantToken(DoubleTy(), "0x432ff973cafa8000", {}, [])
    assert isinstance(parsed, KnownFloatVal)
    self.assertAlmostEqual(parsed.value, 4.5 * 10**15)

  def testParseFuncPtr(self):
    self.assertEqual(
      parseConstantToken(PointerTy(0), "@cool_function", {}, []),
      GlobalPtrVal(PointerTy(0), "cool_function"))

  def testParseNullPtr(self):
    self.assertEqual(
      parseConstantToken(PointerTy(0), "zeroinitializer", {}, []),
      NullPtrVal(PointerTy(0)))

  def testParseStruct(self):
    self.assertEqual(
      parseTypeConstantTokens(parseUntilEnd("type <{ i1, ptr addrspace(7) }> <{ i1 true, ptr addrspace(7) null }>"), {}, []),
      (KnownStructVal(
        StructTy(is_packed=True, members=[IntegerTy(width=1), PointerTy(addrspace=7)]),
        values=[
          KnownIntVal(type=IntegerTy(width=1), value=1, width=1),
          NullPtrVal(type=PointerTy(addrspace=7))]), []))

  def testParseZeroInitStruct(self):
    self.assertEqual(
      parseTypeConstantTokens(parseUntilEnd("type <{ i1, [ 2 x float ] }> zeroinitializer"), {}, []),
      (KnownStructVal(
        StructTy(True, [IntegerTy(1), ArrayTy(FloatTy(), 2)]),
        values=[
          KnownIntVal(IntegerTy(width=1), value=0, width=1),
          KnownArrVal(ArrayTy(inner=FloatTy(), size=2), values=[
            KnownFloatVal(type=FloatTy(), value=0.0),
            KnownFloatVal(type=FloatTy(), value=0.0)])]), []))

  def testParseArray(self):
    self.assertEqual(
      parseTypeConstantTokens(parseUntilEnd("[2 x float] [ float 3.14, float 3.141592 ]"), {}, []),
      (KnownArrVal(type=ArrayTy(inner=FloatTy(), size=2), values=[KnownFloatVal(type=FloatTy(), value=3.14), KnownFloatVal(type=FloatTy(), value=3.141592)]), []))

  def testParseCharArray(self):
    self.assertEqual(
      parseTypeConstantTokens(parseUntilEnd('[3 x i8] c"hi\00"'), {}, []),
      (KnownArrVal(type=ArrayTy(inner=IntegerTy(width=8), size=3), values=[KnownIntVal(type=IntegerTy(width=8), value=104, width=8), KnownIntVal(type=IntegerTy(width=8), value=105, width=8), KnownIntVal(type=IntegerTy(width=8), value=0, width=8)]), []))

  def testParseSplatVec(self):
    self.assertEqual(
      parseTypeConstantTokens(parseUntilEnd("<2 x i32> splat (i32 3)"), {}, []),
      (KnownVecVal(
        type=VecTy(inner=IntegerTy(width=32), size=2),
        values=[
          KnownIntVal(type=IntegerTy(width=32), value=3, width=32),
          KnownIntVal(type=IntegerTy(width=32), value=3, width=32)]), []))

  def testParseGEP(self):
    self.assertEqual(
      parseTypeConstantTokens(parseUntilEnd("ptr getelementptr (i8, ptr @hi, i64 8)"), {}, []),
      (ConstExprVal(
        type=PointerTy(addrspace=0),
        expr=GetElementPtr(
          result=ResultLocalVar(name=""),
          base_ptr_type=IntegerTy(width=8),
          base_ptr=GlobalPtrVal(type=PointerTy(addrspace=0), name="hi"),
          indices=[KnownIntVal(type=IntegerTy(width=64), value=8, width=64)],
          is_inbounds=False, is_nusw=False, is_nuw=False)), []))

  def testParseConv(self):
    self.assertEqual(
      parseTypeConstantTokens(parseUntilEnd("i8 trunc (i16 32 to i8)"), {}, []),
      (ConstExprVal(
        type=IntegerTy(width=8),
        expr=Conversion(result=ResultLocalVar(name=""),
          opcode=ConvOpcode.Trunc,
          value=KnownIntVal(type=IntegerTy(width=16), value=32, width=16),
          res_type=IntegerTy(width=8), is_nuw=False, is_nsw=False)), []))

  def parseAddConv(self):
    self.assertEqual(
      parseTypeConstantTokens(parseUntilEnd("i8 add (i8 1, i8 2)"), {}, []),
      (ConstExprVal(
        type=IntegerTy(width=8),
        expr=BinaryOp(
          result=ResultLocalVar(name=""),
          opcode=BinaryOpcode.Add,
          left=KnownIntVal(type=IntegerTy(width=8), value=1, width=8),
          right=KnownIntVal(type=IntegerTy(width=8), value=2, width=8),
          is_nuw=False, is_nsw=False, is_exact=False, is_disjoint=False)), []))


if __name__ == "__main__":
  _ = unittest.main()
