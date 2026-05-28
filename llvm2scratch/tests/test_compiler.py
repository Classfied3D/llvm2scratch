import unittest
from llvm2scratch.compiler import *
from llvm2scratch import target


WIDTH: int = 32
BINOP_TEST_MASKS: list[int] = [
  0b11111111111111111111111111111111,
  0b00000000000000000000000000000000,
  0b11111111111111111011010100100100,
  0b11111111100000000001000000000001,
  0b01001010101001010101001010101001,
  0b00000000001111111111110000000000,
  0b00000000000000001000000000000000,
  0b01000000000000001000000000000000,
]

class TestBinOp(unittest.TestCase):
  def testExtract(self):
    for in_place in [True, False]:
      for mask in BINOP_TEST_MASKS:
        for start, bits in [(0, 32), (0, 2), (1, 2), (31, 1), (16, 16), (1, 31)]:
          end = start + bits
          shift = WIDTH - end
          expected = (mask >> shift) & ((1 << bits) - 1)
          if in_place: expected <<= shift
          got, _ = extractBits(sb3.Known(mask), WIDTH, start, bits, in_place)
          got = opt.simplifyValue(got)
          self.assertIsInstance(got, sb3.Known)
          self.assertIsInstance(got.known, float)
          self.assertEqual(float(expected), got.known, f"Mask: {mask:032b}, Start: {start}, Length: {bits}")

  def testBinOp(self):
    for opt_target in target.DEFAULT_TARGETS:
      ctx = Context(sb3.Project(sb3.ScratchConfig()), Config(opt_target=target.getTarget(opt_target)))
      unknown_masks = [
        0b11111111111111111111111111111111,
        0b00000000000000000000000000000000,
        0b11010110111100000000010110111011,
      ]
      lookup_func = lambda n, i: tableLookup(n, i, ctx)
      for op in ["and", "or", "xor"]:
        for known in BINOP_TEST_MASKS:
          for unknown in unknown_masks:
            match op:
              case "and": expected = known & unknown
              case "or":  expected = known | unknown
              case "xor": expected = known ^ unknown
              case _:     assert False

            # binOp can optimize values internally - ensure this doesn't happen by using a variable on one side
            got, _ = binOp(op, sb3.Known(known), sb3.GetVar("unknown"), WIDTH, ctx)
            got, _ = opt.assignmentElisionValue(got, {"var:unknown": sb3.Known(unknown)})
            got = opt.simplifyValue(got, lookup_func=lookup_func)
            self.assertIsInstance(got, sb3.Known)
            self.assertIsInstance(got.known, float)
            self.assertEqual(float(expected), got.known, f"Op: {op}, Known: {known:032b}, Unknown: {unknown:032b}")

            # Also test when both sides are unknown
            got, _ = binOp(op, sb3.GetVar("lft"), sb3.GetVar("rgt"), WIDTH, ctx)
            got, _ = opt.assignmentElisionValue(got, {
              "var:lft": sb3.Known(unknown),
              "var:rgt": sb3.Known(known),
            })
            got = opt.simplifyValue(got, lookup_func=lookup_func)
            self.assertIsInstance(got, sb3.Known)
            self.assertIsInstance(got.known, float)
            self.assertEqual(float(expected), got.known, f"Op: {op}, Lft: {unknown:032b}, Known: {known:032b}")

if __name__ == "__main__":
  _ = unittest.main()
