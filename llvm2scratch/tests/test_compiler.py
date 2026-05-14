import unittest
from llvm2scratch.compiler import *


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

if __name__ == "__main__":
  unittest.main() # type: ignore
