from . import compiler

from pathlib import Path
import argparse

def main():
  parser = argparse.ArgumentParser(description="Process a file with optional optimization.")
  parser.add_argument("input", type=Path, help="Path to the input ir file")
  parser.add_argument(
    "-o", "--output",
    type=Path,
    default="out.sprite3",
    help="Path to the output sprite3 file"
  )
  parser.add_argument(
    "--opti",
    type=lambda x: x.lower() in ["true", "1", "yes", "y"],
    default=True,
    help="Enable optimizations (enabled by default)"
  )

  args = parser.parse_args()
  cfg = compiler.Config(opti=args.opti)

  with open(args.input, "r") as file:
    proj, _ = compiler.compile(file.read(), cfg)
    proj.export(args.output)

if __name__ == "__main__":
  main()
