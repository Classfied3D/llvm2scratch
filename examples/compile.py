import llvm2scratch as l2s
from pathlib import Path
import subprocess
import argparse
import os

INPUT_FOLDER = "input"
OUTPUT_IR = "output/out.ll"
OUTPUT_PROJ = "output/out.sb3"
OUTPUT_SCRATCHBLOCKS = "output/blocks.txt"

def main():
  parser = argparse.ArgumentParser(
    description="Compile a C file into a scratch project or sprite",
  )
  parser.add_argument("input", type=Path, help="C file to compile", nargs="?", default=Path("demo.c"))
  parser.add_argument("--optlevel", "-O", type=str, help="C optimization level", default="1")
  args = parser.parse_args()

  llvm_prefix = os.environ.get("LLVM_PREFIX", "")
  llvm_suffix = os.environ.get("LLVM_SUFFIX", "")
  cc = os.environ.get("CC", f"{llvm_prefix}clang{llvm_suffix}")

  script_dir = os.path.dirname(os.path.abspath(__file__))
  os.chdir(script_dir)

  if not os.path.exists("./output"):
    os.mkdir("output")

  subprocess.run([cc, "-S", "-m32", f"-O{args.optlevel}", "-fno-vectorize", "-fno-slp-vectorize", "-emit-llvm", "-I", "sb3api.h", args.input, "-o", os.path.join(script_dir, OUTPUT_IR)],
                 cwd=os.path.join(script_dir, "input"))

  with open(OUTPUT_IR, "r") as file:
    proj = l2s.compile(file.read(), l2s.Config(compiler_opt=True, gen_lut_runtime=True))

  with open(OUTPUT_SCRATCHBLOCKS, "w") as file:
    file.write(proj.stringify(scratchblocks=True))
  proj.export(OUTPUT_PROJ, l2s.Format.Project3)

if __name__ == "__main__":
  main()
