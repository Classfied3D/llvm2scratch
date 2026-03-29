import llvm2scratch
import subprocess
import os

def main():
  script_dir = os.path.dirname(os.path.abspath(__file__))
  os.chdir(script_dir)

  # --target=i386-none-elf will remove standard lib, preferable when adding own
  # Optimisations are disabled as they might change the behavior of buffer overflows.
  subprocess.run(["clang", "-S", "-m32", "-fno-vectorize", "-fno-slp-vectorize", "-emit-llvm", "-I", "sb3api.h", "overflow.c", "-o", "overflow.ll"],
                 cwd=os.path.join(script_dir, "input"))

  with open("input/overflow.ll", "r") as file:
    proj, _ = llvm2scratch.compile(file.read(), llvm2scratch.Config(opti=False))
    proj.export("output/out.sprite3")

if __name__ == "__main__":
  main()
