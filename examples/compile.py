import llvm2scratch
import subprocess
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# --target=i386-none-elf will remove standard lib, preferable when adding own
subprocess.run(["clang", "-S", "-m32", "-O0", "-emit-llvm", "main.c"],
               cwd=os.path.join(script_dir, "input"))

with open("input/main.ll", "r") as file:
  proj, _ = llvm2scratch.compile(file.read())
  proj.export("output/out.sprite3")