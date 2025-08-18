import llvm2scratch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# --target=i386-none-elf will remove standard lib, preferable when adding own
os.system("clang -S -m32 -emit-llvm -O0 input/main.c")

with open("main.ll", "r") as file:
  proj, _ = llvm2scratch.compile(file.read())
  proj.export("output/out.sprite3")