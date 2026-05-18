import llvm2scratch
import subprocess
import os

def main():
  llvm_prefix = os.environ.get("LLVM_PREFIX", "")
  llvm_suffix = os.environ.get("LLVM_SUFFIX", "")
  cc = os.environ.get("CC", f"{llvm_prefix}clang{llvm_suffix}")

  script_dir = os.path.dirname(os.path.abspath(__file__))
  os.chdir(script_dir)

  if not os.path.exists("./output"):
    os.mkdir("output")

  subprocess.run([cc, "-S", "-m32", "-Oz", "-fno-vectorize", "-fno-slp-vectorize", "-emit-llvm", "-I", "sb3api.h", "demo.c", "-o", "demo.ll"],
                 cwd=os.path.join(script_dir, "input"))

  with open("input/demo.ll", "r") as file:
    proj, _ = llvm2scratch.compile(file.read(), llvm2scratch.Config(opti=True, gen_lut_runtime=True))
    proj.export("output/out.sb3", llvm2scratch.Format.Project3)

if __name__ == "__main__":
  main()
