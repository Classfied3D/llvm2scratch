from . import compiler
from . import optimizer
from . import scratch

from pathlib import Path
import argparse

class CustomFormatter(argparse.HelpFormatter):
  def add_item(self, name: str, desc: str):
    self._add_item(self._format_action, [
      argparse.Action(
        option_strings=[],
        dest=name,
        help=desc,
      )
    ])

  def format_help(self):
    self.start_section("optimization options")
    for name, desc in [
        ("all, none", "Self-explanatory"),
        ("compiler", "Enable compiler-level optimizations (e.g. addressing globals with address instead of by variable)"),
        *((o.name, o.description) for o in optimizer.Optimization),
    ]:
      self.add_item(name, desc)
    self.end_section()
    self.start_section("minify options")
    for name, desc in [
        ("all, none", "Self-explanatory"),
        ("general", "Optimize project.json's size by simplifing uids, removing falsy fields, etc"),
        ("break-glow", "Removing the parent key when minifing prevents blocks in the same sprite from "
                       "glowing correctly due to a js error - minify futher and allow this error to occur")
    ]:
      self.add_item(name, desc)
    self.end_section()
    return super().format_help()

def main():
  parser = argparse.ArgumentParser(
    description="Compile an LLVM 19 IR (.ll) file into a scratch sprite (.sprite3)",
    formatter_class=CustomFormatter
  )
  parser.add_argument("input", type=Path, help="Path to the input LLVM 19 IR (.ll) file")
  parser.add_argument(
    "-o", "--output",
    type=Path,
    default="out.sprite3",
    help="Path to the output sprite3 file"
  )
  parser.add_argument(
    "-O",
    choices=["all", "none", "compiler", *(o.name for o in optimizer.Optimization)],
    action="append",
    dest="optimizations",
    default=None,
    help="Optimizations to apply; defaults to all; see below"
  )
  parser.add_argument(
    "-M",
    choices=["all", "none", "general", "break-glow"],
    action="append",
    dest="minify",
    default=None,
    help="Minify settings to apply; defaults to general; see below"
  )
  parser.add_argument(
    "--hide-blocks",
    action="store_true",
    default=False,
    help="Prevent blocks from rendering in the editor by setting shadow: true on top level blocks; stops editor lag"
  )
  parser.add_argument("--memory-size", type=int, default=1024,
    help="Number of 'bytes' on 'memory' list; max value is 200,000; default is 1024")
  parser.add_argument("--local-stack-size", type=int, default=512,
    help="Number of 'bytes' on local stack list for storing registers when recursing; max value is 200,000; default is 512")
  parser.add_argument("--max-branch-recursion", type=int, default=1_000_000,
    help="Maximum depth of scratch's call stack before resetting it; defaults to 1,000,000")

  args = parser.parse_args()

  opti_opts = args.optimizations or ["all"]
  compiler_opti = "all" in opti_opts or "compiler" in opti_opts
  passes: set[optimizer.Optimization] = set()
  if "none" in opti_opts:
    pass
  elif "all" in opti_opts:
    passes = optimizer.ALL_OPTIMIZATIONS
  else:
    passes = {[*filter(lambda x: x.name == o, optimizer.ALL_OPTIMIZATIONS)][0] for o in opti_opts if o != "compiler"}

  minify_opts = args.minify or ["general"]
  minify = minify_break_glow = False
  if "none" in minify_opts:
    pass
  elif "all" in minify_opts:
    minify = minify_break_glow = True
  else:
    minify = "general" in minify_opts
    minify_break_glow = "break-glow" in minify_opts

  scfg = scratch.ScratchConfig(
    minify=minify,
    minify_break_glow=minify_break_glow,
    hide_blocks=args.hide_blocks,
  )

  cfg = compiler.Config(
    opti=compiler_opti,
    opti_passes=passes,
    memory_size=args.memory_size,
    local_stack_size=args.local_stack_size,
    max_branch_recursion=args.local_stack_size,
    scratch_config=scfg,
  )

  with open(args.input, "r") as file:
    proj, _ = compiler.compile(file.read(), cfg)
    proj.export(args.output)

if __name__ == "__main__":
  main()
