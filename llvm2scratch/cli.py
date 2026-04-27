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
    default="out.sb3",
    help="Path to the output file (.sb3 or .sprite3)"
  )
  parser.add_argument(
    "--format",
    choices=["infer", *(f.value for f in scratch.Format)],
    default="infer",
    help="File format of output file. By default this infered by the output file's extension."
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
  parser.add_argument("--memory-size", type=int, default=4096,
    help="Number of 'bytes' on 'memory' list; max value is 200,000; default is 4096")
  parser.add_argument("--local-stack-size", type=int, default=512,
    help="Number of 'bytes' on local stack list for storing registers when recursing; max value is 200,000; default is 512")
  parser.add_argument("--max-branch-recursion", type=int, default=1_000_000,
    help="Maximum depth of scratch's call stack before resetting it; defaults to 1,000,000")
  parser.add_argument("--no-accurate-byte-spacing", action="store_true", default=False,
    help="Disable extra padding bytes added to each value in memory so that it takes up the space it would normally in bytes. "
         "This allows byte indexing to be more accurate at the cost of requiring ~3x more space in the memory list. "
         "Disabling this may break programs that rely on an 8-bit byte size, like memcpy on an array of i32s or optimized IR.")
  parser.add_argument(
    "--debug-scratch-code",
    type=Path,
    default=None,
    help="Output scratch code to a text file so it can be viewed"
  )
  parser.add_argument(
    "--replace-hacked-blocks",
    action="store_true",
    default=False,
    help="Remove 'hacked' blocks not normally accessible from the editor such as 'counter' and 'while' "
         "by replacing them with workarounds. See https://en.scratch-wiki.info/wiki/Hidden_Blocks. This "
         "may lead to a reduction in performance."
  )
  parser.add_argument(
    "--hide-blocks",
    action="store_true",
    default=False,
    help="Prevent blocks from rendering in the editor by setting shadow: true on top level blocks; "
         "stops editor lag. Not recommended due to increased project size and this seems to stop some "
         "projects from running. Instead export to a project instead of a sprite and don't click on the "
         "sprite."
  )

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
    allow_hacked_blocks=not args.replace_hacked_blocks,
  )

  cfg = compiler.Config(
    opti=compiler_opti,
    opti_passes=passes,
    memory_size=args.memory_size,
    local_stack_size=args.local_stack_size,
    max_branch_recursion=args.max_branch_recursion,
    accurate_byte_spacing=not args.no_accurate_byte_spacing,
    scratch_config=scfg,
  )

  with open(args.input, "r") as file:
    proj, _ = compiler.compile(file.read(), cfg)

  if args.debug_scratch_code is not None:
    with open(args.debug_scratch_code, "w") as file:
      file.write(proj.stringify())

  if args.format == "infer":
    extension = str(args.output).rsplit(".")[-1]
    if extension == "sb3":       format = scratch.Format.Project3
    elif extension == "sprite3": format = scratch.Format.Sprite3
    else: raise ValueError(f"Could not infer output file format from extension \"{extension}\". "
                           "Either use a valid extension or set --format")
  else:
    format = args.format

  proj.export(args.output, format)

if __name__ == "__main__":
  main()
