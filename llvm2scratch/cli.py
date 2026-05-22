from pathlib import Path
import argparse

from . import compiler
from . import optimizer
from . import scratch
from . import target

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
    self.start_section("targets")
    for id in target.listTargets():
      t = target.getTarget(id)
      fmt_targets = ", ".join(t.info.formats)
      self.add_item(id, f"{t.info.name} ({t.info.url}): {t.info.desc} Supports formats: {fmt_targets}")
    self.end_section()

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
                       "glowing correctly due to a js error - minify futher and allow this error to occur"),
        ("gen-lut-runtime", "Generate AND/OR/XOR tables at runtime rather than adding pregenerated ones to "
                            "the file. This reduces file size significantly (by ~0.7MB) at the cost of "
                            "~0.4s spent generating the lookup tables on the first time running the "
                            "project (~0.01s on TurboWarp)"),
    ]:
      self.add_item(name, desc)
    self.end_section()
    return super().format_help()

def main():
  defaults = compiler.Config()
  targets = target.listTargets()
  default_targets = [t.id for t in defaults.targets]
  default_opt_target = defaults.opt_target.id

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
    "-f", "--format",
    choices=["infer", *(f.value for f in scratch.Format)],
    default="infer",
    help="File format of output file. By default this infered by the output file's extension."
  )
  parser.add_argument(
    "-T", "--targets",
    metavar="TARGET",
    choices=targets,
    nargs="+",
    default=default_targets,
    help=f"Compile code to support these targets. See list of targets below. Defaults to " + " ".join(default_targets)
  )
  parser.add_argument(
    "-U", "--opt-target",
    metavar="TARGET",
    choices=targets,
    default=None,
    help=f"Optimize code with this target in mind. Defaults to {default_opt_target} if "
         f"available otherwise the first target listed."
  )
  parser.add_argument(
    "-O",
    metavar="OPT_OPTIONS",
    choices=["all", "none", "compiler", *(o.name for o in optimizer.Optimization)],
    nargs="*",
    dest="optimizations",
    default=None,
    help="Optimizations to apply; defaults to all; see below"
  )
  parser.add_argument(
    "-M",
    metavar="MINIFY_OPTIONS",
    choices=["all", "none", "general", "break-glow", "gen-lut-runtime"],
    nargs="*",
    dest="minify",
    default=None,
    help="Minify settings to apply; defaults to general; see below"
  )
  parser.add_argument("--memory-size", type=int, default=defaults.memory_size,
    help=f"Number of 'bytes' on 'memory' list; max value is 200,000; default is {defaults.memory_size}")
  parser.add_argument("--local-stack-size", type=int, default=defaults.local_stack_size,
    help=f"Number of 'bytes' on local stack list for storing registers when recursing; max value is 200,000; default is "
         f"{defaults.local_stack_size}")
  parser.add_argument("--max-branch-recursion", type=int, default=None,
    help=f"Maximum depth of scratch's call stack before resetting it; default depends on targets enabled")
  parser.add_argument("--no-accurate-byte-spacing", action="store_true", default=False,
    help="Disable extra padding bytes added to each value in memory so that it takes up the space it would normally in bytes. "
         "This spacing allows byte indexing to be more accurate at the cost of requiring ~3x more space in the memory list. "
         "Disabling this may break programs that rely on an 8-bit byte size, like memcpy on an array of i32s or optimized IR.")
  parser.add_argument("--entrypoint", type=str, default=defaults.entrypoint,
    help=f"Specify a custom entrypoint function to run once the program starts. Defaults to {defaults.entrypoint}.")
  parser.add_argument(
    "--debug-scratch-text",
    type=Path,
    default=None,
    help="Output readable scratch code to a text file so it can be viewed"
  )
  parser.add_argument(
    "--debug-scratchblocks",
    type=Path,
    default=None,
    help="Output scratchblocks compatible code to a text file so it can be viewed. See https://scratchblocks.github.io/"
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

  targets = [target.getTarget(t) for t in args.targets]
  target_ids = [t.id for t in targets]

  opt_target_id: str | None = args.opt_target
  if args.opt_target is None:
    opt_target_id = target_ids[0]
    if defaults.opt_target.id in target_ids:
      opt_target_id = defaults.opt_target.id
  elif opt_target_id not in target_ids:
    raise ValueError(f"Optimization target (-U/--opt-target) {opt_target_id} should be in supported "
                     f"targets (-T/--targets) " + " ".join(target_ids))
  assert isinstance(opt_target_id, str)
  opt_target = target.getTarget(opt_target_id)

  extension = None
  format_inferred = args.format == "infer"
  if format_inferred:
    extension = str(args.output).rsplit(".")[-1]
    if extension == "sb3":       format = scratch.Format.Project3
    elif extension == "sprite3": format = scratch.Format.Sprite3
    else: raise ValueError(f"Could not infer output file format from extension \"{extension}\". "
                           "Either use a valid extension or set -f/--format")
  else:
    format = scratch.Format(args.format)

  for t in targets:
    if format.value not in t.info.formats:
      msg = f"Target (-T/--targets) {t.id} does not support format (-f/--format) {format.value}, only supports formats "
      msg += " ".join(t.info.formats)
      if format_inferred:
        assert extension is not None
        msg += f" (hint: format inferred from file extension .{extension}, disable by setting manually with -f)"
      raise ValueError(msg)

  max_branch_recursion = args.max_branch_recursion
  max_allowed_branch_recursion, max_abr_reason = min((t.exec.max_branch_recursion, t.info.name) for t in targets)
  if max_branch_recursion is not None:
    if max_branch_recursion > max_allowed_branch_recursion:
      raise ValueError(f"Max branch recursion (--max-branch-recursion) of {max_branch_recursion} exceeds what is allowed "
                       f"by {max_abr_reason} ({max_allowed_branch_recursion})")
  else:
    # Choose preferred, or as close as possible to it
    max_branch_recursion = min(opt_target.exec.preferred_branch_recursion, max_allowed_branch_recursion)
  assert isinstance(max_branch_recursion, int)

  opt_options = args.optimizations or ["all"]
  compiler_opt = "all" in opt_options or "compiler" in opt_options
  passes: set[optimizer.Optimization] = set()
  if "none" in opt_options:
    pass
  elif "all" in opt_options:
    passes = optimizer.ALL_OPTIMIZATIONS
  else:
    passes = {[*filter(lambda x: x.name == o, optimizer.ALL_OPTIMIZATIONS)][0] for o in opt_options if o != "compiler"}

  minify_opts = args.minify or ["general"]
  if "all" in minify_opts or "none" in minify_opts:
    minify = minify_break_glow = gen_lut_runtime = "all" in minify_opts
  else:
    minify = "general" in minify_opts
    minify_break_glow = "break-glow" in minify_opts
    gen_lut_runtime = "gen-lut-runtime" in minify_opts

  scfg = scratch.ScratchConfig(
    minify=minify,
    minify_break_glow=minify_break_glow,
    hide_blocks=args.hide_blocks,
    allow_hacked_blocks=not args.replace_hacked_blocks,
  )

  cfg = compiler.Config(
    compiler_opt=compiler_opt,
    opt_passes=passes,
    opt_target=opt_target,
    memory_size=args.memory_size,
    local_stack_size=args.local_stack_size,
    max_branch_recursion=max_branch_recursion,
    accurate_byte_spacing=not args.no_accurate_byte_spacing,
    entrypoint=args.entrypoint,
    gen_lut_runtime=gen_lut_runtime,
    scratch_config=scfg,
  )

  with open(args.input, "r") as file:
    proj, _ = compiler.compile(file.read(), cfg)

  if args.debug_scratch_text is not None:
    with open(args.debug_scratch_text, "w") as file:
      file.write(proj.stringify())

  if args.debug_scratchblocks is not None:
    with open(args.debug_scratchblocks, "w") as file:
      file.write(proj.stringify(scratchblocks=True))

  proj.export(args.output, format)

if __name__ == "__main__":
  main()
