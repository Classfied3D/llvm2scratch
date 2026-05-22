# why is re-exporting so verbose
from . import optimizer as optimizer
from . import compiler as compiler
from . import scratch as scratch

from .optimizer import Optimization as Optimization, ALL_OPTIMIZATIONS as ALL_OPTIMIZATIONS
from .compiler import Config as Config, DebugInfo as DebugInfo, compile as compile
from .scratch import Project as Project, ScratchConfig as ScratchConfig, Format as Format
from .parser import parseAssembly as parseAssembly
from .target import getTarget as getTarget, listTargets as listTargets, DEFAULT_TARGETS as DEFAULT_TARGETS, DEFAULT_OPT_TARGET as DEFAULT_OPT_TARGET
