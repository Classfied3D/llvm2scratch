from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from importlib.resources.abc import Traversable
from importlib.resources import files
import tomllib
import dacite

DEFAULT_OPT_TARGET = "scratch3"
ESCAPE_KEYWORDS = ["and", "or", "not"]

def getPackageData() -> Traversable:
  return files(__package__).joinpath("data")

def getPackageTargets() -> Traversable:
  return getPackageData().joinpath("targets")

def listTargets() -> list[str]:
  res: list[str] = []
  for target_file in getPackageTargets().iterdir():
    if target_file.is_file():
      res.append(target_file.name.rsplit(".", 1)[0])
  return sorted(res)

def dashToUnderscore(data: Any) -> Any:
  if isinstance(data, dict):
    new = {}
    for k, v in data.items():
      k = k.replace("-", "_")
      if k in ESCAPE_KEYWORDS:
        k += "_"
      new[k] = dashToUnderscore(v)
    return new
  if isinstance(data, list):
    return [dashToUnderscore(v) for v in data]
  return data

_target_cache: dict[str, Target] = {}
def getTarget(name: str) -> Target:
  global _target_cache
  if name in _target_cache: return _target_cache[name]

  raw: str = getPackageTargets().joinpath(f"{name}.toml").read_text()
  data: dict[str, Any] = dashToUnderscore(tomllib.loads(raw))
  data["id"] = name

  res = dacite.from_dict(Target, data)
  _target_cache[name] = res
  return res

@dataclass(frozen=True)
class Target:
  id: str
  info: TargetInfo
  exec: TargetExec
  perf: TargetPerf

  def __repr__(self):
    return f"Target({self.id})"

@dataclass(frozen=True)
class TargetInfo:
  name: str
  formats: list[str]

@dataclass(frozen=True)
class TargetExec:
  max_recursion_counter: int
  prefered_recursion_counter: int

@dataclass(frozen=True)
class TargetPerf:
  # Looks
  cost_num: float
  cost_name: float

  # Control
  counter: float

  # Sensing
  answer: float

  # Operators
  add: float
  sub: float
  mul: float
  div: float
  rand: float
  gt: float
  lt: float
  eq: float
  and_: float
  or_: float
  not_: float
  join: float
  letter_of: float
  length_of_str: float
  contains_str: float
  mod: float
  round: float
  abs: float
  floor: float
  ceil: float
  sqrt: float
  sin: float
  cos: float
  tan: float
  asin: float
  acos: float
  atan: float
  ln: float
  log: float
  exp: float
  pow10: float

  # Variables
  get_var: float
  set_var: float

  # Lists
  get_list: float
  at_index: float
  index_of: float
  length_of_list: float

  # Procedures
  param: float
