from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

import zipfile
import hashlib
import json
import uuid

EMPTY_SVG = """<svg version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="0" height="0" viewBox="0,0,0,0"></svg>"""
EMPTY_SVG_HASH = hashlib.md5(EMPTY_SVG.encode("utf-8")).hexdigest()

Id = str

@dataclass
class ScratchContext:
  vars: dict[str, tuple[Id, KnownValue]] = field(default_factory=dict)
  # TODO lists: dict[str, tuple[Id, list[KnownValue]]] = field(default_factory=dict)
  blocks: dict[Id, dict] = field(default_factory=dict)

  def addBlock(self, id: Id, block: Block, meta: BlockMeta) -> None:
    metaless, self = block.getRaw(self)
    self.blocks[id] = meta.addRawMeta(metaless)
  
  def addOrGetVar(self, var_name: str) -> Id:
    if not var_name in self.vars:
      id = genId()
      self.vars.update({var_name: (id, KnownValue("0"))})
    else:
      id = self.vars[var_name][0]
    return id
  
  def getRaw(self) -> dict:
    """Returns json for blocks and vars defined"""
    raw_vars = {}
    for name, (id, value) in self.vars.items():
      raw_vars.update({
        id: [name, int(value.known)] # TODO: serialise value correctly
      })
    
    raw_blocks = {}
    for id, block in self.blocks.items():
      raw_blocks.update({id: block})

    return {
      "variables": raw_vars,
      "blocks": raw_blocks,
    }

@dataclass
class Block:
  def getRaw(self, _: ScratchContext) -> tuple[dict, ScratchContext]:
    raise ScratchCompException("Cannot export for generic type 'Block'; must be a derived class")

@dataclass
class ModifyVariable(Block):
  op: Literal["set", "change"]
  var_name: str
  value: Value

  def getRaw(self, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    id = ctx.addOrGetVar(self.var_name)
    raw_val, ctx = self.value.getRawValue(ctx)
    return ({
      "opcode": "data_setvariableto" if self.op == "set" else "data_changevariableby",
      "inputs": {"VALUE": raw_val},
      "fields": {"VARIABLE": [self.var_name, id]}
    }, ctx)

@dataclass
class BlockMeta:
  parent: Id | None = None
  next: Id | None = None
  x: int = 0
  y: int = 0

  def addRawMeta(self, metaless: dict) -> dict:
    metaless.update({
      "parent": self.parent, "next": self.next,
      "topLevel": self.parent is None, "shadow": False,
      "x": self.x, "y": self.y,
    })
    return metaless

@dataclass
class BlockList:
  blocks: list[Block]

  def export(self, ctx: ScratchContext) -> ScratchContext:
    lastId = None
    currId = genId()
    nextId = genId()

    for (i, block) in enumerate(self.blocks):
      if i == len(self.blocks) - 1: nextId = None

      assert currId is not None

      meta = BlockMeta(lastId, nextId)
      ctx.addBlock(currId, block, meta)

      lastId = currId
      currId = nextId
      nextId = genId()
    
    return ctx

@dataclass
class Value:
  def getRawValue(self, _: ScratchContext) -> tuple[list, ScratchContext]:
    """Gets the json that can be put in the "inputs" field of a block"""
    raise ScratchCompException("Cannot export for generic type 'Value'; must be a derived class")

@dataclass
class KnownValue(Value):
  """Something that can be typed in a block input e.g. x in Say(x)"""
  known: str

  def getRawValue(self, ctx: ScratchContext) -> tuple[list, ScratchContext]:
    return ([1, [10, self.known]], ctx)

class ScratchCompException(Exception):
  """Exception when compiling to scratch"""
  pass

def genId() -> Id:
  return uuid.uuid4().hex

def exportSpriteData(ctx: ScratchContext) -> str:
  res = {
    "isStage": False,
    "name": "",
    "variables": {},
    "lists": {},
    "broadcasts": {},
    "blocks": {},
    "currentCostume": 0,
    "costumes":[
      # A costume must be defined for the sprite to load
      {
        "name": "",
        "bitmapResolution": 1,
        "dataFormat": "svg",
        "assetId": EMPTY_SVG_HASH,
        "md5ext": f"{EMPTY_SVG_HASH}.svg",
        "rotationCenterX": 0,
        "rotationCenterY": 0
      }
    ],
    "sounds": [],
    "volume": 100,
    "visible": False,
    "x": 0,
    "y": 0,
    "size": 100,
    "direction": 90,
    "draggable": False,
    "rotationStyle": "all around"
  }

  res.update(ctx.getRaw())

  print(res)

  return json.dumps(res)

def exportSpriteFile(ctx: ScratchContext, file: zipfile.ZipFile) -> None:
  file.writestr("Sprite/sprite.json", exportSpriteData(ctx))
  file.writestr(f"Sprite/{EMPTY_SVG_HASH}.svg", EMPTY_SVG)

def main():
  ctx = ScratchContext()
  BlockList([
    ModifyVariable("set", "hello", KnownValue("10")),
    ModifyVariable("set", "hello2", KnownValue("10"))
  ]).export(ctx)
  with zipfile.ZipFile("out.sprite3", "w") as zipf:
    exportSpriteFile(ctx, zipf)

if __name__ == "__main__":
  main()