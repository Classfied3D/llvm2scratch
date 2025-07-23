from __future__ import annotations
from dataclasses import dataclass, field

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
  lists: dict[str, tuple[Id, list[KnownValue]]] = field(default_factory=dict)
  blocks: dict[Id, dict] = field(default_factory=dict)

  def addBlock(self, id: Id, block: Block, meta: BlockMeta) -> None:
    metaless, self = block.getRaw(self)
    self.blocks[id] = meta.addRawMeta(metaless)

@dataclass
class Block:
  def getRaw(self, _: ScratchContext) -> tuple[dict, ScratchContext]:
    raise ScratchCompException("Cannot export for generic type 'Block'; must be a derived class")

@dataclass
class SetVariable(Block):
  var_name: str
  value: Value
  def getRaw(self, ctx: ScratchContext) -> tuple[dict, ScratchContext]:
    return ({
      "opcode": "data_setvariableto",
      "fields": {"VARIABLE": [self.var_name, ctx.vars[self.var_name][0]]}
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
      
class Value:
  pass

class KnownValue(Value):
  pass

class ScratchCompException(Exception):
  """Exception when compiling to scratch"""
  pass

def genId() -> Id:
  return uuid.uuid4().hex

def exportSpriteData() -> str:
  return json.dumps({
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
  })

def exportSpriteFile(file: zipfile.ZipFile) -> None:
  file.writestr("Sprite/sprite.json", exportSpriteData())
  file.writestr(f"Sprite/{EMPTY_SVG_HASH}.svg", EMPTY_SVG)

def main():
  with zipfile.ZipFile("lol.sprite3", "w") as zipf:
    exportSpriteFile(zipf)

if __name__ == "__main__":
  main()