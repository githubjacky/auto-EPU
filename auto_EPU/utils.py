import orjson
from pathlib import Path
from typing import List, Dict


def read_jsonl(path: str | Path,
               n: int = -1,
               return_str: bool = False) -> List[Dict] | List[str]:
    return (
        [
            orjson.loads(i)
            for i in Path(path).read_text().split("\n")[:n]
        ]
        if not return_str
        else
        [
            i
            for i in Path(path).read_text().split("\n")[:n]
        ]
    )
