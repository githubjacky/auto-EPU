import orjson
from pathlib import Path
from typing import List, Dict


def read_jsonl(path: str | Path,
               n: int = -1,
               return_str: bool = False) -> List[Dict] | List[str]:

    res = Path(path).read_text().split('\n')
    if n == len(res):
        n = -1

    if n == -1 and res[-1] != '' and not return_str:
        return [orjson.loads(i) for i in res]
    if n == -1 and res[-1] != '' and return_str:
        return res
    else:
         return (
            [orjson.loads(i) for i in res[:n]]
            if not return_str
            else
            res[:n]
        )
