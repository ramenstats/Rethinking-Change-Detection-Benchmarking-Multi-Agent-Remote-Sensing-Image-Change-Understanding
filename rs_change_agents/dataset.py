from __future__ import annotations
from pathlib import Path
from .utils import read_jsonl

class ManifestDataset:
    def __init__(self, manifest: str):
        self.rows = read_jsonl(manifest)
        base = Path(manifest).resolve().parent
        for r in self.rows:
            for key in ["pre_path", "post_path", "mask_path"]:
                if r.get(key):
                    p = Path(r[key]).expanduser()
                    if not p.is_absolute():
                        p = (base / p).resolve()
                    r[key] = str(p)

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx): return self.rows[idx]
