import json
import yaml
from pathlib import Path
from typing import List, Dict, Any

REQUIRED = {"id", "input", "model_output_v1", "model_output_v2"}

def load_cases(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(p.read_text())
    else:
        data = json.loads(p.read_text())
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of cases")
    for case in data:
        missing = REQUIRED - set(case)
        if missing:
            raise ValueError(f"Case {case.get('id', '?')} missing fields: {missing}")
    return data