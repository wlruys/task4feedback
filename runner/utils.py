import importlib
import os
import yaml
import itertools
from typing import List, Any, Dict, Tuple, Mapping

def import_entry_point(entry_point: str):
    """Imports a function from a string 'module:function'."""
    if ":" not in entry_point:
        raise ValueError(f"Entry point must be in format 'module:function', got '{entry_point}'")
    
    parts = entry_point.split(":")
    if len(parts) != 2:
        raise ValueError(f"Entry point must be in format 'module:function', got '{entry_point}'")
        
    module_name, func_name = parts
        
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        return func
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}': {e}")
    except AttributeError as e:
        raise AttributeError(f"Could not find function '{func_name}' in module '{module_name}': {e}")

def load_sweep_file(path: str) -> List[str]:
    """
    Loads a sweep YAML file and returns a list of Hydra override strings.
    Example YAML:
        +size: [b4, b8]
        algorithm.lr: 3e-4
    Returns:
        ['+size=b4,b8', 'algorithm.lr=3e-4']
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sweep file not found: {path}")

    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    
    overrides = []
    for k, v in data.items():
        if isinstance(v, list):
            # Join with commas for parse_overrides to handle
            # We convert each element to string. 
            # If element is a list [1,2], it becomes "[1, 2]"
            v_str = ",".join(str(x) for x in v)
            overrides.append(f"{k}={v_str}")
        else:
            overrides.append(f"{k}={v}")
            
    return overrides

def split_preserving_brackets(s: str, delimiter: str = ",") -> List[str]:
    """
    Splits a string by delimiter, but respects brackets [] and quotes "" ''.
    Used to correctly parse comma-separated lists that may contain lists.
    Example: "[1,2],[3,4]" -> ["[1,2]", "[3,4]"]
    """
    parts = []
    current = []
    depth = 0
    quote = None
    
    for char in s:
        if quote:
            if char == quote:
                quote = None
            current.append(char)
        elif char in "\"'":
            quote = char
            current.append(char)
        elif char == "[":
            depth += 1
            current.append(char)
        elif char == "]":
            depth -= 1
            current.append(char)
        elif char == delimiter and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
            
    if current:
        parts.append("".join(current))
        
    return [p.strip() for p in parts if p.strip()]

def parse_overrides(overrides: List[str]) -> List[List[str]]:
    """
    Parses a list of Hydra overrides, expanding comma-separated values into multiple configurations.
    Example: ['+a=1,2', 'b=3'] -> [['+a=1', 'b=3'], ['+a=2', 'b=3']]
    """
    # Group by key
    parsed = {}
    for override in overrides:
        if "=" not in override:
            continue 
        key, value = override.split("=", 1)
        
        # Use robust splitter instead of simple split
        # This handles cases like key=[1,2],[3,4] correctly
        values = split_preserving_brackets(value)
        parsed[key] = values
             
    # Cartesian product
    keys = list(parsed.keys())
    values_list = [parsed[k] for k in keys]
    
    configs = []
    for combination in itertools.product(*values_list):
        cfg = []
        for k, v in zip(keys, combination):
            cfg.append(f"{k}={v}")
        configs.append(cfg)
        
    return configs

def flatten_dict(d: Mapping[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dicts into Hydra dotted keys."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def load_packs(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads a packs definition file.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def expand_pack_overrides(overrides: List[str], packs: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Replaces pack references (e.g., +group=pack) with their actual overrides from the packs definition.
    """
    expanded = []
    for override in overrides:
        if "=" not in override:
            expanded.append(override)
            continue
            
        key, value = override.split("=", 1)
        
        # Check if this is a pack reference
        clean_key = key.lstrip("+")
        
        if clean_key in packs and value in packs[clean_key]:
            # Found a pack!
            pack_content = packs[clean_key][value]
            # Flatten it to dot notation
            flat_pack = flatten_dict(pack_content)
            for k, v in flat_pack.items():
                expanded.append(f"{k}={v}")
        else:
            expanded.append(override)
            
    return expanded
