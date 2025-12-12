import torch
from tensordict.tensordict import TensorDict
import hashlib
from typing import Optional, Any

def _make_node_tensor(nodes, dim, single=False):
    if single:
        glb_shape = 1
        attr_shape = dim
    else:
        glb_shape = (nodes,)
        attr_shape = (nodes, dim)

    return TensorDict(
        {
            "glb": torch.zeros(glb_shape, dtype=torch.int64),
            "attr": torch.zeros(attr_shape, dtype=torch.float32),
            "count": torch.zeros((1), dtype=torch.int64),
        }
    )


def _make_edge_tensor(edges, dim, edge_feature=True):
    if edge_feature:
        return TensorDict(
            {
                "glb": torch.zeros((2, edges), dtype=torch.int64),
                "idx": torch.zeros((2, edges), dtype=torch.int64),
                "attr": torch.zeros((edges, dim), dtype=torch.float32),
                "count": torch.zeros((1), dtype=torch.int64),
            }
        )
    else:
        return TensorDict(
            {
                "glb": torch.zeros((2, edges), dtype=torch.int64),
                "idx": torch.zeros((2, edges), dtype=torch.int64),
                "count": torch.zeros((1), dtype=torch.int64),
            }
        )


def _make_index_tensor(n):
    return TensorDict(
        {
            "idx": torch.zeros((n), dtype=torch.int64),
            "count": torch.zeros((1,), dtype=torch.int64),
        }
    )


class HashHolder:
    def __init__(self):
        self.cache = {}

    def _keys_to_hash(self, keys):
        h = 0
        
        def update_hash(k):
            nonlocal h
            if isinstance(k, torch.Tensor):
                # Check device to avoid unnecessary synchronization
                if k.device.type != 'cpu':
                    # WARN: This syncs GPU-CPU. 
                    # Ideally, callers should pass CPU tensors or we should cache on device.
                    data = k.detach().cpu().numpy().tobytes()
                else:
                    data = k.numpy().tobytes()
                h ^= hash(data)
            elif isinstance(k, (list, tuple)):
                for item in k:
                    update_hash(item)
            else:
                h ^= hash(str(k))

        update_hash(keys)
        return h

    def add(self, keys: tuple, feature: tuple, value: TensorDict):
        h = self._keys_to_hash(keys)
        if h not in self.cache:
            self.cache[h] = {}
        self.cache[h][feature] = value

    def get(self, keys: tuple, feature: tuple) -> Optional[TensorDict]:
        h = self._keys_to_hash(keys)
        if h in self.cache and feature in self.cache[h]:
            y = self.cache[h][feature].detach().clone()
            y.requires_grad_ = True
            return y
        return None

    def prune(self, entries=100):
        if len(self.cache) > entries:
            sorted_keys = sorted(self.cache.keys())
            for k in sorted_keys[:-entries]:
                del self.cache[k]
    
    def clear(self):
        self.cache = {}
