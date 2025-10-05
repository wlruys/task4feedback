import pytest

from runner.expgen import ConfigError, ConfigSpace, Pack


def _pack(name: str, option: dict) -> Pack:
    return Pack.from_dict(name, [option])


def test_nested_dict_merge_allows_disjoint_leaves():
    packs = [
        _pack("size", {"tag": "b4", "graph": {"config": {"n": 4}}}),
        _pack(
            "block_time",
            {
                "tag": "1:0",
                "graph": {"config": {"block_time": 0, "task_time": 30000}},
            },
        ),
    ]

    configs = list(ConfigSpace(packs, strict=False).iter_configs())
    assert len(configs) == 1

    cfg = configs[0]
    assert cfg.params == {
        "graph": {
            "config": {
                "block_time": 0,
                "n": 4,
                "task_time": 30000,
            }
        }
    }
    assert cfg.tags == ["1:0", "b4"]


def test_nested_dict_conflict_on_leaf_mismatch():
    packs = [
        _pack("size", {"graph": {"config": {"n": 4}}}),
        _pack("other", {"graph": {"config": {"n": 5}}}),
    ]

    with pytest.raises(ConfigError) as exc_info:
        list(ConfigSpace(packs, strict=False).iter_configs())

    assert "graph.config.n" in str(exc_info.value)
