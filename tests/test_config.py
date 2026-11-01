"""asymvlm.config registry consistency with the released checkpoints."""

import pytest
import torch

config = pytest.importorskip("asymvlm.config")

EXPECTED_HIDDEN = {"qwen25": 2048, "llava_next": 4096, "phi35": 3072}


def test_registry_keys_and_aliases():
    assert set(config.MODELS) == set(EXPECTED_HIDDEN)
    for alias, key in config.ALIASES.items():
        assert config.get_spec(alias).key == key
    with pytest.raises(KeyError):
        config.get_spec("not-a-model")


@pytest.mark.parametrize("key", sorted(EXPECTED_HIDDEN))
def test_spec_defaults(key):
    spec = config.get_spec(key)
    assert spec.hf_id and len(spec.revision) == 40
    if spec.gate_threshold is not None:
        assert 0.0 < spec.gate_threshold < 1.0
    assert config.GATE_THRESHOLDS[key] == spec.gate_threshold


def test_gate_and_generation_defaults():
    assert tuple(config.GATE_PROBES) == tuple(sorted(config.GATE_PROBES))
    assert max(config.GATE_PROBES) < config.GATE_CAP <= 100
    assert config.GENERATION_DEFAULTS["max_new_tokens"] >= 1
    assert config.TEXT_EVICTION_BUDGET >= 1


@pytest.mark.parametrize("key", sorted(EXPECTED_HIDDEN))
def test_spec_checkpoint_matches_backbone(key):
    spec = config.get_spec(key)
    path = spec.scorer_path()
    if not path.exists():
        pytest.skip(f"{spec.scorer_checkpoint} not present")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    assert state["visual.weight"].shape == (32, EXPECTED_HIDDEN[key])
    assert payload["hidden_size"] == EXPECTED_HIDDEN[key]
    fmt = payload.get("format", "")
    if spec.scorer_format == "query_selector_v24":
        assert "query_selector_v24" in fmt
    else:
        assert fmt == "asymvlm_mlsys_scorer_v1" and payload.get("variant") == spec.scorer_format
    ref = spec.reference_path()
    if ref is not None:
        assert ref.exists(), f"{spec.reference_checkpoint} listed but missing"
