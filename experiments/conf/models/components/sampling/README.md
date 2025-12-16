# Sampling Distribution Configurations

This directory contains Hydra configuration files for action sampling distributions used in RL policies.

## Available Distributions

### Multi-Head Categorical Masked
**File:** `multi_head_categorical_masked.yaml`
**Class:** `task4feedback.ml.models.distributions.MultiHeadCategoricalMasked`
**Use Case:** PPO and on-policy algorithms with multi-head action spaces and masking

Features:
- Multiple independent categorical decisions (heads)
- Per-head masking support (some heads can be inactive)
- Masked heads always return the `inactive_action`
- Zero log probability/entropy contribution from masked heads

Example: Multi-task scheduling where some tasks/devices may be unavailable.

### Multi-Head Categorical
**File:** `multi_head_categorical.yaml`
**Class:** `task4feedback.ml.models.distributions.MultiHeadCategorical`
**Use Case:** Multi-head action spaces without masking needs

Features:
- Multiple independent categorical decisions
- No masking support (all heads always active)
- Simple wrapper around `Independent(Categorical, reinterpreted_batch_ndims=1)`

### Masked Categorical
**File:** `masked_categorical.yaml`
**Class:** `torchrl.modules.distributions.MaskedCategorical`
**Use Case:** SAC and off-policy algorithms with single-head action spaces

Features:
- Single categorical decision
- Action-level masking (invalid actions have zero probability)
- Built-in TorchRL distribution

## Usage in Model Configs

Distributions are typically configured via the `runtime.distribution` parameter in model builders:

```yaml
# PPO with multi-head masked categorical (default)
models:
  _target_: task4feedback.ml.builders.build_actor_critic
  runtime:
    distribution:
      class: task4feedback.ml.models.distributions.MultiHeadCategoricalMasked
      kwargs:
        inactive_action: 0
        reinterpreted_batch_ndims: 1
      mask_key_name: head_mask
      mask_key_value: ["observation", "aux", "candidate_mask"]
      return_log_prob: true

# SAC with masked categorical
models:
  _target_: task4feedback.ml.builders.build_off_policy_model
  runtime:
    distribution:
      class: torchrl.modules.distributions.MaskedCategorical
      kwargs: {}
      mask_key_name: mask
      mask_key_value: ["observation", "aux", "candidate_mask"]
      return_log_prob: false
```

## Implementation Details

### Distribution Configuration Schema

The `runtime.distribution` dict supports:

- **class**: Distribution class to instantiate (e.g., `MultiHeadCategoricalMasked`)
- **kwargs**: Arguments passed to distribution constructor
  - `inactive_action`: Action index for masked heads (multi-head only)
  - `reinterpreted_batch_ndims`: Number of batch dims treated as event dims
  - `validate_args`: Enable PyTorch distribution argument validation
- **mask_key_name**: Name of mask key in ProbabilisticActor's `in_keys` dict
  - `"head_mask"` for multi-head distributions
  - `"mask"` for single-head distributions
- **mask_key_value**: TensorDict key path to mask tensor
  - e.g., `("observation", "aux", "candidate_mask")`
- **return_log_prob**: Whether to return log probabilities with samples

### Default Configurations

**PPO (build_actor_critic):**
```python
distribution:
  class: MultiHeadCategoricalMasked
  kwargs: {inactive_action: 0}
  mask_key_name: head_mask
  mask_key_value: ("observation", "aux", "candidate_mask")
  return_log_prob: true
```

**SAC (build_off_policy_model):**
```python
distribution:
  class: MaskedCategorical
  kwargs: {}
  mask_key_name: mask
  mask_key_value: ("observation", "aux", "candidate_mask")
  return_log_prob: false
```

**Generic (build_model):**
```python
distribution:
  class: MultiHeadCategoricalMasked
  kwargs: {inactive_action: 0, reinterpreted_batch_ndims: 0}
  mask_key_name: head_mask
  mask_key_value: ("observation", "aux", "candidate_mask")
  return_log_prob: false
```

## Adding New Distributions

To add a new sampling distribution:

1. Implement the distribution class in `src/task4feedback/ml/models/distributions.py`
2. Export it in `src/task4feedback/ml/models/__init__.py`
3. Create a configuration file here documenting usage
4. Update this README with the new distribution

### Distribution Interface Requirements

Distributions used with `ProbabilisticActor` should implement:

- `sample(sample_shape)`: Sample actions
- `log_prob(value)`: Compute log probability of actions
- `entropy()`: Compute distribution entropy
- `mode` or `deterministic_sample`: Deterministic action selection

See [PyTorch Distributions](https://pytorch.org/docs/stable/distributions.html) for the full API.
