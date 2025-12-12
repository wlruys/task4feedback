# Feature Configuration System

This directory contains the declarative configuration system for observer features in the task4feedback scheduler.

## Quick Start

To use a feature set in your experiment:

```yaml
defaults:
  - feature: mlp  # or cnn, gnn
```

To override the default feature set:

```yaml
defaults:
  - feature: cnn

feature:
  defaults:
    - sets/cnn/read_full  # Choose your feature set
```

## Directory Structure

```
feature/
├── README.md              # This file
├── base.yaml             # Base feature configuration
├── mlp.yaml              # MLP observer configuration (default: io_basic)
├── cnn.yaml              # CNN observer configuration (default: read_basic)
├── gnn.yaml              # GNN observer configuration (default: io_data_mapped)
│
├── observer/             # Observer factory configurations
│   ├── README.md          # Observer documentation
│   ├── mlp_observer.yaml  # MLP task observer factory
│   ├── cnn_observer.yaml  # CNN grid observer factory
│   └── gnn_observer.yaml  # GNN heterogeneous graph observer factory
│
└── sets/                 # Feature set definitions
    ├── README.md          # Feature set catalog
    ├── mlp/              # MLP-specific feature sets (3 sets)
    ├── cnn/              # CNN-specific feature sets (8 sets)
    └── gnn/              # GNN-specific feature sets (4 sets)
```

## Feature Set Catalog

### MLP Feature Sets
- **io_basic** - Basic I/O features (minimal)
- **io_data_size** - I/O + data sizes
- **io_data_coords** - I/O + data sizes + coordinates (recommended)

### CNN Feature Sets
- **read_basic** - Temporal read history only
- **read_data_size** - Read history + data sizes
- **read_coords** - Read history + spatial coordinates
- **read_device** - Read history + device mapping
- **read_data_coords** - Read history + data + coords (recommended)
- **read_data_device** - Read history + data + device
- **read_coords_device** - Read history + coords + device
- **read_full** - All features (maximum expressiveness)

### GNN Feature Sets
- **io_data_mapped** - Task I/O + data mapping (baseline)
- **io_data_coords** - Task I/O + data coordinates
- **io_data_full** - Task I/O + complete data info (recommended)
- **io_state_data_full** - Task I/O + state + data

See `sets/README.md` and architecture-specific READMEs for detailed feature descriptions.

## Hot-Swapping Features

### Command-Line Override
```bash
python train.py feature.defaults='[sets/mlp/io_data_coords]'
```

### Multi-Run Sweep
```bash
python train.py -m feature.defaults='[sets/mlp/io_basic]','[sets/mlp/io_data_coords]'
```

### In Experiment Config
```yaml
feature:
  defaults:
    - sets/cnn/read_full  # Override default feature set
```

## Migration from Version Parameter

### Old Style (DEPRECATED - No Longer Supported)
```yaml
feature:
  observer:
    version: A  # This will cause an error!
```

### New Style
```yaml
defaults:
  - feature: mlp

feature:
  defaults:
    - sets/mlp/io_basic  # Equivalent to old version A
```

### Version Mapping

**MLP:**
- Version A → `sets/mlp/io_basic`
- Version B → `sets/mlp/io_data_size`
- Version C/E → `sets/mlp/io_data_coords`

**CNN:**
- Version A → `sets/cnn/read_basic`
- Version B → `sets/cnn/read_data_size`
- Version C → `sets/cnn/read_coords`
- Version D → `sets/cnn/read_device`
- Version E → `sets/cnn/read_data_coords`
- Version F → `sets/cnn/read_data_device`
- Version G → `sets/cnn/read_coords_device`
- Version H → `sets/cnn/read_full`

**GNN:**
- Version A → `sets/gnn/io_data_mapped`
- Version B → `sets/gnn/io_data_coords`
- Version C → `sets/gnn/io_data_full`
- Version D → `sets/gnn/io_state_data_full`

## Configuration Hierarchy

```
experiment config (e.g., 4x4x16_static_mlp.yaml)
  ├── feature: mlp  →  feature/mlp.yaml
  │   ├── normalization: task_normalization
  │   ├── observer: mlp_observer  →  observer/mlp_observer.yaml
  │   └── sets/mlp/io_basic  →  sets/mlp/io_basic.yaml (default)
  │       └── Provides: feature.observer.features
  └── Override: feature.defaults: [sets/mlp/io_data_coords]
```

## Examples

### Standard Usage
```yaml
# experiments/conf/my_experiment.yaml
defaults:
  - config.yaml
  - feature: cnn
  - models: cond_dilation_cnn_batch

# Uses default feature set (read_basic)
```

### Override Feature Set
```yaml
defaults:
  - config.yaml
  - feature: cnn
  - models: cond_dilation_cnn_batch

feature:
  defaults:
    - sets/cnn/read_full  # Use comprehensive features
```

### Override Parameters
```yaml
feature:
  defaults:
    - sets/cnn/read_basic
  observer:
    prev_frames: 3  # Use 3 frames of temporal history instead of 1
```

## Creating Custom Feature Sets

To create a custom feature set:

1. Create a new YAML file in the appropriate directory (e.g., `sets/mlp/custom.yaml`)
2. Add the `@package` directive to specify where features are injected
3. Define your features under the `features:` key
4. Document the feature set with inline comments

Example:
```yaml
# @package _global_.feature.observer
# Custom Feature Set
#
# Description:
#   Your custom combination of features
#
# Features:
#   - List your features here

features:
  task:
    - InputOutputTaskFeature
    - YourCustomFeature
```

## Troubleshooting

### Error: "features parameter is required"
You need to select a feature set. Add to your config:
```yaml
feature:
  defaults:
    - sets/mlp/io_basic  # or your preferred feature set
```

### Error: "version parameter is no longer supported"
The `version` parameter has been removed. Use the new feature set system:
```yaml
# Instead of: version: A
# Use: feature.defaults: [sets/mlp/io_basic]
```

### Features not being loaded
Check that:
1. The `@package _global_.feature.observer` directive is in the feature set file
2. The path in defaults is correct (e.g., `sets/mlp/io_basic` not `/sets/mlp/io_basic`)
3. The feature set file defines a `features:` dictionary

## Further Reading

- `observer/README.md` - Observer factory documentation
- `sets/README.md` - Complete feature set catalog
- `sets/{mlp,cnn,gnn}/README.md` - Architecture-specific documentation
