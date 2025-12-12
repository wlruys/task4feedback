# Observer Factory Configurations

This directory contains observer factory configurations that define how state observations are collected from the simulator.

## Observer Types

### MLPTaskObserverFactory (`mlp_observer.yaml`)

**Python Class:** `task4feedback.observers.mlp.MLPTaskObserverFactory`
**Observer Type:** `CandidateTaskObserver`

**Description:**
Observes only the current candidate tasks (tasks ready to be scheduled). Returns a simple vector of features for each candidate.

**Observation Shape:** `(1, feature_dim)` per candidate task

**Use Cases:**
- MLP-based policies
- Vector-based neural networks
- Simple feature representations
- Minimal observation overhead

**Parameters:**
- `spec`: Graph specification (buffer sizes)
- `features`: Feature definitions (from feature sets)

**Example Usage:**
```yaml
defaults:
  - feature: mlp
  - models: mlp_baseline
```

---

### GridObserverFactory (`cnn_observer.yaml`)

**Python Class:** `task4feedback.observers.cnn.GridObserverFactory`
**Observer Type:** `GridTaskObserver`

**Description:**
Observes all tasks arranged in a 2D grid structure. Returns a flattened grid of task features suitable for CNN processing. Captures spatial-temporal patterns.

**Observation Shape:** `(width × length, feature_dim)`

**Use Cases:**
- CNN-based policies
- Spatially-structured problems (Jacobi, stencils)
- Grid-based computations
- Temporal pattern recognition

**Parameters:**
- `spec`: Graph specification (buffer sizes)
- `width`: Grid width (typically from `${graph.config.n}`)
- `length`: Grid length (typically from `${graph.config.n}`)
- `prev_frames`: Number of temporal frames to observe (default: 1)
- `features`: Feature definitions (from feature sets)

**Example Usage:**
```yaml
defaults:
  - feature: cnn
  - models: cond_dilation_cnn_batch

feature:
  observer:
    prev_frames: 3  # Override to use 3 frames
```

**Important:**
The grid dimensions (`width × length`) must match `spec.max_candidates`.

---

### GNNObserverFactory (`gnn_observer.yaml`)

**Python Class:** `task4feedback.observers.gnn.GNNObserverFactory`
**Observer Type:** `ExternalObserver` (with caching)

**Description:**
Observes the complete heterogeneous graph structure including tasks, data, devices, and their relationships. Most expressive observer type.

**Observation Type:** `HeteroData` graph with multiple node and edge types

**Node Types:**
- Task nodes: Ready and candidate tasks
- Data nodes: Data blocks accessed by tasks
- Device nodes: Compute devices

**Edge Types:**
- Task-Task: Task dependencies
- Task-Data: Read/write data access
- Data-Device: Data location mappings
- Task-Device: Candidate task-device pairs

**Use Cases:**
- GNN-based policies
- Complex task dependencies
- Multi-data-object tasks
- Heterogeneous graph neural networks

**Parameters:**
- `spec`: Graph specification (buffer sizes for nodes and edges)
- `add_degree`: Whether to add degree features to task nodes (default: false)
- `features`: Feature definitions (from feature sets)

**Example Usage:**
```yaml
defaults:
  - feature: gnn
  - models: gnn_baseline

feature:
  observer:
    add_degree: true  # Add in/out degree features
```

---

## Graph Specification

All observers require a `spec` parameter that defines buffer sizes for observation collection:

```yaml
spec:
  _target_: task4feedback.interface.create_graph_spec
  max_tasks: 64             # Maximum number of tasks to observe
  max_data: 16              # Maximum number of data objects
  max_devices: 5            # Maximum number of devices
  max_edges_tasks_tasks: 256    # Task dependency edges
  max_edges_tasks_data: 256     # Task-data access edges
  max_edges_data_devices: 128   # Data-device location edges
  max_edges_tasks_devices: 64   # Task-device candidate edges
  max_candidates: 64        # Maximum candidate tasks
```

These limits must be set appropriately for your problem size. Exceeding them will cause runtime errors.

## Feature System Integration

Observers work with the feature system to extract specific features:

1. Feature sets define which features to extract (e.g., `sets/mlp/io_basic.yaml`)
2. Observer configs reference these via `features: ???` (required from feature sets)
3. The `ConfigurableObserverFactory` base class parses feature configurations
4. Features are instantiated and attached to the observer

## Context Variables

Feature sets can use context variables that are resolved by observer configs:

**CNN Observers provide:**
- `$width`: Grid width
- `$length`: Grid length
- `$prev_frames`: Number of temporal frames

**Example:**
```yaml
# In CNN feature set
features:
  task:
    - name: PrevReadSizeFeature
      args: ["$width", "$length", true, "$prev_frames"]
```

These variables are resolved at instantiation time using values from the observer config.

## Choosing an Observer

| Use Case | Observer | Feature Set | Network |
|----------|----------|-------------|---------|
| Simple scheduling | MLP | io_basic | mlp_baseline |
| Grid-based tasks | CNN | read_data_coords | cond_dilation_cnn_batch |
| Complex dependencies | GNN | io_data_full | gnn_baseline |
| Multi-device | CNN/GNN | *_device | * |
| Minimal overhead | MLP | io_basic | mlp_baseline |
| Maximum expressiveness | GNN | io_state_data_full | gnn_baseline |

## Implementation Details

Observer factories are instantiated via Hydra:

```python
# In experiments/helper/env.py
observer_factory = hydra.utils.instantiate(
    cfg.feature.observer,
    # Additional parameters injected at runtime
)
```

The factory's `create()` method is called to instantiate actual observer instances when environments are created.

## See Also

- `../README.md` - Feature system overview
- `../sets/README.md` - Feature set catalog
- `src/task4feedback/observers/` - Observer implementation code
- `src/task4feedback/interface/observer.py` - Observer base classes
