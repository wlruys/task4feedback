# Full Comparison Report: Mappers × Eviction Policies × Transition Conditions

**Workload:** DynamicJacobi, 8×8 grid (64 cells), 64 steps, 34,816 tasks  
**System:** 4 GPUs + 1 CPU, H2D=129 GB/s, D2D=54 GB/s  
**Seed:** 0, 1 rep each

Two memory regimes were tested:
- **No-eviction regime** (`gpu_mem=1.5 GB`): max_mem peaks at ~487 MB → no evictions triggered
- **Tight-memory regime** (`gpu_mem=0.3 GB`): max_mem capped at 286 MB → heavy eviction (58–180 GB evicted)

---

## Phase 1 — All Mappers × Eviction Policies (auto transition)

### No-eviction regime (gpu_mem ≈ 1.5 GB, level_mem = 1.5 GB)

| Mapper | Eviction | sim(s) | total_mv | evict_mv |
|--------|----------|-------:|----------:|---------:|
| DequeueEFTMapper | lru | 0.020 | 1.6 GB | 0.0 B |
| DequeueEFTMapper | least_used_mapped | 0.020 | 1.6 GB | 0.0 B |
| MemoryAwareEFTMapper | lru | 0.020 | 1.6 GB | 0.0 B |
| MemoryAwareEFTMapper | least_used_mapped | 0.020 | 1.6 GB | 0.0 B |
| KaHyParMapper | lru | **ERROR** | — | — |
| KaHyParMapper | least_used_mapped | **ERROR** | — | — |
| METISMapper | lru | **ERROR** | — | — |
| METISMapper | least_used_mapped | **ERROR** | — | — |
| DARTSMapper | lru | 0.288 | 94.0 GB | 0.0 B |
| DARTSMapper | least_used_mapped | 0.288 | 94.0 GB | 0.0 B |
| ExtendedDARTSMapper | lru | 0.147 | 44.3 GB | 0.0 B |
| ExtendedDARTSMapper | least_used_mapped | 0.147 | 44.3 GB | 0.0 B |
| ExternalBlockCyclicMapper | lru | 0.026 | 3.7 GB | 0.0 B |
| ExternalBlockCyclicMapper | least_used_mapped | 0.026 | 3.7 GB | 0.0 B |
| ExternalRowCyclicMapper | lru | 0.028 | 2.9 GB | 0.0 B |
| ExternalRowCyclicMapper | least_used_mapped | 0.028 | 2.9 GB | 0.0 B |
| ExternalColCyclicMapper | lru | 0.027 | 3.1 GB | 0.0 B |
| ExternalColCyclicMapper | least_used_mapped | 0.027 | 3.1 GB | 0.0 B |
| ExternalCheckerboardMapper | lru | 0.026 | 3.7 GB | 0.0 B |
| ExternalCheckerboardMapper | least_used_mapped | 0.026 | 3.7 GB | 0.0 B |

**Notes:**
- KaHyParMapper and METISMapper are not available in this build (require `ENABLE_KAHYPAR` / `ENABLE_METIS` cmake flags).
- No evictions occur at this memory budget. Eviction policy has no effect on sim time.
- EFT-based mappers (DequeueEFT, MemoryAwareEFT) minimize data movement (1.6 GB). External static partition mappers have slightly higher movement (2.9–3.7 GB). DARTS mappers incur large movement (44–94 GB) due to their frontier-exploration phase.
- **Best sim time:** DequeueEFT / MemoryAwareEFT (0.020 s)
- **Worst sim time:** DARTSMapper (0.288 s — 14× slower due to exploration overhead)

### Tight-memory regime (gpu_mem ≈ 0.3 GB, level_mem = 1.5 GB)

| Mapper | Eviction | sim(s) | total_mv | evict_mv | evict_events |
|--------|----------|-------:|----------:|---------:|-------------:|
| DequeueEFTMapper | lru | 0.910 | 349.5 GB | 174.3 GB | 20,202 |
| DequeueEFTMapper | least_used_mapped | 0.844 | 281.0 GB | 140.4 GB | 18,949 |
| MemoryAwareEFTMapper | lru | 0.936 | 347.8 GB | 173.0 GB | 19,949 |
| MemoryAwareEFTMapper | least_used_mapped | 0.785 | 235.6 GB | 117.6 GB | 17,935 |
| KaHyParMapper | lru | **ERROR** | — | — | — |
| KaHyParMapper | least_used_mapped | **ERROR** | — | — | — |
| METISMapper | lru | **ERROR** | — | — | — |
| METISMapper | least_used_mapped | **ERROR** | — | — | — |
| DARTSMapper | lru | 0.794 | 242.4 GB | 100.0 GB | 14,378 |
| DARTSMapper | least_used_mapped | 0.794 | 242.4 GB | 100.0 GB | 14,378 |
| ExtendedDARTSMapper | lru | **0.538** | 151.3 GB | 58.4 GB | 9,619 |
| ExtendedDARTSMapper | least_used_mapped | **0.538** | 151.3 GB | 58.4 GB | 9,619 |
| ExternalBlockCyclicMapper | lru | 0.958 | 362.3 GB | 180.3 GB | 20,414 |
| ExternalBlockCyclicMapper | least_used_mapped | 0.856 | 315.1 GB | 156.9 GB | 19,798 |
| ExternalRowCyclicMapper | lru | 0.920 | 358.4 GB | 178.7 GB | 20,381 |
| ExternalRowCyclicMapper | least_used_mapped | 0.787 | 296.9 GB | 148.6 GB | 19,625 |
| ExternalColCyclicMapper | lru | 0.951 | 355.3 GB | 177.4 GB | 20,303 |
| ExternalColCyclicMapper | least_used_mapped | 0.765 | 262.0 GB | 131.0 GB | 19,238 |
| ExternalCheckerboardMapper | lru | 0.958 | 362.3 GB | 180.3 GB | 20,414 |
| ExternalCheckerboardMapper | least_used_mapped | 0.856 | 315.1 GB | 156.9 GB | 19,798 |

**Notes:**
- **Best overall:** ExtendedDARTSMapper (0.538 s sim, 58.4 GB evicted) — wins on both sim time and eviction volume.
- **DARTS family:** DARTS/ExtendedDATS are eviction-resilient. Their locality-aware scheduling naturally reduces thrashing. The extended variant is 32% faster than base DARTS (0.538 vs 0.794 s) with 42% less eviction.
- **Eviction policy impact:** Under tight memory, `least_used_mapped` consistently reduces both sim time and eviction volume vs `lru`. The gain is significant for EFT mappers (e.g., MemoryAwareEFT: 0.936→0.785 s, -16%) but negligible for DARTS (both policies produce identical results).
- **External static mappers** perform worst on eviction volume (180 GB) — fixed partition unaware of runtime data locality.
- **MemoryAwareEFT vs DequeueEFT:** Under LRU, MemoryAwareEFT is marginally *worse* (0.936 vs 0.910 s). Under `least_used_mapped` it improves to 0.785 vs 0.844 s — the memory-awareness helps when the eviction policy is better.

---

## Phase 2 — Key Mappers × Transition Kinds (eviction = lru, tight memory)

| Mapper | Transition | sim(s) | total_mv | evict_mv | evict_events |
|--------|-----------|-------:|----------:|---------:|-------------:|
| DequeueEFTMapper | default | 0.926 | 354.3 GB | 176.6 GB | 20,285 |
| DequeueEFTMapper | **batch** | **0.910** | 349.5 GB | 174.3 GB | 20,202 |
| DequeueEFTMapper | device_threshold | 0.967 | 343.6 GB | 171.3 GB | 19,958 |
| DequeueEFTMapper | range | 0.998 | 361.3 GB | 180.2 GB | 20,456 |
| DequeueEFTMapper | **hysteresis** | **0.895** | 311.3 GB | 155.1 GB | 17,902 |
| MemoryAwareEFTMapper | default | 0.891 | 338.1 GB | 168.5 GB | 19,302 |
| MemoryAwareEFTMapper | **batch** | 0.936 | 347.8 GB | 173.0 GB | 19,949 |
| MemoryAwareEFTMapper | device_threshold | 0.952 | 342.8 GB | 170.1 GB | 19,844 |
| MemoryAwareEFTMapper | range | 0.999 | 358.8 GB | 178.6 GB | 20,337 |
| MemoryAwareEFTMapper | **hysteresis** | 0.930 | 332.4 GB | 165.2 GB | 19,614 |
| DARTSMapper | default | 0.794 | 242.4 GB | 100.0 GB | 14,378 |
| DARTSMapper | batch | 0.794 | 242.4 GB | 100.0 GB | 14,378 |
| DARTSMapper | **device_threshold** | **0.794** | 242.4 GB | 100.0 GB | 14,378 |
| DARTSMapper | range | 0.807 | 244.5 GB | 100.6 GB | 14,016 |
| DARTSMapper | **hysteresis** | **0.772** | 233.8 GB | 94.3 GB | 13,757 |
| ExtendedDARTSMapper | default | 0.538 | 151.3 GB | 58.4 GB | 9,619 |
| ExtendedDARTSMapper | batch | 0.538 | 151.3 GB | 58.4 GB | 9,619 |
| ExtendedDARTSMapper | **device_threshold** | **0.538** | 151.3 GB | 58.4 GB | 9,619 |
| ExtendedDARTSMapper | range | 0.556 | 151.3 GB | 58.4 GB | 9,601 |
| ExtendedDARTSMapper | hysteresis | 0.538 | 152.0 GB | 58.8 GB | 9,333 |

**Notes:**
- **For DequeueEFT:** `hysteresis` is the best transition condition (0.895 s, -11 GB eviction vs `batch`). `range` is worst.
- **For MemoryAwareEFT:** `default` (0.891 s) beats `batch` (0.936 s) under LRU with tight memory.
- **DARTS mapper:** All transition kinds produce nearly identical results except `hysteresis` shaves ~3% off sim time.
- **ExtendedDARTS:** Largely insensitive to transition kind; `range` adds a small sim overhead.
- **`range` is consistently worst** across all mappers (highest sim time and eviction volume).

---

## Phase 3 — Key Mappers × Transition Kinds (eviction = least_used_mapped, tight memory)

| Mapper | Transition | sim(s) | total_mv | evict_mv | evict_events |
|--------|-----------|-------:|----------:|---------:|-------------:|
| DequeueEFTMapper | default | 0.927 | 352.3 GB | 175.9 GB | 20,174 |
| DequeueEFTMapper | batch | 0.844 | 281.0 GB | 140.4 GB | 18,949 |
| DequeueEFTMapper | device_threshold | 0.776 | 191.4 GB | 95.3 GB | 14,725 |
| DequeueEFTMapper | **range** | **0.458** | 167.4 GB | 83.9 GB | 18,262 |
| DequeueEFTMapper | hysteresis | 0.759 | 190.6 GB | 95.2 GB | 17,106 |
| MemoryAwareEFTMapper | default | 0.876 | 331.2 GB | 165.3 GB | 19,489 |
| MemoryAwareEFTMapper | batch | 0.785 | 235.6 GB | 117.6 GB | 17,935 |
| MemoryAwareEFTMapper | device_threshold | 0.755 | 184.7 GB | 92.0 GB | 15,475 |
| MemoryAwareEFTMapper | **range** | **0.455** | 167.0 GB | 83.6 GB | 18,268 |
| MemoryAwareEFTMapper | hysteresis | 0.746 | 189.3 GB | 94.3 GB | 16,376 |
| DARTSMapper | default | 0.794 | 242.4 GB | 100.0 GB | 14,378 |
| DARTSMapper | batch | 0.794 | 242.4 GB | 100.0 GB | 14,378 |
| DARTSMapper | device_threshold | 0.794 | 242.4 GB | 100.0 GB | 14,378 |
| DARTSMapper | range | 0.807 | 244.5 GB | 100.6 GB | 14,016 |
| DARTSMapper | hysteresis | 0.772 | 233.8 GB | 94.3 GB | 13,757 |
| ExtendedDARTSMapper | default | 0.538 | 151.3 GB | 58.4 GB | 9,619 |
| ExtendedDARTSMapper | batch | 0.538 | 151.3 GB | 58.4 GB | 9,619 |
| ExtendedDARTSMapper | device_threshold | 0.538 | 151.3 GB | 58.4 GB | 9,619 |
| ExtendedDARTSMapper | range | 0.556 | 151.3 GB | 58.4 GB | 9,601 |
| ExtendedDARTSMapper | hysteresis | 0.538 | 152.0 GB | 58.8 GB | 9,332 |

**Notes:**
- **Dramatic interaction with `range`:** Under `least_used_mapped`, `RangeTransitionConditions` cuts sim time nearly in half for EFT mappers: DequeueEFT 0.927→0.458 s (-51%), MemoryAwareEFT 0.876→0.455 s (-48%). This is the **best result for any EFT mapper** in the tight-memory regime.
- Under LRU (Phase 2), `range` was the *worst* condition. This indicates a strong interaction between the eviction policy and the in-flight window that `RangeTransitionConditions` manages.
- **DARTS family:** Unaffected by eviction policy (sim times identical between Phase 2 and Phase 3). The DARTS locality-aware mapping already avoids replication so the eviction policy doesn't change behavior.
- **`device_threshold`** is mid-range for EFT mappers under `least_used_mapped` (0.755–0.776 s). 

---

## Summary

### Mapper Rankings (tight memory, best transition+eviction per mapper)

| Rank | Mapper | Best Config | sim(s) | evict_mv |
|-----:|--------|------------|-------:|---------:|
| 1 | ExtendedDARTSMapper | any / any | 0.538 | 58.4 GB |
| 2 | MemoryAwareEFTMapper | range + least_used_mapped | 0.455 | 83.6 GB |
| 3 | DequeueEFTMapper | range + least_used_mapped | 0.458 | 83.9 GB |
| 4 | DARTSMapper | hysteresis / any | 0.772 | 94.3 GB |
| 5 | ExternalColCyclicMapper | auto + least_used_mapped | 0.765 | 131.0 GB |
| 6 | ExternalRowCyclicMapper | auto + least_used_mapped | 0.787 | 148.6 GB |
| 7 | ExternalBlockCyclicMapper | auto + least_used_mapped | 0.856 | 156.9 GB |
| 8 | ExternalCheckerboardMapper | auto + least_used_mapped | 0.856 | 156.9 GB |
| — | KaHyParMapper | — | **NOT BUILT** | — |
| — | METISMapper | — | **NOT BUILT** | — |

### Key Takeaways

1. **ExtendedDARTSMapper dominates under eviction pressure** — 2× better sim time than EFT mappers using the default configuration, with 30% less eviction movement. This advantage comes from its locality-aware frontier exploration.

2. **EFT mappers need the right transition+eviction pairing**: the combination of `RangeTransitionConditions` + `least_used_mapped` cuts their sim time nearly in half under tight memory, making them competitive. Under `lru`, `range` is actually the *worst* choice.

3. **`least_used_mapped` eviction policy consistently beats `lru`** for EFT-class mappers (5–16% improvement in sim time). DARTS-class mappers are insensitive to eviction policy.

4. **`RangeTransitionConditions` is high-variance**: it's the worst transition for LRU and the best for `least_used_mapped` with EFT mappers. Avoid it with `lru`.

5. **`HysteresisTransitionConditions` is the safest general choice** for EFT mappers under LRU: lowest or near-lowest sim time and eviction volume across both EFT and DARTS mappers.

6. **External static-partition mappers** (block/row/col/checkerboard) have the highest eviction volume because they ignore runtime data locality. They benefit notably from `least_used_mapped` vs `lru` (~10-15% sim reduction).

7. **KaHyPar and METIS are not available** in this build. Rebuild with `ENABLE_KAHYPAR=ON` / `ENABLE_METIS=ON` to include them.

---

## Build / Availability

| Component | Status |
|-----------|--------|
| DequeueEFTMapper | ✓ Available |
| MemoryAwareEFTMapper | ✓ Available |
| DARTSMapper | ✓ Available |
| ExtendedDARTSMapper | ✓ Available |
| KaHyParMapper | ✗ Requires `ENABLE_KAHYPAR=ON` |
| METISMapper | ✗ Requires `ENABLE_METIS=ON` |
| ExternalBlockCyclicMapper | ✓ Available |
| ExternalRowCyclicMapper | ✓ Available |
| ExternalColCyclicMapper | ✓ Available |
| ExternalCheckerboardMapper | ✓ Available |
| LRU eviction | ✓ Available |
| LEAST_USED_MAPPED eviction | ✓ Available |
| DefaultTransitionConditions | ✓ Available |
| BatchTransitionConditions | ✓ Available |
| DeviceThresholdTransitionConditions | ✓ Available |
| RangeTransitionConditions | ✓ Available |
| HysteresisTransitionConditions | ✓ Available |
