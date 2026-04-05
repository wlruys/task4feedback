from __future__ import annotations

from render_bench_eviction_mapper import main


if __name__ == "__main__":
    main(
        default_benchmark="jacobi",
        default_mappers="kahypar",
        default_stem="bench_eviction_kahypar",
    )
