#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import yaml


# =========================
# Utilities
# =========================

def stable_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no spaces) for hashing/serialization."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


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


def serialize_value_for_hydra(v: Any) -> str:
    """Render Python value as Hydra CLI literal."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, dict)):
        return stable_json(v)
    return str(v)


def sh_single_quote(s: str) -> str:
    """Shell-safe single-quoting: ' -> '"'"' """
    return "'" + s.replace("'", "'\"'\"'") + "'"

# =========================
# Validation
# =========================

class ConfigError(ValueError):
    pass


def validate_yaml_structure(data: Any) -> None:
    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML must be a mapping (dict).")
    if "cli_base" not in data or not isinstance(data["cli_base"], str) or not data["cli_base"].strip():
        raise ConfigError("Missing or invalid 'cli_base' (non-empty string).")
    if "packs" not in data or not isinstance(data["packs"], list) or not data["packs"]:
        raise ConfigError("Missing or invalid 'packs' (non-empty list).")
    for i, p in enumerate(data["packs"]):
        if not isinstance(p, dict):
            raise ConfigError(f"Pack #{i}: expected dict, got {type(p).__name__}.")
        if "name" not in p or not isinstance(p["name"], str) or not p["name"].strip():
            raise ConfigError(f"Pack #{i}: missing/invalid 'name' (non-empty string).")
        if "options" not in p or not isinstance(p["options"], list) or not p["options"]:
            raise ConfigError(f"Pack '{p.get('name','?')}': 'options' must be a non-empty list.")
        for j, opt in enumerate(p["options"]):
            if not isinstance(opt, dict):
                raise ConfigError(f"Pack '{p['name']}', option #{j}: expected dict, got {type(opt).__name__}.")
            if "tag" in opt and not (isinstance(opt["tag"], str) and opt["tag"].strip()):
                raise ConfigError(f"Pack '{p['name']}', option #{j}: 'tag' must be a non-empty string if present.")


# =========================
# Core data structures
# =========================

@dataclass(frozen=True)
class Option:
    payload: Dict[str, Any]

    @property
    def tag(self) -> Optional[str]:
        t = self.payload.get("tag", None)
        return str(t) if t is not None else None

    def params(self) -> Dict[str, Any]:
        return {k: v for k, v in self.payload.items() if k != "tag"}


@dataclass
class Pack:
    name: str
    options: List[Option]

    @classmethod
    def from_dict(cls, name: str, options: List[Dict[str, Any]]) -> "Pack":
        return cls(name=name, options=[Option(dict(opt)) for opt in options])

    def merged_with(self, other: "Pack") -> "Pack":
        if self.name != other.name:
            raise ValueError("Pack names must match to merge")
        seen = {stable_json(opt.payload) for opt in self.options}
        new_opts = list(self.options)
        for opt in other.options:
            key = stable_json(opt.payload)
            if key not in seen:
                new_opts.append(opt)
                seen.add(key)
        return Pack(name=self.name, options=new_opts)


class Config:
    """Immutable configuration with deterministic ID and derived metadata."""
    def __init__(self, params: Dict[str, Any], tags: Sequence[str]):
        self._params: Dict[str, Any] = json.loads(stable_json(params))
        self._tags: List[str] = sorted({str(t) for t in tags if t})
        serialized = stable_json({"params": self._params, "tags": self._tags})
        self._id: str = hashlib.sha1(serialized.encode("utf-8")).hexdigest()

    @property
    def id(self) -> str:
        return self._id

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    @property
    def tags(self) -> List[str]:
        return list(self._tags)

    @property
    def wandb_name(self) -> str:
        prefix = "_".join(self._tags) if self._tags else "exp"
        return f"{prefix}_{self._id[:8]}"

    def manifest_entry(self) -> Dict[str, Any]:
        return {"id": self._id, "wandb_name": self.wandb_name, "tags": self._tags, "params": self._params}

    def __repr__(self) -> str:
        return f"Config(id={self._id[:8]}, tags={self._tags})"


class ConfigSpace:
    """Cartesian product over packs with conflict checking."""
    def __init__(self, packs: Sequence[Pack], strict: bool = False):
        self._packs: List[Pack] = list(packs)
        self._strict = bool(strict)

    @property
    def packs(self) -> List[Pack]:
        return list(self._packs)

    def iter_configs(self) -> Iterator[Config]:
        if not self._packs:
            return
        option_lists: List[List[Option]] = [p.options for p in self._packs]
        pack_names: List[str] = [p.name for p in self._packs]

        for combo in itertools.product(*option_lists):
            merged: Dict[str, Any] = {}
            tags: List[str] = []
            provenance: Dict[Tuple[str, ...], Tuple[str, int, Dict[str, Any]]] = {}

            for pack_idx, opt in enumerate(combo):
                if opt.tag:
                    tags.append(opt.tag)
                self._merge_params(
                    merged,
                    opt.params(),
                    provenance,
                    pack_names[pack_idx],
                    self._option_index(option_lists[pack_idx], opt),
                    opt.payload,
                )

            yield Config(merged, tags)

    def _merge_params(
        self,
        merged: Dict[str, Any],
        new_params: Dict[str, Any],
        provenance: Dict[Tuple[str, ...], Tuple[str, int, Dict[str, Any]]],
        pack_name: str,
        option_idx: int,
        option_payload: Dict[str, Any],
    ) -> None:
        if not new_params:
            return
        self._merge_dicts(
            merged,
            new_params,
            provenance,
            pack_name,
            option_idx,
            option_payload,
            strict=self._strict,
            path=(),
        )

    @staticmethod
    def _merge_dicts(
        target: Dict[str, Any],
        updates: Dict[str, Any],
        provenance: Dict[Tuple[str, ...], Tuple[str, int, Dict[str, Any]]],
        pack_name: str,
        option_idx: int,
        option_payload: Dict[str, Any],
        *,
        strict: bool,
        path: Tuple[str, ...],
    ) -> None:
        for key, value in updates.items():
            current_path = path + (key,)
            if key not in target:
                target[key] = deepcopy(value)
                ConfigSpace._register_provenance(
                    value,
                    provenance,
                    current_path,
                    pack_name,
                    option_idx,
                    option_payload,
                )
                continue

            existing = target[key]
            if isinstance(existing, dict) and isinstance(value, dict):
                ConfigSpace._merge_dicts(
                    existing,
                    value,
                    provenance,
                    pack_name,
                    option_idx,
                    option_payload,
                    strict=strict,
                    path=current_path,
                )
                continue

            if strict or existing != value:
                prev_pack, prev_idx, prev_payload = ConfigSpace._lookup_provenance(provenance, current_path)
                path_str = ConfigSpace._format_path(current_path)
                raise ConfigError(
                    "Parameter conflict detected:\n"
                    f"  path: {path_str}\n"
                    f"  previous: value={existing!r} from pack='{prev_pack}' option_index={prev_idx} option={prev_payload}\n"
                    f"  current : value={value!r} from pack='{pack_name}' option_index={option_idx} option={option_payload}\n"
                    "Resolve by adjusting packs/options or disable strict mode if values are identical."
                )

            ConfigSpace._register_provenance(
                value,
                provenance,
                current_path,
                pack_name,
                option_idx,
                option_payload,
            )

    @staticmethod
    def _register_provenance(
        value: Any,
        provenance: Dict[Tuple[str, ...], Tuple[str, int, Dict[str, Any]]],
        path: Tuple[str, ...],
        pack_name: str,
        option_idx: int,
        option_payload: Dict[str, Any],
    ) -> None:
        if path not in provenance:
            provenance[path] = (pack_name, option_idx, option_payload)
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                ConfigSpace._register_provenance(
                    sub_value,
                    provenance,
                    path + (sub_key,),
                    pack_name,
                    option_idx,
                    option_payload,
                )

    @staticmethod
    def _lookup_provenance(
        provenance: Dict[Tuple[str, ...], Tuple[str, int, Dict[str, Any]]],
        path: Tuple[str, ...],
    ) -> Tuple[str, int, Dict[str, Any]]:
        current = path
        while current:
            if current in provenance:
                return provenance[current]
            current = current[:-1]
        if () in provenance:
            return provenance[()]
        raise ConfigError(f"Missing provenance for path '{ConfigSpace._format_path(path)}'")

    @staticmethod
    def _format_path(path: Tuple[str, ...]) -> str:
        return ".".join(path) if path else "<root>"

    @staticmethod
    def _option_index(options: List[Option], target: Option) -> int:
        for i, o in enumerate(options):
            if o is target:
                return i
        return -1


# =========================
# CLI builders (Hydra)
# =========================

def to_hydra_cli(cfg: Config, base_cmd: str) -> str:
    """
    Convert a Config to a Hydra CLI string with shell-safe quoting:
      base 'k=v' 'k=v' 'wandb.tags=[...]' 'wandb.name=...'
    """
    flat = flatten_dict(cfg.params)
    tokens: List[str] = []
    for k in sorted(flat.keys()):
        token = f"{k}={serialize_value_for_hydra(flat[k])}"
        tokens.append(sh_single_quote(token))
    if cfg.tags:
        tokens.append(sh_single_quote(f"wandb.tags=[{','.join(cfg.tags)}]"))
    tokens.append(sh_single_quote(f"wandb.name={cfg.wandb_name}"))
    return " ".join([base_cmd] + tokens)


# =========================
# Experiment builder
# =========================

class ExperimentBuilder:
    def __init__(self, cli_base: str, packs: Sequence[Pack]):
        self.cli_base = str(cli_base)
        self._packs = self._merge_same_name_packs(packs)
        # Minimal default: allow equal-value overlaps, error on mismatches.
        self._space = ConfigSpace(self._packs, strict=False)

    @staticmethod
    def _merge_same_name_packs(packs: Sequence[Pack]) -> List[Pack]:
        by_name: Dict[str, Pack] = {}
        for p in packs:
            if p.name in by_name:
                by_name[p.name] = by_name[p.name].merged_with(p)
            else:
                by_name[p.name] = p
        return list(by_name.values())

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentBuilder":
        try:
            data = yaml.safe_load(Path(path).read_text())
        except Exception as e:
            raise ConfigError(f"Failed to parse YAML '{path}': {e}") from e
        validate_yaml_structure(data)
        packs = [Pack.from_dict(p["name"], p["options"]) for p in data["packs"]]
        return cls(cli_base=data["cli_base"], packs=packs)

    def build(
        self,
        outdir: str | Path,
        batch_size: int,
    ) -> List[Path]:
        """Generate batch files and manifests for the full experiment."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        outdir = Path(outdir)
        batch_dir = outdir / "batches"
        manifest_jsonl = outdir / "manifest.jsonl"
        manifest_csv = outdir / "manifest.csv"

        # No resume/append. Require a clean output directory.
        if batch_dir.exists() and any(batch_dir.glob("batch_*.txt")):
            raise ConfigError(f"{batch_dir} already contains batches. Remove OUT to rebuild.")
        if manifest_jsonl.exists() or manifest_csv.exists():
            raise ConfigError(f"{outdir} already contains a manifest. Remove OUT to rebuild.")

        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_index = 0
        written_batches: List[Path] = []

        with manifest_jsonl.open("w") as mj, manifest_csv.open("w", newline="") as mc_file:
            csv_writer = csv.DictWriter(mc_file, fieldnames=["id", "wandb_name", "tags", "params"])
            csv_writer.writeheader()

            selected_batch: List[Config] = []
            for cfg in self._space.iter_configs():
                selected_batch.append(cfg)
                if len(selected_batch) == batch_size:
                    path = self._flush_batch(selected_batch, batch_dir, batch_index, mj, csv_writer)
                    written_batches.append(path)
                    batch_index += 1
                    selected_batch = []
            if selected_batch:
                path = self._flush_batch(selected_batch, batch_dir, batch_index, mj, csv_writer)
                written_batches.append(path)

        print(f"Manifest written to: {manifest_jsonl}")
        print(f"Manifest written to: {manifest_csv}")
        return written_batches

    def _flush_batch(
        self,
        batch: List[Config],
        batch_dir: Path,
        batch_index: int,
        mj,  # JSONL file handle
        csv_writer: Optional[csv.DictWriter],
    ) -> Path:
        path = batch_dir / f"batch_{batch_index:03d}.txt"
        with path.open("w") as bf:
            for cfg in batch:
                bf.write(to_hydra_cli(cfg, self.cli_base) + "\n")
                entry = cfg.manifest_entry()
                mj.write(json.dumps(entry) + "\n")
                if csv_writer:
                    csv_writer.writerow(entry)
        print(f"Wrote {path} with {len(batch)} commands")
        return path

# =========================
# SLURM integration
# =========================

def write_slurm_script(
    job_name: str,
    batch_files: List[Path],
    *,
    time: str = "01:00:00",
    cpus_per_task: int = 1,
    partition: Optional[str] = None,
    gres: Optional[str] = None,
    account: Optional[str] = None,
    qos: Optional[str] = None,
    output_dir: Path = Path("slurm_logs"),
    launcher: Path = Path("run_tmux_launcher.sh"),
    k_per_session: int = 1,
) -> Path:
    """Create a SLURM job-array script that runs one batch file per array task via the tmux launcher."""
    if not batch_files:
        raise ValueError("No batch files provided for SLURM job-array.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arr_max = len(batch_files) - 1
    files_array = " ".join(sh_single_quote(str(p)) for p in batch_files)
    launcher_abs = Path(launcher).expanduser().resolve()

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{arr_max}",
        f"#SBATCH --nodes=1",
        f"#SBATCH --ntasks-per-node=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={time}",
        f"#SBATCH --output={output_dir}/{job_name}_%A_%a.out",
        f"#SBATCH --error={output_dir}/{job_name}_%A_%a.err",
    ]
    if partition: lines.append(f"#SBATCH --partition={partition}")
    if gres:      lines.append(f"#SBATCH --gres={gres}")
    if account:   lines.append(f"#SBATCH --account={account}")
    if qos:       lines.append(f"#SBATCH --qos={qos}")

    body = f"""
set -euo pipefail

micromamba activate pyt4f

SLURM_JOB_ID="${{SLURM_JOB_ID:-nojid}}"
SLURM_ARRAY_TASK_ID="${{SLURM_ARRAY_TASK_ID:-0}}"

files=({files_array})

idx="$SLURM_ARRAY_TASK_ID"
if (( idx < 0 || idx >= ${{#files[@]}} )); then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=$idx is out of range [0, $((${{#files[@]}}-1))]." >&2
  exit 2
fi

cmdfile="${{files[$idx]}}"
if [[ ! -f "$cmdfile" ]]; then
  echo "[ERROR] Command file not found: $cmdfile" >&2
  exit 3
fi

echo "[INFO] Job $SLURM_JOB_ID ArrayTask $SLURM_ARRAY_TASK_ID -> file: $cmdfile"

export TMUX_LOG_DIR="{output_dir}/{job_name}/tmux/${{SLURM_JOB_ID}}/${{SLURM_ARRAY_TASK_ID}}"
export TMUX_PREFIX="{job_name}_${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p "$TMUX_LOG_DIR"

LAUNCHER="{launcher_abs}"
if [[ ! -f "$LAUNCHER" ]]; then
  echo "[ERROR] Launcher not found at: $LAUNCHER" >&2
  exit 4
fi

bash "$LAUNCHER" "$cmdfile" {k_per_session} "$TMUX_PREFIX" "$TMUX_LOG_DIR"
"""
    script_path = output_dir / f"{job_name}.slurm"
    script_path.write_text("\n".join(lines) + "\n" + body)
    return script_path


def submit_sbatch(slurm_script: Path) -> None:
    print(f"[INFO] SLURM script at: {slurm_script}")
    subprocess.run(["sbatch", str(slurm_script)], check=True)


# =========================
# Local (no SLURM) launcher helpers
# =========================

def _ensure_executable(path: Path) -> List[str]:
    """Return argv to execute launcher (direct if executable, otherwise via bash)."""
    p = Path(path)
    print(f"[INFO] Using launcher: {p}")
    if p.is_file() and os.access(str(p), os.X_OK):
        return [str(p)]
    return ["bash", str(p)]


def _list_batch_files(outdir: str | Path) -> List[Path]:
    batch_dir = Path(outdir) / "batches"
    batch_files = sorted(batch_dir.glob("batch_*.txt"))
    if not batch_files:
        raise ConfigError(f"No existing batches found in {batch_dir}. Run 'build' first.")
    return [Path(p) for p in batch_files]


def _write_experiment_copy(yaml_path: str | Path, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "experiment.yaml").write_text(Path(yaml_path).read_text())


def _launch_local_batches(
    batch_files: List[Path],
    *,
    outdir: str | Path,
    launcher: str | Path,
    k_per_session: int,
    job_name: str,
    log_dir: Optional[str],
    cores: Optional[str],
) -> None:
    env = os.environ.copy()
    if cores:
        env["CORES"] = cores
    log_path = Path(log_dir) if log_dir else Path(outdir) / "tmux_logs"
    log_path.mkdir(parents=True, exist_ok=True)
    env["TMUX_LOG_DIR"] = str(log_path)

    base_prefix = job_name
    launcher_argv = _ensure_executable(Path(launcher))

    for i, bfile in enumerate(batch_files):
        prefix = f"{base_prefix}_b{i:03d}"
        env["TMUX_PREFIX"] = prefix

        argv = launcher_argv + [str(bfile), str(k_per_session), prefix, env["TMUX_LOG_DIR"]]

        print(f"[LOCAL] Launching batch {i} -> {bfile}")
        print(f"        K/core-session={k_per_session}, prefix={prefix}, log_dir={env['TMUX_LOG_DIR']}")
        if cores:
            print(f"        CORES={cores}")
        subprocess.run(argv, check=True, env=env)


def _submit_slurm_batches(
    batch_files: List[Path],
    *,
    outdir: str | Path,
    job_name: str,
    time: str,
    cpus_per_task: int,
    partition: Optional[str],
    gres: Optional[str],
    account: Optional[str],
    qos: Optional[str],
    slurm_logs: Optional[str],
    launcher: str | Path,
    k_per_session: int,
) -> None:
    slurm_logs_path = Path(slurm_logs) if slurm_logs else Path(outdir) / "slurm_logs"
    slurm_script = write_slurm_script(
        job_name=job_name,
        batch_files=batch_files,
        time=time,
        cpus_per_task=cpus_per_task,
        partition=partition,
        gres=gres,
        account=account,
        qos=qos,
        output_dir=slurm_logs_path,
        launcher=Path(launcher),
        k_per_session=k_per_session,
    )
    submit_sbatch(slurm_script)


# =========================
# CLI commands
# =========================


def cli_build(args: argparse.Namespace) -> None:
    eb = ExperimentBuilder.from_yaml(args.yaml)
    _write_experiment_copy(args.yaml, args.out)
    eb.build(
        outdir=args.out,
        batch_size=args.batch_size,
    )


def cli_local(args: argparse.Namespace) -> None:
    """
    Launch existing batches locally via the tmux launcher.
    This command will NOT build batches; it requires OUT/batches to already exist.
    """
    batch_files = _list_batch_files(args.out)
    _launch_local_batches(
        batch_files,
        outdir=args.out,
        launcher=args.launcher,
        k_per_session=args.k_per_session,
        job_name=args.job_name,
        log_dir=args.log_dir,
        cores=args.cores,
    )

    print("[LOCAL] Done.")


def cli_slurm(args: argparse.Namespace) -> None:
    """
    Build a SLURM array script that references existing batch files (OUT/batches).
    This command will NOT build batches; it requires OUT/batches to already exist.
    """
    batch_files = _list_batch_files(args.out)
    _submit_slurm_batches(
        batch_files,
        outdir=args.out,
        job_name=args.job_name,
        time=args.time,
        cpus_per_task=args.cpus_per_task,
        partition=args.partition,
        gres=args.gres,
        account=args.account,
        qos=args.qos,
        slurm_logs=args.slurm_logs,
        launcher=args.launcher,
        k_per_session=args.k_per_session,
    )


def cli_run(args: argparse.Namespace) -> None:
    """One-step build + launch."""
    eb = ExperimentBuilder.from_yaml(args.yaml)
    _write_experiment_copy(args.yaml, args.out)
    eb.build(outdir=args.out, batch_size=args.batch_size)
    batch_files = _list_batch_files(args.out)

    if args.mode == "local":
        _launch_local_batches(
            batch_files,
            outdir=args.out,
            launcher=args.launcher,
            k_per_session=args.k_per_session,
            job_name=args.job_name,
            log_dir=args.log_dir,
            cores=args.cores,
        )
        print("[RUN] Local launch complete.")
        return

    _submit_slurm_batches(
        batch_files,
        outdir=args.out,
        job_name=args.job_name,
        time=args.time,
        cpus_per_task=args.cpus_per_task,
        partition=args.partition,
        gres=args.gres,
        account=args.account,
        qos=args.qos,
        slurm_logs=args.slurm_logs,
        launcher=args.launcher,
        k_per_session=args.k_per_session,
    )
    print("[RUN] Slurm submission complete.")


# =========================
# CLI glue
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="expgen", description="Experiment batch generator + local/SLURM launch.")
    sub = p.add_subparsers(dest="cmd", required=True)

    build_common = argparse.ArgumentParser(add_help=False)
    build_common.add_argument("--yaml", required=True, help="Experiment YAML (cli_base + packs).")
    build_common.add_argument("--out", required=True, help="Output directory.")
    build_common.add_argument("--batch-size", type=int, required=True, help="Commands per batch file.")

    pb = sub.add_parser("build", parents=[build_common], help="Generate batches + manifests for full experiment.")
    pb.set_defaults(func=cli_build)

    # run (build + launch)
    prun = sub.add_parser("run", parents=[build_common], help="Build full experiment and launch locally or on SLURM.")
    prun.add_argument("--mode", choices=["local", "slurm"], required=True, help="Where to run after build.")
    prun.add_argument("--launcher", default="run_tmux_launcher.sh", help="Path to tmux launcher script.")
    prun.add_argument("--k-per-session", type=int, default=1, help="CPU cores per tmux session.")
    prun.add_argument("--job-name", required=True, help="Name stem for tmux sessions / SLURM job.")
    prun.add_argument("--log-dir", default=None, help="Local tmux logs dir (default: OUT/tmux_logs).")
    prun.add_argument("--cores", default=None, help='Local CPU set, e.g. "0-31,48-63".')
    prun.add_argument("--time", default="02:00:00", help="SLURM time, e.g., 02:00:00.")
    prun.add_argument("--cpus-per-task", type=int, default=16)
    prun.add_argument("--partition", help="SLURM partition")
    prun.add_argument("--gres", help="SLURM GRES, e.g., gpu:1")
    prun.add_argument("--account", help="SLURM account")
    prun.add_argument("--qos", help="SLURM QoS")
    prun.add_argument("--slurm-logs", default=None, help="SLURM logs dir (default: OUT/slurm_logs).")
    prun.set_defaults(func=cli_run)

    # slurm
    ps = sub.add_parser("slurm", help="Create and submit SLURM array from existing batches (no build).")
    ps.add_argument("--out", required=True, help="Output directory containing batches/.")
    ps.add_argument("--job-name", required=True, help="SLURM job name prefix.")
    ps.add_argument("--time", default="02:00:00", help="SLURM time, e.g., 02:00:00.")
    ps.add_argument("--cpus-per-task", type=int, default=16)
    ps.add_argument("--partition", help="SLURM partition")
    ps.add_argument("--gres", help="SLURM GRES, e.g., gpu:1")
    ps.add_argument("--account", help="SLURM account")
    ps.add_argument("--qos", help="SLURM QoS")
    ps.add_argument("--slurm-logs", default=None, help="Directory for SLURM stdout/err and script (default: OUT/slurm_logs).")
    ps.add_argument("--launcher", default="run_tmux_launcher.sh", help="Path to tmux launcher script.")
    ps.add_argument("--k-per-session", type=int, default=1, help="CPU cores per tmux session on the node.")
    ps.set_defaults(func=cli_slurm)

    # local
    pl = sub.add_parser("local", help="Launch existing batches locally via tmux (no SLURM).")
    pl.add_argument("--out", required=True, help="Output directory containing batches/.")
    pl.add_argument("--launcher", default="run_tmux_launcher.sh", help="Path to tmux launcher script.")
    pl.add_argument("--k-per-session", type=int, default=1, help="CPU cores per tmux session.")
    pl.add_argument("--job-name", default="localrun", help="Name stem for tmux session prefixes.")
    pl.add_argument("--log-dir", default=None, help="Directory for per-session logs (default: OUT/tmux_logs).")
    pl.add_argument("--cores", default=None, help='CPU set for the node, e.g. "0-31,48-63".')
    pl.set_defaults(func=cli_local)

    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ConfigError as ce:
        print(f"[config error] {ce}")
        raise SystemExit(2)
    except Exception as e:
        print(f"[error] {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
