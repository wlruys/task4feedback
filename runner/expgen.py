#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

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


def read_existing_hashes(manifest_jsonl: Path) -> set[str]:
    ids: set[str] = set()
    if manifest_jsonl.exists():
        with manifest_jsonl.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    ids.add(rec["id"])
                except Exception:
                    continue
    return ids


def next_batch_start_index(batch_dir: Path) -> int:
    """Continue numbering if batch_###.txt already exist."""
    if not batch_dir.exists():
        return 0
    existing = sorted(p for p in batch_dir.glob("batch_*.txt") if p.is_file())
    return len(existing)


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
    def __init__(self, packs: Sequence[Pack], strict: bool = True):
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
            provenance: Dict[str, Tuple[str, int, Dict[str, Any]]] = {}

            for pack_idx, opt in enumerate(combo):
                if opt.tag:
                    tags.append(opt.tag)
                opt_params = opt.params()
                for k, v in opt_params.items():
                    if k in merged:
                        if self._strict or merged[k] != v:
                            prev_pack, prev_i, prev_payload = provenance[k]
                            curr_pack, curr_i = pack_names[pack_idx], self._option_index(option_lists[pack_idx], opt)
                            raise ConfigError(
                                "Parameter conflict detected:\n"
                                f"  key: {k}\n"
                                f"  previous: value={merged[k]!r} from pack='{prev_pack}' option_index={prev_i} option={prev_payload}\n"
                                f"  current : value={v!r} from pack='{curr_pack}' option_index={curr_i} option={opt.payload}\n"
                                "Resolve by adjusting packs/options or disable strict mode if values are identical."
                            )
                    else:
                        merged[k] = v
                        provenance[k] = (pack_names[pack_idx], self._option_index(option_lists[pack_idx], opt), opt.payload)

            yield Config(merged, tags)

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
    def __init__(self, cli_base: str, packs: Sequence[Pack], strict: bool = True):
        self.cli_base = str(cli_base)
        self._strict = bool(strict)
        self._packs = self._merge_same_name_packs(packs)
        self._space = ConfigSpace(self._packs, strict=self._strict)

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
    def from_yaml(cls, path: str | Path, strict: bool = True) -> "ExperimentBuilder":
        try:
            data = yaml.safe_load(Path(path).read_text())
        except Exception as e:
            raise ConfigError(f"Failed to parse YAML '{path}': {e}") from e
        validate_yaml_structure(data)
        packs = [Pack.from_dict(p["name"], p["options"]) for p in data["packs"]]
        return cls(cli_base=data["cli_base"], packs=packs, strict=strict)

    def build(
        self,
        outdir: str | Path,
        batch_size: int,
        *,
        only_hashes: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        skip_existing: bool = False,
        write_csv: bool = True,
    ) -> List[Path]:
        """Generate batch files and manifest. Return list of batch file paths."""
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        outdir = Path(outdir)
        batch_dir = outdir / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        manifest_jsonl = outdir / "manifest.jsonl"
        manifest_csv = outdir / "manifest.csv"

        allow_ids: Optional[set[str]] = set(only_hashes) if only_hashes else None
        require_tags: Optional[set[str]] = set(tags) if tags else None
        existing_ids: set[str] = read_existing_hashes(manifest_jsonl) if skip_existing else set()

        batch_index = next_batch_start_index(batch_dir)
        mj_mode = "a" if manifest_jsonl.exists() else "w"
        wrote_any = False
        written_batches: List[Path] = []

        with manifest_jsonl.open(mj_mode) as mj:
            csv_writer = None
            mc_file = None
            try:
                if write_csv:
                    mc_mode = "a" if manifest_csv.exists() else "w"
                    mc_file = manifest_csv.open(mc_mode, newline="")
                    csv_writer = csv.DictWriter(mc_file, fieldnames=["id", "wandb_name", "tags", "params"])
                    if mc_mode == "w":
                        csv_writer.writeheader()

                selected_batch: List[Config] = []
                for cfg in self._space.iter_configs():
                    if allow_ids and cfg.id not in allow_ids:
                        continue
                    if require_tags and not require_tags.issubset(set(cfg.tags)):
                        continue
                    if skip_existing and cfg.id in existing_ids:
                        continue
                    selected_batch.append(cfg)
                    if len(selected_batch) == batch_size:
                        path = self._flush_batch(selected_batch, batch_dir, batch_index, mj, csv_writer)
                        written_batches.append(path)
                        wrote_any = True
                        batch_index += 1
                        selected_batch = []
                if selected_batch:
                    path = self._flush_batch(selected_batch, batch_dir, batch_index, mj, csv_writer)
                    written_batches.append(path)
                    wrote_any = True
            finally:
                if mc_file is not None:
                    mc_file.close()

        if not wrote_any:
            msg = "No matching configurations to write."
            if only_hashes: msg += " (filtered by hashes)"
            if tags: msg += " (filtered by tags)"
            if skip_existing: msg += " (skipped existing IDs from manifest)"
            print(msg)
        else:
            print(f"Manifest written to: {manifest_jsonl}")
            if write_csv:
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

    def rerun(self, outdir: str | Path, batch_size: int, *, hashes=None, tags=None, write_csv: bool = True) -> List[Path]:
        return self.build(outdir, batch_size, only_hashes=hashes, tags=tags, write_csv=write_csv)

    def extend(self, outdir: str | Path, batch_size: int, *, new_packs: Sequence[Pack], write_csv: bool = True) -> List[Path]:
        merged = self._merge_same_name_packs(self._packs + list(new_packs))
        extended = ExperimentBuilder(self.cli_base, merged, strict=self._strict)
        return extended.build(outdir=outdir, batch_size=batch_size, skip_existing=True, write_csv=write_csv)


# =========================
# SLURM integration
# =========================

def write_slurm_script(
    job_name: str,
    batch_files: List[Path],
    *,
    time: str = "01:00:00",
    nodes: int = 1,
    ntasks_per_node: int = 1,
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

    output_dir.mkdir(parents=True, exist_ok=True)

    arr_max = len(batch_files) - 1
    files_array = " ".join(sh_single_quote(str(p)) for p in batch_files)
    log_dir = output_dir / job_name
    log_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array=0-{arr_max}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={ntasks_per_node}",
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

files=({files_array})
cmdfile="${{files[$SLURM_ARRAY_TASK_ID]}}"

echo "[INFO] Job $SLURM_JOB_ID ArrayTask $SLURM_ARRAY_TASK_ID -> file: $cmdfile"

export TMUX_LOG_DIR={sh_single_quote(str(log_dir))}
export TMUX_PREFIX={sh_single_quote(job_name + "_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}")}

bash {sh_single_quote(str(launcher))} "$cmdfile" {k_per_session} "$TMUX_PREFIX" "$TMUX_LOG_DIR"
"""
    script_path = output_dir / f"{job_name}.slurm"
    script_path.write_text("\n".join(lines) + "\n" + body)
    return script_path


def submit_sbatch(slurm_script: Path, dry_run: bool = False) -> None:
    print(f"[INFO] SLURM script at: {slurm_script}")
    if dry_run:
        print("[DRY-RUN] sbatch", slurm_script)
    else:
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


# =========================
# CLI commands
# =========================

def _read_hashes_arg(hashes: Optional[List[str]], hashes_file: Optional[str]) -> Optional[List[str]]:
    acc: List[str] = []
    if hashes:
        acc.extend(hashes)
    if hashes_file:
        for line in Path(hashes_file).read_text().splitlines():
            s = line.strip()
            if s:
                acc.append(s)
    return acc or None


def cli_build(args: argparse.Namespace) -> None:
    eb = ExperimentBuilder.from_yaml(args.yaml, strict=not args.nonstrict)
    eb.build(
        outdir=args.out,
        batch_size=args.batch_size,
        skip_existing=args.skip_existing,
        write_csv=not args.no_csv,
    )


def cli_rerun(args: argparse.Namespace) -> None:
    eb = ExperimentBuilder.from_yaml(args.yaml, strict=not args.nonstrict)
    hashes = _read_hashes_arg(args.hashes, args.hashes_file)
    eb.rerun(
        outdir=args.out,
        batch_size=args.batch_size,
        hashes=hashes,
        tags=args.tags,
        write_csv=not args.no_csv,
    )


def cli_extend(args: argparse.Namespace) -> None:
    base = ExperimentBuilder.from_yaml(args.yaml, strict=not args.nonstrict)
    data = yaml.safe_load(Path(args.new).read_text())
    validate_yaml_structure(data)
    new_packs = [Pack.from_dict(p["name"], p["options"]) for p in data["packs"]]
    base.extend(outdir=args.out, batch_size=args.batch_size, new_packs=new_packs, write_csv=not args.no_csv)


def cli_local(args: argparse.Namespace) -> None:
    """
    Launch existing batches locally via the tmux launcher.
    This command will NOT build batches; it requires OUT/batches to already exist.
    """
    batch_dir = Path(args.out) / "batches"
    batch_files = sorted(batch_dir.glob("batch_*.txt"))
    if not batch_files:
        raise ConfigError(f"No existing batches found in {batch_dir}. Run 'build' first.")

    if args.batch_index is not None:
        try:
            batch_files = [batch_files[args.batch_index]]
        except IndexError:
            raise ConfigError(f"--batch-index {args.batch_index} out of range (0..{len(batch_files)-1})")

    env = os.environ.copy()
    if args.cores:
        env["CORES"] = args.cores
    if args.log_dir:
        env["TMUX_LOG_DIR"] = args.log_dir

    base_prefix = args.prefix or args.job_name
    launcher_argv = _ensure_executable(Path(args.launcher))

    for i, bfile in enumerate(batch_files):
        prefix = f"{base_prefix}_b{i:03d}"
        env["TMUX_PREFIX"] = prefix

        argv = launcher_argv + [str(bfile), str(args.k_per_session), prefix, env.get("TMUX_LOG_DIR", "logs/tmux_sessions")]

        print(f"[LOCAL] Launching batch {i} -> {bfile}")
        print(f"        K/core-session={args.k_per_session}, prefix={prefix}, log_dir={env.get('TMUX_LOG_DIR','logs/tmux_sessions')}")
        if args.cores:
            print(f"        CORES={args.cores}")

        if args.dry_run:
            print("        (dry-run) would exec:", " ".join(argv))
            continue

        subprocess.run(argv, check=True, env=env)

    print("[LOCAL] Done.")


def cli_slurm(args: argparse.Namespace) -> None:
    """
    Build a SLURM array script that references existing batch files (OUT/batches).
    This command will NOT build batches; it requires OUT/batches to already exist.
    """
    batch_dir = Path(args.out) / "batches"
    batch_files = sorted(batch_dir.glob("batch_*.txt"))
    if not batch_files:
        raise ConfigError(f"No existing batches found in {batch_dir}. Run 'build' first.")

    batch_files = [Path(p) for p in batch_files]

    slurm_script = write_slurm_script(
        job_name=args.job_name,
        batch_files=batch_files,
        time=args.time,
        nodes=args.nodes,
        ntasks_per_node=args.ntasks_per_node,
        cpus_per_task=args.cpus_per_task,
        partition=args.partition,
        gres=args.gres,
        account=args.account,
        qos=args.qos,
        output_dir=Path(args.slurm_logs),
        launcher=Path(args.launcher),
        k_per_session=args.k_per_session,
    )
    submit_sbatch(slurm_script, dry_run=args.dry_run)


def cli_clean(args: argparse.Namespace) -> None:
    """
    Remove batches/ and manifest files under OUT.
    Minimal, deliberate deletion: only OUT/batches, OUT/manifest.jsonl, OUT/manifest.csv.
    """
    outdir = Path(args.out)
    batch_dir = outdir / "batches"
    manifest_jsonl = outdir / "manifest.jsonl"
    manifest_csv = outdir / "manifest.csv"

    if batch_dir.exists():
        shutil.rmtree(batch_dir)
        print(f"[CLEAN] Removed {batch_dir}")
    for f in [manifest_jsonl, manifest_csv]:
        if f.exists():
            f.unlink()
            print(f"[CLEAN] Removed {f}")


# =========================
# CLI glue
# =========================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="expgen", description="Experiment batch generator + local/SLURM launch.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--yaml", required=True, help="Experiment YAML (cli_base + packs).")
    common.add_argument("--out", required=True, help="Output directory.")
    common.add_argument("--batch-size", type=int, required=True, help="Commands per batch file.")
    common.add_argument("--no-csv", action="store_true", help="Do not write manifest.csv (only JSONL).")
    common.add_argument("--nonstrict", action="store_true", help="Allow equal-value overlaps (fail on unequal).")

    # build
    pb = sub.add_parser("build", parents=[common], help="Generate batches + manifest.")
    pb.add_argument("--skip-existing", action="store_true", help="Skip configs already in manifest.jsonl.")
    pb.set_defaults(func=cli_build)

    # rerun
    pr = sub.add_parser("rerun", parents=[common], help="Generate rerun batches (by hashes/tags).")
    pr.add_argument("--hashes", nargs="*", help="Only include configs with these hashes.")
    pr.add_argument("--hashes-file", help="File with hashes, one per line.")
    pr.add_argument("--tags", nargs="*", help="Require configs to include all these tags.")
    pr.set_defaults(func=cli_rerun)

    # extend
    pe = sub.add_parser("extend", parents=[common], help="Extend sweep with new packs/options (merge by name).")
    pe.add_argument("--new", required=True, help="YAML containing additional packs/options.")
    pe.set_defaults(func=cli_extend)

    # slurm
    ps = sub.add_parser("slurm", parents=[common], help="Create SLURM array script from existing batches (no build).")
    ps.add_argument("--use-existing-batches", action="store_true", help=argparse.SUPPRESS)  # kept for backward-compat; ignored
    ps.add_argument("--skip-existing", action="store_true", help=argparse.SUPPRESS)
    ps.add_argument("--hashes", nargs="*", help=argparse.SUPPRESS)
    ps.add_argument("--hashes-file", help=argparse.SUPPRESS)
    ps.add_argument("--tags", nargs="*", help=argparse.SUPPRESS)
    ps.add_argument("--job-name", required=True, help="SLURM job name prefix.")
    ps.add_argument("--time", default="02:00:00", help="SLURM time, e.g., 02:00:00.")
    ps.add_argument("--nodes", type=int, default=1)
    ps.add_argument("--ntasks-per-node", type=int, default=1)
    ps.add_argument("--cpus-per-task", type=int, default=16)
    ps.add_argument("--partition", help="SLURM partition")
    ps.add_argument("--gres", help="SLURM GRES, e.g., gpu:1")
    ps.add_argument("--account", help="SLURM account")
    ps.add_argument("--qos", help="SLURM QoS")
    ps.add_argument("--slurm-logs", default="slurm_logs", help="Directory for SLURM stdout/err and script.")
    ps.add_argument("--launcher", default="run_tmux_launcher.sh", help="Path to tmux launcher script.")
    ps.add_argument("--k-per-session", type=int, default=1, help="CPU cores per tmux session on the node.")
    ps.add_argument("--dry-run", action="store_true", help="Show sbatch command but do not submit.")
    ps.set_defaults(func=cli_slurm)

    # local
    pl = sub.add_parser("local", parents=[common], help="Launch existing batches locally via tmux (no SLURM).")
    pl.add_argument("--use-existing-batches", action="store_true", help=argparse.SUPPRESS)  # kept for backward-compat; ignored
    pl.add_argument("--skip-existing", action="store_true", help=argparse.SUPPRESS)
    pl.add_argument("--hashes", nargs="*", help=argparse.SUPPRESS)
    pl.add_argument("--hashes-file", help=argparse.SUPPRESS)
    pl.add_argument("--tags", nargs="*", help=argparse.SUPPRESS)
    pl.add_argument("--launcher", default="run_tmux_launcher.sh", help="Path to tmux launcher script.")
    pl.add_argument("--k-per-session", type=int, default=1, help="CPU cores per tmux session.")
    pl.add_argument("--prefix", default=None, help="TMUX session prefix (default: job-name).")
    pl.add_argument("--job-name", default="localrun", help="Name stem for default prefix.")
    pl.add_argument("--log-dir", default="logs/tmux_sessions", help="Directory for per-session logs.")
    pl.add_argument("--cores", default=None, help='CPU set for the node, e.g. "0-31,48-63".')
    pl.add_argument("--batch-index", type=int, default=None, help="Only launch a single batch by index (0-based).")
    pl.add_argument("--dry-run", action="store_true", help="Print launcher command(s) but do not execute.")
    pl.set_defaults(func=cli_local)

    # clean
    pc = sub.add_parser("clean", help="Remove all batches and manifest files in OUT directory.")
    pc.add_argument("--out", required=True, help="Output directory to clean.")
    pc.set_defaults(func=cli_clean)

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