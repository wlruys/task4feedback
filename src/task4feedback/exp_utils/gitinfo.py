from .definitions import *
from hydra.experimental.callbacks import Callback
from omegaconf import DictConfig
from pathlib import Path
import git

from .logging_helpers import get_helper_logger


logger = get_helper_logger(__name__)


class GitInfo(Callback):
    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        try:
            repo = git.Repo(search_parent_directories=True)
            outdir = Path(config.hydra.runtime.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "git_sha.txt").write_text(repo.head.commit.hexsha)
            (outdir / "git_dirty.txt").write_text(str(repo.is_dirty()))
            diff = repo.git.diff(None)
            (outdir / "git_diff.patch").write_text(diff)

            logger.info(
                "Git SHA: %s %s",
                repo.head.commit.hexsha,
                "(dirty)" if repo.is_dirty() else "(clean)",
            )

        except Exception as e:
            logger.warning("GitInfo callback failed: %s", e)
