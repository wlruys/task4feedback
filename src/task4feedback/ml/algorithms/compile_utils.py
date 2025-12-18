
from torchrl._utils import compile_with_warmup
from typing import Any, Callable
import functools
import warnings
from task4feedback.logging import training


def _safe_compile_with_warmup(
    obj: Callable[..., Any],
    *,
    warmup: int,
    label: str = "",
    **compile_kwargs: Any,
) -> Callable[..., Any]:
    compiled = compile_with_warmup(obj, warmup=warmup, **compile_kwargs)

    @functools.wraps(obj)
    def wrapped(*args: Any, **kwargs: Any):
        nonlocal compiled
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            dynamo_exc: tuple[type[BaseException], ...] = ()
            try:
                import torch._dynamo.exc as _dynamo_exc

                dynamo_exc = (
                    _dynamo_exc.InternalTorchDynamoError,
                    _dynamo_exc.Unsupported,
                    RecursionError,
                )
            except Exception:
                dynamo_exc = (RecursionError,)

            if dynamo_exc and not isinstance(exc, dynamo_exc):
                raise

            exc_summary = str(exc).splitlines()[0] if str(exc) else ""
            training.warning(
                "torch.compile failed for %s; falling back to eager (%s: %s)",
                label,
                type(exc).__name__,
                exc_summary,
            )
            compiled = obj
            return obj(*args, **kwargs)

    return wrapped