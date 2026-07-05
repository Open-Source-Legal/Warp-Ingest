"""CPU-aware service settings.

The parse workload is CPU-bound pure Python, and three layers of parallelism
share the same cores: the uvicorn web workers (each running one parse at a
time), the front-end page-striping pool inside each web worker
(``WARP_FE_WORKERS``), and onnxruntime's intra-op threads during OCR
(``WARP_OCR_THREADS``).  The launcher computes one budget at start-up from the
CPUs actually available to the process — CPU affinity *and* the cgroup quota,
so a docker ``--cpus`` / K8s ``limits.cpu`` container on a big node budgets for
its slice, not the node — and exports it to the workers via the environment.
A user-set environment variable always wins over the computed default.

Defaults (``W`` = web workers):

- ``W = cpus`` — one CPU-bound worker per core is the throughput maximum under
  saturation; the kernel balances accepted connections across workers.
- ``FE = max(1, min(8, cpus // W))`` — 1 when ``W == cpus`` (all parallelism is
  across requests); pinning ``W`` low hands the slack to page striping instead
  (latency-oriented deploys).
- ``OCR threads = max(1, cpus // (W * FE))`` — an onnx session never spins more
  runnable threads than its share of the box.
"""

import logging
import math
import os
from dataclasses import asdict, dataclass

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5001


def _cgroup_cpu_quota():
    """CPU quota (in cores, float) from cgroup v2 or v1, or None if unlimited."""
    try:  # cgroup v2
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota_s, period_s = fh.read().split()[:2]
        if quota_s != "max" and int(quota_s) > 0 and int(period_s) > 0:
            return int(quota_s) / int(period_s)
        return None
    except (OSError, ValueError, IndexError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read())
        if quota > 0 and period > 0:
            return quota / period
    except (OSError, ValueError):
        pass
    return None


def effective_cpu_count():
    """CPUs actually available to this process: affinity capped by cgroup quota."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    quota = _cgroup_cpu_quota()
    if quota is not None:
        cpus = min(cpus, math.ceil(quota))
    return max(1, cpus)


def _env_int(env, key, default, minimum=1):
    raw = env.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return max(minimum, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid %s=%r; using %r", key, raw, default)
        return default


@dataclass(frozen=True)
class ServiceSettings:
    cpus: int
    web_workers: int
    fe_workers: int
    ocr_threads: int
    parse_slots: int
    host: str
    port: int

    def as_dict(self):
        return asdict(self)


def compute_settings(cpus=None, env=None):
    """Resolve the full service budget; pure given explicit *cpus* and *env*."""
    env = os.environ if env is None else env
    cpus = effective_cpu_count() if cpus is None else max(1, int(cpus))
    web_workers = _env_int(
        env, "WARP_WEB_WORKERS", _env_int(env, "WEB_CONCURRENCY", cpus)
    )
    fe_workers = _env_int(env, "WARP_FE_WORKERS", max(1, min(8, cpus // web_workers)))
    ocr_threads = _env_int(
        env, "WARP_OCR_THREADS", max(1, cpus // (web_workers * fe_workers))
    )
    parse_slots = _env_int(env, "WARP_WORKER_PARSE_SLOTS", 1)
    host = env.get("WARP_HOST", DEFAULT_HOST)
    port = _env_int(env, "WARP_PORT", _env_int(env, "PORT", DEFAULT_PORT))
    return ServiceSettings(
        cpus=cpus,
        web_workers=web_workers,
        fe_workers=fe_workers,
        ocr_threads=ocr_threads,
        parse_slots=parse_slots,
        host=host,
        port=port,
    )
