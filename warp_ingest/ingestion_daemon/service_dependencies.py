"""Helpers for optional service-runtime dependencies."""

import importlib
from collections.abc import Callable, Sequence
from types import ModuleType

SERVICE_EXTRA_INSTALL_HINT = (
    "Warp-Ingest service dependencies are not installed. "
    'Install them with `pip install "warp-ingest[service]"` '
    'or `pip install "warp-ingest[all]"`.'
)


class MissingServiceExtraError(RuntimeError):
    """Raised when the service extra is needed but not installed."""


def _missing_service_dependency(package_name: str) -> MissingServiceExtraError:
    return MissingServiceExtraError(
        f"{SERVICE_EXTRA_INSTALL_HINT} Missing dependency: {package_name}."
    )


def require_service_dependency(
    module_name: str,
    package_name: str | None = None,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    """Import a service dependency or raise an install-extra error.

    Only the top-level requested module is converted to the install-extra error;
    missing transitive imports are left untouched because they indicate a broken
    or incompatible installation rather than a missing optional extra.
    """
    try:
        return importer(module_name)
    except ModuleNotFoundError as exc:
        root_module = module_name.partition(".")[0]
        if exc.name == root_module:
            raise _missing_service_dependency(package_name or root_module) from None
        raise


def require_any_service_dependency(
    module_names: Sequence[str],
    package_name: str,
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> ModuleType:
    """Import the first available module for a service dependency package."""
    last_missing = None
    for module_name in module_names:
        try:
            return require_service_dependency(module_name, package_name, importer)
        except MissingServiceExtraError as exc:
            last_missing = exc
    if last_missing is not None:
        raise last_missing
    raise ValueError("module_names must not be empty")
