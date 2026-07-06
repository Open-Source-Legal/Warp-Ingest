"""Package metadata guards for optional install profiles."""

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from warp_ingest.ingestion_daemon import __main__ as service_main
from warp_ingest.ingestion_daemon.service_dependencies import (
    MissingServiceExtraError,
    require_service_dependency,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _metadata():
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def _requirement_names(requirements):
    names = set()
    for requirement in requirements:
        name = re.split(r"\s*(?:\[|[<>=!~; ])", requirement, maxsplit=1)[0]
        names.add(name.lower().replace("_", "-"))
    return names


def test_base_install_is_parser_only():
    project = _metadata()["project"]
    base = _requirement_names(project["dependencies"])

    assert {
        "beautifulsoup4",
        "nltk",
        "numpy",
        "pdfplumber",
        "pypdfium2",
    } <= base
    assert {"fastapi", "python-multipart", "uvicorn"}.isdisjoint(base)
    assert project["optional-dependencies"]["parser"] == []


def test_service_and_all_extras_are_explicit_profiles():
    extras = _metadata()["project"]["optional-dependencies"]
    service = _requirement_names(extras["service"])
    ocr = _requirement_names(extras["ocr"])
    all_extra = _requirement_names(extras["all"])

    assert {"fastapi", "python-multipart", "uvicorn"} <= service
    assert service <= all_extra
    assert ocr <= all_extra
    assert "pymupdf4llm" not in all_extra


def test_missing_service_dependency_has_extra_install_hint():
    def missing_dependency(_module_name):
        raise ModuleNotFoundError(name="uvicorn")

    with pytest.raises(MissingServiceExtraError) as exc:
        require_service_dependency("uvicorn", importer=missing_dependency)

    message = str(exc.value)
    assert 'pip install "warp-ingest[service]"' in message
    assert 'pip install "warp-ingest[all]"' in message
    assert "Missing dependency: uvicorn." in message


def test_transitive_import_errors_are_not_masked():
    def missing_transitive(_module_name):
        raise ModuleNotFoundError(name="h11")

    with pytest.raises(ModuleNotFoundError) as exc:
        require_service_dependency("uvicorn", importer=missing_transitive)

    assert exc.value.name == "h11"


def test_service_launcher_exits_with_install_hint(monkeypatch):
    def missing_service_dependency(_module_name, package_name=None):
        raise MissingServiceExtraError(
            f"install service extra for {package_name or _module_name}"
        )

    monkeypatch.setattr(
        service_main,
        "require_service_dependency",
        missing_service_dependency,
    )

    with pytest.raises(SystemExit) as exc:
        service_main.main()

    assert str(exc.value) == "install service extra for uvicorn"
