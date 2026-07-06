"""Package metadata guards for optional install profiles."""

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


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
