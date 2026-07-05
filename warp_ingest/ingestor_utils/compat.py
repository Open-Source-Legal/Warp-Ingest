"""Small self-contained helpers vendored from the legacy ``nlm_utils`` package.

The historical build depended on the internal ``nlm_utils`` wheel solely for a
handful of trivial utilities (``ensure_bool``, ``generate_version`` and
``file_utils.extract_file_properties``).  Importing ``nlm_utils`` pulled in a
large server-side dependency tree (openai, tiktoken, redis, pymongo, minio,
pinned numpy==1.24.4, ...) that the document-parsing path never touches.

To keep this library a clean, standalone, pure-Python package these helpers are
re-implemented here with no third-party requirements beyond the standard
library (``magic`` is used only as an optional fallback).
"""

import hashlib
import mimetypes
import os
import re
import unicodedata
from datetime import datetime

__all__ = [
    "ensure_bool",
    "ensure_float",
    "ensure_integer",
    "generate_version",
    "extract_file_properties",
    "guess_mime_type",
    "to_ascii",
]


# --------------------------------------------------------------------------- #
# permissive ASCII transliteration
# Replacement for the GPL-licensed ``unidecode`` package (copyleft, disallowed).
# Uses only the Python standard library (PSF licensed): NFKD-decompose, drop
# combining marks, then keep ASCII.  For the Latin-accented text this codebase
# deals with this is equivalent to unidecode (café -> cafe, Zürich -> Zurich).
# --------------------------------------------------------------------------- #
def to_ascii(text):
    if not text:
        return text
    return (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
    )


# --------------------------------------------------------------------------- #
# scalar casting helpers
# --------------------------------------------------------------------------- #
def ensure_bool(value):
    if isinstance(value, bool):
        return value
    elif isinstance(value, str):
        value = value.lower()
        if value in ("true", "false"):
            return value == "true"
        elif value.isdigit():
            return value != "0"
    elif isinstance(value, int):
        return value != 0
    elif value is None:
        return False
    raise ValueError(f"Can not cast {type(value)}:'{value}' to boolean")


def ensure_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Can not cast {type(value)}:'{value}' to float")


def ensure_integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Can not cast {type(value)}:'{value}' to integer")


# --------------------------------------------------------------------------- #
# deterministic version hashing (content hash of the source tree)
# Adapted from checksumdir (MIT) but using only hashlib so there is no xxhash
# dependency.  Nothing depends on the exact hash value, only its determinism.
# --------------------------------------------------------------------------- #
def _filehash(filepath, hashfunc):
    hasher = hashfunc()
    blocksize = 64 * 1024
    if not os.path.exists(filepath):
        return hasher.hexdigest()
    with open(filepath, "rb") as fp:
        while True:
            data = fp.read(blocksize)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _dirhash(dirname, hashfunc, excluded_extensions=("pyc",), ignore_hidden=True):
    hashvalues = []
    if not os.path.isdir(dirname):
        return hashvalues
    for root, dirs, files in os.walk(dirname, topdown=True):
        if ignore_hidden and re.search(r"/\.", root):
            continue
        dirs.sort()
        files.sort()
        for fname in files:
            if ignore_hidden and fname.startswith("."):
                continue
            if fname.split(".")[-1:][0] in excluded_extensions:
                continue
            hashvalues.append(_filehash(os.path.join(root, fname), hashfunc))
    return hashvalues


def generate_version(paths, main_version=None, version_file="version.txt"):
    hashfunc = hashlib.md5
    if main_version is None:
        main_version = "0.0.0"
        try:
            for vf in [os.path.join(p, version_file) for p in paths] + [version_file]:
                with open(vf) as f:
                    main_version = f.read().strip()
                break
        except FileNotFoundError:
            pass
    if isinstance(paths, str):
        paths = [paths]
    hashvalues = []
    for path in paths:
        hashvalues += _dirhash(path, hashfunc)
    reducer = hashfunc()
    for hv in sorted(hashvalues):
        reducer.update(hv.encode("utf-8"))
    return f"{main_version}+{reducer.hexdigest()}"


# --------------------------------------------------------------------------- #
# file property extraction (extension-first, pure-python fallbacks)
# --------------------------------------------------------------------------- #
_EXT_MIME = {
    ".md": "text/x-markdown",
    ".markdown": "text/x-markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
    ".xml": "text/xml",
    ".txt": "text/plain",
}


def guess_mime_type(filepath):
    _, ext = os.path.splitext(filepath.lower())
    if ext in _EXT_MIME:
        return _EXT_MIME[ext]
    # optional libmagic content sniffing if available
    try:
        import magic  # type: ignore

        return magic.from_file(filepath, mime=True)
    except Exception:
        pass
    mime, _ = mimetypes.guess_type(filepath)
    return mime or "application/octet-stream"


def get_file_sha256(filepath):
    sha2 = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(131072), b""):
            sha2.update(chunk)
    return sha2.hexdigest()


def extract_file_properties(filepath):
    return {
        "fileSize": os.path.getsize(filepath),
        "mimeType": guess_mime_type(filepath),
        "checksum": get_file_sha256(filepath),
        "createdOn": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "isDeleted": False,
    }
