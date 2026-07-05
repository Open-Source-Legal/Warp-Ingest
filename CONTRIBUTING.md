## Maintainers
The key maintainers of this codebase are:
1. Ambika Sukla (@ansukla)
2. Kiran Panicker (@kiran-nlmatics)
3. John Scrudato (@JSv4)

## Contribution guidelines
- For small changes or bug fixes, go ahead and create a PR
- For large changes, create an issue on github with the proposal and @ one of the key maintainers for discussion before working on it
- The full test suite is pure Python and runs with no external services: `make test` (or `uv run pytest tests/`). Add tests for your change; changes to `line_parser` **must** come with tests in `tests/test_line_parser.py`.
- Engine behavior is locked by committed regression baselines (S-1, OpenContracts export, Docling layout, hetero/legal-100). Run `make test` before opening a PR — baselines may improve but must not regress.
- Run `make format` (black + isort) before committing or the lint CI fails.

## Contribution areas
- Bug fixes for specific tables - this is very difficult and time consuming
- Add table test cases

