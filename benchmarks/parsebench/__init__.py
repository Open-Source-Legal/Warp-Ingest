"""ParseBench integration for Warp-Ingest.

Runs the *official* LlamaIndex ParseBench evaluation framework
(https://github.com/run-llama/ParseBench) against Warp-Ingest
and the local-library baselines (liteparse, markitdown, pymupdf, pypdf).

ParseBench scoring is fully deterministic and rule-based (no LLM judge), so the
numbers here are reproducible and require no API keys for any local parser.
"""
