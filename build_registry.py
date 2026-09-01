"""One-time script: build data/knowledge_base/reference_registry.json.
Run this before first pipeline execution and before S4 tests.
Requires ASTR_O_REGISTRY_SECRET environment variable.
"""
from __future__ import annotations

import os
import sys

from registry_builder import build_registry, save_registry


def main() -> None:
    secret = os.environ.get("ASTR_O_REGISTRY_SECRET")
    if not secret:
        print(
            "ERROR: ASTR_O_REGISTRY_SECRET is not set.\n"
            "Add it to your .env file and re-run:\n"
            "  export ASTR_O_REGISTRY_SECRET=<secret>\n"
            "  python build_registry.py"
        )
        sys.exit(1)

    config_path = "config/registry_config.json"
    docs_folder = "data/knowledge_base/"
    output_path = "data/knowledge_base/reference_registry.json"

    print(f"Building registry from {docs_folder} using {config_path} ...")
    registry = build_registry(
        config_path=config_path,
        docs_folder=docs_folder,
        secret_key=secret,
    )
    save_registry(registry, output_path)
    print(f"Registry written to {output_path}")


if __name__ == "__main__":
    main()
