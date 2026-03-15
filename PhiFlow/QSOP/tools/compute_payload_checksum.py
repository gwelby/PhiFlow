#!/usr/bin/env python3
"""Compute sha256 checksum string for a payload file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from qsop_packet_lib import compute_sha256_for_path, qsop_root, resolve_payload_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute `sha256:<hex>` for objective payload files.")
    parser.add_argument("payload_path", help="Payload path (absolute, QSOP/... or relative to QSOP root).")
    args = parser.parse_args()

    payload_path = resolve_payload_path(qsop_root(), args.payload_path)
    if not payload_path.exists() or not payload_path.is_file():
        print(f"ERROR: payload file not found: {payload_path}")
        return 1

    digest = compute_sha256_for_path(payload_path)
    print(f"sha256:{digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
