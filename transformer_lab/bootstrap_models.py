"""Command-line helper for downloading transformer lab assets."""

from __future__ import annotations

import sys

try:  # pragma: no cover
    from .bootstrap import cli
except ImportError:  # direct execution without package context
    from bootstrap import cli


if __name__ == "__main__":  # pragma: no cover
    sys.exit(cli())
