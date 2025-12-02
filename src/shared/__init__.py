from __future__ import annotations

"""
Shared, sport-agnostic utilities for the Showdown optimizer codebase.

This package is intended for code that is truly common across NFL and NBA,
such as:

- Project root / path helpers.
- Generic numerical utilities.
- Small helpers that do not depend on any particular sport's rules.

Sport-specific configuration and logic should live under:

- src/nfl/...
- src/nba/...
"""


