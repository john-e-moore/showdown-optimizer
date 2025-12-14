from __future__ import annotations

"""
DraftKings contest metadata + payouts helpers.

This module provides a small, sport-agnostic wrapper for:
  - downloading contest JSON from DraftKings (public endpoint)
  - caching the JSON under <data_dir>/payouts/
  - parsing: field size (num entries), entry fee, and a rank->payout array

It intentionally does NOT depend on sport-specific config modules.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Tuple

import numpy as np


DK_CONTEST_URL_TEMPLATE = (
    "https://api.draftkings.com/contests/v1/contests/{contest_id}?format=json"
)


def _download_contest_json(contest_id: str, dest_path: Path) -> bool:
    """
    Download contest JSON for a contest from DraftKings and write it to dest_path.

    Returns:
        True if the file was downloaded and written successfully; False otherwise.
    """
    url = DK_CONTEST_URL_TEMPLATE.format(contest_id=contest_id)
    print(f"Downloading DraftKings contest JSON from {url} ...")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            status = getattr(resp, "status", None)
            if status is not None and int(status) != 200:
                print(
                    f"Warning: DraftKings contest request for contest {contest_id} "
                    f"returned HTTP status {status}."
                )
                return False
            data = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(
            f"Warning: Failed to download DraftKings contest JSON for contest {contest_id} "
            f"from {url}: {exc}."
        )
        return False

    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        print(
            f"Warning: Failed to parse DraftKings contest JSON for contest {contest_id} "
            f"from {url}: {exc}."
        )
        return False

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except OSError as exc:
        print(
            f"Warning: Failed to write contest JSON to {dest_path}: {exc}."
        )
        return False

    print(f"Wrote DraftKings contest JSON to {dest_path}")
    return True


def _get_int_maybe(obj: Any) -> int | None:
    try:
        if obj is None:
            return None
        # DK sometimes uses floats/strings; normalize.
        val = int(float(obj))
        if val <= 0:
            return None
        return val
    except (TypeError, ValueError):
        return None


def _extract_field_size(contest_detail: dict[str, Any]) -> int | None:
    """
    Best-effort extraction of contest field size from the DraftKings contestDetail.
    """
    # Common keys we've seen / expect across DK payloads.
    direct_keys = [
        "numEntries",
        "entryCount",
        "entries",
        "contestSize",
        "maxEntries",
        "maxEntryCount",
        "maximumEntries",
        "totalEntries",
    ]
    for k in direct_keys:
        v = _get_int_maybe(contest_detail.get(k))
        if v is not None:
            return v

    # Sometimes nested under "contest" or "entries" objects.
    nested_candidates: list[Any] = []
    for k in ["contest", "entries", "entryCounts", "entryCount"]:
        nested_candidates.append(contest_detail.get(k))

    for nested in nested_candidates:
        if isinstance(nested, dict):
            for k in direct_keys:
                v = _get_int_maybe(nested.get(k))
                if v is not None:
                    return v

    return None


def _parse_payouts_and_fee(
    contest_detail: dict[str, Any],
    *,
    num_entries: int,
) -> Tuple[np.ndarray, float]:
    """
    Parse DraftKings payout structure and entry fee from contestDetail.
    """
    try:
        entry_fee = float(contest_detail.get("entryFee", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "DraftKings contest JSON has non-numeric 'entryFee' field."
        ) from exc

    payout_summary = contest_detail.get("payoutSummary", [])
    payouts = np.zeros(num_entries, dtype=float)
    if not isinstance(payout_summary, list):
        raise ValueError(
            "DraftKings contest JSON has unexpected 'payoutSummary' format; expected a list."
        )

    for tier in payout_summary:
        if not isinstance(tier, dict):
            continue
        try:
            min_pos = int(tier.get("minPosition", 0))
            max_pos = int(tier.get("maxPosition", min_pos))
        except (TypeError, ValueError):
            continue
        if max_pos < 1 or min_pos < 1:
            continue

        payout_descs = tier.get("payoutDescriptions") or []
        value: float = 0.0
        if isinstance(payout_descs, list):
            for desc in payout_descs:
                if not isinstance(desc, dict):
                    continue
                if "value" in desc:
                    try:
                        value = float(desc["value"])
                    except (TypeError, ValueError):
                        continue
                    break
        if value <= 0.0:
            continue

        for pos in range(min_pos, max_pos + 1):
            idx = pos - 1
            if 0 <= idx < num_entries:
                payouts[idx] = value

    return payouts, entry_fee


def load_contest_payouts_and_size(
    contest_id: str,
    *,
    data_dir: Path,
    payouts_json: str | None = None,
) -> tuple[int, float, np.ndarray]:
    """
    Load contest payout structure and derived field size for a DK contest id.

    Caching:
      - If payouts_json is provided, reads that file (JSON).
      - Otherwise reads/writes data_dir/payouts/payouts-{contest_id}.json.

    Returns:
        field_size, entry_fee, payouts_by_rank (0-indexed ranks).
    """
    if payouts_json is not None:
        path = Path(payouts_json)
        if not path.is_file():
            raise FileNotFoundError(
                f"Specified payout/contest JSON file does not exist: {path} (from --payouts-json)."
            )
    else:
        payouts_dir = data_dir / "payouts"
        path = payouts_dir / f"payouts-{contest_id}.json"
        if not path.is_file():
            if not _download_contest_json(str(contest_id), path):
                raise RuntimeError(
                    f"Failed to download DraftKings contest JSON for contest {contest_id}."
                )

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse contest JSON at {path}: {exc}") from exc

    try:
        detail = raw["contestDetail"]
    except (TypeError, KeyError) as exc:
        raise ValueError(
            f"Contest JSON at {path} missing 'contestDetail' section."
        ) from exc
    if not isinstance(detail, dict):
        raise ValueError(f"Contest JSON at {path} has invalid 'contestDetail' type.")

    field_size = _extract_field_size(detail)
    if field_size is None:
        raise ValueError(
            "Could not infer field size from DraftKings contest JSON. "
            "Pass --field-size explicitly or provide a payouts JSON with a supported schema."
        )

    payouts, entry_fee = _parse_payouts_and_fee(detail, num_entries=field_size)
    return field_size, entry_fee, payouts


__all__ = [
    "DK_CONTEST_URL_TEMPLATE",
    "load_contest_payouts_and_size",
]


