"""
savant_fetcher.py  v3
─────────────────────
Downloads Statcast pitch-level data from Baseball Savant.

Key insight on the 40k row cap:
  Savant silently truncates responses at ~40k rows. A 7-day window for ALL
  pitchers can easily exceed this (30 teams × 7 games × 300 pitches ≈ 63k).
  Fix: 3-day windows for bulk training (~13.5k rows max), and always fetch
  individual pitchers by pitcher_id (max ~500 rows/week — never truncated).
"""

import os, time, io
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, datetime, timedelta

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

# ── Windows ────────────────────────────────────────────────────────────────────

def _make_windows(start_str: str, end_str: str, days: int):
    start, end = date.fromisoformat(start_str), date.fromisoformat(end_str)
    windows, cur = [], start
    while cur <= end:
        w_end = min(cur + timedelta(days=days - 1), end)
        windows.append((cur.isoformat(), w_end.isoformat()))
        cur = w_end + timedelta(days=1)
    return windows

# 3-day windows for bulk: max ~13.5k rows, well under Savant's 40k cap
WINDOWS_2025 = _make_windows("2025-03-27", "2025-09-28", days=3)
WINDOWS_2026 = _make_windows("2026-03-26", "2026-09-27", days=3)

NUMERIC_COLS = [
    "release_speed", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z", "release_extension",
    "plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "release_spin_rate", "spin_rate_deprecated", "spin_axis",
    "delta_run_exp", "estimated_woba_using_speedangle", "zone",
    "spin_efficiency",
]

# Columns required for physics scoring — dropna only on these
PHYSICS_COLS = [
    "release_speed", "pfx_x", "pfx_z",
    "release_pos_x", "release_pos_z",
    "vx0", "vy0", "vz0", "ax", "ay", "az",
    "plate_x", "plate_z",
]


def _params(start: str, end: str, pitcher_id=None) -> dict:
    # game_date_lt is EXCLUSIVE on Savant → add 1 day to include window end
    lt = (date.fromisoformat(end) + timedelta(days=1)).isoformat()
    p = {
        "all":          "true",
        "hfGT":         "R|",       # Regular season
        "hfSea":        f"{start[:4]}|",
        "hfPT":         "",         # All pitch types
        "hfAB":         "",         # All at-bat results
        "hfBBT":        "",         # All batted ball types
        "hfBBL":        "",         # All ballpark locations
        "game_date_gt": start,
        "game_date_lt": lt,
        "player_type":  "pitcher",
        "min_pitches":  "0",
        "min_results":  "0",
        "min_pas":      "0",
        "type":         "details",  # Pitch-by-pitch rows
        "sort_col":     "game_date",
        "sort_order":   "asc",
    }
    if pitcher_id:
        p["pitchers_lookup[]"] = str(pitcher_id)
    return p


def _fetch_chunk(start: str, end: str,
                 pitcher_id=None, retries: int = 3) -> pd.DataFrame | None:
    params = _params(start, end, pitcher_id)
    for attempt in range(retries):
        try:
            r = requests.get(BASE_URL, params=params, timeout=120)
            r.raise_for_status()
            text = r.text.strip()
            if not text or len(text) < 200 or text.startswith("Error"):
                return None
            df = pd.read_csv(io.StringIO(text), low_memory=False)
            if df.empty or "pitch_type" not in df.columns:
                return None
            # Warn if suspiciously close to the 40k cap
            if len(df) >= 39_000:
                print(f"  ⚠️  {start}→{end}: {len(df):,} rows — near 40k cap, may be truncated!")
            return df
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(8 * (attempt + 1))
            else:
                print(f"  ✗ {start}–{end}: {e}")
                return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pitch_type" in df.columns:
        df["pitch_type"] = df["pitch_type"].fillna("").str.upper().str.strip()
        df = df[df["pitch_type"].str.len() > 0]
    present = [c for c in PHYSICS_COLS if c in df.columns]
    df = df.dropna(subset=present, how="any")
    return df


def _dedup(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in ["game_pk", "at_bat_number", "pitch_number"]
               if c in df.columns]
    if len(id_cols) == 3:   # only dedup when all 3 IDs present
        df = df.drop_duplicates(subset=id_cols)
    return df


# ─── BULK SEASON DOWNLOAD (for training calibration) ──────────────────────────

def fetch_season(year: int = 2025,
                 force: bool = False,
                 status_fn=None) -> pd.DataFrame:
    """
    Download entire season for training calibration.
    Uses 3-day windows to stay under Savant's 40k row cap.
    """
    cache = CACHE_DIR / f"statcast_{year}.parquet"
    if cache.exists() and not force:
        if status_fn:
            status_fn(f"✓ Loading cached {year} season…", 1.0)
        return pd.read_parquet(cache)

    windows = WINDOWS_2025 if year == 2025 else WINDOWS_2026
    today   = date.today()
    windows = [(s, e) for s, e in windows if date.fromisoformat(s) <= today]

    if not windows:
        raise RuntimeError(f"{year} season has not started yet.")

    chunks = []
    for i, (start, end) in enumerate(windows):
        end_capped = min(date.fromisoformat(end), today).isoformat()
        if status_fn:
            status_fn(
                f"Window {i+1}/{len(windows)}: {start} → {end_capped}",
                i / len(windows)
            )
        chunk = _fetch_chunk(start, end_capped)
        if chunk is not None:
            cleaned = _clean(chunk)
            chunks.append(cleaned)
            if status_fn:
                status_fn(
                    f"Window {i+1}/{len(windows)}: {start} → {end_capped} "
                    f"({len(cleaned):,} pitches)",
                    (i + 0.9) / len(windows)
                )
        time.sleep(1.5)

    if not chunks:
        raise RuntimeError(f"No data returned for {year}.")

    df = _dedup(pd.concat(chunks, ignore_index=True))
    df.to_parquet(cache, index=False)
    if status_fn:
        status_fn(f"✓ {year} cached — {len(df):,} pitches", 1.0)
    return df


# ─── INDIVIDUAL PITCHER FETCH ─────────────────────────────────────────────────

def fetch_pitcher_season(pitcher_id: int, year: int = 2025) -> pd.DataFrame:
    """
    Fetch a specific pitcher's full season.

    Priority order (fastest first):
      1. Per-pitcher parquet cache (< 24h old) — instant
      2. Bulk season parquet (statcast_YYYY.parquet) — instant, built by Train tab
      3. Live fetch from Savant by pitcher_id — slow (~60s) but guaranteed complete

    The live fetch is only used when the bulk cache doesn't exist yet or the
    pitcher genuinely isn't in it (e.g. called up after training ran).
    """
    pitcher_cache = CACHE_DIR / f"pitcher_{pitcher_id}_{year}.parquet"
    now           = datetime.now()

    # ── 1. Per-pitcher cache (fresh) ──────────────────────────────────────────
    if pitcher_cache.exists():
        age_h = (now.timestamp() - pitcher_cache.stat().st_mtime) / 3600
        if age_h < 24:
            return pd.read_parquet(pitcher_cache)

    # ── 2. Bulk season cache ───────────────────────────────────────────────────
    bulk = CACHE_DIR / f"statcast_{year}.parquet"
    if bulk.exists():
        df_bulk = pd.read_parquet(bulk)
        for col in ["pitcher", "pitcher_id", "player_id"]:
            if col in df_bulk.columns:
                sub = df_bulk[df_bulk[col] == pitcher_id]
                if len(sub) > 0:
                    result = sub.reset_index(drop=True)
                    # Save as per-pitcher cache so next load is even faster
                    result.to_parquet(pitcher_cache, index=False)
                    return result

    # ── 3. Live fetch by pitcher_id (slow but complete) ───────────────────────
    # Pitcher-specific queries: max ~500 rows/week, never hits 40k cap
    windows = WINDOWS_2025 if year == 2025 else WINDOWS_2026
    today   = date.today()
    active  = [(s, e) for s, e in windows if date.fromisoformat(s) <= today]

    chunks = []
    for start, end in active:
        end_capped = min(date.fromisoformat(end), today).isoformat()
        chunk = _fetch_chunk(start, end_capped, pitcher_id=pitcher_id)
        if chunk is not None:
            chunks.append(_clean(chunk))
        time.sleep(0.8)

    if not chunks:
        return pd.DataFrame()

    df = _dedup(pd.concat(chunks, ignore_index=True))
    df.to_parquet(pitcher_cache, index=False)
    return df


def fetch_pitcher_ytd(pitcher_id: int,
                      year: int = 2026,
                      force_refresh: bool = False) -> pd.DataFrame:
    """Year-to-date fetch for current season (2026+). 6-hour cache."""
    cache = CACHE_DIR / f"pitcher_{pitcher_id}_{year}.parquet"
    now   = datetime.now()

    if cache.exists() and not force_refresh:
        age_h = (now.timestamp() - cache.stat().st_mtime) / 3600
        if age_h < 6:
            return pd.read_parquet(cache)

    windows = WINDOWS_2026 if year == 2026 else WINDOWS_2025
    today   = date.today()
    active  = [(s, e) for s, e in windows if date.fromisoformat(s) <= today]

    chunks = []
    for start, end in active:
        end_capped = min(date.fromisoformat(end), today).isoformat()
        chunk = _fetch_chunk(start, end_capped, pitcher_id=pitcher_id)
        if chunk is not None:
            chunks.append(_clean(chunk))
        time.sleep(0.8)

    if not chunks:
        return pd.DataFrame()

    df = _dedup(pd.concat(chunks, ignore_index=True))
    df.to_parquet(cache, index=False)
    return df


# ─── SEARCH ───────────────────────────────────────────────────────────────────

def _search_local_cache(name: str, year: int = 2025) -> list[dict]:
    """
    Search pitcher names directly from the local season parquet.
    Statcast CSV includes a 'player_name' column (format: "Last, First")
    and a 'pitcher' column (numeric MLB ID).
    Returns list of {id, name} dicts sorted by name.
    """
    cache = CACHE_DIR / f"statcast_{year}.parquet"
    if not cache.exists():
        return []
    try:
        df = pd.read_parquet(cache, columns=None)
        # Find pitcher name column — Savant uses 'player_name' (pitcher perspective)
        name_col = next((c for c in df.columns
                         if c.lower() in ("player_name", "pitcher_name")), None)
        id_col   = next((c for c in df.columns
                         if c.lower() in ("pitcher", "pitcher_id", "player_id")), None)
        if name_col is None or id_col is None:
            return []

        query = name.lower().strip()
        mask  = df[name_col].str.lower().str.contains(query, na=False, regex=False)
        hits  = df[mask][[name_col, id_col]].dropna().drop_duplicates()

        results = []
        for _, row in hits.iterrows():
            raw_name = str(row[name_col]).strip()
            # Savant stores as "Last, First" — convert to "First Last" for display
            if "," in raw_name:
                parts    = raw_name.split(",", 1)
                display  = f"{parts[1].strip()} {parts[0].strip()}"
            else:
                display  = raw_name
            results.append({"id": int(row[id_col]), "name": display})

        # Sort by name, deduplicate by id
        seen = set()
        out  = []
        for r in sorted(results, key=lambda x: x["name"]):
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
        return out
    except Exception:
        return []


def search_pitcher(name: str, year: int = 2025) -> list[dict]:
    """
    Search for a pitcher by name.
    Checks local cache first (instant, no network).
    Falls back to Savant suggest API if cache is empty or no match found.
    """
    # Always try local cache first
    local = _search_local_cache(name, year=year)
    if local:
        return local

    # Fall back to Savant API
    try:
        r = requests.get(
            "https://baseballsavant.mlb.com/player-services/suggest",
            params={"query": name, "type": "pitcher"},
            timeout=10
        )
        data = r.json()
        results = []
        for item in data:
            pid   = item.get("xba_id") or item.get("id") or item.get("player_id")
            pname = item.get("name_display_first_last") or item.get("name")
            if pid and pname:
                results.append({"id": int(pid), "name": str(pname)})
        return results
    except Exception:
        return []


def cache_status() -> dict:
    files = list(CACHE_DIR.glob("*.parquet"))
    total = sum(f.stat().st_size for f in files) / 1e6
    return {
        "Cached files": len(files),
        "Total size":   f"{total:.1f} MB",
        "Location":     str(CACHE_DIR),
    }
