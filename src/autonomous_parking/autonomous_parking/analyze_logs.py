#!/usr/bin/env python3
"""
Analyze the latest training log in ./logs and print a detailed summary.

- Automatically finds the newest *.log file in logs/
- Parses:
  - Old 3-level curriculum: "*** CURRICULUM LEVEL UP: L1 (Easy) ***"
  - v15 micro-curriculum: "📚 CURRICULUM STAGE ADVANCE" blocks
  - Episode starts from: "[WaypointEnv.reset] lot=..., goal_bay=..."
  - Waypoint hits: "✓ Waypoint i/N reached (bonus: ...)"
  - Parking success lines: "✓ PARKING SUCCESS! dist=..., cx=..., cy=..., yaw_err=..."
  - SB3 progress: "|    total_timesteps      | 419840        |"
  - Reward trend: "ep_rew_mean"
  - Episode length trend: "ep_len_mean"

Usage:
    python -m autonomous_parking.analyze_logs
    python -m autonomous_parking.analyze_logs --logs-dir logs --target-steps 700000
    python -m autonomous_parking.analyze_logs --target-steps 700000 --live
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List


def find_latest_log(logs_dir: Path) -> Optional[Path]:
    """Return the newest *.log file in logs_dir, or None if none exist."""
    if not logs_dir.exists() or not logs_dir.is_dir():
        return None

    log_files = sorted(
        [p for p in logs_dir.glob("*.log") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return log_files[0] if log_files else None


def parse_log_file(log_path: Path) -> Dict[str, Any]:
    """
    Parse a single log file and extract metrics.

    Returns a dict with:
      - old_curriculum_levels: list[str]
      - curriculum_advances: list[dict] (v15 stages)
      - total_timesteps_last: int | None
      - episode_count: int
      - reset_count: int
      - lot_counts: dict[lot_name -> int]
      - bay_counts: dict[bay_id -> int]
      - waypoint_hits: int
      - waypoint_max_index: int
      - waypoint_max_total: int
      - success_dists: list[float]
      - success_cx: list[float]
      - success_cy: list[float]
      - success_yaw_err: list[float]
      - ep_rew_mean: list[float]
    """
    # Regexes
    re_total_steps = re.compile(r"total_timesteps\s*\|\s*([0-9]+)")
    re_ep_rew = re.compile(r"ep_rew_mean\s*\|\s*([-0-9.e+]+)")
    re_ep_len = re.compile(r"ep_len_mean\s*\|\s*([0-9.e+]+)")

    re_old_curriculum = re.compile(r"\*\*\* CURRICULUM LEVEL UP:\s*(.+?)\s*\*\*\*")

    re_wp_reset = re.compile(
        r"\[WaypointEnv\.reset\]\s+lot=(\S+),\s+goal_bay=([^,]+)"
    )

    # Waypoint reached (old or new formats)
    # Examples:
    #   "✓ Waypoint 5/6 reached (bonus: 253.1)"
    #   "✓ Waypoint 5/12 reached (bonus: 253.1, threshold: 3.4m)"
    re_waypoint = re.compile(
        r"Waypoin[tT]\s+(\d+)\s*/\s*(\d+)\s+reached"
    )

    # Flexible success-field regexes
    re_success_dist = re.compile(r"dist\s*=\s*([0-9.]+)")
    re_success_cx = re.compile(r"cx\s*=\s*([-0-9.]+)")
    re_success_cy = re.compile(r"cy\s*=\s*([-0-9.]+)")
    re_success_yaw = re.compile(r"yaw_err\s*=\s*([-0-9.]+)")

    # v15 curriculum block parsing
    re_cur_from = re.compile(r"From:\s*(.+)")
    re_cur_to = re.compile(r"To:\s*(.+)")
    re_cur_steps = re.compile(r"Steps:\s*([0-9,]+)")
    re_cur_recent = re.compile(r"Recent Success:\s*([0-9.]+)%")
    re_cur_replay = re.compile(r"Replay Prob:\s*([0-9.]+)%")

    # Metrics containers
    old_curriculum_levels: List[str] = []
    curriculum_advances: List[Dict[str, Any]] = []

    total_timesteps_last: Optional[int] = None
    ep_rew_mean: List[float] = []
    ep_len_mean: List[float] = []

    episode_count = 0   # episodes = number of WaypointEnv.reset calls
    reset_count = 0     # same as above but kept separate in case we later add other reset markers
    lot_counts: Dict[str, int] = {}
    bay_counts: Dict[str, int] = {}

    waypoint_hits = 0
    waypoint_max_index = 0
    waypoint_max_total = 0

    success_dists: List[float] = []
    success_cx: List[float] = []
    success_cy: List[float] = []
    success_yaw_err: List[float] = []

    # Curriculum block state
    in_curr_block = False
    curr_block: Dict[str, Any] = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_stripped = line.strip()

            # -------------------
            # SB3 training stats
            # -------------------
            m_ts = re_total_steps.search(line)
            if m_ts:
                try:
                    total_timesteps_last = int(m_ts.group(1))
                except ValueError:
                    pass

            m_rew = re_ep_rew.search(line)
            if m_rew:
                try:
                    ep_rew_mean.append(float(m_rew.group(1)))
                except ValueError:
                    pass

            m_len = re_ep_len.search(line)
            if m_len:
                try:
                    ep_len_mean.append(float(m_len.group(1)))
                except ValueError:
                    pass

            # -------------------
            # Old 3-level curriculum (ParkingEnv)
            # "*** CURRICULUM LEVEL UP: L1 (Easy) ***"
            # -------------------
            m_old = re_old_curriculum.search(line)
            if m_old:
                level_name = m_old.group(1).strip()
                old_curriculum_levels.append(level_name)

            # -------------------
            # v15 curriculum block
            # -------------------
            if "📚 CURRICULUM STAGE ADVANCE" in line:
                # Start a new block
                in_curr_block = True
                curr_block = {
                    "from": None,
                    "to": None,
                    "steps": None,
                    "recent_success": None,
                    "replay_prob": None,
                    "_lines_seen": 0,  # safety counter
                }
                continue

            if in_curr_block:
                curr_block["_lines_seen"] = curr_block.get("_lines_seen", 0) + 1

                # Safety: abandon malformed / truncated block
                if curr_block["_lines_seen"] > 20:
                    in_curr_block = False
                    curr_block = {}
                    continue

                if line_stripped.startswith("*") and "CURRICULUM STAGE ADVANCE" not in line:
                    # Decorative lines allowed
                    pass

                m_from = re_cur_from.search(line)
                if m_from:
                    curr_block["from"] = m_from.group(1).strip()

                m_to = re_cur_to.search(line)
                if m_to:
                    curr_block["to"] = m_to.group(1).strip()

                m_steps = re_cur_steps.search(line)
                if m_steps:
                    steps_str = m_steps.group(1).replace(",", "")
                    try:
                        curr_block["steps"] = int(steps_str)
                    except ValueError:
                        curr_block["steps"] = None

                m_recent = re_cur_recent.search(line)
                if m_recent:
                    try:
                        curr_block["recent_success"] = float(m_recent.group(1))
                    except ValueError:
                        curr_block["recent_success"] = None

                m_replay = re_cur_replay.search(line)
                if m_replay:
                    try:
                        curr_block["replay_prob"] = float(m_replay.group(1))
                    except ValueError:
                        curr_block["replay_prob"] = None
                    # Assume this is the last line of the block
                    curriculum_advances.append(curr_block)
                    in_curr_block = False
                    curr_block = {}
                # Continue to next line regardless

            # -------------------
            # Episode starts from WaypointEnv.reset or ParkingEnv.reset
            # "[WaypointEnv.reset] lot=lot_a, goal_bay=A1, ..."
            # "[ParkingEnv.reset] ..."
            # -------------------
            m_reset = re_wp_reset.search(line)
            if m_reset:
                episode_count += 1
                reset_count += 1
                lot = m_reset.group(1)
                bay = m_reset.group(2)

                lot_counts[lot] = lot_counts.get(lot, 0) + 1
                bay_counts[bay] = bay_counts.get(bay, 0) + 1
            elif "[ParkingEnv.reset]" in line:
                # Also count ParkingEnv resets (for compatibility)
                reset_count += 1

            # -------------------
            # Waypoint hits
            # -------------------
            m_wp = re_waypoint.search(line)
            if m_wp:
                waypoint_hits += 1
                try:
                    idx = int(m_wp.group(1))
                    total = int(m_wp.group(2))
                    if idx > waypoint_max_index:
                        waypoint_max_index = idx
                    if total > waypoint_max_total:
                        waypoint_max_total = total
                except ValueError:
                    pass

            # -------------------
            # Parking successes
            # "✓ PARKING SUCCESS! dist=0.91, cx=-0.67, cy=-0.60, yaw_err=1.04"
            # -------------------
            if "✓ PARKING SUCCESS!" in line:
                m_d = re_success_dist.search(line)
                m_cx = re_success_cx.search(line)
                m_cy = re_success_cy.search(line)
                m_ye = re_success_yaw.search(line)
                if m_d and m_cx and m_cy and m_ye:
                    try:
                        d = float(m_d.group(1))
                        cx = float(m_cx.group(1))
                        cy = float(m_cy.group(1))
                        ye = float(m_ye.group(1))
                        success_dists.append(d)
                        success_cx.append(cx)
                        success_cy.append(cy)
                        success_yaw_err.append(ye)
                    except ValueError:
                        pass

    return {
        "old_curriculum_levels": old_curriculum_levels,
        "curriculum_advances": curriculum_advances,
        "total_timesteps_last": total_timesteps_last,
        "episode_count": episode_count,
        "reset_count": reset_count,
        "lot_counts": lot_counts,
        "bay_counts": bay_counts,
        "waypoint_hits": waypoint_hits,
        "waypoint_max_index": waypoint_max_index,
        "waypoint_max_total": waypoint_max_total,
        "success_dists": success_dists,
        "success_cx": success_cx,
        "success_cy": success_cy,
        "success_yaw_err": success_yaw_err,
        "ep_rew_mean": ep_rew_mean,
        "ep_len_mean": ep_len_mean,
    }


def _fmt_stats(values: List[float]) -> str:
    """Return 'min / mean / max' string for a list of floats."""
    if not values:
        return "n/a"
    vmin = min(values)
    vmax = max(values)
    vmean = sum(values) / len(values)
    return f"{vmin:.3f} / {vmean:.3f} / {vmax:.3f}"


def print_report(log_path: Path, m: Dict[str, Any], target_steps: Optional[int]) -> None:
    """Pretty-print a human-readable report based on parsed metrics."""
    # Get log file metadata
    mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
    file_age_seconds = time.time() - log_path.stat().st_mtime
    
    print("=" * 80)
    print(f"📄 LOG ANALYSIS REPORT")
    print(f"File: {log_path}")
    print(f"Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({file_age_seconds:.0f}s ago)")
    print("=" * 80)
    print()

    # --- High-level run summary ---
    print("1) RUN PROGRESS")
    ts = m["total_timesteps_last"]
    if ts is not None:
        if target_steps and target_steps > 0:
            pct = 100.0 * ts / target_steps
            print(f"  - Total timesteps (last seen): {ts:,}   "
                  f"(target: {target_steps:,}, {pct:5.1f}%)")
        elif target_steps is not None:
            print(f"  - Total timesteps (last seen): {ts:,}   "
                  f"(WARNING: invalid target_steps={target_steps})")
        else:
            print(f"  - Total timesteps (last seen): {ts:,}")
    else:
        print("  - Total timesteps: not found in log")

    if m["ep_rew_mean"]:
        first = m["ep_rew_mean"][0]
        last = m["ep_rew_mean"][-1]
        delta = last - first
        print(f"  - Reward trend (ep_rew_mean): {first:.2e} → {last:.2e} (Δ = {delta:+.2e})")
    else:
        print("  - Reward trend: no ep_rew_mean entries found")

    if m["ep_len_mean"]:
        first_len = m["ep_len_mean"][0]
        last_len = m["ep_len_mean"][-1]
        delta_len = last_len - first_len
        print(f"  - Episode length trend: {first_len:.0f} → {last_len:.0f} steps (Δ = {delta_len:+.0f})")
    else:
        print("  - Episode length trend: no ep_len_mean entries found")

    print()

    # --- Curriculum info ---
    print("2) CURRICULUM PROGRESSION")

    # Old L1/L2/L3 auto-curriculum
    old_levels = m["old_curriculum_levels"]
    if old_levels:
        from collections import Counter
        level_counts = Counter(old_levels)
        unique_old = list(dict.fromkeys(old_levels))  # preserve order
        print(f"  - Legacy L1/L2/L3 curriculum levels reached: {', '.join(unique_old)}")
        print(f"    (counts: {dict(level_counts)})")
    else:
        print("  - No legacy (L1/L2/L3) curriculum level prints found.")

    # v15 micro-curriculum
    adv = m["curriculum_advances"]
    if adv:
        print(f"  - v15 micro-curriculum stage advances: {len(adv)} event(s)")
        last = adv[-1]
        to_stage = last.get("to", "unknown")
        steps = last.get("steps")
        recent = last.get("recent_success")
        replay = last.get("replay_prob")
        print(f"    • Last advance → {to_stage}")
        if steps is not None:
            print(f"      at steps={steps:,}")
        if recent is not None:
            print(f"      recent success window ≈ {recent:.1f}%")
        if replay is not None:
            print(f"      replay probability in new stage ≈ {replay:.0f}%")
    else:
        print("  - No v15 '📚 CURRICULUM STAGE ADVANCE' events found.")

    print()

    # --- Episodes, resets, lots, bays ---
    print("3) EPISODES / LOTS / BAYS")
    eps = m["episode_count"]
    resets = m["reset_count"]
    succ = len(m["success_dists"])

    print(f"  - Episodes (WaypointEnv.reset calls): {eps}")
    print(f"  - Reset count (same marker):          {resets}")

    if eps > 0:
        success_rate = 100.0 * succ / eps
        resets_per_episode = resets / eps if resets else 0.0
        print(f"  - Parking successes: {succ}  "
              f"→ success rate = {succ} / {eps} = {success_rate:5.2f}%")
        print(f"  - Resets per episode: {resets_per_episode:.2f}")
    else:
        print(f"  - Parking successes: {succ} (no episodes detected → cannot compute rate)")

    # Lot usage
    lot_counts = m["lot_counts"]
    if lot_counts:
        print("  - Lots used:")
        total_lot = sum(lot_counts.values())
        for lot, c in sorted(lot_counts.items(), key=lambda kv: kv[0]):
            frac = 100.0 * c / total_lot if total_lot else 0.0
            print(f"      • {lot:8s}: {c:4d} episodes ({frac:5.1f}%)")
    else:
        print("  - No [WaypointEnv.reset] lot=... entries found (lot usage unknown).")

    # Bay usage
    bay_counts = m["bay_counts"]
    if bay_counts:
        top_n = min(10, len(bay_counts))
        print(f"  - Goal bays sampled (top {top_n}):")
        total_bays = sum(bay_counts.values())
        for bay, c in sorted(bay_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]:
            frac = 100.0 * c / total_bays if total_bays else 0.0
            print(f"      • {bay:6s}: {c:4d} episodes ({frac:5.1f}%)")
    else:
        print("  - No goal_bay=... info found in [WaypointEnv.reset] lines.")

    print()

    # --- Waypoint-following stats ---
    print("4) WAYPOINT-FOLLOWING PERFORMANCE")

    hits = m["waypoint_hits"]
    w_idx = m["waypoint_max_index"]
    w_tot = m["waypoint_max_total"]

    print(f"  - Waypoint hits logged: {hits}")
    if w_tot > 0:
        print(f"  - Max waypoint index reached: {w_idx} / {w_tot}")
    else:
        print("  - No '✓ Waypoint i/N reached' lines found (or format changed).")

    print()

    # --- Parking quality stats ---
    print("5) PARKING QUALITY (from '✓ PARKING SUCCESS!' logs)")

    dists = m["success_dists"]
    cxs = m["success_cx"]
    cys = m["success_cy"]
    yaws = m["success_yaw_err"]

    print(f"  - Number of logged successes: {len(dists)}")
    print(f"  - Final distance to goal center   (min / mean / max): {_fmt_stats(dists)}  [m]")
    print(f"  - Center offset along bay axis cx (min / mean / max): {_fmt_stats(cxs)}  [m]")
    print(f"  - Center offset across bay  cy    (min / mean / max): {_fmt_stats(cys)}  [m]")
    print(f"  - Yaw error at success            (min / mean / max): {_fmt_stats(yaws)} [rad]")

    print()
    print("=" * 80)
    print("End of report.")
    print("=" * 80)


def analyze_once(logs_dir: Path, target_steps: Optional[int]) -> None:
    """Find latest log, parse it, and print a report once."""
    latest = find_latest_log(logs_dir)
    if latest is None:
        print(f"No .log files found in {logs_dir.resolve()}")
        return

    metrics = parse_log_file(latest)
    print_report(latest, metrics, target_steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze latest training log in ./logs")
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs",
        help="Directory containing training .log files (default: ./logs)",
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        default=None,
        help="Target total timesteps for this run (e.g. 700000)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live mode: repeatedly re-analyze latest log until Ctrl+C",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Refresh interval in seconds for --live mode (default: 10s)",
    )

    args = parser.parse_args()
    logs_dir = Path(args.logs_dir)

    if args.live:
        # LIVE MODE: loop until Ctrl+C
        try:
            while True:
                # Clear screen for a dashboard feel
                print("\033c", end="")  # ANSI clear
                print(f"[Live mode] Refresh interval = {args.interval:.1f}s")
                analyze_once(logs_dir, args.target_steps)
                time.sleep(max(args.interval, 0.5))
        except KeyboardInterrupt:
            print("\nLive mode stopped by user (Ctrl+C).")
    else:
        # SNAPSHOT MODE: single run
        analyze_once(logs_dir, args.target_steps)


if __name__ == "__main__":
    main()


# cd ~/autonomous_parking_ws/src/autonomous_parking
# source ../../.venv/bin/activate
# ../../.venv/bin/python -m

# # Quick snapshot
# ../../.venv/bin/python -m autonomous_parking.analyze_logs --target-steps 700000
# python -m autonomous_parking.analyze_logs --target-steps 700000

# # Live dashboard (10s refresh)
# python -m autonomous_parking.analyze_logs --target-steps 700000 --live

# # Live dashboard (5s refresh for active training)
# python -m autonomous_parking.analyze_logs --target-steps 700000 --live --interval 5

# # Analyze specific logs directory
# python -m autonomous_parking.analyze_logs --logs-dir old_runs --target-steps 500000