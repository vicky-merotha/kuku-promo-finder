"""Compare candidate coverage and judging cost on episodes 1, 26, and 50."""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import judge

EPISODES = {1, 26, 50}
OUTPUTS = Path("outputs")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_picks(path, events):
    picks = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            episode, start, end = (int(row[key]) for key in ("episode", "start_ms", "end_ms"))
            if episode not in EPISODES or not 20000 <= end - start <= 60000:
                raise ValueError("Each reference pick must be from episode 1, 26, or 50 and last 20–60 seconds.")
            hook = next(
                (event for event in events[str(episode)] if event["event_id"] == row["hook_event_id"].strip()),
                None,
            )
            if hook is None or not start <= hook["start_ms"] < start + 3000:
                raise ValueError(f"Check the first-three-second hook ID for reference row {len(picks) + 1}.")
            picks.append((episode, start, end, hook["start_ms"]))
    if not 3 <= len(picks) <= 5 or {pick[0] for pick in picks} != EPISODES:
        raise ValueError("Enter 3–5 reference picks total, with at least one per episode.")
    if len(set(picks)) != len(picks):
        raise ValueError("Remove duplicate reference picks.")
    return picks


def matches(clip, pick):
    episode, start, end, hook_start = pick
    overlap = min(clip["end_ms"], end) - max(clip["start_ms"], start)
    return (
        clip["episode"] == episode
        and clip["start_ms"] <= hook_start < clip["start_ms"] + 3000
        and overlap * 2 >= end - start
    )


def run_judge(clips, folder):
    folder.mkdir()
    (folder / "retrieved_candidates.json").write_text(json.dumps(clips), encoding="utf-8")
    old_outputs, old_argv = judge.OUTPUTS, sys.argv
    try:
        judge.OUTPUTS = folder
        sys.argv = ["judge.py", "--all"]
        exit_code = judge.main()
    finally:
        judge.OUTPUTS, sys.argv = old_outputs, old_argv
    log = folder / "judge_attempts.jsonl"
    attempts = [
        json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
    ] if log.exists() else []
    valid = {
        row["clip_id"] for row in attempts
        if row.get("raw") is not None and not row.get("error")
    }
    ranked = folder / "ranked.csv"
    kept = set()
    if ranked.exists():
        with ranked.open(encoding="utf-8-sig", newline="") as file:
            kept = {row["clip_id"] for row in csv.DictReader(file)}
    return {
        "judged_successfully": len(valid),
        "model_requests": sum(type(row.get("attempt")) is int for row in attempts),
        "kept_ids": kept,
        "exit_code": exit_code,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--judge", action="store_true", help="Also run the existing judge on both routes.")
    args = parser.parse_args()
    try:
        picks = load_picks(Path("eval/human_picks.csv"), load_json(OUTPUTS / "parsed_events.json"))
        routes = {
            "all_windows": load_json(args.windows),
            "retrieval_shortlist": load_json(OUTPUTS / "retrieved_candidates.json"),
        }
        if any(not isinstance(rows, list) for rows in routes.values()):
            raise ValueError("Both inputs must be JSON arrays of clip objects.")
        folder = Path("eval") / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        folder.mkdir()
        lines = [
            "# Phase 6 retrieval evaluation", "",
            "Match rule: same episode, reference hook onset within the first 3 seconds, "
            "and at least 50% of the human reference covered.", "",
        ]
        for name, rows in routes.items():
            clips = [clip for clip in rows if clip["episode"] in EPISODES]
            if not clips:
                raise ValueError(f"No evaluation clips found for {name}.")
            eligible = [clip for clip in clips if judge.evidence_for(clip)["opening_evidence_ids"]]
            lines += [
                f"## {name}", "",
                f"- Candidate windows: {len(clips)}",
                f"- Eligible for judge after opening screen: {len(eligible)}",
                f"- Human picks matched before screening: "
                f"{sum(any(matches(clip, pick) for clip in clips) for pick in picks)}/{len(picks)}",
                f"- Human picks matched after screening: "
                f"{sum(any(matches(clip, pick) for clip in eligible) for pick in picks)}/{len(picks)}",
            ]
            if args.judge:
                result = run_judge(clips, folder / name)
                kept = [clip for clip in clips if clip["clip_id"] in result["kept_ids"]]
                lines += [
                    f"- Successfully judged (including cache): {result['judged_successfully']}",
                    f"- Model attempts this run (including retries): {result['model_requests']}",
                    f"- Human picks matched in kept clips: "
                    f"{sum(any(matches(clip, pick) for clip in kept) for pick in picks)}/{len(picks)}",
                    f"- Judge exit code: {result['exit_code']}",
                ]
            lines.append("")
        report = "\n".join(lines)
        (folder / "retrieval_metrics.md").write_text(report, encoding="utf-8")
        print(report)
        print(f"Saved {folder / 'retrieval_metrics.md'}")
        return 0
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
