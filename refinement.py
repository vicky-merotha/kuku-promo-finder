"""Test a single refinement decision and save a separate comparison report."""

import argparse
import json
import sys
from pathlib import Path

from judge import RUBRIC, SCORES, evidence_for, timestamp, validate_result
from llm import ProviderError, complete, configuration

OUTPUTS = Path("outputs")


def read_json(name):
    return json.loads((OUTPUTS / name).read_text(encoding="utf-8-sig"))


def find_nearby_cuts(clip_id, anchor_event_id):
    original = next(
        (clip for clip in read_json("retrieved_candidates.json") if clip["clip_id"] == clip_id),
        None,
    )
    if original is None:
        raise ValueError(f"Clip {clip_id} was not found in the shortlist.")
    episode = original["episode"]
    events = read_json("parsed_events.json")[str(episode)]
    anchor = next((event for event in events if event["event_id"] == anchor_event_id), None)
    if anchor is None:
        raise ValueError(f"Anchor {anchor_event_id} was not found in episode {episode}.")
    scene = [event for event in events if event["scene"] == anchor["scene"]]
    dialogue = [event for event in events if event["type"] == "dialogue"]

    def safe_boundary(time):
        return all(not line["start_ms"] < time < line["end_ms"] for line in dialogue)

    original_bounds = original["start_ms"], original["end_ms"]
    starts = sorted({
        event["start_ms"] for event in scene
        if event["type"] in ("dialogue", "action")
        and 0 <= anchor["start_ms"] - event["start_ms"] <= 8000
        and safe_boundary(event["start_ms"])
    })
    ends = sorted({
        event["end_ms"] for event in scene
        if event["end_ms"] >= anchor["end_ms"]
        and abs(event["end_ms"] - original["end_ms"]) <= 8000
        and safe_boundary(event["end_ms"])
    })
    alternatives = []
    for start in starts:
        valid_ends = [
            end for end in ends
            if 20000 <= end - start <= 60000 and (start, end) != original_bounds
        ]
        if not valid_ends:
            continue
        end = min(valid_ends, key=lambda value: (abs(value - original["end_ms"]), value))
        clip = {
            "clip_id": f"E{episode:03d}_{start}_{end}",
            "episode": episode,
            "start_ms": start,
            "end_ms": end,
            "events": [event for event in scene if event["start_ms"] < end and event["end_ms"] > start],
        }
        if evidence_for(clip)["opening_evidence_ids"]:
            alternatives.append(clip)
    alternatives.sort(key=lambda clip: (
        anchor["end_ms"] - clip["start_ms"] > 3000,
        anchor["start_ms"] - clip["start_ms"],
        abs(clip["end_ms"] - original["end_ms"]),
    ))
    return original, alternatives[:3]


def original_decision(original, anchor, provider, model):
    matching = None
    with (OUTPUTS / "judge_attempts.jsonl").open(encoding="utf-8-sig") as log:
        for line in log:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("clip_id") == original["clip_id"] and row.get("provider") == provider
                and row.get("model") == model and not row.get("error") and row.get("raw")
            ):
                matching = json.loads(row["raw"])
    if matching is None:
        raise ValueError("No validated original judge result for the current provider and model.")
    validate_result(matching, evidence_for(original))
    if matching["decision"] != "refine" or matching["refinement_event_id"] != anchor:
        raise ValueError("The requested anchor does not match the original refine decision.")
    return matching


def score(result):
    return sum(result[field] for field in SCORES), result["hook_score"]


def describe(clip, result=None):
    row = {field: clip[field] for field in ("clip_id", "episode", "start_ms", "end_ms")}
    row.update(start=timestamp(clip["start_ms"]), end=timestamp(clip["end_ms"]))
    if result is not None:
        row.update(judgment=result, total_score=score(result)[0])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--anchor-event-id", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Print candidate cuts without model calls.")
    args = parser.parse_args()
    try:
        original, alternatives = find_nearby_cuts(args.clip_id, args.anchor_event_id)
        print(f"Original: {args.clip_id}. Anchor: {args.anchor_event_id}. Alternatives: {len(alternatives)}.")
        for clip in alternatives:
            print(f"  {clip['clip_id']}: {timestamp(clip['start_ms'])} to {timestamp(clip['end_ms'])}")
        if args.dry_run:
            return 0

        provider, model = configuration()
        baseline = original_decision(original, args.anchor_event_id, provider, model)
        report = {
            "tool_call": {
                "name": "find_nearby_cuts",
                "clip_id": args.clip_id,
                "anchor_event_id": args.anchor_event_id,
            },
            "provider": provider,
            "model": model,
            "original": describe(original, baseline),
            "alternatives": [],
            "model_requests": 0,
            "manual_review": None,
        }
        winner = original if all(baseline[field] > 0 for field in SCORES) else None
        winning_result = baseline if winner else None
        for clip in alternatives:
            evidence = evidence_for(clip)
            prompt = RUBRIC + "\nClip evidence in JSON, with times relative to the cut:\n"
            prompt += json.dumps(evidence, ensure_ascii=False)
            row = describe(clip)
            row["attempts"] = []

            def record(attempt, raw, error):
                row["attempts"].append({"attempt": attempt, "raw": raw, "error": error})
                if type(attempt) is int:
                    report["model_requests"] += 1

            try:
                result = complete(prompt, lambda value: validate_result(value, evidence), record)
            except (ProviderError, ValueError) as error:
                row["error"] = str(error)
                print(f"  {clip['clip_id']}: {error}")
            else:
                row.update(judgment=result, total_score=score(result)[0])
                print(f"  {clip['clip_id']}: {result['decision']} | {score(result)[0]}/9")
                if result["decision"] == "keep" and (
                    winning_result is None or score(result) > score(winning_result)
                ):
                    winner, winning_result = clip, result
            report["alternatives"].append(row)

        report["selected"] = describe(winner, winning_result) if winner else None
        report["outcome"] = (
            "rejected" if winner is None else "original_retained" if winner is original else "revised"
        )
        path = OUTPUTS / f"refinement_{args.clip_id}.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Outcome: {report['outcome']}. Model requests: {report['model_requests']}.")
        print(f"Saved {path}")
        return 1 if any("error" in row for row in report["alternatives"]) else 0
    except (OSError, KeyError, ValueError, ProviderError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
