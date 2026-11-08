import json
from pathlib import Path

INPUT = Path("outputs/parsed_events.json")
OUTPUT = Path("outputs/candidate_windows.json")
TARGET_DURATIONS = (25_000, 40_000, 55_000)


def generate_windows(episode, events):
    dialogue = [event for event in events if event["type"] == "dialogue"]

    def safe_cut(time):
        return not any(line["start_ms"] < time < line["end_ms"] for line in dialogue)

    starts = sorted({
        event["start_ms"] for event in events
        if event["type"] in ("dialogue", "action")
    })
    ends = sorted({event["end_ms"] for event in events if safe_cut(event["end_ms"])})

    for start in starts:
        if not safe_cut(start):
            continue

        # Filter for valid durations before choosing the closest end.
        valid_ends = [end for end in ends if 20_000 <= end - start <= 60_000]
        if not valid_ends:
            continue

        # A set removes duplicate endings for the same start.
        selected_ends = {
            min(valid_ends, key=lambda end: (abs(end - start - target), end))
            for target in TARGET_DURATIONS
        }

        for end in sorted(selected_ends):
            included = [
                event for event in events
                if event["start_ms"] < end and event["end_ms"] > start
            ]
            opening = [
                {
                    **event,
                    "opening_status": (
                        "ongoing_at_start" if event["start_ms"] < start
                        else "starts_in_opening"
                    ),
                    "fully_inside_opening": (
                        start <= event["start_ms"] and event["end_ms"] <= start + 3_000
                    ),
                }
                for event in included
                if event["start_ms"] < start + 3_000
            ]
            yield {
                "clip_id": f"E{int(episode):03d}_{start}_{end}",
                "episode": int(episode),
                "start_ms": start,
                "end_ms": end,
                "duration_ms": end - start,
                "events": included,
                "opening_events": opening,
                "text": "\n".join(
                    f"{event['label']}: {event['text']}" for event in included
                    if start <= event["start_ms"] and event["end_ms"] <= end
                ),
            }


def main():
    episodes = json.loads(INPUT.read_text(encoding="utf-8"))
    windows = [
        clip for episode, events in episodes.items()
        for clip in generate_windows(episode, events)
    ]
    OUTPUT.write_text(json.dumps(windows, ensure_ascii=False), encoding="utf-8")

    first_episode = [clip for clip in windows if clip["episode"] == 1]
    print(f"Generated {len(windows)} windows across {len(episodes)} episodes.")
    print(f"Episode 1: {len(first_episode)} windows")

    for clip in first_episode[:5]:
        start = clip["start_ms"]
        print(
            f"\n{clip['clip_id']} | {start / 1000:.2f}s to {clip['end_ms'] / 1000:.2f}s"
            f" | duration {clip['duration_ms'] / 1000:.2f}s"
        )
        for event in clip["opening_events"]:
            print(
                f"  {event['event_id']} | {event['label']}"
                f" | {event['start_ms'] / 1000:.2f}s to {event['end_ms'] / 1000:.2f}s"
                f" | {event['opening_status']}"
            )

    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()