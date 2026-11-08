"""Phase 1 CLI: parse all episodes and preview one episode."""

import argparse
import json
import sys
from pathlib import Path
from zipfile import BadZipFile

from parser import parse_zip


def main():
    cli = argparse.ArgumentParser(description="Parse timestamped screenplay ZIP files")
    cli.add_argument("zip_path", type=Path)
    cli.add_argument("--episode", type=int, default=1)
    cli.add_argument("--preview", type=int, default=15)
    cli.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = cli.parse_args()
    if args.preview < 1:
        cli.error("--preview must be at least 1")
    try:
        episodes, failures = parse_zip(args.zip_path)
    except (OSError, ValueError, RuntimeError, BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = args.output_dir / "parsed_events.json"
    failures_path = args.output_dir / "parse_failures.json"
    parsed_path.write_text(json.dumps(episodes, ensure_ascii=False, indent=2), encoding="utf-8")
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    for episode, events in episodes.items():
        print(f"Episode {episode:03d}: {len(events)} events")
    total = sum(len(events) for events in episodes.values())
    print(f"\nParsed {len(episodes)} episodes, {total} events, {len(failures)} failed files.")
    for failure in failures:
        print(f"FAILED: {failure['file']} -- {failure['error']}", file=sys.stderr)
    if args.episode not in episodes:
        print(f"ERROR: requested episode {args.episode} was not parsed", file=sys.stderr)
        return 1
    preview = episodes[args.episode][:args.preview]
    print(f"\nFirst {len(preview)} events of Episode {args.episode}:")
    print(json.dumps(preview, ensure_ascii=True, indent=2))
    print(f"\nSaved {parsed_path} and {failures_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
