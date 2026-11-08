"""Read timestamped screenplay events directly from a ZIP archive."""

import re
from pathlib import PurePosixPath
from zipfile import ZipFile


TIME_PATTERN = r"\d+:[0-5]\d(?:[.,]\d{1,3})?"
TIMING = re.compile(
    rf"Start\s*:\s*\(?\s*(?P<start>{TIME_PATTERN})\s*\)?"
    rf"\s*\|?\s*End\s*:\s*\(?\s*(?P<end>{TIME_PATTERN})\s*\)?",
    re.IGNORECASE,
)
MARKER = re.compile(
    r"^[ \t]*(?:"
    r"SCENE[ \t]+(?P<scene>\d+)[ \t]*:?"
    r"|\*\*(?P<label>[^\n]+?)\*\*"
    r"|(?P<end>---[ \t]*SCRIPT[ \t]+END[ \t]*---)"
    r")[ \t]*$",
    re.MULTILINE | re.IGNORECASE,
)
EPISODE_HEADER = re.compile(
    r"^[ \t]*EPISODE[ \t]+NUMBER\s*:\s*(?:Episode\s*)?(\d+)\b",
    re.MULTILINE | re.IGNORECASE,
)


def timestamp_to_ms(timestamp):
    """Example: 01:02.35 -> 62350, without floating point rounding."""
    timestamp = timestamp.strip().replace(",", ".")
    if not re.fullmatch(TIME_PATTERN, timestamp):
        raise ValueError(f"Invalid timestamp: {timestamp!r}")
    minutes, seconds = timestamp.split(":")
    whole_seconds, _, fraction = seconds.partition(".")
    return (int(minutes) * 60 + int(whole_seconds)) * 1000 + int(
        fraction.ljust(3, "0")
    )


def parse_episode(text, filename):
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    header = EPISODE_HEADER.search(text)
    filename_match = re.search(
        r"episode[-_ ]+(\d+)\b", PurePosixPath(filename).name, re.IGNORECASE
    )
    if not header and not filename_match:
        raise ValueError("Cannot identify the episode number")
    if header and filename_match and int(header[1]) != int(filename_match[1]):
        raise ValueError("Episode header and filename disagree")
    episode = int((header or filename_match)[1])
    markers = list(MARKER.finditer(text))
    events = []
    scene = None

    for index, marker in enumerate(markers):
        if marker.group("scene") is not None:
            scene = int(marker.group("scene"))
            continue
        if marker.group("end"):
            break
        label = marker.group("label").strip().rstrip(":").strip()
        next_position = len(text)
        if index + 1 < len(markers):
            next_position = markers[index + 1].start()
        block = text[marker.end():next_position].strip()
        timing = TIMING.match(block)
        line_number = text.count("\n", 0, marker.start()) + 1
        if scene is None:
            raise ValueError(f"Line {line_number}: event appears before a scene")
        if timing is None:
            raise ValueError(f"Line {line_number}: invalid/missing event timing")
        start_ms = timestamp_to_ms(timing.group("start"))
        end_ms = timestamp_to_ms(timing.group("end"))
        content = block[timing.end():].strip()
        if end_ms <= start_ms:
            raise ValueError(f"Line {line_number}: end must be after start")
        if not content:
            raise ValueError(f"Line {line_number}: empty event text")

        # In this screenplay convention, all other labels name dialogue speakers.
        event_type = {
            "CAMERA": "camera",
            "ACTION": "action",
            "SOUND FX": "sound",
            "SFX": "sound",
            "SOUND": "sound",
            "MUSIC": "sound",
        }.get(" ".join(label.upper().split()), "dialogue")
        events.append({
            "event_id": f"ep{episode:03d}_event{len(events) + 1:04d}",
            "episode": episode,
            "scene": scene,
            "type": event_type,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": content,
            "label": label,
        })

    # Catch timestamped blocks accidentally skipped due to a missing header.
    timing_lines = re.findall(r"^[ \t]*Start\s*:", text, re.MULTILINE | re.IGNORECASE)
    if not events or len(events) != len(timing_lines):
        raise ValueError(
            f"Found {len(timing_lines)} event starts but parsed {len(events)} events"
        )
    # Python's stable sort keeps source order for equal start times.
    events.sort(key=lambda event: event["start_ms"])
    return episode, events


def parse_zip(zip_path):
    episodes = {}
    failures = []
    sources = {}
    with ZipFile(zip_path) as archive:
        members = [
            member for member in archive.infolist()
            if not member.is_dir()
            and member.filename.lower().endswith(".txt")
            and "__MACOSX" not in PurePosixPath(member.filename).parts
            and not PurePosixPath(member.filename).name.startswith("._")
        ]
        if not members:
            raise ValueError("ZIP contains no screenplay .txt files")
        for member in sorted(members, key=lambda item: item.filename):
            try:
                text = archive.read(member).decode("utf-8-sig")
                episode, events = parse_episode(text, member.filename)
            except (ValueError, OSError, RuntimeError) as error:
                failures.append({"file": member.filename, "error": str(error)})
                continue
            if episode in episodes:
                raise ValueError(
                    f"Duplicate episode {episode}: {sources[episode]} and {member.filename}"
                )
            episodes[episode] = events
            sources[episode] = member.filename
    return dict(sorted(episodes.items())), failures
