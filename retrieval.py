import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INPUT = Path("outputs/candidate_windows.json")
OUTPUT = Path("outputs/retrieved_candidates.json")

QUERIES = {
    "betrayal": "A lover publicly betrays or abandons their partner.",
    "identity reveal": "A clue exposes a secret identity or unexpected family relationship.",
    "proof withheld": "Someone demands proof of an accusation during a tense confrontation.",
    "humiliation": "A shocking slap, insult, or public humiliation in front of witnesses.",
    "danger": "An urgent threat leaves someone trapped or in immediate danger.",
}


def event_texts(clip):
    # Only use events that fit completely inside the clip.
    return [
        f"{event['label']}: {event['text']}"
        for event in clip["events"]
        if event["type"] in ("dialogue", "action", "camera")
        and clip["start_ms"] <= event["start_ms"]
        and event["end_ms"] <= clip["end_ms"]
    ]


def encode_events(model, texts):
    payload = json.dumps([MODEL, texts], ensure_ascii=False).encode("utf-8")
    key = hashlib.sha256(payload).hexdigest()[:16]
    cache = Path("cache") / f"events_{key}.npy"

    if cache.exists():
        print("Loading cached event embeddings.")
        return np.load(cache, allow_pickle=False)

    vectors = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    cache.parent.mkdir(exist_ok=True)
    np.save(cache, vectors, allow_pickle=False)
    return vectors


def overlap_fraction(first, second):
    shared = min(first["end_ms"], second["end_ms"]) - max(
        first["start_ms"], second["start_ms"]
    )
    return max(0, shared) / min(first["duration_ms"], second["duration_ms"])


def shortlist_episode(clips, scores, episode):
    rankings = np.argsort(-scores, axis=0, kind="stable")[:12].T.tolist()
    selected = []

    if episode == 1:
        for query, ranking in zip(QUERIES, rankings):
            ids = [clips[index]["clip_id"] for index in ranking[:3]]
            print(f"  {query}: " + ", ".join(ids))

    # Each query gets a turn before any query gets a second turn.
    for _ in range(2):
        for column, (query, ranking) in enumerate(zip(QUERIES, rankings)):
            while ranking:
                index = ranking.pop(0)
                clip = clips[index]
                if all(overlap_fraction(clip, other) <= 0.60 for other in selected):
                    selected.append({
                        **clip,
                        "retrieval_query": query,
                        "retrieval_score": float(scores[index, column]),
                    })
                    break
            if len(selected) == 6:
                return selected

    return selected


def main():
    clips = json.loads(INPUT.read_text(encoding="utf-8"))
    clip_texts = [event_texts(clip) for clip in clips]
    unique_texts = list(dict.fromkeys(text for texts in clip_texts for text in texts))
    if not unique_texts:
        raise SystemExit("No complete dialogue or visual events found in the windows.")

    model = SentenceTransformer(MODEL)
    event_vectors = encode_events(model, unique_texts)
    text_rows = {text: index for index, text in enumerate(unique_texts)}
    query_vectors = model.encode(list(QUERIES.values()), normalize_embeddings=True)

    grouped = defaultdict(list)
    for clip, texts in zip(clips, clip_texts):
        if texts:
            rows = [text_rows[text] for text in texts]
            vector = event_vectors[rows].mean(axis=0)
            norm = np.linalg.norm(vector)
            if norm > 0:
                grouped[clip["episode"]].append((clip, vector / norm))

    selected = []
    for episode, entries in sorted(grouped.items()):
        episode_clips, vectors = zip(*entries)
        # With normalized vectors, the dot product equals cosine similarity.
        scores = np.stack(vectors) @ query_vectors.T

        if episode == 1:
            print("\nEpisode 1 top 3 per query, before overlap filtering:")

        picks = shortlist_episode(episode_clips, scores, episode)
        selected.extend(picks)

        if episode == 1:
            print("\nEpisode 1 final shortlist:")
            for clip in picks:
                print(
                    f"  {clip['clip_id']} | {clip['retrieval_query']} | "
                    f"{clip['retrieval_score']:.3f}"
                )

    OUTPUT.write_text(json.dumps(selected, ensure_ascii=False), encoding="utf-8")
    episode_count = len({clip["episode"] for clip in selected})
    print(f"\nShortlisted {len(selected)} windows from {episode_count} episodes.")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()