"""Data loading for the grooming evaluation.

Sources:
- examples/synthetic-grooming-conversations/ (synthetic 1-line snippets, 10-line
  conversations, and expert annotations)
- ../PervertedJusticeDataset/predators/ (real predator-side chat logs; read-only)

Leakage rules enforced here:
- Index seeds come only from UNannotated 1-line snippets.
- Expert-annotated 1-line snippets are reserved for composing eval conversations.
- The innocuous 1-line pool is split by seed: one half seeds the index negatives,
  the other half is background material for composed eval conversations.
"""

import csv
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# This harness lives outside the Sentinel repo; point it at the datasets via
# env vars (defaults match the local ROOST working-folder layout).
SYNTH_ROOT = Path(os.environ.get(
    "SYNTH_GROOMING_DATA",
    str(Path.home() / "ROOST/Sentinel/examples/synthetic-grooming-conversations")))
PJ_ROOT = Path(os.environ.get(
    "PJ_DATASET", str(Path.home() / "ROOST/PervertedJusticeDataset")))

_SPEAKER_RE = re.compile(r"^\s*[A-Za-z][A-Za-z .'-]{0,19}:\s*")


def strip_speaker(line: str) -> str:
    """Remove a leading 'Name: ' speaker tag from a synthetic chat line."""
    return _SPEAKER_RE.sub("", line).strip()


def _read_conversation(path: Path) -> List[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = strip_speaker(raw)
        if len(text) > 1:
            lines.append(text)
    return lines


@dataclass
class GroomingLinePools:
    """Expert-labeled single grooming lines, bucketed by annotated risk."""

    high: List[str] = field(default_factory=list)
    med: List[str] = field(default_factory=list)
    low: List[str] = field(default_factory=list)

    def tier(self, name: str) -> List[str]:
        return getattr(self, name)


def load_annotated_singles() -> Dict[str, GroomingLinePools]:
    """Load the expert-annotated 1-line snippets.

    Returns a dict with:
      pools: GroomingLinePools of confirmed grooming lines by risk tier
      filenames: set of every annotated filename (excluded from index seeds)
    """
    path = SYNTH_ROOT / "annotations" / "grooming_single_annotated.csv"
    pools = GroomingLinePools()
    filenames = set()
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            filenames.add(row["filename"].strip())
            label = row["label"].strip()
            risk = row["grooming risk"].strip()
            if label == "grooming" and risk in ("high", "med", "low"):
                text = strip_speaker(row["content"])
                if len(text) > 1:
                    pools.tier(risk).append(text)
    return {"pools": pools, "filenames": filenames}


def load_single_line_snippets(kind: str, exclude_filenames=frozenset()) -> List[str]:
    """Load 1-line snippets of a kind ('grooming' or 'innocuous'), one text per file."""
    directory = SYNTH_ROOT / "conversations" / "synth_chat_snippets_1l" / kind
    texts = []
    for path in sorted(directory.glob("*.txt")):
        if path.name in exclude_filenames:
            continue
        lines = _read_conversation(path)
        if lines:
            texts.append(lines[0])
    return texts


def split_pool(items: List[str], seed: int, fraction: float = 0.5):
    """Deterministically split a pool in two disjoint parts."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * fraction)
    return shuffled[:cut], shuffled[cut:]


def load_generated_10l(n_per_class: int, seed: int):
    """Sample shipped 10-line conversations as (name, label, lines).

    NOTE: the expert annotations in grooming_multiple_annotated.csv cannot be
    used - every filename in multiple_mapping.csv refers to a Dec 9-10
    generation batch that is not shipped in this repo (the shipped 10-line
    conversations are Dec 18+). Labels here are therefore the *generated*
    labels (which directory Gemma3 was asked to write into), which the 1-line
    annotation pass suggests are noisy. Metrics on this set are a lower bound.
    """
    rng = random.Random(seed)
    conversations = []
    for kind, label in (("grooming", 1), ("innocuous", 0)):
        directory = SYNTH_ROOT / "conversations" / "synth_chat_snippets_10l" / kind
        paths = sorted(directory.glob("*.txt"))
        for path in rng.sample(paths, min(n_per_class, len(paths))):
            lines = _read_conversation(path)
            if lines:
                conversations.append((f"{kind}_{path.stem[-8:]}", label, lines))
    return conversations


def load_pj_conversations(n: int, max_lines: int, seed: int):
    """Sample real predator-side chat logs as (name, lines). Read-only dataset."""
    paths = sorted((PJ_ROOT / "predators").glob("*.txt"))
    rng = random.Random(seed)
    chosen = rng.sample(paths, min(n, len(paths)))
    conversations = []
    for path in chosen:
        lines = []
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = raw.strip()
            if len(text) > 1:
                lines.append(text)
            if len(lines) >= max_lines:
                break
        if len(lines) >= 20:
            conversations.append((path.stem, lines))
    return conversations
