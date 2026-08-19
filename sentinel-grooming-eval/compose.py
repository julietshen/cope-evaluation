"""Compose long multi-turn eval conversations from expert-labeled single lines.

The synthetic dataset ships 1-line snippets (index seeds) and 10-line
conversations (short tests), but nothing long. Real grooming plays out over
hundreds of messages in which almost everything is mundane. This module builds
that shape deliberately: innocuous background lines with expert-confirmed
grooming lines injected at a controlled density, risk tier, and onset, so every
composed conversation has conversation-level ground truth by construction.

Lines are sampled with replacement across conversations (the labeled pools are
small), but never within one conversation.
"""

import random
from dataclasses import dataclass
from typing import List, Optional

from data import GroomingLinePools


@dataclass
class ComposedConversation:
    name: str
    label: int                      # 1 if any grooming lines injected
    lines: List[str]
    grooming_positions: List[int]   # indices of injected lines, ascending
    length: int
    density: float
    tier: str                       # high / med / low / none
    onset: str                      # uniform / late


def _sample(pool: List[str], k: int, rng: random.Random) -> List[str]:
    """Sample k lines, without replacement while the pool lasts."""
    if k <= len(pool):
        return rng.sample(pool, k)
    return list(pool) + rng.choices(pool, k=k - len(pool))


def compose_conversation(
    name: str,
    length: int,
    density: float,
    tier: str,
    innocuous_pool: List[str],
    grooming_pools: Optional[GroomingLinePools],
    rng: random.Random,
    onset: str = "uniform",
) -> ComposedConversation:
    n_grooming = 0 if density == 0 else max(1, round(length * density))
    n_innocuous = length - n_grooming

    lines = _sample(innocuous_pool, n_innocuous, rng)

    positions: List[int] = []
    if n_grooming:
        grooming_lines = _sample(grooming_pools.tier(tier), n_grooming, rng)
        # "late" onset: grooming only appears in the final 40% of the
        # conversation, mimicking escalation after rapport-building.
        first_eligible = int(length * 0.6) if onset == "late" else 0
        positions = sorted(rng.sample(range(first_eligible, length), n_grooming))
        for position, line in zip(positions, grooming_lines):
            lines.insert(position, line)
        lines = lines[:length]

    return ComposedConversation(
        name=name,
        label=1 if n_grooming else 0,
        lines=lines,
        grooming_positions=positions,
        length=length,
        density=density,
        tier=tier if n_grooming else "none",
        onset=onset,
    )


def build_grid(
    innocuous_pool: List[str],
    grooming_pools: GroomingLinePools,
    lengths=(10, 25, 50, 100, 200),
    densities=(0.05, 0.10, 0.20),
    tiers=("high", "med", "low"),
    positives_per_cell: int = 20,
    negatives_per_length: int = 40,
    seed: int = 7,
) -> List[ComposedConversation]:
    rng = random.Random(seed)
    conversations = []
    for length in lengths:
        for i in range(negatives_per_length):
            conversations.append(compose_conversation(
                f"neg_L{length}_{i:02d}", length, 0.0, "high",
                innocuous_pool, None, rng))
        for density in densities:
            for tier in tiers:
                for i in range(positives_per_cell):
                    conversations.append(compose_conversation(
                        f"pos_L{length}_d{int(density * 100):02d}_{tier}_{i:02d}",
                        length, density, tier, innocuous_pool, grooming_pools, rng))
    return conversations


def build_late_onset_set(
    innocuous_pool: List[str],
    grooming_pools: GroomingLinePools,
    length: int = 100,
    density: float = 0.10,
    tier: str = "high",
    n: int = 30,
    seed: int = 11,
) -> List[ComposedConversation]:
    rng = random.Random(seed)
    return [
        compose_conversation(
            f"late_L{length}_{i:02d}", length, density, tier,
            innocuous_pool, grooming_pools, rng, onset="late")
        for i in range(n)
    ]
