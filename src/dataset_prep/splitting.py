"""Relationship-aware splitting utilities.

The pilot splits *before* windows are finalized: every identifiable participant
of a generated example (source speaker, target speaker, source/target reference
recordings) must live in the same partition as the example.  This module
provides:

* a small union-find over relationship keys,
* a deterministic greedy partitioner that balances window counts across
  train/validation while respecting component integrity,
* a per-group window cap helper.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Hashable, Iterable, Mapping, Sequence

RelationshipKey = tuple[str, str]


class UnionFind:
    """Minimal union-find over hashable keys."""

    def __init__(self) -> None:
        self._parent: dict[Hashable, Hashable] = {}

    def find(self, key: Hashable) -> Hashable:
        parent = self._parent.setdefault(key, key)
        while parent != self._parent[parent]:
            self._parent[parent] = self._parent[self._parent[parent]]
            parent = self._parent[parent]
        return parent

    def union(self, left: Hashable, right: Hashable) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            # Deterministic root choice: smallest key wins.
            if str(root_left) <= str(root_right):
                self._parent[root_right] = root_left
            else:
                self._parent[root_left] = root_right


def relationship_components(
    items: Sequence[Mapping[str, Any]],
    keys_of: Callable[[Mapping[str, Any]], Iterable[RelationshipKey]],
) -> list[list[int]]:
    """Group item indices into connected components over their keys."""
    finder = UnionFind()
    item_roots: list[Hashable] = []
    for index, item in enumerate(items):
        keys = list(keys_of(item))
        if not keys:
            # An item with no relationship keys forms a singleton component.
            sentinel = ("item", str(index))
            finder.union(sentinel, sentinel)
            item_roots.append(sentinel)
            continue
        first = keys[0]
        finder.union(first, first)
        for key in keys[1:]:
            finder.union(first, key)
        item_roots.append(first)

    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, root in enumerate(item_roots):
        grouped[finder.find(root)].append(index)
    return [sorted(indices) for _, indices in sorted(grouped.items(), key=lambda kv: str(kv[0]))]


def component_keys(
    items: Sequence[Mapping[str, Any]],
    components: Sequence[Sequence[int]],
    keys_of: Callable[[Mapping[str, Any]], Iterable[RelationshipKey]],
) -> list[set[RelationshipKey]]:
    """Return the union of relationship keys for each component."""
    result: list[set[RelationshipKey]] = []
    for component in components:
        keys: set[RelationshipKey] = set()
        for index in component:
            keys.update(keys_of(items[index]))
        result.append(keys)
    return result


def split_components(
    components: Sequence[Sequence[int]],
    window_counts: Sequence[int],
    *,
    val_fraction: float,
    seed: int,
) -> dict[int, str]:
    """Assign component indices to ``train``/``val`` by greedy balance.

    ``window_counts[i]`` is the number of selected windows contributed by
    component ``i``.  Components are visited largest-first with a seeded
    shuffle among equal sizes, so a tiny number of dense components cannot make
    the validation side empty.  Returns a mapping component index -> split.
    """
    if len(components) != len(window_counts):
        raise ValueError("components and window_counts must be aligned.")
    if not components:
        return {}
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")

    total = sum(window_counts)
    target_val = max(1, int(round(total * val_fraction))) if len(components) > 1 else 0
    order = sorted(range(len(components)), key=lambda i: (-window_counts[i], i))
    rng = random.Random(seed)
    # Stable shuffle among equal-weight components.
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in order:
        grouped[window_counts[index]].append(index)
    shuffled: list[int] = []
    for weight in sorted(grouped, reverse=True):
        bucket = grouped[weight]
        rng.shuffle(bucket)
        shuffled.extend(bucket)

    assignment: dict[int, str] = {}
    val_total = 0
    for index in shuffled:
        count = window_counts[index]
        if val_total < target_val and (total - val_total - count) >= target_val:
            assignment[index] = "val"
            val_total += count
        else:
            assignment[index] = "train"
    # Ensure every component ends up somewhere and val is not empty when possible.
    if target_val > 0 and val_total == 0 and len(shuffled) > 1:
        for index in shuffled:
            assignment[index] = "val"
            break
    return assignment


def cap_per_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: Callable[[Mapping[str, Any]], Hashable],
    cap: int,
    seed: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Keep at most ``cap`` rows per group, deterministically; return (kept, dropped)."""
    if cap < 1:
        raise ValueError("cap must be positive.")
    grouped: dict[Hashable, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[group_key(row)].append(index)
    rng = random.Random(seed)
    kept: list[Mapping[str, Any]] = []
    dropped: list[Mapping[str, Any]] = []
    for _, indices in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            (kept if position < cap else dropped).append(rows[index])
    return kept, dropped


__all__ = [
    "UnionFind",
    "cap_per_group",
    "component_keys",
    "relationship_components",
    "split_components",
]
