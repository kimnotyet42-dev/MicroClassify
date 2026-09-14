from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from dataio.datasets import Sample


@dataclass(frozen=True)
class FewShotSplit:
    support_groups: List[List[int]]
    query_indices: List[int]
    query_labels: np.ndarray


@dataclass(frozen=True)
class FSCILSplit:
    sessions: List[List[str]]
    support_by_class: Dict[str, List[int]]
    test_by_class: Dict[str, List[int]]


MICROSCOPY_TARGETS = {
    "golden_mussel": (
        "淡水壳菜踯行幼虫",
        "淡水壳菜壳顶幼虫",
        "淡水壳菜D型幼虫",
        "淡水壳菜担轮幼虫",
        "淡水壳菜受精卵",
        "淡水壳菜n_cell阶段",
    ),
    "Daphnia": ("大型溞",),
}


def group_indices(samples: Sequence[Sample]) -> Dict[str, List[int]]:
    grouped = defaultdict(list)  # type: Dict[str, List[int]]
    for index, sample in enumerate(samples):
        grouped[sample.label].append(index)
    return dict(grouped)


def zoolake_episodes(
    samples: Sequence[Sample],
    way: int = 32,
    shot: int = 5,
    query: int = 15,
    episodes: int = 200,
    seed: int = 0,
    class_names: Optional[Sequence[str]] = None,
) -> List[FewShotSplit]:
    grouped = group_indices(samples)
    required = shot + query
    usable = sorted(label for label, indices in grouped.items() if len(indices) >= required)
    if class_names is not None:
        requested = list(class_names)
        missing = [label for label in requested if label not in usable]
        if missing:
            raise ValueError("ZooLake2 classes lack required images: {0}".format(missing))
        usable = requested
        way = len(usable)
    if len(usable) < way:
        raise ValueError(f"ZooLake2 has {len(usable)} eligible classes; requested {way}")
    rng = np.random.RandomState(seed)
    output = []
    for _ in range(episodes):
        selected = (
            np.asarray(usable)
            if class_names is not None
            else rng.choice(usable, size=way, replace=False)
        )
        support_groups = []
        query_indices = []
        query_labels = []
        for class_index, label in enumerate(selected):
            chosen = rng.choice(grouped[label], size=required, replace=False)
            support_groups.append(chosen[:shot].tolist())
            query_indices.extend(chosen[shot:].tolist())
            query_labels.extend([class_index] * query)
        output.append(
            FewShotSplit(
                support_groups,
                query_indices,
                np.asarray(query_labels, dtype=np.int64),
            )
        )
    return output


def emds_allway_splits(
    samples: Sequence[Sample],
    shot: int = 4,
    seeds: int = 5,
) -> List[FewShotSplit]:
    grouped = group_indices(samples)
    classes = sorted(label for label, indices in grouped.items() if len(indices) > shot)
    output = []
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        support_groups = []
        query_indices = []
        query_labels = []
        for class_index, label in enumerate(classes):
            indices = list(grouped[label])
            rng.shuffle(indices)
            support_groups.append(indices[:shot])
            query_indices.extend(indices[shot:])
            query_labels.extend([class_index] * (len(indices) - shot))
        output.append(
            FewShotSplit(
                support_groups,
                query_indices,
                np.asarray(query_labels, dtype=np.int64),
            )
        )
    return output


def microscopy_split(
    samples: Sequence[Sample],
    task: str,
    seed: int,
    shot: int = 3,
) -> FewShotSplit:
    if task not in MICROSCOPY_TARGETS:
        raise KeyError(f"Unknown microscopy task: {task}")
    grouped = group_indices(samples)
    rng = np.random.RandomState(seed)
    support_groups = []
    support_indices = set()
    for positive_class in MICROSCOPY_TARGETS[task]:
        candidates = grouped.get(positive_class, [])
        if len(candidates) < shot:
            raise ValueError(f"{positive_class} has fewer than {shot} images")
        chosen = rng.choice(candidates, size=shot, replace=False).tolist()
        support_groups.append(chosen)
        support_indices.update(chosen)
    query_indices = [index for index in range(len(samples)) if index not in support_indices]
    positive = set(MICROSCOPY_TARGETS[task])
    query_labels = np.asarray(
        [int(samples[index].label in positive) for index in query_indices],
        dtype=np.int64,
    )
    return FewShotSplit(support_groups, query_indices, query_labels)


def emds_class_pool(
    samples: Sequence[Sample],
    class_pool_size: int,
    repeat: int,
    minimum_class_size: int = 12,
) -> Tuple[List[int], List[str]]:
    grouped = group_indices(samples)
    candidates = sorted(
        label for label, indices in grouped.items() if len(indices) >= minimum_class_size
    )
    if class_pool_size > len(candidates):
        raise ValueError(
            f"Requested {class_pool_size} classes from {len(candidates)} candidates"
        )
    selected = (
        candidates
        if class_pool_size == len(candidates)
        else np.random.RandomState(1000 + repeat)
        .choice(candidates, class_pool_size, replace=False)
        .tolist()
    )
    selected_set = set(selected)
    indices = [index for index, sample in enumerate(samples) if sample.label in selected_set]
    return indices, selected


def zoolake_shot_splits(
    samples: Sequence[Sample],
    shot: int,
    maximum_shot: int = 24,
    query: int = 15,
    episodes: int = 100,
    seed: int = 0,
) -> Tuple[List[str], List[FewShotSplit]]:
    """Return the fixed-class ZooLake2 protocol used by the shot curve."""
    grouped = group_indices(samples)
    required = maximum_shot + query
    classes = sorted(
        label for label, indices in grouped.items() if len(indices) >= required
    )
    splits = zoolake_episodes(
        samples,
        shot=shot,
        query=query,
        episodes=episodes,
        seed=seed,
        class_names=classes,
    )
    return classes, splits


def emds_fscil_splits(
    samples: Sequence[Sample],
    base_shot: int = 25,
    seeds: int = 10,
    random_partition: bool = True,
    base_classes: int = 20,
    incremental_way: int = 3,
    incremental_shot: int = 3,
    test_fraction: float = 0.4,
) -> List[FSCILSplit]:
    """Build the whole-field EMDS-7 FSCIL protocol used in the manuscript."""
    grouped = group_indices(samples)

    def train_pool_size(label: str) -> int:
        count = len(grouped[label])
        return count - max(int(test_fraction * count), 1)

    eligible = sorted(
        [label for label in grouped if train_pool_size(label) >= base_shot],
        key=lambda label: -len(grouped[label]),
    )
    by_size = sorted(grouped, key=lambda label: -len(grouped[label]))
    if len(eligible) < base_classes:
        raise ValueError(
            "EMDS-7 has {0} eligible base classes; required {1}".format(
                len(eligible), base_classes
            )
        )

    output = []
    for seed in range(seeds):
        rng = np.random.RandomState(seed)
        if random_partition:
            base = list(rng.choice(eligible, size=base_classes, replace=False))
        else:
            base = list(eligible[:base_classes])
        incremental = [label for label in by_size if label not in base]
        sessions = [base]
        sessions.extend(
            incremental[offset : offset + incremental_way]
            for offset in range(0, len(incremental), incremental_way)
        )

        train_by_class = {}
        test_by_class = {}
        for label in grouped:
            indices = list(grouped[label])
            rng.shuffle(indices)
            test_count = max(int(test_fraction * len(indices)), 1)
            test_by_class[label] = indices[:test_count]
            train_by_class[label] = indices[test_count:]

        support = {}
        for label in base:
            support[label] = train_by_class[label][:base_shot]
            if len(support[label]) != base_shot:
                raise AssertionError("Insufficient base support for {0}".format(label))
        for label in incremental:
            pool = train_by_class[label]
            count = min(incremental_shot, len(pool))
            support[label] = list(rng.choice(pool, size=count, replace=False))

        output.append(FSCILSplit(sessions, support, test_by_class))
    return output
