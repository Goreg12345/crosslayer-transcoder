"""Deterministic conversation peaks shared by activation and logit views."""

import numpy as np


def example_peak_indices(positions, values, offsets):
    conversations = np.searchsorted(offsets, positions, side="right") - 1
    peaks = {}
    for j in np.argsort(-values, kind="stable"):
        peaks.setdefault(int(conversations[j]), int(j))
    ranked = list(peaks.values())
    remaining = ranked[12:]
    middle = remaining[len(remaining) // 2 : len(remaining) // 2 + 6]
    weak = [j for j in remaining[-6:] if j not in middle]
    return conversations, [
        ("Strongest examples (one per conversation)", ranked[:12]),
        ("Medium-strength examples", middle),
        ("Lower-strength examples", weak),
    ]
