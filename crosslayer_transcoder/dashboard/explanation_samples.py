"""CPU-only, conversation-disjoint examples from the existing sparse gate cache."""

import numpy as np


def index_features(cache, n_features):
    """Index a large sparse cache once instead of scanning it for every feature."""
    order = np.argsort(cache["features"], kind="stable")
    cuts = np.searchsorted(cache["features"][order], np.arange(n_features + 1))
    cache["feature_index"] = (order, cuts)


def feature_order(stats):
    """Visit every observed transform once, interleaving ranks by frequency."""
    ranks = sorted({s["rank"] for s in stats}, reverse=True)
    pools = [
        [
            s["id"]
            for s in sorted(stats, key=lambda s: (-s["count"], s["id"]))
            if s["rank"] == rank and s["count"] > 0
        ]
        for rank in ranks
    ]
    return [
        pool[i]
        for i in range(max(map(len, pools), default=0))
        for pool in pools
        if i < len(pool)
    ]


def split_conversations(count, seed=42):
    order = np.random.default_rng(seed).permutation(count)
    cut = max(1, min(count - 1, int(count * 0.75)))
    return set(order[:cut].tolist()), set(order[cut:].tolist())


def feature_examples(
    cache,
    feature,
    vocab,
    train_conversations,
    test_conversations,
    n_train=12,
    n_test=5,
    context=64,
    seed=42,
):
    """Use one peak per conversation and prefixes ending at that peak.

    Negatives are zero-gate windows in held-out conversations. No future tokens
    are shown: these explanations describe gate triggers, not output effects.
    """
    if train_conversations & test_conversations:
        raise ValueError("Explanation and evaluation conversations overlap")
    tokens, offsets = cache["tokens"], cache["offsets"]
    if "feature_index" in cache:
        order, cuts = cache["feature_index"]
        chosen = order[cuts[feature] : cuts[feature + 1]]
    else:
        chosen = cache["features"] == feature
    positions, values = cache["positions"][chosen], cache["values"][chosen]
    # Python integer search keys otherwise promote an int32 array to int64 on
    # every search. Dense gates can require thousands of negative-window probes.
    positions = positions.astype(np.int64, copy=False)
    convs = np.searchsorted(offsets, positions, side="right") - 1
    peaks = {}
    for j in np.argsort(-values, kind="stable"):
        peaks.setdefault(int(convs[j]), int(positions[j]))

    def window(conv, stop):
        start = max(int(offsets[conv]), stop - context)
        ids = tokens[start:stop]
        acts = np.zeros(len(ids), dtype=np.float32)
        left, right = np.searchsorted(positions, [start, stop])
        acts[positions[left:right] - start] = values[left:right]
        return dict(
            conversation=conv,
            start=start,
            stop=stop,
            tokens=ids.astype(int).tolist(),
            activations=acts.tolist(),
            str_tokens=[vocab[str(int(t))] for t in ids],
        )

    train = [(c, p) for c, p in peaks.items() if c in train_conversations]
    test = [(c, p) for c, p in peaks.items() if c in test_conversations]
    # Mix the strongest examples with lower-strength examples, without repeats.
    top = min(8, n_train, len(train))
    if len(train) > n_train:
        rest = np.linspace(top, len(train) - 1, n_train - top, dtype=int)
        train = train[:top] + [train[i] for i in rest]
    if len(test) > n_test:
        test = [test[i] for i in np.linspace(0, len(test) - 1, n_test, dtype=int)]
    train = [window(c, p + 1) for c, p in train]
    test = [window(c, p + 1) for c, p in test]
    rng = np.random.default_rng(seed + feature)
    negatives = []
    used = {x["conversation"] for x in test}
    for conv in rng.permutation(sorted(test_conversations)):
        conv = int(conv)
        if conv in used:
            continue
        lo, hi = int(offsets[conv]), int(offsets[conv + 1])
        first = min(lo + context, hi)
        candidates = np.unique(
            np.concatenate([[hi], rng.integers(first, hi + 1, size=32)])
        )
        rng.shuffle(candidates)
        for stop in candidates:
            start = max(lo, int(stop) - context)
            left, right = np.searchsorted(positions, [start, stop])
            if left == right:
                negatives.append(window(conv, int(stop)))
                break
        if len(negatives) >= n_test:
            break
    return dict(
        train=train,
        test=test,
        negatives=negatives,
        active_conversations=len(peaks),
        active_tokens=len(positions),
    )


def detection_metrics(rows):
    """Keep parse failures visible rather than treating them as correct negatives."""
    valid = [r for r in rows if r["prediction"] is not None]
    tp = sum(r["activating"] and r["prediction"] for r in valid)
    tn = sum(not r["activating"] and not r["prediction"] for r in valid)
    fp = sum(not r["activating"] and r["prediction"] for r in valid)
    fn = sum(r["activating"] and not r["prediction"] for r in valid)
    return dict(
        total=len(rows),
        valid=len(valid),
        parse_failures=len(rows) - len(valid),
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        accuracy=(tp + tn) / len(valid) if valid else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        balanced_accuracy=((tp / (tp + fn)) + (tn / (tn + fp))) / 2
        if (tp + fn) and (tn + fp)
        else None,
    )
