"""Sampling and metrics for a fixed, reviewable MOLT explanation pilot."""

import json
import hashlib
import math
import re

import numpy as np

from .explanation_samples import detection_metrics, feature_examples


def spread(items, count, top=0):
    if len(items) <= count:
        return list(items)
    return list(items[:top]) + [
        items[i] for i in np.linspace(top, len(items) - 1, count - top, dtype=int)
    ]


def unique_examples(examples, excluded=None, positive=False):
    seen = set(excluded or ())
    result = []
    for example in examples:
        key = tuple(example["tokens"])
        if key in seen or len(key) < 16:
            continue
        if positive and not any(a == 0 for a in example["activations"]):
            continue
        seen.add(key)
        result.append(example)
    return result


def select_pilot(cache, stats, vocab, train, test, count=32, seed=42):
    """Freeze a cohort by rank and frequency before any LLM judgment is seen."""
    ranks = sorted({s["rank"] for s in stats}, reverse=True)
    order, cuts = cache["feature_index"]
    is_train = np.zeros(len(cache["offsets"]) - 1, dtype=bool)
    is_train[list(train)] = True
    pools = {}
    for rank in ranks:
        candidates = []
        for stat in sorted(
            (s for s in stats if s["rank"] == rank),
            key=lambda s: (-s["count"], s["id"]),
        ):
            ix = order[cuts[stat["id"]] : cuts[stat["id"] + 1]]
            if len(ix) < 232:
                continue
            conv = np.unique(
                np.searchsorted(cache["offsets"], cache["positions"][ix], side="right")
                - 1
            )
            n_train = int(is_train[conv].sum())
            if n_train >= 64 and len(conv) - n_train >= 100:
                candidates.append(stat["id"])
        pools[rank] = candidates
    rng = np.random.default_rng(seed)
    chosen, samples, rejected = [], {}, []
    for rank_index, rank in enumerate(ranks):
        quota = count // len(ranks) + int(rank_index < count % len(ranks))
        bins = np.array_split(pools[rank], quota)
        for bin_index, ids in enumerate(bins):
            accepted = False
            for feature in rng.permutation(ids).tolist():
                evidence = feature_examples(
                    cache,
                    feature,
                    vocab,
                    train,
                    test,
                    n_train=64,
                    n_test=100,
                    context=64,
                    seed=seed,
                )
                training = spread(
                    unique_examples(evidence["train"], positive=True), 32, top=8
                )
                used = {tuple(e["tokens"]) for e in training}
                testing = spread(
                    unique_examples(evidence["test"], used, positive=True), 50
                )
                used.update(tuple(e["tokens"]) for e in testing)
                negative = unique_examples(evidence["negatives"], used)[:50]
                if len(training) != 32 or len(testing) != 50 or len(negative) != 50:
                    rejected.append(
                        dict(
                            id=feature,
                            rank=rank,
                            frequency_bin=bin_index,
                            reason="insufficient unique contexts",
                            train=len(training),
                            test=len(testing),
                            negatives=len(negative),
                        )
                    )
                    continue
                evidence.update(train=training, test=testing, negatives=negative)
                samples[feature] = evidence
                chosen.append(
                    dict(
                        id=feature,
                        rank=rank,
                        frequency_bin=bin_index,
                        rank_bins=quota,
                        **{
                            k: stats[feature][k]
                            for k in ["count", "frequency", "maximum", "mean_active"]
                        },
                    )
                )
                accepted = True
                break
            if not accepted:
                raise ValueError(
                    f"No supported feature in rank {rank}, frequency bin {bin_index}; review sampling before proceeding"
                )
    return (
        chosen,
        samples,
        dict(
            eligible_by_rank={str(r): len(p) for r, p in pools.items()},
            rejected=rejected,
        ),
    )


def wilson_interval(correct, total):
    if not total:
        return None
    z = 1.959963984540054
    p = correct / total
    denom = 1 + z * z / total
    midpoint = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0, midpoint - half), min(1, midpoint + half)]


def binary_metrics(rows):
    result = detection_metrics(rows)
    result["accuracy_ci95"] = wilson_interval(
        result["tp"] + result["tn"], result["valid"]
    )
    return result


def parse_simulation(text, tokens):
    """Never silently replace invalid/misaligned simulations with zero predictions."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    body = json.loads(text)
    entries = body["activations"]
    if len(entries) != len(tokens):
        raise ValueError("Simulation token count mismatch")
    values = []
    for entry, token in zip(entries, tokens):
        value = entry["activation"]
        if entry["token"] != token:
            raise ValueError("Simulation token identity mismatch")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 10
        ):
            raise ValueError("Invalid simulated activation")
        values.append(float(value))
    return values


def response_key(task, messages, max_tokens):
    return hashlib.sha256(
        json.dumps([task, messages, max_tokens], sort_keys=True).encode()
    ).hexdigest()


def load_response_cache(path):
    """Replay only validated responses for exactly the same scored prompt."""
    cache = {}
    if not path.exists():
        return cache
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            task, text = row["task"], row["response"]
            if row["finish_reason"] != "stop" or not text:
                continue
            if task == "binary" and text.strip() not in ("[0]", "[1]"):
                continue
            if task == "explanation" and not re.search(r"\[EXPLANATION\]:\s*\S", text):
                continue
            if task == "intruder" and not re.fullmatch(
                r"\s*\[RESPONSE\]:\s*[0-4]\s*", text
            ):
                continue
            if task == "simulation":
                document = json.JSONDecoder().raw_decode(
                    row["messages"][-1]["content"]
                )[0]
                parse_simulation(
                    text, [entry["token"] for entry in document["activations"]]
                )
            key = response_key(task, row["messages"], row["max_tokens"])
            cache.setdefault(key, text)
        except (ValueError, KeyError, TypeError):
            continue
    return cache


def simulation_metrics(rows):
    valid = [row for row in rows if row.get("prediction") is not None]
    result = dict(
        total=len(rows), valid=len(valid), parse_failures=len(rows) - len(valid)
    )
    if not valid:
        return result
    actual = np.concatenate([r["actual"] for r in valid]).astype(float)
    predicted = np.concatenate([r["prediction"] for r in valid]).astype(float)
    mse = float(np.square(actual - predicted).mean())
    zero = float(np.square(actual).mean())
    variance = float(actual.var())
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if actual.std() > 0 and predicted.std() > 0
        else None
    )
    result.update(
        tokens=len(actual),
        pearson=correlation,
        mse=mse,
        zero_baseline_mse=zero,
        mse_skill_vs_zero=1 - mse / zero if zero else None,
        r_squared=1 - mse / variance if variance else None,
    )
    return result
