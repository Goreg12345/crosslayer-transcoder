"""Evidence selection and compact progress records for a whole MOLT layer."""

import copy

from .pilot import spread, unique_examples

METHODS = ("detection", "fuzz", "active_fuzz", "intruder")
TERMINAL = ("complete", "limited_evidence", "not_observed", "discarded")


def unique_with_fallback(examples, excluded=(), positive=False, target=50):
    """Prefer pilot-quality windows, then retain rare short/fully active evidence."""
    preferred = unique_examples(examples, excluded, positive=positive)
    if len(preferred) >= target:
        return preferred
    seen = set(excluded) | {tuple(e["tokens"]) for e in preferred}
    for example in examples:
        key = tuple(example["tokens"])
        if key and key not in seen:
            preferred.append(example)
            seen.add(key)
            if len(preferred) >= target:
                break
    return preferred


def layer_samples(raw):
    """Keep held-out conversations disjoint, including the one-observation case."""
    data = copy.deepcopy(raw)
    exceptions = []
    if not data["train"] and data["test"]:
        # This transform has no evidence in the global explanation partition.
        # Allocate one conversation to its explanation and never score it.
        chosen = data["test"][0]
        data["train"] = [chosen]
        for key in ("test", "negatives"):
            data[key] = [
                e for e in data[key] if e["conversation"] != chosen["conversation"]
            ]
        exceptions.append(
            "One evaluation conversation reassigned to explanation; excluded from every score"
        )
    train = spread(
        unique_with_fallback(data["train"], positive=True, target=32), 32, top=8
    )
    used = {tuple(e["tokens"]) for e in train}
    test = spread(unique_with_fallback(data["test"], used, positive=True), 50)
    used.update(tuple(e["tokens"]) for e in test)
    negatives = unique_with_fallback(data["negatives"], used)[:50]
    data.update(
        train=train, test=test, negatives=negatives, sampling_exceptions=exceptions
    )
    return data


def evidence_counts(samples):
    counts = {key: len(samples[key]) for key in ("train", "test", "negatives")}
    counts["active_fuzz_positives"] = sum(
        any(a == 0 for a in e["activations"]) for e in samples["test"]
    )
    counts["short_windows"] = sum(
        len(e["tokens"]) < 16
        for key in ("train", "test", "negatives")
        for e in samples[key]
    )
    counts["full_budget"] = (
        counts["train"] == 32
        and counts["test"]
        == counts["negatives"]
        == counts["active_fuzz_positives"]
        == 50
        and not counts["short_windows"]
    )
    return counts


def compact_result(result):
    """Keep per-case token arrays out of the live layer-wide summary."""
    out = {k: v for k, v in result.items() if k != "methods"}
    out["methods"] = {
        name: {k: v for k, v in method.items() if k != "rows"}
        for name, method in result.get("methods", {}).items()
        if name in METHODS
    }
    return out


def result_status(result):
    if result.get("status") == "not_observed":
        return "not_observed"
    if result.get("discarded_stage"):
        return "discarded"
    if not result.get("explanation") or not result.get("title"):
        return "incomplete"
    statuses = [result.get("methods", {}).get(m, {}).get("status") for m in METHODS]
    if any(s not in ("complete", "unavailable", "discarded") for s in statuses):
        return "incomplete"
    if "discarded" in statuses:
        return "discarded"
    if "unavailable" in statuses or not result.get("evidence_counts", {}).get(
        "full_budget"
    ):
        return "limited_evidence"
    return "complete"
