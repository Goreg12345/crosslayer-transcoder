#!/usr/bin/env python3
"""Explain every observed gate in a completed layer and run four Delphi scorers."""

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from delphi.explainers import DefaultExplainer
from delphi.latents import LatentRecord
from delphi.scorers import DetectionScorer, IntruderScorer

from crosslayer_transcoder.dashboard.delphi_pilot import AuditedFuzzingScorer
from crosslayer_transcoder.dashboard.explanation_samples import (
    feature_examples,
    feature_order,
    index_features,
    split_conversations,
)
from crosslayer_transcoder.dashboard.layer_explanations import (
    METHODS,
    TERMINAL,
    compact_result,
    evidence_counts,
    layer_samples,
    result_status,
)
from crosslayer_transcoder.dashboard.pilot import (
    binary_metrics,
    load_response_cache,
    wilson_interval,
)
from crosslayer_transcoder.dashboard.token_display import escape_vocab
from tools.explain_molt_dashboard import Endpoint, atomic_json
from tools.run_molt_pilot import TaskClient, digest, make_record, now, row_data


def pipeline_settings(root, meta, args, cache_hash):
    """Use the approved protocol without requiring a separate pilot in every layer."""
    pilot = root / "pilot32"
    pilot_manifest = None
    if (pilot / "manifest.json").exists():
        pilot_manifest = json.loads((pilot / "manifest.json").read_text())
        if any(
            pilot_manifest["settings"][key] != value
            for key, value in {
                "cache_sha256": cache_hash,
                "model": args.model,
                "base_url": args.base_url,
                "checkpoint": meta["checkpoint"],
                "global_step": meta["global_step"],
            }.items()
        ):
            raise ValueError("Pilot and full-layer provenance disagree")
        settings = dict(pilot_manifest["settings"])
    else:
        settings = dict(json.loads(args.method_reference.read_text())["settings"])
        train, test = split_conversations(meta["conversations"])
        settings.update(
            checkpoint=meta["checkpoint"],
            global_step=meta["global_step"],
            cache_sha256=cache_hash,
            tokens=meta["tokens"],
            layer=meta["layer"],
            n_features=meta["n_features"],
            train_conversations=sorted(train),
            test_conversations=sorted(test),
            model=args.model,
            base_url=args.base_url,
        )
    settings.pop("simulator_output_tokens", None)
    settings.pop("pilot_version", None)
    settings.update(
        scope=f"all transforms in layer {meta['layer']}",
        methods={m: settings["methods"][m] for m in METHODS},
        selection="Every transform, including explicit unobserved and limited-evidence records",
        exclusions="Exact duplicate windows; preserve rare short windows and report sample counts; active fuzz requires zero-gate distractors",
        evidence_policy="Up to 32 explanation, 50 positive and 50 negative evaluation windows; one evaluation-only conversation may be reassigned exclusively to explanation",
        title_policy="LLM title of 3–8 words from the explanation, without new claims; approved pilot titles reused",
        special_token_display="every tokenizer added token (incl. <think>, <tool_call>) shown as ⟪name⟫ in evidence and prompts",
        pipeline_version=1,
    )
    return settings, pilot_manifest


def prepare(root, output, args):
    meta = json.loads((root / "metadata.json").read_text())
    if (
        meta["tokens"] != 10_000_000
        or not meta.get("target_reached")
        or meta["global_step"] != 100_000
    ):
        raise ValueError(
            "This run requires a completed 10M-token cache from a finished campaign checkpoint"
        )
    cache_hash = digest(root / "activations.npz")
    settings, pilot_manifest = pipeline_settings(root, meta, args, cache_hash)
    pilot = root / "pilot32"
    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError(
                "Full-layer configuration changed; preserve existing results"
            )
        return manifest
    with np.load(root / "activations.npz") as archive:
        cache = {k: archive[k] for k in archive.files}
    if not (root / "feature_stats.json").exists():
        counts = np.bincount(cache["features"], minlength=meta["n_features"])
        totals = np.bincount(
            cache["features"], weights=cache["values"], minlength=meta["n_features"]
        )
        maximum = np.zeros(meta["n_features"])
        np.maximum.at(maximum, cache["features"], cache["values"])
        atomic_json(
            root / "feature_stats.json",
            [
                dict(
                    id=f,
                    rank=rank,
                    count=int(counts[f]),
                    frequency=float(counts[f] / len(cache["tokens"])),
                    maximum=float(maximum[f]),
                    mean_active=float(totals[f] / counts[f]) if counts[f] else 0.0,
                )
                for f, rank in enumerate(meta["ranks"])
            ],
        )
    stats = json.loads((root / "feature_stats.json").read_text())
    stats = [
        {
            k: s[k]
            for k in ("id", "rank", "count", "frequency", "maximum", "mean_active")
        }
        for s in stats
    ]
    if len(cache["tokens"]) != 10_000_000:
        raise ValueError("Sparse cache token count disagrees with metadata")
    index_features(cache, meta["n_features"])
    train, test = split_conversations(meta["conversations"])
    vocab = escape_vocab(json.loads((root / "vocab.json").read_text()))
    pilot_ids = (
        {s["id"] for s in pilot_manifest["selected"]} if pilot_manifest else set()
    )
    counts = Counter(
        not_observed=0,
        observed=0,
        full_evidence_budget=0,
        limited_evidence_budget=0,
        reused_pilot=0,
    )
    for stat in stats:
        feature = stat["id"]
        result = dict(**stat, fingerprint=fingerprint, methods={}, status="pending")
        if not stat["count"]:
            result.update(
                status="not_observed",
                title="Not observed in 10M tokens",
                label="Not observed in 10M tokens",
                explanation=None,
            )
            result["methods"] = {
                m: dict(
                    status="unavailable",
                    reason="No positive activation evidence in this cache",
                )
                for m in METHODS
            }
            counts["not_observed"] += 1
        else:
            if feature in pilot_ids:
                samples = json.loads(
                    (pilot / "evidence" / f"{feature}.json").read_text()
                )
                old = json.loads((pilot / "features" / f"{feature}.json").read_text())
                if old["fingerprint"] != pilot_manifest["fingerprint"]:
                    raise ValueError("Pilot feature provenance mismatch")
                result.update(
                    explanation=old["explanation"],
                    title=old["label"],
                    label=old["label"],
                    source_pilot_fingerprint=old["fingerprint"],
                )
                result["methods"] = {m: old["methods"][m] for m in METHODS}
                # Keep exact prompts available for any retry, without simulation requests.
                source_audit = pilot / "audit" / f"{feature}.jsonl"
                if source_audit.exists():
                    with (output / "audit" / f"{feature}.jsonl").open("w") as handle:
                        for line in source_audit.open():
                            item = json.loads(line)
                            if item["task"] != "simulation":
                                handle.write(line)
                            if (
                                item["task"] == "label"
                                and item.get("finish_reason") == "stop"
                                and item.get("response")
                            ):
                                title = item["response"].strip()
                                if title.startswith('"') and title.endswith('"'):
                                    title = title[1:-1]
                                result.update(title=title, label=title)
                counts["reused_pilot"] += 1
            else:
                raw = feature_examples(
                    cache,
                    feature,
                    vocab,
                    train,
                    test,
                    n_train=64,
                    n_test=100,
                    context=64,
                )
                samples = layer_samples(raw)
            result["evidence_counts"] = evidence_counts(samples)
            result["sampling_exceptions"] = samples.get("sampling_exceptions", [])
            if not samples["train"]:
                raise ValueError(
                    f"Observed transform {feature} has no explanation evidence"
                )
            atomic_json(output / "evidence" / f"{feature}.json", samples)
            if feature in pilot_ids:
                result["status"] = result_status(result)
            counts["observed"] += 1
            counts[
                "full_evidence_budget"
                if result["evidence_counts"]["full_budget"]
                else "limited_evidence_budget"
            ] += 1
        atomic_json(output / "features" / f"{feature}.json", result)
        if feature % 100 == 0:
            atomic_json(
                output / "preparation-progress.json",
                dict(
                    state="preparing",
                    transforms=feature + 1,
                    total=len(stats),
                    counts=dict(counts),
                    updated_at=now(),
                ),
            )
            print(
                json.dumps(
                    dict(event="preparing", transforms=feature + 1, counts=dict(counts))
                ),
                flush=True,
            )
    manifest = dict(
        settings=settings,
        fingerprint=fingerprint,
        selected=stats,
        preparation_counts=dict(counts),
        created_at=now(),
    )
    # A manifest commits preparation only after every evidence/result file exists.
    atomic_json(manifest_path, manifest)
    atomic_json(
        output / "preparation-progress.json",
        dict(
            state="complete",
            transforms=len(stats),
            total=len(stats),
            counts=dict(counts),
            updated_at=now(),
        ),
    )
    return manifest


def scoring_record(record, name):
    """Balance available evidence without inventing cases for rare transforms."""
    positives = list(record.test)
    negatives = list(record.not_active)
    if name == "active_fuzz":
        positives = [e for e in positives if bool((e.activations == 0).any())]
        if not positives:
            return None, "No held-out positive window has a zero-gate distractor token"
    elif name == "intruder":
        if len(positives) < 4 or not negatives:
            return None, "Need four held-out positive contexts and one negative"
        bins = min(5, len(positives) // 4)
        for i, example in enumerate(positives):
            example.quantile = min(bins - 1, i * bins // len(positives))
        negatives = negatives[:20]
    else:
        n = min(len(positives), len(negatives))
        if not n:
            return None, "No conversation-disjoint positive/negative evaluation pair"
        positives, negatives = positives[:n], negatives[:n]
    return LatentRecord(
        latent=record.latent,
        explanation=record.explanation,
        test=positives,
        not_active=negatives,
    ), None


async def score_method(name, record, client, feature):
    trial, reason = scoring_record(record, name)
    if trial is None:
        return dict(status="unavailable", reason=reason)
    if name == "intruder":
        score = await IntruderScorer(
            client("intruder"), n_examples_shown=5, cot=False, seed=42 + feature
        )(trial)
        rows = [asdict(row) for row in score.score]
        valid = [r for r in rows if r["sample"] is not None]
        correct = sum(r["correct"] for r in valid)
        metrics = dict(
            total=len(rows),
            valid=len(valid),
            parse_failures=len(rows) - len(valid),
            correct=correct,
            accuracy=correct / len(valid) if valid else None,
            accuracy_ci95=wilson_interval(correct, len(valid)),
            chance=0.2,
        )
    else:
        if name == "detection":
            scorer = DetectionScorer(
                client("binary"), n_examples_shown=1, max_tokens=32
            )
        else:
            scorer = AuditedFuzzingScorer(
                client("binary"),
                n_examples_shown=1,
                max_tokens=32,
                fuzz_type="active" if name == "active_fuzz" else "default",
                sample_seed=42 + feature,
            )
        score = await scorer(trial)
        rows = [row_data(row) for row in score.score]
        metrics = binary_metrics(rows)
    return dict(
        status="complete" if metrics["parse_failures"] == 0 else "incomplete",
        metrics=metrics,
        rows=rows,
    )


async def main(args):
    torch.set_num_threads(2)
    root = args.directory.resolve()
    output = root / "autointerp-layer"
    for name in ("evidence", "features", "audit"):
        (output / name).mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = prepare(root, output, args)
    settings, fingerprint = manifest["settings"], manifest["fingerprint"]
    features = {}
    for stat in manifest["selected"]:
        row = json.loads((output / "features" / f"{stat['id']}.json").read_text())
        if row["fingerprint"] != fingerprint:
            raise ValueError("Saved feature configuration mismatch")
        features[str(row["id"])] = compact_result(row)
    order = feature_order(manifest["selected"])
    endpoint = (
        None
        if args.prepare_only
        else Endpoint(
            args.model, args.base_url, os.environ[args.api_key_env], 32, 10000
        )
    )
    start = time.monotonic()
    last_save = 0
    phase = "prepared"

    def progress(force=False, state="running"):
        nonlocal last_save
        if not force and time.monotonic() - last_save < 10:
            return
        last_save = time.monotonic()
        counts = dict(Counter(row["status"] for row in features.values()))
        summary = dict(
            settings=settings,
            fingerprint=fingerprint,
            features=features,
            counts=counts,
            preparation_counts=manifest["preparation_counts"],
            usage=endpoint.usage_snapshot() if endpoint else {},
            run=dict(
                state=state,
                phase=phase,
                updated_at=now(),
                elapsed_seconds=time.monotonic() - start,
                total=len(features),
                observed=len(order),
                explained=sum(bool(r.get("explanation")) for r in features.values()),
            ),
        )
        atomic_json(output / "summary.json", summary)
        # Small shared annotation file; full per-case arrays live in feature JSONs.
        annotations = {}
        for key, row in features.items():
            annotation = dict(
                status=row["status"],
                label=row.get("title", "Explanation pending"),
                title=row.get("title"),
                explanation=row.get("explanation"),
                review_url=f"../autointerp-layer/features/{key}.html",
            )
            if row.get("explanation"):
                annotation["status"] = "explained"
                annotation["detection"] = (
                    row.get("methods", {}).get("detection", {}).get("metrics")
                )
            annotations[key] = annotation
        (root / "autointerp").mkdir(exist_ok=True)
        atomic_json(
            root / "autointerp" / "summary.json",
            dict(
                settings=settings,
                fingerprint=fingerprint,
                features=annotations,
                review_url="autointerp-layer/",
                full_layer=True,
            ),
        )

    def save(result):
        result["updated_at"] = now()
        atomic_json(output / "features" / f"{result['id']}.json", result)
        features[str(result["id"])] = compact_result(result)
        progress()

    progress(True, "prepared" if args.prepare_only else "running")
    if args.prepare_only:
        print(
            json.dumps(dict(event="prepared", **manifest["preparation_counts"])),
            flush=True,
        )
        return

    workers = asyncio.Semaphore(32)

    async def feature_job(feature, stage):
        async with workers:
            result = json.loads((output / "features" / f"{feature}.json").read_text())
            if stage == "explain" and result.get("explanation") and result.get("title"):
                return
            if stage == "score" and result["status"] in TERMINAL:
                return
            samples = json.loads((output / "evidence" / f"{feature}.json").read_text())
            record = make_record(settings["layer"], feature, samples)
            audit = output / "audit" / f"{feature}.jsonl"
            cache = load_response_cache(audit)

            def client(task):
                return TaskClient(endpoint, task, audit, cache)

            try:
                if stage == "explain":
                    result["status"] = "explaining"
                    save(result)
                    if not result.get("explanation"):
                        explanation = (
                            await DefaultExplainer(
                                client("explanation"), activations=True, cot=False
                            )(record)
                        ).explanation
                        if explanation == "Explanation could not be parsed.":
                            raise ValueError("Unparseable explanation")
                        result["explanation"] = explanation
                        save(result)
                    if not result.get("title"):
                        response = await client("label").generate(
                            [
                                dict(
                                    role="system",
                                    content="Give this MOLT gate-activation explanation a specific, concise title of 3–8 words. Name its trigger or pattern directly. Omit introductory phrases like 'Gate activates on'. Do not add claims. Return only the title.",
                                ),
                                dict(role="user", content=result["explanation"]),
                            ],
                            max_tokens=96,
                        )
                        title = response.text.strip()
                        if title.startswith('"') and title.endswith('"'):
                            title = title[1:-1]
                        result.update(title=title, label=title)
                    result["status"] = "ready_to_score"
                else:
                    if not result.get("explanation") or not result.get("title"):
                        return
                    record.explanation = result["explanation"]
                    result["status"] = "scoring"
                    save(result)

                    async def evaluate(name):
                        if result["methods"].get(name, {}).get("status") in (
                            "complete",
                            "unavailable",
                        ):
                            return
                        try:
                            result["methods"][name] = await score_method(
                                name, record, client, feature
                            )
                        except Exception as exc:
                            result["methods"][name] = dict(
                                status="error", error=type(exc).__name__
                            )
                        save(result)
                        print(
                            json.dumps(
                                dict(
                                    event="method_finished",
                                    id=feature,
                                    method=name,
                                    status=result["methods"][name]["status"],
                                )
                            ),
                            flush=True,
                        )

                    await asyncio.gather(*(evaluate(name) for name in METHODS))
                    result["status"] = result_status(result)
                result.pop("error", None)
                result.pop("error_message", None)
            except Exception as exc:
                result.update(
                    status="error",
                    error=type(exc).__name__,
                    error_message=str(exc)[:500],
                )
            save(result)
            print(
                json.dumps(
                    dict(
                        event="feature_finished",
                        stage=stage,
                        id=feature,
                        status=result["status"],
                        title=result.get("title"),
                    )
                ),
                flush=True,
            )

    async def heartbeat():
        while True:
            await asyncio.sleep(15)
            progress(True)

    ticker = asyncio.create_task(heartbeat())
    try:
        for attempt in range(1, 4):
            phase = f"explaining (pass {attempt}/3)"
            progress(True)
            await asyncio.gather(*(feature_job(f, "explain") for f in order))
            phase = f"scoring (pass {attempt}/3)"
            progress(True)
            await asyncio.gather(*(feature_job(f, "score") for f in order))
            if all(row["status"] in TERMINAL for row in features.values()):
                break
        finished = all(row["status"] in TERMINAL for row in features.values())
        phase = "finished"
        progress(True, "complete" if finished else "incomplete")
        print(
            json.dumps(
                dict(
                    event="layer_finished",
                    counts=dict(Counter(r["status"] for r in features.values())),
                    usage=endpoint.usage,
                )
            ),
            flush=True,
        )
    finally:
        ticker.cancel()
        await asyncio.gather(ticker, return_exceptions=True)
        await endpoint.client.aclose()
    if not finished:
        raise RuntimeError(
            "Some layer methods remain incomplete; rerun to resume exact cached prompts"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--base-url", default="https://geor.taild80f17.ts.net/v1")
    parser.add_argument("--model", default="qwen3.8-27b-nvfp4")
    parser.add_argument("--api-key-env", default="vLLM_API_KEY")
    parser.add_argument(
        "--method-reference",
        type=Path,
        default=Path(
            "results/molt-qwen-dashboard-campaign-layer22-10m/autointerp-layer/manifest.json"
        ),
    )
    asyncio.run(main(parser.parse_args()))
