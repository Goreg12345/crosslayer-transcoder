#!/usr/bin/env python3
"""Freeze, explain, and evaluate exactly 32 MOLT transforms for human review."""

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from delphi.clients.client import Client
from delphi.explainers import DefaultExplainer
from delphi.latents import ActivatingExample, NonActivatingExample, Latent, LatentRecord
from delphi.scorers import DetectionScorer, IntruderScorer

from crosslayer_transcoder.dashboard.token_display import (
    escape_added_tokens,
    escape_vocab,
)
from crosslayer_transcoder.dashboard.delphi_pilot import (
    AuditedFuzzingScorer,
    StrictNeuronSimulator,
)
from crosslayer_transcoder.dashboard.explanation_samples import (
    index_features,
    split_conversations,
)
from crosslayer_transcoder.dashboard.pilot import (
    binary_metrics,
    load_response_cache,
    select_pilot,
    simulation_metrics,
    spread,
    validate_response,
    wilson_interval,
)
from crosslayer_transcoder.dashboard.output_limits import RequestDiscarded
from tools.explain_molt_dashboard import Endpoint, atomic_json

METHODS = ("detection", "fuzz", "active_fuzz", "intruder", "simulation")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, output):
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["tokens"] != 10000000:
        raise ValueError("This pilot requires the completed 10,000,000-token sample")
    with np.load(root / "activations.npz") as archive:
        cache = {k: archive[k] for k in archive.files}
    if len(cache["tokens"]) != metadata["tokens"]:
        raise ValueError("Token count disagrees with collection metadata")
    vocab = escape_vocab(json.loads((root / "vocab.json").read_text()))
    if (root / "feature_stats.json").exists():
        stats = json.loads((root / "feature_stats.json").read_text())
    else:
        counts = np.bincount(cache["features"], minlength=metadata["n_features"])
        totals = np.bincount(
            cache["features"], weights=cache["values"], minlength=metadata["n_features"]
        )
        maximum = np.zeros(metadata["n_features"])
        np.maximum.at(maximum, cache["features"], cache["values"])
        stats = [
            dict(
                id=f,
                rank=rank,
                count=int(counts[f]),
                frequency=float(counts[f] / len(cache["tokens"])),
                maximum=float(maximum[f]),
                mean_active=float(totals[f] / counts[f]) if counts[f] else 0.0,
            )
            for f, rank in enumerate(metadata["ranks"])
        ]
        atomic_json(root / "feature_stats.json", stats)
    index_features(cache, metadata["n_features"])
    train, test = split_conversations(metadata["conversations"])
    chosen, samples, eligibility = select_pilot(cache, stats, vocab, train, test)
    settings = dict(
        checkpoint=metadata["checkpoint"],
        global_step=metadata["global_step"],
        cache_sha256=digest(root / "activations.npz"),
        tokens=metadata["tokens"],
        layer=metadata["layer"],
        n_features=metadata["n_features"],
        delphi_revision=subprocess.check_output(
            ["git", "-C", "local/delphi", "rev-parse", "HEAD"], text=True
        ).strip(),
        seed=42,
        train_conversations=sorted(train),
        test_conversations=sorted(test),
        n_train=32,
        n_test_per_class=50,
        context_tokens=64,
        normalization="ceil(10*gate/max_explanation_gate), clipped 0..10; no evaluation fitting",
        selection="Rank quotas 7/7/6/6/6, frequency quantile bins, seeded selection before LLM calls; fixed cohort regardless of scores",
        exclusions="Exact token-window duplicates, contexts under 16 tokens, positive windows without zero-gate tokens; require 32 train + 50 test + 50 negative unique contexts",
        methods=dict(
            detection="Delphi DetectionScorer, 100 one-example decisions",
            fuzz="Delphi FuzzingScorer with audited zero-gate distractors, 100 decisions",
            active_fuzz="Same scoring prompt, incorrect highlights on zero-gate tokens in positive contexts, 100 decisions",
            intruder="Delphi IntruderScorer, 20 groups of 4 positive contexts + 1 negative; evaluates coherence without the explanation",
            simulation="Delphi NeuronSimulator JSON prompt, 10 positive + 10 negative windows; strict token identity and numeric validation",
        ),
        simulator_output_tokens=3072,
        explanation_max_words=80,
        limitations=[
            "Same endpoint model generates and evaluates explanations.",
            "Windows end at a conversation peak; token simulation and fuzzing can see later tokens within that prefix when judging earlier tokens.",
            "Random zero-gate negative windows are not semantically matched.",
            "Intruder groups reuse positive contexts; their decisions are not independent.",
            "Embedding and surprisal scorers are not run: they need an embedding backend or teacher-forced model scoring, respectively.",
        ],
        pilot_version=1,
    )
    payload = dict(settings=settings, selected=chosen, eligibility=eligibility)
    atomic_json(output / "selection.json", payload)
    for feature, evidence in samples.items():
        atomic_json(output / "evidence" / f"{feature}.json", evidence)
    print(
        json.dumps(
            dict(
                event="cohort_frozen",
                ids=[x["id"] for x in chosen],
                ranks=dict(Counter(x["rank"] for x in chosen)),
                eligibility=eligibility,
            )
        ),
        flush=True,
    )
    return payload


class TaskClient(Client):
    def __init__(self, endpoint, task, audit, response_cache, *, output_limits=None):
        super().__init__(endpoint.model)
        self.endpoint, self.task, self.audit = endpoint, task, audit
        self.response_cache = response_cache
        self.output_limits = output_limits
        self.discarded_requests = []

    async def generate(self, prompt, **kwargs):
        kwargs.update(
            task=self.task, audit_path=self.audit, response_cache=self.response_cache
        )
        if self.output_limits is not None:
            kwargs["output_limits"] = self.output_limits
        if self.task == "explanation":
            kwargs.update(max_tokens=768, explanation_words=80)
            prompt = [dict(m) for m in prompt]
            prompt[0]["content"] += (
                "\nDescribe token-level triggers and their preceding context. Cover strong and weaker examples; acknowledge mixed evidence rather than forcing one semantic theme. Put the final explanation on one [EXPLANATION]: line."
            )
        elif self.task == "simulation":
            kwargs["max_tokens"] = 3072
        elif self.task == "intruder":
            kwargs["max_tokens"] = 64
        for attempt in range(3):
            try:
                response = await self.endpoint.generate(prompt, **kwargs)
            except RequestDiscarded as exc:
                # Delphi scorers turn client exceptions into missing predictions;
                # retain the reason so the layer worker can finish this method.
                self.discarded_requests.append(exc.record())
                raise
            try:
                validate_response(self.task, response.text, prompt)
                self.response_cache[response.cache_key] = response.text
                return response
            except (ValueError, KeyError, TypeError):
                self.response_cache.pop(response.cache_key, None)
                if attempt == 2:
                    raise
                self.endpoint.usage["validation_retries"] = (
                    self.endpoint.usage.get("validation_retries", 0) + 1
                )


def make_record(layer, feature, samples):
    scale = max(max(e["activations"]) for e in samples["train"])
    # Evidence prepared before every added token was escaped still holds e.g. </think>.
    samples = {
        key: [
            dict(e, str_tokens=[escape_added_tokens(t) for t in e["str_tokens"]])
            for e in samples[key]
        ]
        for key in ("train", "test", "negatives")
    }

    def active(example, quantile=0):
        acts = torch.tensor(example["activations"])
        return ActivatingExample(
            tokens=torch.tensor(example["tokens"]),
            activations=acts,
            normalized_activations=(10 * acts / scale).ceil().clamp(0, 10),
            str_tokens=example["str_tokens"],
            quantile=quantile,
        )

    record = LatentRecord(latent=Latent(f"qwen3.layer{layer}.molt_gate", feature))
    record.train = [active(e) for e in samples["train"]]
    record.examples = record.train
    record.test = [active(e, min(4, i // 10)) for i, e in enumerate(samples["test"])]
    record.not_active = [
        NonActivatingExample(
            tokens=torch.tensor(e["tokens"]),
            activations=torch.tensor(e["activations"]),
            str_tokens=e["str_tokens"],
        )
        for e in samples["negatives"]
    ]
    return record


def row_data(row):
    data = asdict(row)
    for key in ("highlighted_positions", "shown_text"):
        if hasattr(row, key):
            data[key] = getattr(row, key)
    return data


async def main(args):
    torch.set_num_threads(2)
    root = args.directory
    output = root / "pilot32"
    for name in ("evidence", "features", "audit"):
        (output / name).mkdir(parents=True, exist_ok=True)
    selection = output / "selection.json"
    payload = (
        json.loads(selection.read_text())
        if selection.exists()
        else prepare(root, output)
    )
    if payload["settings"]["cache_sha256"] != digest(root / "activations.npz"):
        raise ValueError("Pilot selection belongs to a different cache")
    if args.prepare_only:
        return
    settings = dict(
        payload["settings"],
        model=args.model,
        base_url=args.base_url,
        concurrency=32,
        context_limit=10000,
    )
    fingerprint = hashlib.sha256(
        json.dumps(settings, sort_keys=True).encode()
    ).hexdigest()
    if (output / "manifest.json").exists() and json.loads(
        (output / "manifest.json").read_text()
    )["fingerprint"] != fingerprint:
        raise ValueError("Pilot configuration changed; preserve existing results")
    atomic_json(
        output / "manifest.json",
        dict(fingerprint=fingerprint, settings=settings, selected=payload["selected"]),
    )
    endpoint = Endpoint(
        args.model, args.base_url, os.environ[args.api_key_env], 32, 10000
    )
    endpoint.usage["validation_retries"] = 0
    results = {}
    start = time.monotonic()
    labels_ready = asyncio.Event()
    prepared_features = 0
    pending_simulations = []

    def save_progress(state="running"):
        summary = dict(
            settings=settings,
            fingerprint=fingerprint,
            selection=payload["selected"],
            features=results,
            run=dict(
                state=state, updated_at=now(), elapsed_seconds=time.monotonic() - start
            ),
            usage=endpoint.usage,
        )
        atomic_json(output / "summary.json", summary)

    async def feature_job(selected):
        nonlocal prepared_features
        feature = selected["id"]
        path = output / "features" / f"{feature}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["fingerprint"] != fingerprint:
                raise ValueError("Feature result fingerprint mismatch")
        else:
            result = dict(
                **selected, fingerprint=fingerprint, status="pending", methods={}
            )
        results[str(feature)] = result
        samples = json.loads((output / "evidence" / f"{feature}.json").read_text())
        record = make_record(settings["layer"], feature, samples)
        audit = output / "audit" / f"{feature}.jsonl"
        response_cache = load_response_cache(audit)

        def client(task):
            return TaskClient(endpoint, task, audit, response_cache)

        def checkpoint():
            result["updated_at"] = now()
            atomic_json(path, result)
            save_progress()

        label_counted = False
        try:
            if not result.get("explanation"):
                result["status"] = "explaining"
                checkpoint()
                explanation = (
                    await DefaultExplainer(
                        client("explanation"), activations=True, cot=False
                    )(record)
                ).explanation
                if explanation == "Explanation could not be parsed.":
                    raise ValueError("Unparseable explanation")
                result["explanation"] = explanation
                checkpoint()
            record.explanation = result["explanation"]
            if not result.get("label"):
                label = await client("label").generate(
                    [
                        dict(
                            role="system",
                            content="Summarize this gate-activation hypothesis as a precise label of at most 12 words. Return only the label. Do not add new claims.",
                        ),
                        dict(role="user", content=result["explanation"]),
                    ],
                    max_tokens=96,
                )
                result["label"] = label.text.strip()
                if result["label"].startswith('"') and result["label"].endswith('"'):
                    result["label"] = result["label"][1:-1]
                checkpoint()
            result["status"] = "scoring"
            checkpoint()
            prepared_features += 1
            label_counted = True
            if prepared_features == len(payload["selected"]):
                labels_ready.set()
            await labels_ready.wait()

            async def score_method(name):
                old = result["methods"].get(name, {})
                if old.get("status") == "complete":
                    return
                try:
                    if name in ("detection", "fuzz", "active_fuzz"):
                        if name == "detection":
                            scorer = DetectionScorer(
                                client("binary"), n_examples_shown=1, max_tokens=32
                            )
                        else:
                            scorer = AuditedFuzzingScorer(
                                client("binary"),
                                n_examples_shown=1,
                                max_tokens=32,
                                fuzz_type="active"
                                if name == "active_fuzz"
                                else "default",
                                sample_seed=42 + feature,
                            )
                        score = await scorer(record)
                        rows = [row_data(row) for row in score.score]
                        metrics = binary_metrics(rows)
                    elif name == "intruder":
                        trial = LatentRecord(
                            latent=record.latent,
                            test=record.test,
                            not_active=record.not_active[:20],
                        )
                        score = await IntruderScorer(
                            client("intruder"),
                            n_examples_shown=5,
                            cot=False,
                            seed=42 + feature,
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
                        simulator = StrictNeuronSimulator(
                            client("simulation"), record.explanation, use_logprobs=False
                        )
                        examples = [(True, e) for e in spread(record.test, 10)] + [
                            (False, e) for e in record.not_active[:10]
                        ]

                        async def simulate(activating, example):
                            actual = (
                                example.normalized_activations.tolist()
                                if activating
                                else [0.0] * len(example.tokens)
                            )
                            row = dict(
                                activating=activating,
                                tokens=example.tokens.tolist(),
                                str_tokens=example.str_tokens,
                                actual=actual,
                            )
                            try:
                                sim = await simulator.simulate(example.str_tokens)
                                row["prediction"] = sim.expected_activations
                            except Exception as exc:
                                row.update(prediction=None, error=type(exc).__name__)
                            return row

                        rows = await asyncio.gather(
                            *(simulate(a, e) for a, e in examples)
                        )
                        metrics = simulation_metrics(rows)
                    result["methods"][name] = dict(
                        status="complete"
                        if metrics["parse_failures"] == 0
                        else "incomplete",
                        metrics=metrics,
                        rows=rows,
                    )
                except Exception as exc:
                    result["methods"][name] = dict(
                        status="error", error=type(exc).__name__
                    )
                checkpoint()
                print(
                    json.dumps(
                        dict(
                            event="method_finished",
                            id=feature,
                            method=name,
                            status=result["methods"][name]["status"],
                            metrics=result["methods"][name].get("metrics"),
                        )
                    ),
                    flush=True,
                )

            await asyncio.gather(
                *(score_method(name) for name in METHODS if name != "simulation")
            )
            result["status"] = "waiting_simulation"
            checkpoint()

            async def finish_simulation():
                result["status"] = "simulating"
                checkpoint()
                await score_method("simulation")
                result["status"] = (
                    "complete"
                    if all(
                        result["methods"].get(name, {}).get("status") == "complete"
                        for name in METHODS
                    )
                    else "incomplete"
                )
                checkpoint()

            pending_simulations.append(finish_simulation)
        except Exception as exc:
            if not label_counted:
                prepared_features += 1
                if prepared_features == len(payload["selected"]):
                    labels_ready.set()
            result.update(status="error", error=type(exc).__name__)
        checkpoint()
        print(
            json.dumps(
                dict(
                    event="feature_finished",
                    id=feature,
                    status=result["status"],
                    label=result.get("label"),
                )
            ),
            flush=True,
        )

    save_progress()
    try:
        await asyncio.gather(
            *(feature_job(selected) for selected in payload["selected"])
        )
        await asyncio.gather(*(finish() for finish in pending_simulations))
    finally:
        finished = len(results) == 32 and all(
            row["status"] == "complete" for row in results.values()
        )
        save_progress("complete" if finished else "incomplete")
        await endpoint.client.aclose()
    # Publish only this fixed cohort into the shared feature explorer.
    statuses = {
        str(i): dict(
            status="not_selected", label="Outside the 32-transform review pilot"
        )
        for i in range(settings["n_features"])
    }
    for key, result in results.items():
        if result.get("explanation") and result["methods"].get("detection", {}).get(
            "metrics"
        ):
            statuses[key] = dict(
                status="explained",
                label=result["label"],
                explanation=result["explanation"],
                detection=result["methods"]["detection"]["metrics"],
                pilot_url=f"../pilot32/features/{key}.html",
            )
    (root / "autointerp").mkdir(exist_ok=True)
    atomic_json(
        root / "autointerp" / "summary.json",
        dict(
            settings=settings,
            fingerprint=fingerprint,
            features=statuses,
            pilot_only=True,
        ),
    )
    print(
        json.dumps(
            dict(
                event="pilot_finished",
                counts=dict(Counter(row["status"] for row in results.values())),
                usage=endpoint.usage,
                elapsed_seconds=time.monotonic() - start,
            )
        ),
        flush=True,
    )
    if not finished:
        raise RuntimeError("Pilot contains incomplete methods; rerun to resume them")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--base-url", default="https://geor.taild80f17.ts.net/v1")
    parser.add_argument("--model", default="qwen3.8-27b-nvfp4")
    parser.add_argument("--api-key-env", default="vLLM_API_KEY")
    asyncio.run(main(parser.parse_args()))
