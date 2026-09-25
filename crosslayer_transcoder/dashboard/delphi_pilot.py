"""Audited adapters around Delphi's remote scorers for the MOLT pilot."""

import numpy as np
import torch

from delphi.scorers import FuzzingScorer
from delphi.scorers.classifier.sample import (
    ClassifierOutput,
    Sample,
    _highlight,
    examples_to_samples,
)
from delphi.scorers.simulator.simulation.simulator import NeuronSimulator

from .pilot import parse_simulation


class AuditedFuzzingScorer(FuzzingScorer):
    """Use Delphi classification/prompts with reproducible, genuinely zero-gate distractors.

    Upstream fixes one distractor at 75% of each context, can highlight weak
    positives, and can substitute unrelated fallback text. Avoid those artifacts.
    """

    def __init__(self, *args, sample_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_seed = sample_seed

    def _prepare(self, record):
        rng = np.random.default_rng(self.sample_seed)
        samples = []
        for index, example in enumerate(record.test):
            positive = examples_to_samples([example], highlighted=True)[0]
            marked = (
                torch.nonzero(example.activations >= 0.3 * example.max_activation)
                .flatten()
                .tolist()
            )
            positive.data.highlighted_positions = marked
            positive.data.shown_text = positive.text
            source = (
                record.not_active[index] if self.fuzz_type == "default" else example
            )
            zeros = torch.nonzero(source.activations == 0).flatten().numpy()
            if not len(zeros):
                raise ValueError(
                    "No zero-gate distractor positions in evaluation example"
                )
            indices = sorted(
                rng.choice(zeros, min(len(marked), len(zeros)), replace=False).tolist()
            )
            chosen = set(indices)
            text = _highlight(source.str_tokens, lambda i: i in chosen)
            negative = ClassifierOutput(
                str_tokens=source.str_tokens,
                activations=source.activations.tolist(),
                distance=example.quantile,
                activating=False,
            )
            negative.highlighted_positions = indices
            negative.shown_text = text
            samples.extend([positive, Sample(text, negative)])
        return samples


class StrictNeuronSimulator(NeuronSimulator):
    async def simulate(self, tokens):
        # Propagate errors; the upstream convenience wrapper silently returns zeros.
        # Qwen's output reasoning parser can consume literal thinking delimiters.
        # Preserve a one-to-one token mapping while displaying those as plain text.
        displayed = [
            t.replace("<think>", "⟪think⟫").replace("</think>", "⟪/think⟫")
            for t in tokens
        ]
        return await self._simulate_with_json(displayed)

    def _parse_json_response(self, response, tokens):
        return parse_simulation(response.text, tokens)
