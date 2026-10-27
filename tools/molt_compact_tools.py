#!/usr/bin/env python3
"""Compact, readable MCP surface for agentic Delphi MOLT investigations.

Same 17 tools and arguments as molt_agentic_full_mcp, backed by the same state
objects, but every tool returns plain text written for the explainer: examples
are their text with <<active spans>> and a one-line header. Token/ID/activation
arrays, per-example boilerplate, echoed inputs and scorer audit tables are never
returned. Window references (conversation:offset) keep examples auditable.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crosslayer_transcoder.dashboard.token_display import escape_added_tokens
from molt_agentic_full_mcp import TOOL_NAMES, bounded_text, now

# Native Delphi DefaultExplainer threshold for corpus windows.
CORPUS_HIGHLIGHT = 0.3
DETECTION_ERRORS_SHOWN = 4  # per error direction
FUZZ_ERRORS_SHOWN = 2
LOGITS_SHOWN = 6
PROBE_TOKENS_SHOWN = 8
WINDOW_PATTERN = re.compile(r'^#\d+ · max ([0-9.]+) · window (\d+):(\d+)$', re.M)


def highlight(tokens, activations, fraction):
    """Delphi's contiguous <<span>> marking: tokens above fraction × window maximum."""
    threshold = max(activations, default=0) * fraction
    text, inside = [], False
    for token, value in zip(tokens, activations):
        active = value > threshold
        if active != inside:
            text.append('<<' if active else '>>')
            inside = active
        text.append(token)
    if inside:
        text.append('>>')
    return ''.join(text)


def number(value):
    return f'{value:.2f}'


def example_text(row):
    tokens = row.get('str_tokens') or row.get('tokens')
    acts = row.get('activations')
    if isinstance(tokens, list) and isinstance(acts, list) and len(tokens) == len(acts):
        return highlight(tokens, acts, CORPUS_HIGHLIGHT)
    return row.get('marked_text') or row.get('text') or ''


def format_examples(rows, first=1):
    blocks = []
    for index, row in enumerate(rows, first):
        peak = row.get('max_activation') or max(row.get('activations') or [0])
        parts = [f'#{index}']
        if peak > 0:
            parts.append(f'max {number(peak)}')
        if 'conversation' in row and 'start' in row:
            parts.append(f"window {row['conversation']}:{row['start']}")
        blocks.append(' · '.join(parts) + '\n' + example_text(row))
    return '\n\n'.join(blocks)


def corpus_text(payload, title, offset=0):
    rows = payload.get('examples', [])
    total = payload.get('total_available')
    span = f'{offset+1}–{offset+len(rows)}' if rows else 'none'
    header = f"Feature {payload['feature']} · {title} {span}" + (f' of {total}' if total is not None else '')
    if not rows:
        return header + '\nNo examples at this offset.'
    if any(max(r.get('activations') or [0]) > 0 for r in rows):
        header += f'\n<<…>> marks tokens with gate > {CORPUS_HIGHLIGHT} × that example’s max.'
    return header + '\n\n' + format_examples(rows, offset + 1)


def quantile_text(feature, groups):
    blocks = [f'Feature {feature} · training examples near activation quantiles (q=1 strongest, q=0 weakest)\n'
              f'<<…>> marks tokens with gate > {CORPUS_HIGHLIGHT} × that example’s max.']
    number_from = 1
    for group in groups:
        rows = group.get('examples', [])
        blocks.append(f"## q={group.get('quantile')}\n\n" + format_examples(rows, number_from))
        number_from += len(rows)
    return '\n\n'.join(blocks)


SKIPPED_INFO = {'feature', 'layer', 'model', 'coordinate_system', 'evidence_scope', 'initial_examples'}


def latent_text(info):
    lines = [f"Feature {info['feature']} · layer {info.get('layer')} · {info.get('model')}"]
    for key, value in info.items():
        if key in SKIPPED_INFO or isinstance(value, (list, dict)):
            continue
        lines.append(f"{key.replace('_', ' ')}: {number(value) if isinstance(value, float) else value}")
    if info.get('initial_examples'):
        lines.append(f"initial training examples: {len(info['initial_examples'])} "
                     '(read them with get_top_activating_examples / get_quantile_examples)')
    return '\n'.join(lines)


def verify_text(payload):
    blocks = [f"Feature {payload['feature']} · {len(payload['results'])} probe(s), {payload['prompt_format']} format, "
              'actual Qwen3-4B + MOLT gate\n<<…>> marks every token with gate > 0.']
    for index, row in enumerate(payload['results'], 1):
        acts = row['activations']
        if not any(a > 0 for a in acts):
            blocks.append(f'#{index} · no activation\n' + ''.join(row['tokens']))
            continue
        active = sorted((i for i, a in enumerate(acts) if a > 0), key=lambda i: -acts[i])
        listed = ', '.join(f"{json.dumps(row['tokens'][i], ensure_ascii=False)} {number(acts[i])}"
                           for i in active[:PROBE_TOKENS_SHOWN])
        if len(active) > PROBE_TOKENS_SHOWN:
            listed += f', … {len(active) - PROBE_TOKENS_SHOWN} more'
        blocks.append(f'#{index} · max {number(max(acts))} · active: {listed}\n' + highlight(row['tokens'], acts, 0))
    return '\n\n'.join(blocks)


def steer_text(payload):
    plan = payload['intervention']
    where = 'all prompt tokens' if payload.get('all_positions') else f"prompt positions {plan['prompt_positions']}"
    lines = [f"Feature {payload['feature']} · {plan['mode']} {plan['strength']} on {where} and every generated token · "
             f"{plan['max_new_tokens']} new tokens, greedy"]
    trace = (payload.get('steered') or {}).get('trace') or []
    if trace:
        live = [t['before'] for t in trace if t.get('before', 0) > 0]
        lines.append(f'Gate was active at {len(live)} of {len(trace)} intervened positions'
                     + (f' (max {number(max(live))})' if live else '; multiply has no effect where the gate is 0'))
    baseline, steered = payload['baseline']['text'], payload['steered']['text']
    lines.append('Baseline: ' + baseline)
    lines.append('Steered: ' + ('(identical to baseline)' if steered == baseline else steered))
    first = payload.get('first_prediction') or {}
    rows = sorted(first.get('tokens', []), key=lambda r: -abs(r['delta']))[:LOGITS_SHOWN]
    if rows:
        lines.append(f"First-token logits, max |Δ| {number(first.get('max_abs_logit_change', 0))}: " + '; '.join(
            f"{json.dumps(r['token'], ensure_ascii=False)} {number(r['baseline'])}→{number(r['steered'])}" for r in rows))
    return '\n'.join(lines)


SCORE_LABELS = {
    'detection': ('activating window', 'inactive window'),
    'fuzz': ('real highlights on an activating window', 'random highlights on an inactive window'),
}


def score_text(result):
    if 'raw_tool_artifact' in result:
        # The wire result drops token arrays; the server-local artifact keeps true gates.
        path = Path(result['raw_tool_artifact']['path'])
        if path.exists():
            result = json.loads(path.read_text())
    lines = ['Training diagnostic only — not the held-out benchmark; do not overfit to it.']
    if result.get('status') not in (None, 'complete'):
        lines.append(f"Status: {result['status']} (some predictions missing)")
    for name, method in result.get('methods', {}).items():
        metrics = method.get('metrics', {})
        rows = [r for r in method.get('rows', []) if r.get('prediction') is not None]
        recall = {label: (sum(bool(r['correct']) for r in rows if r['activating'] == label),
                          sum(r['activating'] == label for r in rows)) for label in (True, False)}
        positive, negative = SCORE_LABELS.get(name, ('positive', 'negative'))
        balanced = metrics.get('balanced_accuracy', metrics.get('accuracy'))
        lines.append(f"\n{name}: balanced accuracy {number(balanced) if balanced is not None else 'n/a'} · "
                     f"{positive}s {recall[True][0]}/{recall[True][1]} correct · "
                     f"{negative}s {recall[False][0]}/{recall[False][1]} correct"
                     + (f" · {metrics['invalid']} invalid" if metrics.get('invalid') else ''))
        shortcuts = [k for k, v in (method.get('positional_audit') or {}).items()
                     if isinstance(v, dict) and v.get('exceeds_0_9')]
        if shortcuts:
            lines.append(f"Warning: positional shortcut available ({', '.join(shortcuts)})")
        limit = DETECTION_ERRORS_SHOWN if name == 'detection' else FUZZ_ERRORS_SHOWN
        for label, verdict in ((True, 'missed'), (False, 'false alarm')):
            wrong = [r for r in rows if r['activating'] == label and not r['correct']]
            if not wrong:
                continue
            true_marks = name == 'detection' and label and all(r.get('str_tokens') and r.get('activations') for r in wrong[:limit])
            lines.append(f'{verdict.capitalize()} — {positive if label else negative}, scorer said '
                         f"{'no' if label else 'yes'} (showing {min(limit, len(wrong))} of {len(wrong)})"
                         + ('; true active tokens marked, the scorer saw them unmarked:' if true_marks else ':'))
            for row in wrong[:limit]:
                text = (highlight(row['str_tokens'], row['activations'], CORPUS_HIGHLIGHT) if true_marks
                        else row.get('shown_text', ''))
                lines.append('- ' + text.replace('\n', '\n  '))
    return '\n'.join(lines)


def inspected_windows(text):
    """Activating corpus windows a compact tool result showed: {(conversation, start)}."""
    return {(int(c), int(s)) for peak, c, s in WINDOW_PATTERN.findall(text) if float(peak) > 0}


def build_server(state):
    from mcp.server.mcpserver import MCPServer
    server = MCPServer('molt-agentic-compact')

    def tool(function):
        # Plain text only: structured output would duplicate the payload on the wire.
        return server.tool(structured_output=False)(function)

    def text(value):
        return escape_added_tokens(value)

    @tool
    def ping(feature: int) -> str:
        """Health check for the assigned feature."""
        state._feature(feature)
        return f'pong · feature {feature} · {now()}'

    @tool
    def echo(feature: int, message: str) -> str:
        """Echo text for protocol debugging; control-token strings are escaped."""
        state._feature(feature)
        return text(bounded_text(message, 'message'))

    @tool
    def secret_word(feature: int) -> str:
        """Upstream debug handshake; returns its public test word, not a credential."""
        state._feature(feature)
        return 'pineapple'

    @tool
    def get_server_info(feature: int) -> str:
        """Describe tool availability and the MOLT-specific adaptations."""
        state._feature(feature)
        return text(
            f'Model {state.model_name}, layer {state.layer}, feature {feature} only.\n'
            'Evidence: training conversations and your own probes; memory and drafts are per feature and run.\n'
            f"Training corpus: {'yes' if state.corpus is not None else 'no'} · training scorer: {'yes' if state.scorer is not None else 'no'}\n"
            'Unavailable: get_delphi_explanation (would reveal the blinded baseline).\n'
            'score_explanation is training-only feedback, not held-out evaluation.\n'
            'Steering acts on the context-dependent MOLT transform (multiply=1 no-op, set=0 ablates); '
            'there is no fixed decoder direction.')

    @tool
    def list_latents(feature: int) -> str:
        """Return the assigned feature only; cross-feature investigation is intentionally isolated."""
        state._feature(feature)
        return f'Feature {feature} (layer {state.layer}) is the only available latent.'

    @tool
    def latent_info(feature: int) -> str:
        """Training statistics for the feature's gate. MOLT has no fixed decoder direction; use steering for effects."""
        return text(latent_text(state.get_latent_info(feature)))

    @tool
    def get_top_activating_examples(feature: int, count: int = 5, offset: int = 0) -> str:
        """Strongest training examples as text with <<active spans>>, paginated by offset. Headers give max gate and a window reference (conversation:offset)."""
        payload = state.corpus_call('positive_examples', feature, count=count, offset=offset, selection='top')
        return text(corpus_text(payload, 'top activating training examples', offset))

    @tool
    def get_quantile_examples(feature: int, quantiles: list[float] | None = None, examples_per_quantile: int = 2) -> str:
        """Training examples near activation quantiles (q=1 strongest, q=0 weakest), as text with <<active spans>>."""
        quantiles = [.25, .5, .75, .95] if quantiles is None else quantiles
        if not 1 <= len(quantiles) <= 5:
            raise ValueError('Use 1–5 quantiles')
        groups = [dict(state.corpus_call('positive_examples', feature, count=examples_per_quantile,
                                         selection='quantile', quantile=q), quantile=q) for q in quantiles]
        return text(quantile_text(feature, groups))

    @tool
    def get_non_activating_examples(feature: int, count: int = 5, seed: int = 42) -> str:
        """Zero-gate training examples as plain text, for negative evidence."""
        payload = state.corpus_call('negative_examples', feature, count=count, seed=seed)
        return text(corpus_text(payload, 'non-activating training examples'))

    @tool
    def get_random_activating_examples(feature: int, count: int = 5, seed: int = 42) -> str:
        """Random activating training examples as text with <<active spans>>."""
        payload = state.corpus_call('positive_examples', feature, count=count, seed=seed, selection='random')
        return text(corpus_text(payload, 'random activating training examples'))

    @tool
    def verify_hypothesis(feature: int, texts: list[str], prompt_format: str = 'raw') -> str:
        """Run 1–8 invented probes through the actual Qwen3-4B+MOLT. Returns each probe with <<active tokens>> and their exact gate values. Display forms like ⟪think⟫ are restored to subject control tokens."""
        return text(verify_text(state.verify_hypothesis(feature, texts, prompt_format)))

    @tool
    def steer_generation(feature: int, prompt: str, mode: str = 'multiply', strength: float = 2.0,
                         prompt_format: str = 'raw', positions: list[int] | None = None,
                         max_new_tokens: int = 32, seed: int = 42) -> str:
        """Compare greedy baseline and MOLT-intervened continuations plus the largest first-token logit changes. multiply=1 no-op; set=0 ablates. Intervenes on selected prompt positions (None=all) and every generated token; max 64 new tokens."""
        result = state.steer_generation(feature, prompt, mode, strength, prompt_format, positions, max_new_tokens, seed)
        return text(steer_text(dict(result, all_positions=positions is None)))

    @tool
    async def score_explanation(feature: int, explanation: str) -> str:
        """Training-only detection and fuzz feedback for a candidate explanation: balanced accuracies and a few misclassified examples. Never touches held-out evaluation."""
        state._feature(feature)
        bounded_text(explanation, 'explanation', 4000)
        if state.scorer is None:
            raise RuntimeError('Training-only scorer is not configured')
        result = state.scorer.score_explanation(feature, explanation)
        if inspect.isawaitable(result):
            result = await result
        return text(score_text(result))

    @tool
    def save_explanation(feature: int, explanation: str) -> str:
        """Persist one draft explanation for this feature; does not finalize."""
        return text(saved_text(state.save_explanations(feature, [explanation]), state, feature))

    @tool
    def save_explanations(feature: int, explanations: list[str]) -> str:
        """Persist 1–10 alternative draft explanations for this feature; does not finalize."""
        return text(saved_text(state.save_explanations(feature, explanations), state, feature))

    @tool
    def remember_insight(feature: int, insight: str, category: str = 'general') -> str:
        """Persist an observation or hypothesis for this feature and run only."""
        state.remember_insight(feature, insight, category)
        total = len(state.read_state(feature)['insights'])
        return text(f'Saved insight ({category}); {total} stored for feature {feature}.')

    @tool
    def recall_insights(feature: int, category: str | None = None, limit: int = 20) -> str:
        """Recall this feature's saved insights, newest first."""
        rows = state.recall_insights(feature, category, limit)['insights']
        if not rows:
            return 'No saved insights.'
        return text(f'{len(rows)} insight(s), newest first:\n' +
                    '\n'.join(f"- [{r['category']}] {r['insight']}" for r in rows))

    return server


def saved_text(result, state, feature):
    total = len(state.read_state(feature)['explanations'])
    return (f"Saved {len(result['drafts'])} draft(s); {total} stored for feature {feature}. "
            'Drafts do not finalize — end with [EXPLANATION]: when done.')
