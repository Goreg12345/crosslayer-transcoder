#!/usr/bin/env python3
"""Render the agentic pilot as an offline, standalone comparison and transcript report."""
import argparse
from datetime import datetime, timezone
import html
import hashlib
import json
import math
import os
import re
from pathlib import Path

METHODS = ('detection', 'fuzz', 'active_fuzz', 'intruder')
COHORTS = ('near_chance', 'intermediate', 'strong')


def esc(value):
    return html.escape(str(value), quote=True)


def pretty(value):
    return esc(json.dumps(value, indent=2, ensure_ascii=False))


def read(path, errors):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        errors.append(f'{path.name}: {error}')
        return {}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pct(value):
    return f'{value * 100:.1f}%' if number(value) else '—'


def delta(value):
    if not number(value):
        return '—'
    return f'<span class="{"positive" if value > 0 else "negative" if value < 0 else "muted"}">{value * 100:+.1f} pp</span>'


def metric(methods, method):
    value = methods.get(method, {}) if isinstance(methods, dict) else {}
    if number(value):
        return value
    if not isinstance(value, dict):
        return None
    metrics = value.get('metrics', value)
    return metrics.get('accuracy' if method == 'intruder' else 'balanced_accuracy')


def details(title, data):
    return f'<details><summary>{esc(title)}</summary><pre>{pretty(data)}</pre></details>'


def inline_markdown(text):
    # Parse only our own fixed markup; model HTML and link destinations never execute.
    parts = re.split(r"(`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*)", str(text))
    out = []
    for part in parts:
        if part.startswith('`') and part.endswith('`'):
            out.append('<code>'+esc(part[1:-1])+'</code>')
        elif part.startswith('**') and part.endswith('**') or part.startswith('__') and part.endswith('__'):
            out.append('<strong>'+esc(part[2:-2])+'</strong>')
        elif part.startswith('*') and part.endswith('*') and len(part)>2:
            out.append('<em>'+esc(part[1:-1])+'</em>')
        else:
            out.append(esc(part))
    return ''.join(out)


def markdown(text):
    result, paragraph, code = [], [], None
    def flush():
        if paragraph:
            result.append('<p>'+'<br>'.join(inline_markdown(line) for line in paragraph)+'</p>')
            paragraph.clear()
    for line in str(text).splitlines():
        if line.lstrip().startswith('```'):
            flush()
            if code is None:
                code = []
            else:
                result.append('<pre><code>'+esc('\n'.join(code))+'</code></pre>')
                code = None
        elif code is not None:
            code.append(line)
        elif not line.strip():
            flush()
        elif re.match(r'^#{1,6}\s', line):
            flush();result.append('<h4>'+inline_markdown(re.sub(r'^#+\s+', '', line))+'</h4>')
        elif re.match(r'^\s*(?:[-*+] |\d+[.)] )', line):
            flush();result.append('<div class="md-list">'+inline_markdown(line.strip())+'</div>')
        elif line.startswith('> '):
            flush();result.append('<blockquote>'+inline_markdown(line[2:])+'</blockquote>')
        else:
            paragraph.append(line)
    flush()
    if code is not None:
        result.append('<pre><code>'+esc('\n'.join(code))+'</code></pre>')
    return '<div class="markdown">'+''.join(result)+'</div>'



def diagnostic_case(row, index, scorer_input=None):
    sample = row.get('sample') if isinstance(row.get('sample'), dict) else {}
    tokens = row.get('str_tokens', sample.get('str_tokens', []))
    recorded = row.get('shown_text', sample.get('shown_text'))
    if scorer_input:
        shown = scorer_input['exact_example']
        text_label = 'Exact example excerpt from the saved scorer request · literal text'
        links = '<p class="scorer-input-link"><a href="'+esc(scorer_input['url'])+'" target="_blank" rel="noopener">Open exact scorer input · full explanation, user message, system and few-shots ↗</a></p>'
        provenance = '<p class="status">Source: '+esc(scorer_input['audit_path'])+' · audit line '+esc(scorer_input['audit_line'])+' · '+str(scorer_input['matching_attempts'])+' matching attempt(s). '+('Saved response matches the recorded prediction.' if scorer_input['matches_recorded_prediction'] else 'No matching response prediction verified; inspect recorded attempts.')+'</p>'
    else:
        shown = recorded if recorded is not None else ''.join(tokens)
        text_label = 'Saved case text · exact scorer request not verified here' if recorded is not None else 'Joined saved tokens · inspection fallback, not verified scorer input'
        links = '<p class="status">No matching exact scorer-request audit is attached to this view. This fallback is not claimed as the actual API input.</p>'
        provenance = ''
    label = row.get('activating', sample.get('activating', 'not recorded'))
    prediction = row.get('prediction', 'not recorded')
    identity = row.get('training_window', sample.get('training_window', {}))
    acts = row.get('activations', sample.get('activations', []))
    overlay = ''
    if tokens and len(acts)==len(tokens):
        marked=''.join('<span class="'+('gate-active' if number(a) and a>0 else 'gate-zero')+'" title="token '+str(i)+' · gate '+esc(a)+'">'+esc(token).replace('\r','&#13;')+'</span>' for i,(token,a) in enumerate(zip(tokens,acts)))
        overlay='<details class="activation-overlay"><summary>Activation overlay · dashboard visualization only</summary><p class="status">These gate highlights are added for inspection. They are not inserted into the scorer request; earlier tokens can activate as well as the sampled final token.</p><div class="marked-tokens">'+marked+'</div></details>'
    literal=esc(shown or 'No displayed text recorded.').replace('\r','&#13;')
    return f'<section class="diagnostic-case"><h4>Case {index+1} · {"correct" if row.get("correct") else "wrong / invalid"}</h4><p><b>Label:</b> {esc(label)} · <b>Prediction:</b> {esc(prediction)} · <b>Probability:</b> {esc(row.get("probability", "—"))}</p><p class="status">Training window: {esc(identity)}</p><h4>{text_label}</h4><pre class="scorer-literal"><span>{literal}</span></pre>{links}{provenance}{overlay}{details("Complete case / nested sample", row)}</section>'


def training_feedback(data, scorer_inputs=None):
    body = '<p class="training-notice"><b>Adaptive training diagnostic · not the final held-out benchmark.</b> '+esc(data.get('warning', ''))+'</p>'
    if data.get('explanation'):
        body += '<h4>Explanation tested</h4>'+markdown(data['explanation'])
    rows = []
    for name, result in data['methods'].items():
        metrics = result.get('metrics', {})
        rows.append('<tr><th>'+esc(name)+'</th><td>'+esc(result.get('status','—'))+'</td><td>'+pct(metrics.get('balanced_accuracy'))+'</td><td>'+pct(metrics.get('accuracy'))+'</td><td>'+pct(metrics.get('f1'))+'</td><td>'+esc(metrics.get('valid','—'))+'/'+esc(metrics.get('total','—'))+'</td><td>'+esc(metrics.get('parse_failures','—'))+'</td></tr>')
    body += '<div class="table-wrap"><table><thead><tr><th>Training method</th><th>Status</th><th>Balanced accuracy</th><th>Accuracy</th><th>F1</th><th>Valid / total</th><th>Parse failures</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    for name, result in data['methods'].items():
        inputs=(scorer_inputs or {}).get(name,{})
        all_rows = result.get('rows', [])
        wrong = result.get('wrong_examples', [r for r in all_rows if not r.get('correct')])
        wrong = [all_rows[r['row_index']] if isinstance(r,dict) and type(r.get('row_index')) is int and 0<=r['row_index']<len(all_rows) else r for r in wrong]
        body += '<details class="diagnostic-errors"><summary>'+esc(name)+f' · {len(wrong)} wrong cases / '+str(len(all_rows))+' total</summary>'+''.join(diagnostic_case(row,i,inputs.get(all_rows.index(row)) if row in all_rows else None) for i,row in enumerate(wrong))+'</details>'
        body += '<details><summary>'+esc(name)+' · all training cases</summary>'+''.join(diagnostic_case(row,i,inputs.get(i)) for i,row in enumerate(all_rows))+'</details>'
        body += details(name+' · complete metrics', result.get('metrics', {}))
    return body


def steering_result(data):
    body = '<p class="caveat">Actual subject-model generation comparison. The intervention changes the input-dependent MOLT contribution; gate-trigger explanations and causal generation effects are separate claims.</p><div class="explanations">'
    for key, title in (('baseline','Baseline generation'),('steered','Steered generation')):
        value = data.get(key, data.get(key+'_generation'))
        text = value.get('text', value.get('generated_text', '')) if isinstance(value, dict) else value
        body += '<section><h3>'+title+'</h3>'+markdown(text or '(No generation text returned.)')+'</section>'
    body += '</div>'
    if data.get('intervention'):
        body += details('Intervention settings / positions', data['intervention'])
    prediction = data.get('first_prediction') or {}
    if isinstance(prediction, dict) and prediction:
        body += '<p>Maximum absolute first-token logit change: <b>'+esc(prediction.get('max_abs_logit_change','—'))+'</b></p>'
        rows = ''.join('<tr><td class="token-cell">'+esc(row.get('token',''))+'</td>'+''.join('<td>'+esc(row.get(k,'—'))+'</td>' for k in ('id','baseline','steered','delta'))+'</tr>' for row in prediction.get('tokens',[]))
        body += '<details><summary>First-token logits · baseline vs steered</summary><div class="table-wrap"><table><thead><tr><th>Token</th><th>ID</th><th>Baseline</th><th>Steered</th><th>Δ logit</th></tr></thead><tbody>'+rows+'</tbody></table></div></details>'
    return body


def tool_result(content, scorer_inputs=None):
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except (ValueError, TypeError):
        return markdown(content)
    if not isinstance(data, dict):
        return '<pre>'+pretty(data)+'</pre>'
    examples = data.get('examples', data.get('results', data.get('initial_examples', [])))
    specialized = ''
    excluded = {'examples', 'results', 'groups', 'initial_examples'}
    if isinstance(data.get('methods'), dict):
        specialized += training_feedback(data, scorer_inputs)
        excluded.update(('methods','explanation','warning'))
    if 'steered' in data or 'steered_generation' in data:
        specialized += steering_result(data)
        excluded.update(('baseline','steered','baseline_generation','steered_generation','intervention','first_prediction'))
    metadata = {k:v for k,v in data.items() if k not in excluded}
    scalar = {k:v for k,v in metadata.items() if not isinstance(v, (dict,list))}
    body = specialized+'<dl class="tool-meta">'+''.join('<dt>'+esc(k.replace('_',' '))+'</dt><dd>'+esc(v)+'</dd>' for k,v in scalar.items())+'</dl>'
    for key, value in metadata.items():
        if isinstance(value, (dict,list)):
            body += details(key.replace('_',' ')+' · structured details', value)
    for i, group in enumerate(data.get('groups', [])):
        body += '<details class="quantile-group"><summary>Quantile group '+str(i+1)+'</summary>'+tool_result(group)+'</details>'
    for i, row in enumerate(examples):
        if not isinstance(row, dict):
            body += '<pre>'+pretty(row)+'</pre>';continue
        tokens, acts, ids = row.get('str_tokens', row.get('tokens', [])), row.get('activations', []), row.get('token_ids', [])
        spans, table = [], []
        for j, token in enumerate(tokens):
            act = acts[j] if j<len(acts) else None
            token_id = ids[j] if j<len(ids) else '—'
            active = number(act) and act>0
            value = f'{act:.6g}' if number(act) else '—'
            spans.append(f'<span class="{"gate-active" if active else "gate-zero"}" title="token {j} · ID {esc(token_id)} · gate {esc(value)}">{esc(token)}</span>')
            table.append(f'<tr class="{"gate-active" if active else ""}"><td>{j}</td><td>{esc(token_id)}</td><td class="token-cell">{esc(token)}</td><td>{esc(value)}</td></tr>')
        scope = row.get('evidence_scope', data.get('scope', 'synthetic probe' if 'results' in data else 'training'))
        title = f'{"Probe" if "results" in data else "Window"} {i+1} · peak gate {row.get("max_activation", "—")}'
        body += f'<section class="example"><h4>{esc(title)}</h4><p class="status">{esc(scope)}</p><div class="marked-tokens">{"".join(spans) if spans else esc(row.get("marked_text", ""))}</div>'
        if not table:
            body += '</section>'
            continue
        body += '<details class="token-table"><summary>Every token · index, token ID and gate</summary><div class="table-wrap"><table><thead><tr><th>Index</th><th>Token ID</th><th>Token</th><th>Gate</th></tr></thead><tbody>'+''.join(table)+'</tbody></table></div></details></section>'
    return body



def artifact_link(root, relative, title):
    if root is None or not isinstance(relative, str):
        return ''
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or ':' in relative or not (root/path).is_file():
        return ''
    return '<a href="'+esc(path.as_posix())+'">'+esc(title)+'</a>'


def request_attempts(agent, root):
    rows = []
    events = [e for e in agent.get('events', []) if e.get('type') == 'model_response']
    for event_index, event in enumerate(events):
        turn = event.get('turn', event_index)
        choice = (event.get('response', {}).get('choices') or [{}])[0]
        truncated = choice.get('finish_reason') == 'length'
        file_turn = f'{turn}-truncated-{event_index}' if truncated else str(turn)
        prefix = f'prompt-debug/{agent.get("id", agent.get("feature", ""))}/turn-{file_turn}'
        links = [artifact_link(root, prefix+'.html', 'Rendered prompt / token IDs'), artifact_link(root, prefix+'.json', 'Render JSON'), artifact_link(root, event.get('request_body_path'), 'Immutable request body JSON')]
        links = ' · '.join(link for link in links if link)
        settings = event.get('request_settings', {})
        usage = event.get('response', {}).get('usage', {})
        status = 'TRUNCATED · rejected; not included in conversation' if truncated else 'accepted model response'
        title = f'Turn {turn} · attempt {event.get("attempt", "legacy")} · {status}'
        reasoning = choice.get('message', {}).get('reasoning_content') or choice.get('message', {}).get('reasoning')
        content = choice.get('message', {}).get('content')
        body = f'<p class="status">Output budget: {esc(settings.get("max_tokens", "—"))} · prompt / completion tokens: {esc(usage.get("prompt_tokens", "—"))} / {esc(usage.get("completion_tokens", "—"))} · finish reason: {esc(choice.get("finish_reason", "—"))}</p><p class="debug-links">{links or "Request/render artifacts not exported yet."}</p>'
        body += '<p class="status">Request SHA-256: '+esc(event.get('request_body_sha256', 'not recorded'))+'<br>Provenance: '+esc(event.get('request_body_provenance', 'captured before dispatch' if event.get('request_body_path') else 'no captured request body recorded'))+'</p>'
        if truncated:
            if reasoning:
                body += '<details class="returned-reasoning"><summary>Truncated returned reasoning · preserved audit, not accepted</summary>'+markdown(reasoning)+'</details>'
            if content:
                body += '<details><summary>Truncated returned content · not accepted</summary>'+markdown(content)+'</details>'
        body += details('Request settings', settings)
        rows.append('<details class="request-attempt"><summary>'+esc(title)+'</summary>'+body+'</details>')
    return '<details class="request-attempts"><summary>Model request attempts &amp; truncated outputs · '+str(len(events))+' recorded responses</summary>'+''.join(rows)+'</details>' if rows else ''


def transcript(agent, root=None, embed_raw=True):
    blocks, turn = [], 0
    messages = agent.get('messages', [])
    tool_names = {}
    returned_reasoning = {}
    for event in agent.get('events', []):
        if event.get('type') == 'model_response':
            response = event.get('response', {})
            for choice in response.get('choices', []):
                provider_message = choice.get('message', {})
                reasoning = provider_message.get('reasoning_content') or provider_message.get('reasoning')
                if reasoning:
                    returned_reasoning[event.get('turn')] = reasoning
    if isinstance(messages, list):
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                blocks.append(details(f'{i+1:02d} · event', message));continue
            role = message.get('role', message.get('type', 'event'))
            label = f'{i+1:02d} · {role}'
            debug = ''
            if role == 'assistant':
                label += f' · model turn {turn}'
                prefix = f'prompt-debug/{agent.get("id", agent.get("feature", ""))}/turn-{turn}'
                if root is not None and (root/(prefix+'.html')).exists():
                    debug = f'<p class="debug-links"><a href="{esc(prefix)}.html">Tools &amp; model input · turn {turn}</a> · <a href="{esc(prefix)}.json">Exact request / token IDs JSON</a></p>'
                elif root is not None:
                    debug = '<p class="status">Exact rendered model-input artifact pending for this turn.</p>'
                turn += 1
            content = message.get('content')
            if role == 'tool':
                name = tool_names.get(message.get('tool_call_id'), 'tool result')
                readable = '<details class="tool-readable"><summary>'+esc(name)+' · readable result / activations</summary>'+tool_result(content)+'</details>'
            else:
                readable = markdown(content) if isinstance(content, str) else ('<pre>'+pretty(content)+'</pre>' if content else '')
            reasoning = message.get('reasoning_content') or message.get('reasoning')
            if role == 'assistant' and not reasoning:
                reasoning = returned_reasoning.get(turn-1)
            if reasoning:
                readable = '<details class="returned-reasoning"><summary>Returned model reasoning · '+str(len(str(reasoning).split()))+' words · click to read</summary>'+markdown(reasoning)+'</details>'+readable
            for call in message.get('tool_calls', []):
                function = call.get('function', {})
                tool_names[call.get('id')] = function.get('name', 'tool')
                arguments = function.get('arguments', '')
                try:
                    arguments = json.loads(arguments)
                except (ValueError, TypeError):
                    pass
                readable += '<div class="tool-call"><b>Call '+esc(function.get('name', 'tool'))+'</b><pre>'+pretty(arguments)+'</pre></div>'
            raw = details('Raw message JSON · all fields', message).replace('<details>', '<details class="raw-json">', 1) if embed_raw else ''
            blocks.append(f'<section class="message role-{esc(role)}"><h3>{esc(label)}</h3>{debug}{readable}{raw}</section>')
    elif messages:
        blocks.append(details('Messages', messages))
    blocks.append(recovery_events(agent))
    blocks.append(request_attempts(agent, root))
    for key in ('events', 'responses', 'transcript'):
        if embed_raw and agent.get(key):
            blocks.append(details(f'Full raw {key} · includes provider responses / returned reasoning', agent[key]).replace('<details>', '<details class="raw-json">', 1))
    if not blocks:
        blocks.append('<p class="muted">No transcript recorded yet.</p>')
    if not embed_raw:
        blocks.append('<p class="links">'+artifact_link(root, f'agents/{agent.get("id", agent.get("feature", ""))}.json', 'Complete raw transcript / provider responses / all fields JSON')+'</p>')
    extra = {k:v for k,v in agent.items() if k not in ('messages', 'events', 'responses', 'transcript')}
    blocks.append(details('Remaining agent artifact fields', extra))
    return '<details class="transcript"><summary>Readable full agent transcript · ' + str(len(messages)) + ' messages</summary>' + ''.join(blocks) + '</details>'



def scoring_attempt_summary(path):
    attempts, valid, budgets, incomplete = 0, 0, {}, 0
    if path.exists():
        with path.open() as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except ValueError:
                    incomplete += 1
                    continue
                if item.get('method') != 'intruder':
                    continue
                attempts += 1
                valid += bool(item.get('valid'))
                budget = item.get('max_tokens', item.get('body', {}).get('max_tokens'))
                if number(budget):
                    budgets[budget] = budgets.get(budget, 0)+1
    return dict(intruder_attempts=attempts, valid_attempts=valid, output_budgets=budgets, incomplete_audit_lines=incomplete)


def recovery_events(agent):
    events = [e for e in agent.get('events', []) if e.get('type') == 'turn_budget_extension']
    body = ''
    for event in events:
        body += '<section class="run-banner"><h4>Audited finalization-turn allowance</h4><p>Turn limit '+esc(event.get('old_limit', '—'))+' → '+esc(event.get('new_limit', '—'))+' · '+esc(event.get('additional_turns', '—'))+' additional turns. Missing requirements: '+esc(', '.join(event.get('missing_requirements', [])) or 'none recorded')+'</p>'+markdown(event.get('user_note', ''))+details('Exact recovery event', event)+'</section>'
    return body


def load_features(root, selection, baseline):
    features = []
    for selected in selection.get('selected', []):
        errors = []
        feature_id = str(selected['id'])
        # IDs originate in local selections but are validated before constructing file paths.
        if not feature_id.isdecimal():
            raise ValueError(f'Invalid feature ID: {feature_id!r}')
        agent = read(root / 'agents' / f'{feature_id}.json', errors)
        score_path = root / 'scored-features' / f'{feature_id}.json'
        if not score_path.exists():
            score_path = root / 'comparisons' / f'{feature_id}.json'
        score = read(score_path, errors)
        intruder_policy = score.get('intruder_retry_policy') or read(root/'scoring-policy'/f'{feature_id}.json', errors)
        scoring_attempts = scoring_attempt_summary(root/'scoring-audit'/f'{feature_id}.jsonl')
        attempts = [(path.name, read(path, errors)) for path in sorted((root/'agent-attempts').glob(f'{feature_id}-*.json'))]
        preflight_path = root/'smoke-before-coordinate-clarification'/'agents'/f'{feature_id}.json'
        preflight = read(preflight_path, errors) if preflight_path.exists() else {}
        original = read(baseline / 'features' / f'{feature_id}.json', errors) if baseline else {}
        old_methods = original.get('methods', selected.get('baseline_scores', selected.get('scores', {})))
        comparisons = score.get('comparisons', {}).get('historical', {})
        scores = {}
        for method in METHODS:
            comparison = comparisons.get(method, {})
            old = comparison.get('baseline', metric(old_methods, method))
            new = comparison.get('candidate', metric(score.get('methods', {}), method))
            fresh = metric(score.get('refreshed_baseline', {}), method)
            scores[method] = dict(old=old, new=new, fresh=fresh,
                                  delta=new-old if number(old) and number(new) else None,
                                  fresh_delta=new-fresh if number(fresh) and number(new) else None)
        errors.extend(str(agent[k]) for k in ('error', 'errors') if agent.get(k))
        errors.extend(str(score[k]) for k in ('error', 'errors') if score.get(k))
        features.append(dict(id=feature_id, selected=selected, agent=agent, score=score, attempts=attempts, preflight=preflight, intruder_policy=intruder_policy, scoring_attempts=scoring_attempts,
                             score_path=score_path, scores=scores, errors=errors,
                             original=original.get('explanation', selected.get('baseline_explanation', 'Original explanation unavailable.')),
                             explanation=agent.get('explanation') or score.get('explanation') or 'Awaiting frozen explanation.',
                             status=agent.get('status', 'pending'), cohort=selected.get('cohort', 'unassigned')))
    return features


def summary(features):
    rows = []
    for cohort in ('all', *sorted({f['cohort'] for f in features})):
        group = [f for f in features if cohort == 'all' or f['cohort'] == cohort]
        for method in METHODS:
            paired = [f['scores'][method] for f in group if number(f['scores'][method]['old']) and number(f['scores'][method]['new'])]
            n = len(paired)
            mean = lambda key: sum(p[key] for p in paired) / n if n else None
            fresh = [p for p in paired if number(p['fresh'])]
            fresh_mean = sum(p['fresh'] for p in fresh)/len(fresh) if fresh else None
            fresh_delta = sum(p['fresh_delta'] for p in fresh)/len(fresh) if fresh else None
            rows.append(f'<tr><td>{esc(cohort.replace("_", " "))}</td><td>{esc(method)}</td><td>{n}/{len(group)}</td><td>{pct(mean("old"))}</td><td>{pct(mean("new"))}</td><td>{delta(mean("delta"))}</td><td>{pct(fresh_mean)} <small>n={len(fresh)}</small></td><td>{delta(fresh_delta)}</td></tr>')
    return '<div class="table-wrap"><table><thead><tr><th>Cohort</th><th>Method</th><th>Paired / selected</th><th>Historical original</th><th>Agentic</th><th>Δ historical</th><th>Fresh original</th><th>Δ fresh · primary</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def feature_html(f, root, embed_raw=True):
    agent, score = f['agent'], f['score']
    rows = []
    for method, values in f['scores'].items():
        metrics = score.get('methods', {}).get(method, {}).get('metrics', {})
        valid = f'{metrics.get("valid", "—")}/{metrics.get("total", "—")}'
        rows.append(f'<tr><td>{esc(method)}<small>{"coherence · accuracy" if method == "intruder" else "balanced accuracy"}</small></td><td>{pct(values["old"])}</td><td>{pct(values["new"])}</td><td>{delta(values["delta"])}</td><td>{pct(values["fresh"])}</td><td>{delta(values["fresh_delta"])}</td><td>{esc(valid)}</td></tr>')
    links = []
    for label, path in [('Agent JSON', root/'agents'/f'{f["id"]}.json'), ('Scores JSON', f['score_path']), ('Frozen scoring plan', root/'scoring-plans'/f'{f["id"]}.json'), ('Scoring audit', root/'scoring-audit'/f'{f["id"]}.jsonl')]:
        if path.exists():
            links.append(f'<a href="{esc(path.relative_to(root).as_posix())}">{label}</a>')
    errors = ''.join(f'<p class="error">{esc(e)}</p>' for e in f['errors'])
    attempts = ''.join('<details><summary>'+esc(f'Archived attempt · {name} · full transcript')+'</summary>'+transcript(data)+'</details>' for name, data in f.get('attempts', []))
    if f.get('preflight'):
        attempts += '<details><summary>Archived preflight · NOT scored · before coordinate/sampling clarification · full artifact and explanation</summary>'+markdown(f['preflight'].get('explanation', 'No frozen explanation recorded.'))+transcript(f['preflight'])+'</details>'
    usage = agent.get('usage', {})
    scoring_attempts = f.get('scoring_attempts', {})
    budget_counts = ', '.join(f'{budget}: {count} attempts' for budget, count in sorted(scoring_attempts.get('output_budgets', {}).items())) or 'none recorded yet'
    scoring_policy_html = details('Intruder adaptive output policy · full audit', f['intruder_policy']) if f.get('intruder_policy') else ''
    revision = score.get('baseline_revision', f['selected'].get('baseline_revision', 'not recorded'))
    corrected = score.get('display_corrected_cases', 'pending')
    scoring_status = 'complete' if all(number(f['scores'][m]['new']) for m in METHODS) else 'pending / incomplete'
    meta = {k:v for k,v in agent.items() if k not in ('messages', 'events', 'responses', 'transcript', 'explanation')}
    return f'''<article class="feature" data-cohort="{esc(f['cohort'])}" data-id="{esc(f['id'])}">
<div class="feature-heading"><h2>Feature {esc(f['id'])} <small>rank {esc(f['selected'].get('rank', '—'))}</small></h2><span class="badge">{esc(f['cohort'].replace('_',' '))}</span></div>
<p class="status">Agent: <b>{esc(f['status'])}</b> · scoring: <b>{scoring_status}</b></p>{errors}
<p class="status">Model turns: {esc(agent.get("turns", len([e for e in agent.get("events", []) if e.get("type") == "model_response"])))} · prompt tokens (summed): {esc(usage.get("prompt_tokens", "—"))} · output tokens: {esc(usage.get("completion_tokens", "—"))} · largest prompt: {esc(agent.get("max_prompt_tokens", "—"))} · distinct training windows inspected: {esc(agent.get("training_windows_inspected", "—"))} · explanation words: {esc(agent.get("word_count", "—"))}{" (over requested 80; preserved)" if agent.get("exceeds_requested_80_words") else ""}</p>
<div class="explanations"><section><h3>Original explanation</h3><div class="explanation">{esc(f['original'])}</div></section><section><h3>Agentic explanation</h3><div class="explanation">{esc(f['explanation'])}</div></section></div>
<div class="table-wrap"><table><thead><tr><th>Method</th><th>Historical original</th><th>Agentic</th><th>Δ historical</th><th>Fresh original</th><th>Δ fresh · primary</th><th>Valid / total</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="status">Baseline revision: {esc(revision)} · display-corrected scoring cases: {esc(corrected)}</p><p class="links">{' · '.join(links)}</p>{transcript(agent, root, embed_raw)}{attempts}{details('Agent run metadata', meta)}<p class="status">Intruder recorded output budgets: {esc(budget_counts)} · {esc(scoring_attempts.get('valid_attempts', 0))} valid attempts. Attempts are not unique evaluation cases; scoring completion is shown separately above.</p>{scoring_policy_html}{details('Scoring result · full metrics, labels and provenance', score) if embed_raw else details('Scoring metadata · full rows linked in Scores JSON', {k:v for k,v in score.items() if k not in ('methods','refreshed_baseline')})}</article>'''


STYLE = '''
.training-notice{background:#fff4d6;border-left:3px solid #c49126;padding:10px}.diagnostic-case{border-top:1px solid #d9e3e9;padding:12px 0}.marked-tokens mark{background:#d8e9fb;color:#173a62}

.returned-reasoning{border-color:#a99bc8;background:#f5f0ff;padding:14px}.returned-reasoning>summary{color:#62458c;font-size:14px}.run-banner{border:2px solid #c68a2b;background:#fff5de;border-radius:10px;padding:18px;margin:16px 0}.settings-panel{border:2px solid #369788}.settings-panel td{white-space:normal}

.message{padding:18px;margin:14px 0;background:#f8fafb;border-left:3px solid #c8d7df;border-radius:6px}.role-assistant{border-left-color:#007d73;background:#f5fbf9}.role-tool{border-left-color:#9a7634}.markdown{font-size:14px;line-height:1.65;max-width:1000px}.markdown h4,.example h4{margin:12px 0 6px;font-size:15px}.markdown p{margin:9px 0}.markdown code{background:#e8eef2;border-radius:3px;padding:1px 4px}.md-list{margin:5px 0 5px 16px}.markdown blockquote{border-left:3px solid #b6c8d1;padding-left:12px;margin-left:0}.tool-call{margin:12px 0}.tool-meta{display:grid;grid-template-columns:minmax(110px,200px) 1fr;gap:5px 12px;font-size:12px}.tool-meta dt{color:#526576}.tool-meta dd{margin:0;overflow-wrap:anywhere}.example{border-top:1px solid #d9e3e9;padding:12px 0}.marked-tokens{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.9;font-family:ui-monospace,monospace;font-size:13px;background:white;padding:12px;border-radius:5px}.gate-active{background:#ffe6a3;color:#523600}.marked-tokens .gate-active{box-shadow:0 2px #d69b25}.token-cell{white-space:pre-wrap;min-width:100px;font-family:ui-monospace,monospace}.debug-links{font-size:12px;font-weight:650}.raw-json{background:#f2f5f7}.tool-readable{background:#fff}

:root{color-scheme:light;--ink:#203344;--muted:#5c7180;--line:#d9e3e9;--accent:#007d73}*{box-sizing:border-box}body{margin:0;overflow-wrap:anywhere;background:#f1f5f7;color:var(--ink);font:14px/1.55 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:28px}header{margin-bottom:26px}h1{font-size:32px;line-height:1.2;margin:8px 0}h2{font-size:21px;margin:0}h3{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 9px}p{margin:10px 0}.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);font-weight:700}.muted,small{color:var(--muted)}small{font-size:11px}td small{display:block}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.card,.feature,.panel{background:white;border:1px solid var(--line);border-radius:12px;padding:20px}.card strong{display:block;font-size:28px}.card span{color:var(--muted)}.panel{margin:16px 0}.feature{margin:20px 0;scroll-margin-top:15px}.feature-heading{display:flex;align-items:center;justify-content:space-between;gap:12px}.badge{padding:4px 11px;border-radius:20px;background:#e7f3f0;color:#00685e;font-size:12px}.status{font-size:12px;color:var(--muted)}.explanations{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin:20px 0}.explanations section{padding:16px;background:#f6f9fa;border-radius:8px}.explanation{white-space:pre-wrap;overflow-wrap:anywhere;font-size:15px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{font-size:11px;color:var(--muted);background:#f6f9fa}td{font-size:13px}.positive{color:#00745c}.negative{color:#b23938}.error{background:#fff0ee;border-left:3px solid #b23938;padding:10px;white-space:pre-wrap;overflow-wrap:anywhere}a{color:var(--accent)}.links{font-size:12px}details{margin:9px 0;border:1px solid var(--line);border-radius:7px;padding:10px 12px}summary{cursor:pointer;font-size:13px;font-weight:600}pre{background:#f6f9fa;border-radius:6px;padding:14px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.5}button,input,select{font:inherit;border:1px solid #bdcdd5;border-radius:6px;padding:8px 10px;background:white;color:var(--ink)}button{cursor:pointer}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:22px 0}.toolbar input{width:210px}[hidden]{display:none!important}.caveat{max-width:1100px;font-size:13px}footer{color:var(--muted);font-size:12px;margin:28px 0}@media(max-width:750px){main{padding:14px}.explanations{grid-template-columns:1fr}.cards{grid-template-columns:1fr 1fr}.feature{padding:14px}h1{font-size:26px}.feature-heading{align-items:flex-start}th,td{padding:8px}.badge{white-space:nowrap}}
'''
SCRIPT = '''
const cohort=document.querySelector('#cohort'),query=document.querySelector('#query');
function filter(){let n=0;for(const card of document.querySelectorAll('.feature')){card.hidden=!!((cohort.value!=='all'&&card.dataset.cohort!==cohort.value)||(query.value.trim()&&!card.textContent.toLowerCase().includes(query.value.trim().toLowerCase())));if(!card.hidden)n++}document.querySelector('#visible').textContent=n+' features shown';}
cohort.addEventListener('change',filter);query.addEventListener('input',filter);
document.querySelector('#expand').addEventListener('click',()=>{for(const card of document.querySelectorAll('.feature:not([hidden])'))for(const d of card.querySelectorAll('.transcript, .tool-readable'))d.open=true});
document.querySelector('#collapse').addEventListener('click',()=>{for(const d of document.querySelectorAll('details'))d.open=false});filter();
'''



def execution_panel(root, features, errors):
    artifacts = {}
    for name in ('run.json', 'configuration.json', 'gate-validation.json', 'scoring-validation.json', 'tool-inventory.json', 'scorer-configuration.json', 'retry-policy.json', 'recovery-policy.json'):
        if (root/name).exists():
            artifacts[name] = read(root/name, errors)
    for name in ('run.json', 'configuration.json'):
        path = root/'smoke-before-coordinate-clarification'/name
        if path.exists():
            artifacts['smoke-before-coordinate-clarification/'+name] = read(path, errors)
    for path in sorted((root/'infrastructure').glob('*.json')):
        artifacts['infrastructure/'+path.name] = read(path, errors)
    run, config = artifacts.get('run.json', {}), artifacts.get('configuration.json', {})
    gate = artifacts.get('gate-validation.json', {})
    agents = [f['agent'] for f in features if f['agent']]
    prompt = sum(a.get('usage', {}).get('prompt_tokens', 0) for a in agents)
    output = sum(a.get('usage', {}).get('completion_tokens', 0) for a in agents)
    maximum = max((a.get('max_prompt_tokens', 0) for a in agents), default=0)
    points = []
    for agent in agents:
        for event in agent.get('events', []):
            if event.get('type') == 'model_response' and event.get('at') and number(event.get('duration_seconds')):
                try:
                    end = datetime.fromisoformat(event['at']).timestamp()
                    points.extend(((end-event['duration_seconds'], 1), (end, -1)))
                except ValueError:
                    pass
    current = peak = 0
    for _, change in sorted(points):
        current += change
        peak = max(current, peak)
    validations = []
    def prompt_counts(value):
        found = []
        if isinstance(value, dict):
            if number(value.get('prompt_tokens')):
                found.append(value['prompt_tokens'])
            for child in value.values():
                found.extend(prompt_counts(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(prompt_counts(child))
        return found
    for name, data in artifacts.items():
        if name.startswith('infrastructure/') and ('context' in name or 'validation' in name):
            counts = prompt_counts(data)
            validations.append(f'<li>{esc(name)}: largest recorded prompt <b>{max(counts):,}</b> tokens</li>' if counts else f'<li>{esc(name)}: see full validation artifact below</li>')
    limits = sorted({a.get('generation', {}).get('context_limit') for a in agents if number(a.get('generation', {}).get('context_limit'))})
    context = ', '.join(f'{v:,}' for v in limits) or 'not yet recorded in agent artifacts'
    full_tools = len(artifacts.get('tool-inventory.json', {}).get('tools', [])) > 4
    evidence_description = ('The full training corpus is available through positive, negative, random and quantile tools, plus invented synthetic probes evaluated against actual subject-model JumpReLU gates. Training diagnostics use 16 positive and 16 negative examples per detection/fuzz method; these diagnostics are distinct from the final held-out scoring cases.' if full_tools else 'Each feature has 32 fixed training windows available, plus invented synthetic probes evaluated against actual subject-model JumpReLU gates.')
    workflow_description = ('The configured workflow requests at least 16 distinct activating windows, negative windows and weaker examples, at least two hypothesis-verification batches, and training diagnostic feedback on at least two alternative explanations. Memory and live steering tools support autonomous investigation.' if full_tools else 'The configured workflow requires at least 16 distinct training windows, weak/middle quantile examples, and at least two successful hypothesis-verification calls before accepting an explanation.')
    statuses = ', '.join(f'{state}: {sum(f["status"] == state for f in features)}' for state in sorted({f['status'] for f in features}))
    gate_note = (f'Gate validation: <b>{"passed" if gate.get("passed") else "not passed"}</b>; feature {esc(gate.get("feature"))}, {esc(gate.get("compared_tokens"))} tokens, maximum absolute error {esc(gate.get("max_absolute_error"))}, activation-support mismatches {esc(gate.get("activation_support_mismatches"))}.' if gate else 'Gate validation has not been recorded yet.')
    return f'''<section class="panel"><h2>Execution and evidence</h2>
<p class="caveat">This is a local OpenAI-compatible tool loop adapted from the agentic Delphi evidence → hypothesis → verification workflow, using a MOLT gate MCP adapter. It does not run the upstream Anthropic connector. {evidence_description} These gates describe token activation, not downstream causal effects.</p><p class="caveat"><b>Sampling caveat:</b> training windows end at a selected activation peak by construction; end-of-window firing is a sampling artifact, not evidence for an end-position trigger. Window start/stop values are global cache offsets, not within-conversation positions; preceding conversation context may be omitted. These coordinate and sampling clarifications were added before the full run. Earlier preflight transcripts are preserved below for audit and are not scored.</p>
<p class="caveat">{workflow_description} Explanations are requested to be at most 80 words, but overlength responses are preserved without truncation, as are the existing baseline explanations. Per-feature metadata records the actual word count and limit exceedance. Scoring reuses matching held-out evidence, labels, examples and order; fresh-original and agentic arms use the same corrected display and scorer settings.</p>
<div class="table-wrap"><table><tbody>
<tr><th>Run status</th><td>{esc(run.get('status', 'not recorded'))} · {esc(statuses)}</td></tr>
<tr><th>Agent / scoring concurrency configured</th><td>{esc(run.get('agent_concurrency', '—'))} / {esc(run.get('scoring_concurrency', '—'))} during agents; {esc(run.get('scoring_concurrency_after_agents', run.get('scoring_concurrency', '—')))} after agents</td></tr>
<tr><th>Observed peak overlapping agent model requests</th><td>{peak if points else 'not recorded'} · reconstructed from completed response timestamps and durations; includes request retries</td></tr>
<tr><th>Agent prompt / output tokens, summed</th><td>{prompt:,} / {output:,} · includes repeated prompt prefixes</td></tr>
<tr><th>Largest actual agent prompt</th><td>{maximum:,} tokens</td></tr>
<tr><th>Configured native context limit</th><td>{context} tokens · configuration is distinct from a full-capacity validation</td></tr>
<tr><th>Model / thinking / output limit</th><td>{esc(config.get('model', '—'))} / {esc(config.get('enable_thinking', '—'))} / {esc(config.get('max_tokens', '—'))}</td></tr>
</tbody></table></div><p class="caveat">{gate_note}</p><ul>{''.join(validations) or '<li>No context validation artifact recorded yet.</li>'}</ul>
{''.join(details(name + ' · full artifact', data) for name, data in artifacts.items())}</section>'''



def tooling_panel(root, errors):
    tools = read(root/'tool-schemas.json', errors)
    manifest = read(root/'prompt-debug'/'index.json', errors)
    inventory = read(root/'tool-inventory.json', errors)
    names = [t.get('function', t).get('name', 'Tool') for t in tools] if isinstance(tools, list) else inventory.get('tools', [])
    rows = []
    if isinstance(tools, list):
        for tool in tools:
            function = tool.get('function', tool)
            rows.append('<section class="tool-call"><h3>'+esc(function.get('name', 'Tool'))+'</h3><p>'+esc(function.get('description', ''))+'</p>'+details('Full API tool schema', function.get('parameters', function.get('input_schema', {})))+'</section>')
    return '<section class="panel" id="tools-model-input"><h2>Tools &amp; model input · '+str(len(names))+' tools</h2><p>'+esc(', '.join(names))+'</p><p class="caveat">Tools are supplied in the API request’s <code>tools</code> field. vLLM applies the model’s chat template, which injects the tool definitions into the system prefix before the user message. Tool-call arguments and tool-result messages then become part of subsequent requests. The readable transcript below displays message content; it is not itself the final chat-template string.</p><p class="caveat">Each available model-turn or retry link opens prompt text and token IDs rendered by the unchanged server’s render endpoint. New requests preserve immutable API bodies captured before dispatch with SHA-256 hashes; older requests may instead use reconstructed bodies from frozen history, schemas and settings. Per-attempt provenance distinguishes them. Rendered prompt strings are reconstructed by the render endpoint, not captured during inference. The debug artifacts report comparisons against original API usage and token decode/re-encode checks. Full token IDs live in separate static artifacts to keep this report manageable.</p>'+''.join(rows)+('<p><a href="prompt-debug/index.json">Prompt-debug manifest JSON</a></p>'+details('Prompt-render export progress and validation', manifest) if manifest else '<p class="status">Prompt rendering exports are pending. Links appear beside assistant turns when their debug artifacts are available.</p>')+'</section>'



def settings_panel(root, features, errors):
    config = read(root/'configuration.json', errors)
    runtime = read(root/'scorer-runtime.json', errors)
    policy = read(root/'retry-policy.json', errors)
    recovery = read(root/'recovery-policy.json', errors)
    intruder_policy = next((f['intruder_policy'] for f in features if f.get('intruder_policy')), {})
    run = read(root/'run.json', errors)
    scorer = read(root/'scorer-configuration.json', errors) or run.get('scorer_settings', {})
    source = 'planned scoring configuration'
    for feature in features:
        if feature['score'].get('scorer_settings'):
            scorer = feature['score']['scorer_settings']
            source = 'executed scoring artifact'
            break
    if not scorer:
        paths = sorted((root/'scoring-config').glob('*.json'))
        if paths:
            scorer = read(paths[0], errors)
            source = 'frozen scoring request configuration'
    def row(label, settings, provenance):
        thinking = settings.get('enable_thinking', settings.get('chat_template_kwargs', {}).get('enable_thinking'))
        thinking = 'ON' if thinking is True else 'OFF' if thinking is False else 'not recorded'
        sampling = ', '.join(f'{k}={settings[k]}' for k in ('temperature','top_p','top_k','min_p') if k in settings) or 'not recorded'
        effort = settings.get('reasoning_effort', settings.get('chat_template_kwargs', {}).get('reasoning_effort'))
        return f'<tr><th>{esc(label)}</th><td><b>{thinking}</b>{" · "+esc(effort) if effort else ""}</td><td>{esc(settings.get("max_tokens", "not recorded"))}</td><td>{esc(sampling)}</td><td>{esc(provenance)}</td></tr>'
    agent_settings = dict(config)
    if policy.get('truncation_budgets'):
        agent_settings['max_tokens'] = str(policy.get('base_max_tokens', config.get('max_tokens', '—')))+' base; retry budgets '+', '.join(str(v) for v in policy['truncation_budgets'])
    rows = row('Agent investigator', agent_settings, 'frozen agent configuration + retry policy' if policy else 'frozen agent configuration')
    rows += row('Detection / fuzz / active fuzz', scorer.get('binary', {}), source)
    intruder = dict(scorer.get('intruder', {}))
    if scorer.get('intruder_reasoning_effort'):
        intruder['reasoning_effort'] = scorer['intruder_reasoning_effort']
    if intruder_policy.get('output_token_budgets'):
        intruder['max_tokens'] = str(intruder.get('max_tokens', '—'))+' base; length-only retries '+', '.join(str(b) for b in intruder_policy['output_token_budgets'])
    if runtime.get('context_limit'):
        intruder['max_tokens'] = str(runtime.get('intruder_max_tokens', 8192))+' maximum initially; bounded by context remaining after the exact prompt'
    rows += row('Intruder coherence control', intruder, source+' + audited adaptive output policy' if intruder_policy else source)
    notice = '<p class="caveat"><b>Intruder is not directly comparable to historical settings:</b> this run enables reasoning and changes its generation budget/sampling. Compare matched fresh controls; historical changes cannot be credited to explanation improvement. Both fresh arms share the same intruder configuration, and intruder never consumes an explanation.</p>' if scorer.get('intruder_changed_from_historical') or scorer.get('intruder_note') else ''
    banner = ''
    if run.get('status') == 'needs_attention':
        banner = '<div class="run-banner"><b>Run needs attention · incomplete</b><p>'+esc(run.get('supervisor_error', 'See supervisor status for details.'))+'</p></div>'
    elif run.get('status') in ('stopped','superseded','cancelled','interrupted') or 'superseded' in str(run.get('status', '')):
        banner = '<div class="run-banner"><b>Archived run · '+esc(run.get('status'))+'</b><p>'+esc(run.get('reason', 'This run is no longer active; partial artifacts are preserved.'))+'</p></div>'
    retry_note = ''
    if policy:
        observed = sorted({e.get('request_settings', {}).get('max_tokens') for f in features for e in f['agent'].get('events', []) if e.get('type') == 'model_response' and number(e.get('request_settings', {}).get('max_tokens'))})
        retry_note = '<p class="caveat"><b>Checkpoint continuation and truncation recovery:</b> length-truncated outputs are retained for audit but never enter the accepted conversation or finalize an explanation. The same turn retries at a larger output budget, with thinking ON / xhigh throughout. Actual observed output budgets: '+esc(', '.join(str(v) for v in observed) or 'none yet')+'. The configured context limit is 262,144 tokens; output budgets are distinct from actual prompt length and do not establish a full-context validation. Request-attempt sections expose each truncated response, its input-debug page and immutable request body when captured.</p>'
    recovery_note = ''
    if recovery:
        recovery_note = '<p class="caveat"><b>Audited required-finalization allowance:</b> the frozen '+esc(recovery.get('base_max_turns', '—'))+'-turn investigation budget has a '+esc(recovery.get('additional_turns', '—'))+'-turn recovery allowance, for '+esc(recovery.get('effective_max_turns', '—'))+' total turns, to complete missing required tools and finalize existing work. Acceptance requirements remain unchanged and thinking remains ON / xhigh. Exact missing requirements and recovery notes are preserved per feature.</p>'
    intruder_note = ''
    if intruder_policy:
        observed = sorted({b for f in features for b in f.get('scoring_attempts', {}).get('output_budgets', {})})
        intruder_note = '<p class="caveat"><b>Intruder adaptive output allowance:</b> only a length-truncated response permits a larger budget; correctness never triggers a retry. The first valid response at any approved budget is reused, and the explanation-independent result is shared between both fresh arms. Observed intruder budgets in scoring audits: '+esc(', '.join(str(b) for b in observed) or 'none yet')+'. Per-feature policy and attempt counts remain separate from completed scores.</p>'
    archive = '<p><a href="v1-superseded/index.html">View superseded v1 report and its preserved transcripts</a></p>' if (root/'v1-superseded'/'index.html').exists() else ''
    routing = ''
    if runtime:
        routing = '<p class="caveat"><b>Current scoring server:</b> '+esc(runtime.get('server', runtime.get('base_url', '—')))+' · '+esc(runtime.get('model', '—'))+' · '+esc(runtime.get('context_limit', '—'))+' tokens total context. Training feedback and both fresh held-out arms use this endpoint. The explainer stays on the local RTX PRO 6000 with thinking ON / xhigh and 262,144-token context. Earlier local scoring attempts and training feedback remain in the audit; final matched results are recomputed on the current endpoint.</p>'+details('Active scorer configuration and migration', runtime)
    return banner+'<section class="panel settings-panel"><h2>Model settings · agent and final scorers</h2>'+routing+'<div class="table-wrap"><table><thead><tr><th>Role</th><th>Thinking</th><th>Output budget</th><th>Sampling</th><th>Evidence</th></tr></thead><tbody>'+rows+'</tbody></table></div>'+notice+retry_note+recovery_note+intruder_note+'<p class="caveat">Output budgets include reasoning and final answer where reasoning is enabled. Agent tools may provide training diagnostics; those are separate from final held-out scorers.</p>'+archive+'</section>'



SPLIT_VERSION = 1
SPLIT_SCRIPT = '''
const rows=[...document.querySelectorAll('.campaign-row')],filters=[...document.querySelectorAll('[data-filter]')],q=document.querySelector('#query');let page=0;const size=100;
function draw(reset=false){if(reset)page=0;const text=q.value.trim().toLowerCase(),selected=rows.filter(r=>filters.every(f=>!f.value||r.dataset[f.dataset.filter]===f.value)&&(!text||r.textContent.toLowerCase().includes(text)));page=Math.min(page,Math.max(0,Math.ceil(selected.length/size)-1));for(const r of rows)r.hidden=true;for(const r of selected.slice(page*size,(page+1)*size))r.hidden=false;document.querySelector('#visible').textContent=selected.length+' matching / '+rows.length+' total · page '+(page+1)+' of '+Math.max(1,Math.ceil(selected.length/size));document.querySelector('#prev').disabled=page===0;document.querySelector('#next').disabled=(page+1)*size>=selected.length;}
for(const f of filters)f.addEventListener('change',()=>draw(true));q.addEventListener('input',()=>draw(true));document.querySelector('#prev').onclick=()=>{page--;draw()};document.querySelector('#next').onclick=()=>{page++;draw()};draw();
'''


def write_report(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def split_shell(title, body, script=''):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{STYLE}\n.campaign-table .description{{white-space:normal;min-width:220px;max-width:350px}}.campaign-table td{{vertical-align:top}}.campaign-table small{{display:block}}.campaign-table td.score-summary{{white-space:nowrap}}.campaign-table th{{position:sticky;top:0}}\n</style></head><body><main>{body}</main>{"<script>"+script+"</script>" if script else ""}</body></html>'


def split_fingerprint(root, selected, baseline):
    feature = str(selected['id'])
    paths = [Path(__file__), root/'agents'/f'{feature}.json', root/'scored-features'/f'{feature}.json', root/'comparisons'/f'{feature}.json', root/'scoring-audit'/f'{feature}.jsonl', root/'scoring-policy'/f'{feature}.json', root/'prompt-debug'/feature, root/'configuration.json']
    paths += list((root/'agent-attempts').glob(f'{feature}-*.json'))
    if baseline:
        paths.append(baseline/'features'/f'{feature}.json')
    stats = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            stats.append((str(path),stat.st_mtime_ns,stat.st_size))
    return hashlib.sha256(json.dumps([SPLIT_VERSION,selected,stats],sort_keys=True).encode()).hexdigest()


def compact_feature(feature):
    # Keep only values needed by the overview; never cache transcripts or scoring rows.
    keys = ('id','selected','scores','errors','original','explanation','status','cohort','intruder_policy','scoring_attempts')
    light = {k:feature[k] for k in keys}
    light['agent'] = {k:v for k,v in feature['agent'].items() if k in ('status','explanation','usage','max_prompt_tokens','generation','turns','word_count','training_windows_inspected')}
    light['agent']['events'] = [dict(type='model_response', request_settings=e.get('request_settings', {})) for e in feature['agent'].get('events', []) if e.get('type') == 'model_response']
    light['score'] = {k:v for k,v in feature['score'].items() if k in ('scorer_settings','intruder_retry_policy','baseline_replay_verified','provenance')}
    return light


def render_split(root, selection, baseline=None, output=None):
    """Bound memory to one transcript; cache lightweight rows and write changed pages only.

    Raw JSON remains the canonical full audit and is linked, not copied into every
    page. Nothing writes to source baseline/pilot artifacts, including reused symlinks.
    """
    destination = Path(output) if output else root/'index.html'
    if destination.parent.resolve() != root.resolve():
        raise ValueError('Split overview must be in the campaign root so artifact links remain valid')
    errors = []
    cache_path = root/'.report-cache.json'
    old_cache = read(cache_path, errors)
    previous = old_cache.get('features', {}) if old_cache.get('version') == SPLIT_VERSION else {}
    cached, features, changed = {}, [], 0
    for selected in selection.get('selected', []):
        feature_id = str(selected['id'])
        if not feature_id.isdecimal():
            raise ValueError(f'Invalid feature ID: {feature_id!r}')
        fingerprint = split_fingerprint(root, selected, baseline)
        prior = previous.get(feature_id, {})
        page = root/'features'/f'{feature_id}.html'
        has_artifacts = (root/'agents'/f'{feature_id}.json').exists() or (root/'scored-features'/f'{feature_id}.json').exists() or (root/'comparisons'/f'{feature_id}.json').exists()
        if prior.get('fingerprint') == fingerprint and (not has_artifacts or page.exists()):
            light = prior['summary']
        else:
            feature = load_features(root, {'selected':[selected]}, baseline)[0]
            light = compact_feature(feature)
            if has_artifacts:
                reuse = selected.get('pilot_reuse')
                provenance = '<p class="caveat">Full readable transcript and returned reasoning appear below. Complete raw messages, provider events and scoring rows remain in the linked JSON artifacts rather than being duplicated in this HTML. Exact model-input pages are linked per turn/attempt when exported. This page works offline with its adjacent artifact folders.</p>'
                if reuse:
                    provenance += '<div class="run-banner"><b>Reused pilot work · not a new campaign investigation</b>'+details('Reuse source and hashes', reuse)+'</div>'
                provenance += details('Feature selection / evidence eligibility', selected)
                # feature_html emits root-relative links; feature pages are one directory deeper.
                content = feature_html(feature, root, embed_raw=False).replace('href="','href="../')
                body = '<p><a href="../index.html">← All layer features</a></p><h1>Layer '+esc(selection.get('layer',22))+' · feature '+esc(feature_id)+'</h1>'+provenance+content
                write_report(page,split_shell(f'MOLT feature {feature_id}',body))
                changed += 1
        light['has_page'] = has_artifacts and page.exists()
        light['evidence_status'] = selected.get('evidence_status','not recorded')
        if selected.get('agent_eligible') is False and light['status'] == 'pending':
            light['status'] = 'ineligible'
        light['reused'] = bool(selected.get('pilot_reuse'))
        features.append(light)
        cached[feature_id] = {'fingerprint':fingerprint,'summary':light}
    write_report(cache_path,json.dumps({'version':SPLIT_VERSION,'features':cached},ensure_ascii=False))
    complete = sum(all(number(f['scores'][m]['new']) for m in METHODS) for f in features)
    frozen = sum(bool(f['agent'].get('explanation')) for f in features)
    unobserved = sum(f['evidence_status']=='not_observed' for f in features)
    reused = sum(f['reused'] for f in features)
    run = read(root/'run.json',errors)
    progress = read(root/'progress.json',errors)
    rows=[]
    for feature in features:
        fid = feature['id']
        title = f'<a href="features/{fid}.html">Feature {fid}</a>' if feature['has_page'] else f'Feature {fid}<small>No investigation artifact yet</small>'
        cells=[]
        for method in METHODS:
            value = feature['scores'][method]
            cells.append('<td class="score-summary">'+pct(value['new'])+'<small>historical '+pct(value['old'])+' · fresh '+pct(value['fresh'])+'</small><small>Δ fresh '+delta(value['fresh_delta'])+' · Δ hist '+delta(value['delta'])+'</small></td>')
        available = feature['selected'].get('available_methods', list(METHODS))
        scoring = ('complete for '+str(len(available))+' available methods' if available and all(number(feature['scores'][m]['new']) for m in available if m in METHODS) else 'not applicable' if not available else 'pending / incomplete')
        unavailable = feature['selected'].get('unavailable_methods', {})
        unavailable_note = '<small>Unavailable: '+esc(', '.join(unavailable))+'</small>' if unavailable else ''
        errors_html = ''.join('<small class="negative">'+esc(e)+'</small>' for e in feature['errors'])
        rows.append(f'<tr class="campaign-row" data-cohort="{esc(feature["cohort"])}" data-status="{esc(feature["status"])}" data-evidence="{esc(feature["evidence_status"])}" data-reuse="{"reused" if feature["reused"] else "new"}"><td>{title}<small>rank {esc(feature["selected"].get("rank","—"))}</small></td><td>{esc(feature["evidence_status"])}<small>{esc(feature["cohort"])}</small></td><td>{esc(feature["status"])}<small>scoring: {scoring}</small>{unavailable_note}<small>{"reused pilot" if feature["reused"] else "campaign work"}</small>{errors_html}</td>{"".join(cells)}<td class="description">{esc(feature["original"])}</td><td class="description">{esc(feature["explanation"])}</td></tr>')
    def dropdown(key, title, values):
        return '<label>'+title+' <select data-filter="'+key+'"><option value="">All</option>'+''.join('<option value="'+esc(v)+'">'+esc(v)+'</option>' for v in sorted(set(values)))+'</select></label>'
    filters = dropdown('evidence','Evidence',[f['evidence_status'] for f in features])+dropdown('status','Agent',[f['status'] for f in features])+dropdown('cohort','Cohort',[f['cohort'] for f in features])+dropdown('reuse','Origin',['reused','new'])
    selection_metadata = {k:v for k,v in selection.items() if k!='selected'}
    body = f'''<header><div class="eyebrow">MOLT · full layer campaign</div><h1>Layer {esc(selection.get('layer',22))} · all {len(features):,} features</h1><p>Every manifest feature is listed, including unobserved and limited-evidence features. Readable investigations live on separate feature pages; this overview never embeds full transcripts.</p><p><b>Run status:</b> {esc(run.get('status','not recorded'))}</p></header>
<div class="cards"><div class="card"><strong>{len(features):,}</strong><span>all layer features</span></div><div class="card"><strong>{frozen:,}</strong><span>frozen explanations · {reused} reused pilot rows</span></div><div class="card"><strong>{complete:,}</strong><span>all four methods scored</span></div><div class="card"><strong>{unobserved:,}</strong><span>not observed in evidence</span></div></div>
<section class="panel"><h2>Coverage and comparison</h2><p class="caveat">An absent score is not zero. Unobserved or limited evidence can prevent investigation or particular evaluations; these states are not automatically errors. Counts show completed pairs, not the full population. Reused pilot work is explicitly marked and preserves its original source/hashes. Fresh-baseline deltas are primary; historical deltas can include scorer and token-display changes. Binary methods use balanced accuracy; intruder uses accuracy and is explanation-independent, so its changes do not measure explanation quality.</p>{summary(features)}</section>
{settings_panel(root,features,errors)}
<section class="panel"><h2>Campaign provenance and progress</h2><p><a href="selection.json">Complete selection / eligibility JSON</a> · <a href="run.json">Run JSON</a></p>{details('Manifest metadata',selection_metadata)}{details('Run state',run)}{details('Progress snapshot',progress)}</section>
<div class="toolbar">{filters}<input id="query" type="search" placeholder="Feature, explanation or status" aria-label="Search all features"><button id="prev">Previous 100</button><button id="next">Next 100</button><span id="visible"></span></div>
<div class="table-wrap"><table class="campaign-table"><thead><tr><th>Feature</th><th>Evidence</th><th>Agent / scoring</th>{''.join('<th>'+esc(m)+'</th>' for m in METHODS)}<th>Original explanation</th><th>Agentic explanation</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{''.join('<p class="error">'+esc(error)+'</p>' for error in errors)}
<footer>Rendered {datetime.now(timezone.utc).isoformat(timespec='seconds')} · {changed} feature pages updated. Each page is regenerated only when its source artifacts change; source pilots and baselines are never modified.</footer>'''
    write_report(destination,split_shell('MOLT full-layer comparison',body,SPLIT_SCRIPT))
    return destination


def render(root, baseline=None, output=None, split=False):
    root = Path(root)
    errors = []
    selection = read(root/'selection.json', errors)
    baseline_hint = selection.get('baseline', selection.get('baseline_dir'))
    if baseline is None and isinstance(baseline_hint, str):
        baseline = Path(baseline_hint)
    if split or len(selection.get('selected', [])) > 64 or root.name == 'molt-agentic-layer22-full':
        return render_split(root, selection, Path(baseline) if baseline else None, output)
    features = load_features(root, selection, Path(baseline) if baseline else None)
    complete = sum(all(number(f['scores'][m]['new']) for m in METHODS) for f in features)
    frozen = sum(bool(f['agent'].get('explanation')) for f in features)
    failed = sum(bool(f['errors']) or f['status'] in ('failed', 'error') for f in features)
    progress = read(root/'progress.json', errors)
    execution = execution_panel(root, features, errors)
    tooling = tooling_panel(root, errors)
    settings = settings_panel(root, features, errors)
    body = f'''<header><div class="eyebrow">MOLT · agentic Delphi pilot</div><h1>Do interactive explanations improve feature understanding?</h1><p class="muted">Corrected existing baseline vs. frozen agentic explanations · all recorded agent transcripts</p></header>
{settings}
<div class="cards"><div class="card"><strong>{len(features)}</strong><span>selected features</span></div><div class="card"><strong>{frozen}</strong><span>frozen explanations</span></div><div class="card"><strong>{complete}/{len(features)}</strong><span>fully scored</span></div><div class="card"><strong>{failed}</strong><span>features with errors</span></div></div>
<section class="panel"><h2>Comparison at a glance</h2><p class="caveat">Means give each feature equal weight and include only features with both original and agentic scores. Paired counts expose incomplete coverage. Fresh-original means and deltas use only their own matched subset. The primary explanation comparison is agentic minus fresh original, using identical corrected text rendering. Detection, fuzz and active fuzz use balanced accuracy (chance 50%); intruder uses accuracy (chance 20%). Deltas are percentage points.</p>{summary(features)}</section>
{execution}
{tooling}
<section class="panel"><h2>How to read this pilot</h2><p class="caveat">Cohorts are selected by existing baseline performance, so improvement estimates are descriptive and subject to selection effects. This is a small pilot, not an independent statistical claim. Historical original scores refer to the existing campaign, including its recorded per-feature revisions. Fresh original and agentic scoring share corrected special-token rendering, so use Δ fresh as the primary comparison. Δ historical can conflate explanation changes, prompt-rendering repairs and scorer variation. The scoring artifacts record exact historical prompt replay checks, changed-display counts and source hashes; fresh controls also help assess regression to the mean.</p><p class="caveat"><b>Intruder is explanation-independent.</b> It measures feature coherence, not explanation quality; any historical-to-current change reflects scorer variation, not an agentic explanation gain. Agent exploration must remain restricted to training evidence; scoring examples and labels are held out. Check per-feature provenance and run metadata for the actual executed configuration.</p><p class="caveat">Transcripts preserve all recorded fields, including tool calls, full tool results, and reasoning when returned by the provider. Unreturned internal reasoning cannot be reconstructed. This HTML embeds all displayed data and works offline; raw JSON links require the adjacent result folders. Progress is a snapshot, regenerated by the pipeline.</p>{details('Selection and provenance', selection)}{details('Pipeline progress', progress)}</section>
<div class="toolbar"><label>Cohort <select id="cohort"><option value="all">All cohorts</option>{''.join(f'<option value="{c}">{c.replace("_", " ")}</option>' for c in COHORTS)}</select></label><input id="query" type="search" placeholder="Feature ID or text" aria-label="Search features"><button id="expand">Expand full transcripts</button><button id="collapse">Collapse details</button><span id="visible" class="muted"></span></div>
{''.join(f'<p class="error">{esc(e)}</p>' for e in errors)}
{''.join(feature_html(f, root) for f in features)}
{'<p class="error">Selection is missing or empty; no features can be displayed yet.</p>' if not features else ''}
<footer>Rendered {datetime.now(timezone.utc).isoformat(timespec='seconds')} · {esc(root)} · <a href="selection.json">Selection JSON</a></footer>'''
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MOLT agentic pilot comparison</title><style>{STYLE}</style></head><body><main>{body}</main><script>{SCRIPT}</script></body></html>'
    if (root/'trajectories/index.html').exists():
        document = document.replace('<main>', '<main><p class="panel"><a href="trajectories/">Open the interactive trajectory explorer: tool usage, token counts, context limits and readable timelines →</a></p>', 1)
    destination = Path(output) if output else root/'index.html'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+f'.{os.getpid()}.tmp')
    temporary.write_text(document)
    temporary.replace(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', '--input', type=Path, default=Path('results/molt-agentic-pilot-v1'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--output', type=Path, help='HTML path; defaults to ROOT/index.html')
    parser.add_argument('--split', action='store_true', help='Lightweight overview plus incremental per-feature transcript pages; automatic above 64 features')
    args = parser.parse_args()
    print(render(args.root, args.baseline, args.output, args.split))


if __name__ == '__main__':
    main()
