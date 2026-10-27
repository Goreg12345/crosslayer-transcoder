#!/usr/bin/env python3
"""Score frozen agent explanations against exact historical campaign audit prompts.

No calls are made without --execute. Historical messages (including historical
special-token rendering and few-shots) are authoritative, not current templates.
All added tokens are escaped in BOTH arms to preserve the campaign bug fix.
Intruder never consumed an explanation: it is retained as a coherence control.
"""
import argparse
import asyncio
from collections import defaultdict
from copy import deepcopy
import hashlib
import http.client
import json
from pathlib import Path
import re
import time
import urllib.request

from crosslayer_transcoder.dashboard.pilot import binary_metrics, response_key, wilson_interval
from crosslayer_transcoder.dashboard.token_display import escape_added_tokens, leftover_added_tokens

METHODS = ('detection', 'fuzz', 'active_fuzz', 'intruder')
BINARY_SUFFIX = '\n\nClassify only the single Example 0 above. Return exactly [1] or [0], with no additional text.'
INTRUDER_SUFFIX = '\nReturn only [RESPONSE]: followed by the integer intruder index.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def prediction(task, text):
    text = (text or '').strip()
    if task == 'binary' and text in ('[0]', '[1]'):
        return text == '[1]'
    match = re.fullmatch(r'\[RESPONSE\]:\s*(\d+)', text)
    if task == 'intruder' and match and int(match[1]) < 5:
        return int(match[1])
    raise ValueError(f'Invalid {task} response: {text!r}')


def metrics(method, rows):
    if method != 'intruder':
        return binary_metrics(rows)
    valid = [r for r in rows if r['sample'] is not None and r['prediction'] is not None]
    correct = sum(r['correct'] for r in valid)
    return dict(total=len(rows), valid=len(valid), parse_failures=len(rows)-len(valid),
                correct=correct, accuracy=correct/len(valid) if valid else None,
                accuracy_ci95=wilson_interval(correct, len(valid)), chance=0.2)


def prepare_feature(baseline, feature, explanation):
    """Return immutable-in-practice plan; raises on ambiguous or missing evidence.

    Rows are matched by entire final user message, including every shown token,
    highlight and intruder permutation. Duplicate audit attempts may be present,
    but their full prompt bodies must agree. Each selected response must reproduce
    the row's historical prediction, and aggregate metrics must agree exactly.
    """
    baseline = Path(baseline)
    source = baseline / 'features' / f'{feature}.json'
    audit = baseline / 'audit' / f'{feature}.jsonl'
    evidence = baseline / 'evidence' / f'{feature}.json'
    original = json.loads(source.read_text())
    if not explanation.strip():
        raise ValueError('Empty final explanation')
    index = defaultdict(list)
    for line_number, line in enumerate(audit.read_text().splitlines(), 1):
        row = json.loads(line)
        if row['task'] in ('binary', 'intruder'):
            index[(row['task'], row['messages'][-1]['content'])].append((line_number, row))
    old = original['explanation']
    cases = []
    for method in METHODS:
        result = original['methods'][method]
        if result['status'] != 'complete':
            raise ValueError(f'{feature}/{method}: baseline incomplete')
        replay_rows = []
        for row_number, label in enumerate(result['rows']):
            task = 'intruder' if method == 'intruder' else 'binary'
            if task == 'binary':
                shown = label.get('shown_text', ''.join(label['str_tokens']))
                suffix = '\n\nText examples:\n\nExample 0: ' + shown + BINARY_SUFFIX
                user = 'Latent explanation: ' + old + suffix
            else:
                user = 'Examples:\n\n' + '\n'.join(
                    f'Example {i}:' + example for i, example in enumerate(label['sample']['examples'])
                ) + '\n' + INTRUDER_SUFFIX
            matches = index.get((task, user), []) or index.get((task, escape_added_tokens(user)), [])
            if not matches:
                raise ValueError(f'{feature}/{method}/{row_number}: no exact audited prompt')
            prompt_keys = {response_key(task, r['messages'], r['max_tokens']) for _, r in matches}
            if len(prompt_keys) != 1:
                raise ValueError(f'{feature}/{method}/{row_number}: ambiguous audited full prompts')
            accepted = []
            for line_number, row in matches:
                try:
                    pred = prediction(task, row.get('response'))
                except ValueError:
                    continue
                if pred == label['prediction'] and row.get('finish_reason') != 'length':
                    accepted.append((line_number, row))
            if not accepted:
                raise ValueError(f'{feature}/{method}/{row_number}: historical prediction not reproduced')
            line_number, row = accepted[-1]
            expected_limit = 64 if task == 'intruder' else 32
            if row['max_tokens'] != expected_limit:
                raise ValueError('Historical scoring output limit changed')
            historical_messages = deepcopy(row['messages'])
            baseline_messages = [dict(m, content=escape_added_tokens(m['content'])) for m in historical_messages]
            messages = deepcopy(baseline_messages)
            if task == 'binary':
                messages[-1]['content'] = escape_added_tokens('Latent explanation: ' + explanation + suffix)
                assert messages[:-1] == baseline_messages[:-1]
                assert messages[-1]['content'][len(escape_added_tokens('Latent explanation: ' + explanation)):] == escape_added_tokens(suffix)
            elif any(old in m['content'] for m in messages):
                raise ValueError('Unexpected explanation-dependent intruder prompt')
            replay = deepcopy(label)
            replay['prediction'] = prediction(task, row['response'])
            replay['correct'] = replay['prediction'] == (label['correct_index'] if task == 'intruder' else label['activating'])
            replay_rows.append(replay)
            cases.append(dict(method=method, row_index=row_number, task=task, audit_line=line_number,
                              baseline_request_key=next(iter(prompt_keys)),
                              candidate_request_key=response_key(task, messages, expected_limit),
                              historical_messages=historical_messages,
                              baseline_messages=baseline_messages, messages=messages,
                              display_corrected=baseline_messages != historical_messages,
                              max_tokens=expected_limit, label=label))
        if metrics(method, replay_rows) != result['metrics']:
            raise ValueError(f'{feature}/{method}: exact baseline metric replay failed')
    return dict(feature=feature, explanation=explanation, baseline_explanation=old,
                baseline=str(baseline.resolve()), seed=42+feature,
                baseline_replay_verified=True, intruder_explanation_independent=True,
                source_sha256={str(p.resolve()): sha(p) for p in (source,audit,evidence)},
                baseline_revision=original.get('revision'),
                display_corrected_cases=sum(c['display_corrected'] for c in cases),
                historical_methods=original['methods'], cases=cases)


async def execute_plan(plan, output, *, base_url='http://127.0.0.1:8000/v1',
                       model='qwen3.8-27b-nvfp4', concurrency=32,
                       refresh_baseline=False, semaphore=None, intruder_thinking=False):
    """Score one frozen explanation; optional semaphore caps a whole batch.

    Durable audit responses resume only when the complete request body matches.
    Baseline and candidate identical prompts share one response (notably intruder).
    No scorer outputs are supplied to the explainer by this module.
    """
    if plan['display_corrected_cases'] and not refresh_baseline:
        raise ValueError('Historical prompts need token-display correction; refresh_baseline=True is mandatory')
    output = Path(output)
    feature = plan['feature']
    output.mkdir(parents=True, exist_ok=True)
    from tools.molt_scorer_runtime import load_scorer_runtime, count_prompt_tokens, bounded_budgets
    runtime=load_scorer_runtime(output)
    if runtime is not None:
        base_url,model=runtime['base_url'],runtime['model']
    binary_settings = dict(temperature=0, max_tokens=32, chat_template_kwargs={'enable_thinking': False})
    intruder_settings = (dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, max_tokens=8192,
                              chat_template_kwargs={'enable_thinking': True})
                         if intruder_thinking else
                         dict(temperature=0, max_tokens=64, chat_template_kwargs={'enable_thinking': False}))
    if runtime is not None and intruder_thinking:
        intruder_settings['max_tokens']=runtime['intruder_max_tokens']
        intruder_settings['chat_template_kwargs']['reasoning_effort']=runtime['intruder_reasoning_effort']
    scorer_settings = dict(model=model, binary=binary_settings, intruder=intruder_settings,
                           intruder_changed_from_historical=bool(intruder_thinking))
    if runtime is not None:scorer_settings.update(base_url=base_url,context_limit=runtime['context_limit'],scorer_runtime=runtime)
    config_path = output/'scoring-config'/f'{feature}.json'
    if config_path.exists() and json.loads(config_path.read_text()) != scorer_settings:
        raise ValueError('Frozen scorer settings changed; use a new output directory')
    write_json(config_path, scorer_settings)
    retry_policy = (dict(version=1, trigger='finish_reason == length only',
                         output_token_budgets=[8192,16384,32768,65536],
                         max_attempts_per_request_per_invocation=10,
                         timeout_seconds=3600, correctness_based_retry=False,
                         valid_response_policy='reuse first valid response at any approved budget',
                         explanation_independent_intruder_shared_between_arms=True)
                    if intruder_thinking else None)
    if runtime is not None and retry_policy is not None:
        retry_policy.update(version=2,output_token_budgets='min(initial cap, context minus rendered prompt), then full remaining context',
                            initial_output_cap=runtime['intruder_max_tokens'],context_limit=runtime['context_limit'],
                            prompt_counter='chat/completions/render max_tokens=1; no inference',scorer_runtime_sha256=runtime['runtime_sha256'])
    if retry_policy is not None:
        policy_path = output/'scoring-policy'/f'{feature}.json'
        if policy_path.exists() and json.loads(policy_path.read_text()) != retry_policy:
            raise ValueError('Intruder retry policy changed; use a new output directory')
        write_json(policy_path, retry_policy)
    plan_path = output/'scoring-plans'/f'{feature}.json'
    audit_path = output/'scoring-audit'/f'{feature}.jsonl'
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen scoring plan changed; use a new output directory')
    for source, digest in plan['source_sha256'].items():
        if sha(source) != digest:
            raise ValueError(f'Baseline changed: {source}')
    write_json(plan_path, plan)
    semaphore = semaphore or asyncio.Semaphore(concurrency)
    cache, length_keys = {}, set()
    if audit_path.exists():
        for line in audit_path.read_text().splitlines():
            row = json.loads(line)
            if row.get('valid'):
                cache.setdefault(row['body_sha256'], row)
            response = row.get('response')
            choices = response.get('choices') if isinstance(response,dict) else None
            if isinstance(choices,list) and choices and isinstance(choices[0],dict) and choices[0].get('finish_reason') == 'length':
                length_keys.add(row['body_sha256'])
    pending = {}
    render_slots=asyncio.Semaphore(16)
    prompt_counts={}

    async def request(case, arm):
        messages = case['baseline_messages'] if arm == 'baseline' else case['messages']
        if any(leftover_added_tokens(m['content']) for m in messages):
            raise ValueError('Unescaped added token in scoring request')
        body = dict(model=model, messages=messages,
                    **(intruder_settings if case['task'] == 'intruder' else binary_settings))
        thinking = case['task'] == 'intruder' and intruder_thinking
        prompt_tokens=None
        if runtime is not None:
            render_key=hashlib.sha256(json.dumps(dict(body,max_tokens=1),sort_keys=True).encode()).hexdigest()
            if render_key not in prompt_counts:
                async with render_slots:
                    if render_key not in prompt_counts:
                        prompt_counts[render_key]=await asyncio.to_thread(count_prompt_tokens,runtime,body)
            prompt_tokens=prompt_counts[render_key]
            budgets=bounded_budgets(runtime,prompt_tokens,thinking,body['max_tokens'])
        else:
            budgets = retry_policy['output_token_budgets'] if thinking else [body['max_tokens']]
        bodies = [dict(body,max_tokens=budget) for budget in budgets]
        keys = [hashlib.sha256(json.dumps([base_url,b],sort_keys=True).encode()).hexdigest() for b in bodies]
        key = keys[0]  # Stable identity for cross-arm and duplicate-case deduplication.
        allowed = set(keys)
        # Dict insertion order preserves the first audited valid result, irrespective
        # of its budget. Never regenerate or select among answers by correctness.
        for cached_key,cached_row in cache.items():
            if cached_key in allowed:
                return cached_row['prediction']
        if key in pending:
            return await pending[key]

        async def run():
            last_error = None
            budget_index = 0
            while budget_index < len(keys)-1 and keys[budget_index] in length_keys:
                budget_index += 1
            escalation_reason = 'prior_audited_length' if budget_index else None
            for attempt in range(1, 11):
                start = time.monotonic()
                actual_body,actual_key = bodies[budget_index],keys[budget_index]
                result, raw_response, pred, error = None, None, None, None
                valid, truncated = False, False
                try:
                    def post():
                        req = urllib.request.Request(base_url.rstrip('/')+'/chat/completions',
                            data=json.dumps(actual_body).encode(),headers={'Content-Type':'application/json'})
                        with urllib.request.urlopen(req, timeout=3600 if thinking else 900) as response:
                            return response.read().decode('utf-8')
                    async with semaphore:
                        raw_response = await asyncio.to_thread(post)
                    result = json.loads(raw_response)
                    choice = result['choices'][0]
                    truncated = choice.get('finish_reason') == 'length'
                    if truncated:
                        raise ValueError('Truncated response')
                    text = choice['message'].get('content')
                    pred = prediction(case['task'],text)
                    valid = True
                except (OSError, ValueError, KeyError, IndexError, TypeError,
                        AttributeError, http.client.HTTPException) as exc:
                    error = f'{type(exc).__name__}: {exc}'
                    last_error = error
                row = dict(feature=feature, arm=arm, method=case['method'],
                           row_index=case['row_index'], body_sha256=actual_key, body=actual_body,
                           base_body_sha256=key, max_tokens=actual_body['max_tokens'],
                           budget_escalation_reason=escalation_reason,
                           endpoint=base_url, response=result, valid=valid, prediction=pred,
                           error=error, attempt=attempt, seconds=time.monotonic()-start)
                if runtime is not None:
                    row.update(context_limit=runtime['context_limit'],prompt_tokens_counted=prompt_tokens,
                               effective_output_budget=actual_body['max_tokens'],scorer_runtime_sha256=runtime['runtime_sha256'])
                if result is None and raw_response is not None:
                    row['raw_response'] = raw_response
                with audit_path.open('a') as handle:
                    handle.write(json.dumps(row,ensure_ascii=False)+'\n')
                if valid:
                    cache[actual_key] = row
                    return pred
                escalation_reason = None
                if truncated:
                    length_keys.add(actual_key)
                    if budget_index < len(keys)-1:
                        budget_index += 1
                        escalation_reason = 'previous_attempt_length'
                if attempt < 10:
                    await asyncio.sleep(min(2 ** (attempt-1), 2))
            raise ValueError(f'{feature}/{case["method"]}/{case["row_index"]}: '
                             f'10 scoring failures; last error: {last_error}')
        pending[key] = asyncio.create_task(run())
        return await pending[key]

    arms = ['baseline','candidate'] if refresh_baseline else ['candidate']
    results = {}
    for arm in arms:
        predictions = await asyncio.gather(*(request(case,arm) for case in plan['cases']), return_exceptions=True)
        failures = [p for p in predictions if isinstance(p, BaseException)]
        if failures:
            raise ValueError(f'{feature}/{arm}: {len(failures)} scoring cases failed after all requests settled; first: {failures[0]}')
        methods = {}
        for method in METHODS:
            rows = []
            for case,pred in zip(plan['cases'],predictions):
                if case['method'] == method:
                    row = deepcopy(case['label'])
                    row['prediction'] = pred
                    row['correct'] = pred == (row['correct_index'] if method == 'intruder' else row['activating'])
                    if method != 'intruder':
                        row['probability'] = float(pred)
                    rows.append(row)
            methods[method] = dict(status='complete',metrics=metrics(method,rows),rows=rows)
        results[arm] = methods
    comparisons = {}
    for reference,methods in [('historical',plan['historical_methods'])] + ([('refreshed',results['baseline'])] if refresh_baseline else []):
        comparisons[reference] = {}
        for method in METHODS:
            metric = 'accuracy' if method == 'intruder' else 'balanced_accuracy'
            old = methods[method]['metrics'][metric]
            new = results['candidate'][method]['metrics'][metric]
            comparisons[reference][method] = dict(metric=metric, baseline=old, candidate=new, delta=new-old)
    result = dict(feature=feature,explanation=plan['explanation'],methods=results['candidate'],
                  refreshed_baseline=results.get('baseline'), comparisons=comparisons,
                  baseline_replay_verified=True,intruder_explanation_independent=True,
                  provenance=plan['source_sha256'],model=model,base_url=base_url,
                  display_corrected_cases=plan['display_corrected_cases'],baseline_revision=plan['baseline_revision'],
                  scorer_settings=scorer_settings)
    if retry_policy is not None:
        result['intruder_retry_policy'] = retry_policy
    write_json(output/'scored-features'/f'{feature}.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=Path('results/molt-qwen-dashboard-campaign-layer22-10m/autointerp-layer'))
    parser.add_argument('--feature',type=int,required=True)
    parser.add_argument('--explanation-file',type=Path,required=True,help='UTF-8 final explanation, or JSON containing explanation')
    parser.add_argument('--output',type=Path,default=Path('results/molt-agentic-pilot-v1'))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--refresh-baseline',action='store_true')
    parser.add_argument('--intruder-thinking',action='store_true',help='Use reasoning intruder control (8192 tokens, recommended Qwen sampling); binary settings stay original')
    parser.add_argument('--concurrency',type=int,default=32)
    args=parser.parse_args()
    raw=args.explanation_file.read_text()
    explanation=json.loads(raw)['explanation'] if args.explanation_file.suffix=='.json' else raw.strip()
    plan=prepare_feature(args.baseline,args.feature,explanation)
    if args.execute:
        result=asyncio.run(execute_plan(plan,args.output,concurrency=args.concurrency,refresh_baseline=args.refresh_baseline,intruder_thinking=args.intruder_thinking))
        print(json.dumps(result['comparisons'],indent=2))
    else:
        write_json(args.output/'scoring-plans'/f'{args.feature}.json',plan)
        print(json.dumps(dict(feature=args.feature,cases=len(plan['cases']),baseline_replay_verified=True,executed=False)))


if __name__=='__main__':
    main()
