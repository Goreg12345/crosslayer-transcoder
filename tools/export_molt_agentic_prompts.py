#!/usr/bin/env python3
"""Reconstruct recorded agent request prompts using the live, unchanged render template.

No generation is performed. These artifacts are reconstructions, not runtime
captures. A turn is accepted only if prompt usage and exact token round-trip match.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crosslayer_transcoder.dashboard.token_display import escape_added_tokens


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def safe_messages(messages):
    # Same escaping as the frozen driver, without importing its scoring runtime.
    result = []
    for message in messages:
        row = dict(message)
        if isinstance(row.get('content'), str):
            row['content'] = escape_added_tokens(row['content'])
        if isinstance(row.get('reasoning_content'), str):
            row['reasoning_content'] = escape_added_tokens(row['reasoning_content'])
        if row.get('tool_calls'):
            row['tool_calls'] = [dict(t, function=dict(t['function'],
                arguments=escape_added_tokens(t['function']['arguments']))) for t in row['tool_calls']]
        result.append(row)
    return result


def request_for_event(agent, event, ordinal, config, tools):
    messages = agent['messages']
    assert messages[0] == {'role': 'system', 'content': config['system']}, 'Frozen system prompt mismatch'
    assistants = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
    if ordinal < len(assistants):
        end = assistants[ordinal]
        recorded = messages[end]
        response = event['response']['choices'][0]['message']
        if event['response']['choices'][0].get('finish_reason') != 'length':
            assert recorded.get('content') == response.get('content'), 'Response/message association mismatch'
            assert recorded.get('tool_calls') == response.get('tool_calls'), 'Tool-call association mismatch'
    else:
        assert ordinal == len(assistants), 'Missing assistant history'
        # A length-truncated response is logged before the driver appends it.
        assert event['response']['choices'][0]['finish_reason'] == 'length', 'Missing corresponding assistant message'
        end = len(messages)
    settings = event['request_settings']
    assert settings['model'] == config['model'] == agent['model'], 'Frozen model mismatch'
    assert settings['max_tokens'] == config['max_tokens'], 'Frozen max_tokens mismatch'
    assert settings['temperature'] == config['temperature'], 'Frozen temperature mismatch'
    assert settings['chat_template_kwargs']['enable_thinking'] == config['enable_thinking'], 'Frozen thinking mismatch'
    return dict(settings, messages=safe_messages(messages[:end]), tools=tools)


def event_request(run, agent, event, ordinal, config, tools):
    if event.get('request_body_path'):
        path = (Path(run) / event['request_body_path']).resolve()
        assert path.is_relative_to(Path(run).resolve() / 'request-bodies'), 'Request body outside capture directory'
        raw = path.read_bytes()
        assert digest(raw) == event['request_body_sha256'], 'Captured request SHA mismatch'
        body = json.loads(raw)
        assert {k:v for k,v in body.items() if k not in ('messages','tools')} == event['request_settings'], 'Captured request settings mismatch'
        assert body['tools'] == tools, 'Captured tool schemas differ from frozen schemas'
        assert body['model'] == config['model'], 'Captured model differs from frozen model'
        source = event.get('request_body_provenance', 'request body captured before HTTP dispatch')
        return body, source + '; token IDs re-rendered using current server template, not captured runtime'
    return request_for_event(agent,event,ordinal,config,tools), 'request body reconstructed from transcript; token IDs re-rendered using current server template, not captured runtime'


PAGE = '''<!doctype html><meta charset="utf-8"><title>Feature FEATURE — model input turn TURN</title>
<style>body{font:16px system-ui;margin:2rem;max-width:1200px;background:#f6f7f9;color:#18212a}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:1rem;border:1px solid #ddd}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ddd;padding:.4rem;text-align:left}td:last-child{white-space:pre-wrap;font-family:monospace}button{margin:.5rem} .note{background:#fff2cc;padding:1rem}</style>
<h1>Feature FEATURE · model input turn TURN</h1>
<p class="note">Request provenance is shown below: newer HTTP request bodies were captured before dispatch; older bodies are reconstructed from transcripts. Token IDs are re-rendered with the current server template, not captured runtime. Validation requires original prompt-token count and exact token round-trip.</p>
<p><a href="turn-TURN.json" download>Download full JSON (request, token IDs, rendered prompt)</a> · <a href="../../index.html">Pilot report</a></p>
<pre id="validation">Loading…</pre>
<h2>Where the tool definitions enter the model input</h2><p>The server chat template inserts the tool-definition block before the investigator’s own system instructions. This section shows that rendered prefix, not an extra tool message.</p><pre id="injection"></pre>
<h2>Full rendered model input</h2><details><summary>Show complete decoded prompt</summary><pre id="prompt"></pre></details>
<p>These are explainer-model input token IDs (Qwen3.8-27B), not the Qwen3-4B subject-model token IDs inside MOLT tool results. Literal control tokens in this rendered prompt were inserted by the chat template; evidence content was escaped by the agent driver before rendering.</p>
<h2>Exact request body</h2><details><summary>Show OpenAI request and message roles</summary><pre id="request"></pre></details>
<h2>Token IDs and decoded pieces</h2><p>Individual token pieces can display replacement characters for byte fragments; the complete decoded prompt above is authoritative.</p>
<button id="prev">Previous 200</button><button id="next">Next 200</button><span id="range"></span>
<table><thead><tr><th>Position</th><th>ID</th><th>Piece</th></tr></thead><tbody id="tokens"></tbody></table>
<script id="artifact" type="application/json">ARTIFACT_JSON</script>
<script>
let data,offset=0;const size=200;const byid=x=>document.getElementById(x);
function render(){const end=Math.min(offset+size,data.token_ids.length);byid('range').textContent=`${offset}–${end-1} / ${data.token_ids.length}`;byid('tokens').replaceChildren();for(let i=offset;i<end;i++){let tr=document.createElement('tr');for(let v of [i,data.token_ids[i],data.token_pieces[i]]){let td=document.createElement('td');td.textContent=v;tr.append(td)}byid('tokens').append(tr)}byid('prev').disabled=offset===0;byid('next').disabled=end===data.token_ids.length}
byid('prev').onclick=()=>{offset=Math.max(0,offset-size);render()};byid('next').onclick=()=>{offset=Math.min(data.token_ids.length-1,offset+size);render()};
Promise.resolve(JSON.parse(byid('artifact').textContent)).then(d=>{data=d;byid('validation').textContent=JSON.stringify({provenance:d.provenance,validation:d.validation,hashes:d.hashes},null,2);byid('injection').textContent=d.tool_injection.rendered_prefix;byid('prompt').textContent=d.rendered_prompt;byid('request').textContent=JSON.stringify(d.request_body,null,2);render()}).catch(e=>byid('validation').textContent=String(e));
</script>'''


def write_page(dest, feature, file_turn, artifact):
    page = PAGE.replace('FEATURE', str(feature)).replace('TURN', str(file_turn))
    # HTML parser cannot interpret data as script/end tags, including tool text.
    embedded = json.dumps(artifact, ensure_ascii=False).replace('<', '\\u003c')
    page = page.replace('ARTIFACT_JSON', embedded)
    tmp = dest.with_suffix('.html.tmp')
    tmp.write_text(page)
    tmp.replace(dest.with_suffix('.html'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--render-url', default='http://127.0.0.1:8002/v1/chat/completions/render')
    parser.add_argument('--tokenizer', type=Path, default=Path('/home/georg/Code/vllm-utils/models/Qwen3.8-27B-NVFP4/tokenizer.json'))
    args = parser.parse_args()
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    config_raw = (args.run / 'configuration.json').read_bytes()
    schema_raw = (args.run / 'tool-schemas.json').read_bytes()
    config, tools = json.loads(config_raw), json.loads(schema_raw)
    out = args.run / 'prompt-debug'
    index = {'created_at': datetime.now(timezone.utc).isoformat(), 'provenance': 'reconstructed with current server template; not captured runtime',
             'render_url': args.render_url, 'configuration_sha256': digest(config_raw), 'tool_schemas_sha256': digest(schema_raw),
             'tokenizer_sha256': digest(args.tokenizer.read_bytes()), 'turns': []}
    snapshots = [(p, p.read_bytes()) for p in sorted((args.run/'agents').glob('*.json'))]
    for path, raw in snapshots:
        agent = json.loads(raw)
        events = [e for e in agent['events'] if e['type'] == 'model_response']
        successful_ordinal = 0
        for event_index, event in enumerate(events):
            ordinal = successful_ordinal
            truncated = event['response']['choices'][0]['finish_reason'] == 'length'
            if not truncated: successful_ordinal += 1
            feature, turn = agent['id'], event['turn']
            file_turn = f'{turn}-truncated-{event_index}' if truncated else str(turn)
            entry = {'feature': feature, 'turn': turn, 'event_index': event_index, 'truncated': truncated, 'json': f'{feature}/turn-{file_turn}.json', 'html': f'{feature}/turn-{file_turn}.html'}
            dest = out / entry['json']
            try:
                body, provenance = event_request(args.run, agent, event, ordinal, config, tools)
                body_hash = digest(canonical(body))
                if dest.exists():
                    previous = json.loads(dest.read_text())
                    if (previous.get('hashes', {}).get('request_body_sha256') == body_hash
                            and previous.get('hashes', {}).get('tokenizer_sha256') == index['tokenizer_sha256']
                            and previous.get('validation', {}).get('passed')
                            and len(previous['token_ids']) == event['response']['usage']['prompt_tokens']):
                        previous['provenance']=provenance
                        previous['source_request_body_path']=event.get('request_body_path')
                        atomic_json(dest,previous)
                        write_page(dest, feature, file_turn, previous)
                        entry.update(status='validated', prompt_tokens=len(previous['token_ids']), cached=True)
                        index['turns'].append(entry)
                        continue
                request = Request(args.render_url, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
                with urlopen(request, timeout=120) as response:
                    rendered = json.load(response)
                ids = rendered['token_ids']
                text = tokenizer.decode(ids, skip_special_tokens=False)
                expected = event['response']['usage']['prompt_tokens']
                checks = {'prompt_tokens_recorded':expected, 'prompt_tokens_rendered':len(ids),
                          'prompt_count_matches':len(ids)==expected,
                          'decode_reencode_matches':tokenizer.encode(text,add_special_tokens=False).ids==ids}
                checks['passed'] = checks['prompt_count_matches'] and checks['decode_reencode_matches']
                assert checks['passed'], f'Prompt reconstruction mismatch: {checks}'
                system = body['messages'][0]['content']
                system_start = text.find(system.strip())
                assert system_start >= 0, 'System prompt not found in rendered input'
                injection = text[:system_start]
                assert '# Tools' in injection, 'Tool definition prefix missing before system instructions'
                artifact = {'feature':feature,'turn':turn,'provenance':provenance,
                            'reconstructed_at':datetime.now(timezone.utc).isoformat(),
                            'source_request_body_path':event.get('request_body_path'),
                            'request_body':body,'token_ids':ids,'token_pieces':[tokenizer.decode([t],skip_special_tokens=False) for t in ids],
                            'rendered_prompt':text,'sampling_params':rendered.get('sampling_params'),
                            'tool_injection':{'system_prompt_character_start':system_start,'rendered_prefix':injection},
                            'validation':checks,'hashes':{'request_body_sha256':body_hash,'token_ids_sha256':digest(canonical(ids)),
                            'rendered_prompt_sha256':digest(text),'agent_snapshot_sha256':digest(raw),
                            'configuration_sha256':index['configuration_sha256'],'tool_schemas_sha256':index['tool_schemas_sha256'],
                            'tokenizer_sha256':index['tokenizer_sha256']}}
                atomic_json(dest,artifact)
                dest.with_name(f'turn-{file_turn}.error.json').unlink(missing_ok=True)
                write_page(dest, feature, file_turn, artifact)
                entry.update(status='validated',prompt_tokens=len(ids))
            except Exception as exc:
                entry.update(status='error',error=f'{type(exc).__name__}: {exc}')
                dest.unlink(missing_ok=True)
                dest.with_suffix('.html').unlink(missing_ok=True)
                atomic_json(dest.with_name(f'turn-{file_turn}.error.json'),entry)
            index['turns'].append(entry)
            print(json.dumps(entry),flush=True)
    index['validated_turns']=sum(x['status']=='validated' for x in index['turns'])
    index['errors']=sum(x['status']=='error' for x in index['turns'])
    atomic_json(out/'index.json',index)
    print(json.dumps({'validated_turns':index['validated_turns'],'errors':index['errors']}))
    if index['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
