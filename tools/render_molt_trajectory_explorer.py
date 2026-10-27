#!/usr/bin/env python3
"""Build a local, lazy-loaded trajectory explorer without inference or source edits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from render_molt_agentic import esc, markdown, pretty, tool_result, METHODS, metric
from molt_training_scorer_input import TrainingScorerInputs

VERSION = 3
ASSETS = Path(__file__).with_name('molt_trajectory_assets')


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded=json.dumps(data,ensure_ascii=False,separators=(',',':'))
    if path.exists() and path.read_text()==encoded:return
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(encoded)
    temporary.replace(path)


def safe_source(root, relative):
    if not isinstance(relative,str):return None
    path=Path(relative)
    if path.is_absolute() or '..' in path.parts or ':' in relative:return None
    return '../'+path.as_posix() if (root/path).is_file() else None



def exact_input_links(root,fid,body):
    key=hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    prefix=f'agent-inputs/{fid}/{key}'
    result={}
    for label,suffix in [('Exact captured model input text','.txt'),('Captured input token IDs / provenance','.json')]:
        url=safe_source(root,prefix+suffix)
        if url:result[label]=url
    return result


def unprojected_artifact(root,result):
    try:data=json.loads(raw_tool_text(result))
    except (ValueError,TypeError):return None
    reference=data.get('raw_tool_artifact') if isinstance(data,dict) else None
    if not isinstance(reference,dict):return None
    path=Path(reference.get('path',''))
    path=(path if path.is_absolute() else root/path).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():return None
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if not reference.get('sha256') or sha!=reference['sha256']:raise ValueError('Unprojected tool artifact hash mismatch')
    return dict(path=path.relative_to(root.resolve()).as_posix(),url=safe_source(root,path.relative_to(root.resolve()).as_posix()),sha256=sha,
                provenance='Hash-verified full unprojected training-feedback artifact. It is separate from the MCP transport return and model-delivered projection.')


TOKENIZER_PATH=Path('/home/georg/Code/vllm-utils/models/Qwen3.8-27B-NVFP4/tokenizer.json')


class PayloadCounter:
    def __init__(self,tokenizer):self.tokenizer=tokenizer;self.cache={}
    def count(self,text):
        key=hashlib.sha256(text.encode()).hexdigest()
        if key not in self.cache:
            self.cache[key]=len(self.tokenizer.encode(text,add_special_tokens=False).ids)
        return self.cache[key]


def captured_tool_payloads(root,fid,agent,counter):
    """Index literal tool-message content; never regenerate it with current helpers."""
    references={};ordinal=0
    for event in agent.get('events',[]):
        if event.get('type')!='model_response':continue
        choice=(event.get('response',{}).get('choices') or [{}])[0]
        suffix=f'{event.get("turn")}-truncated-{ordinal}' if choice.get('finish_reason')=='length' else str(event.get('turn'))
        ordinal+=1
        relative=event.get('request_body_path')
        if relative and safe_source(root,relative):
            references.setdefault(relative,[]).append(dict(event=event,debug=f'prompt-debug/{fid}/turn-{suffix}.html'))
    paths=set((root/'request-bodies'/fid).glob('*.json'))
    paths.update(root/name for name in references)
    by_call={}
    for path in sorted(paths):
        raw=path.read_bytes();body_hash=hashlib.sha256(raw).hexdigest();body=json.loads(raw)
        relative=path.relative_to(root).as_posix();refs=references.get(relative,[])
        for ref in refs:
            expected=ref['event'].get('request_body_sha256')
            if expected and expected!=body_hash:raise ValueError(f'Immutable request hash mismatch: {relative}')
        provenance=[ref['event'].get('request_body_provenance','Captured request body before dispatch') for ref in refs]
        reconstructed=bool(refs) and all('reconstruct' in note.lower() for note in provenance)
        confirmed=any(ref['event'].get('request_body_sha256') and 'reconstruct' not in ref['event'].get('request_body_provenance','').lower() for ref in refs)
        recorded_hash_verified=any(bool(ref['event'].get('request_body_sha256')) for ref in refs)
        source=dict(path=relative,url=safe_source(root,relative),sha256=body_hash,
                    recorded_response=bool(refs),runtime_capture=confirmed,matches_recorded_hash=recorded_hash_verified,
                    provenance='; '.join(dict.fromkeys(provenance)) if refs else 'Saved request body; no associated recorded response. Dispatch/completion not established by this file alone.',
                    exact_input_urls=[safe_source(root,ref['debug']) for ref in refs if safe_source(root,ref['debug'])]+list(exact_input_links(root,fid,body).values()))
        for message in body.get('messages',[]):
            if message.get('role')!='tool' or not isinstance(message.get('content'),str):continue
            call_id=message.get('tool_call_id')
            if not call_id:continue
            text=message['content'];sha=hashlib.sha256(text.encode()).hexdigest()
            variants=by_call.setdefault(call_id,{})
            if sha not in variants:
                variants[sha]=dict(text=text,tokens=counter.count(text),sha256=sha,sources=[],confirmed_runtime_capture=False)
            variant=variants[sha]
            if not any(item['path']==relative for item in variant['sources']):variant['sources'].append(source)
            variant['confirmed_runtime_capture'] |= confirmed
    return {call_id:sorted(variants.values(),key=lambda v:not v['confirmed_runtime_capture']) for call_id,variants in by_call.items()}


def tool_payload_view(call_id,result,captures,counter,tokenizer_note):
    raw=raw_tool_text(result) if result is not None else None
    variants=captures.get(call_id,[]) if call_id else []
    return dict(tool_call_id=call_id,raw_text=raw,raw_tokens=counter.count(raw) if raw is not None else None,
                raw_sha256=hashlib.sha256(raw.encode()).hexdigest() if raw is not None else None,
                model_variants=variants,tokenizer_note=tokenizer_note,
                missing_capture_note=None if variants else 'No matching saved tool-message content was found in immutable request bodies. Current compaction code was not used to invent a historical model input.',
                raw_provenance='Original recorded MCP text blocks, joined with a newline as in the runner.' if raw is not None else 'Original MCP return is not linked for this message; the conversation text is not relabeled as the original return.')


def message_html(message):
    body=''
    reasoning=message.get('reasoning_content') or message.get('reasoning')
    if reasoning:
        body+='<details class="returned-reasoning"><summary>Returned reasoning · '+str(len(str(reasoning).split()))+' words</summary>'+markdown(reasoning)+'</details>'
    if message.get('content'):
        content=message['content']
        body+=markdown(content) if isinstance(content,str) else '<pre>'+pretty(content)+'</pre>'
    for call in message.get('tool_calls',[]):
        function=call.get('function',{})
        arguments=function.get('arguments','')
        try:arguments=json.loads(arguments)
        except (ValueError,TypeError):pass
        body+='<section class="tool-call"><h3>Call '+esc(function.get('name','tool'))+'</h3><pre>'+pretty(arguments)+'</pre></section>'
    return body or '<p class="muted">No text content returned.</p>'


def raw_tool_text(result):
    if isinstance(result,dict):
        content=result.get('content',[])
        if isinstance(content,list):return '\n'.join(c.get('text','') for c in content if c.get('type')=='text')
        if isinstance(content,str):return content
    return json.dumps(result,ensure_ascii=False)


def tool_html(result,arguments=None,scorer_reader=None):
    body=''
    if arguments is not None:
        try:arguments=json.loads(arguments) if isinstance(arguments,str) else arguments
        except ValueError:pass
        body+='<details><summary>Tool arguments</summary><pre>'+pretty(arguments)+'</pre></details>'
    text=raw_tool_text(result);inputs={}
    if scorer_reader:
        try:inputs=scorer_reader.prepare(json.loads(text))
        except (ValueError,TypeError):pass
    return body+tool_result(text,scorer_inputs=inputs)


def event_node(root,feature,event,event_index,model_ordinal,scorer_reader=None):
    kind=event.get('type','event');turn=event.get('turn');role='event';tool=None;usage={};finish=None;links={}
    if kind=='model_response':
        role='assistant';response=event.get('response',{});choice=(response.get('choices') or [{}])[0]
        usage=response.get('usage',{});finish=choice.get('finish_reason');message=choice.get('message',{})
        content=message_html(message);title=f'Model turn {turn}'
        if finish=='length':title+=' · truncated'
        file_turn=f'{turn}-truncated-{model_ordinal}' if finish=='length' else str(turn)
        for name,suffix in [('Exact model input','.html'),('Prompt / token IDs JSON','.json')]:
            path=safe_source(root,f'prompt-debug/{feature}/turn-{file_turn}'+suffix)
            if path:links[name]=path
        path=safe_source(root,event.get('request_body_path'))
        if path:
            links['Immutable request JSON']=path
            body=read(root/event['request_body_path'])
            links.update(exact_input_links(root,str(feature),body))
    elif kind=='tool_result':
        role='tool';tool=event.get('tool','tool');title=tool;content=tool_html(event.get('result',{}),event.get('arguments'),scorer_reader)
        artifact=unprojected_artifact(root,event.get('result',{}))
        if artifact:links['Full unprojected tool artifact · hash verified']=artifact['url']
    else:
        title=kind.replace('_',' ')
        content=markdown(event.get('user_note') or event.get('reason') or event.get('note') or '')+'<pre>'+pretty(event)+'</pre>'
    status='truncated' if finish=='length' else 'error' if event.get('error') or (isinstance(event.get('result'),dict) and event['result'].get('isError')) else 'recovery' if any(k in kind for k in ('recovery','extension','compaction','context','budget_revision')) else 'recorded'
    summary=dict(id=f'e{event_index}',stream='events',kind=kind,role=role,tool=tool,title=title,turn=turn,
                 at=event.get('at'),duration=event.get('duration_seconds'),status=status,usage=usage,
                 settings=event.get('request_settings',{}),finish_reason=finish,
                 search=' '.join(str(v) for v in (title,kind,turn,event.get('arguments',''))))
    return summary,dict(html=content,raw=event,links=links)


def build_feature(root,out,selected,counter,tokenizer_note,scorer_reader):
    fid=str(selected['id']);agent=read(root/'agents'/f'{fid}.json');score=read(root/'scored-features'/f'{fid}.json')
    audit=read(root/'trajectory-audit'/'features'/f'{fid}.json')
    corrected=(root/'baseline/manifest.json').exists()
    baseline=read(root/'baseline/features'/f'{fid}.json')
    run=read(root/'run.json')
    config=read(root/(run.get('active_configuration') or 'configuration.json'))
    captures=captured_tool_payloads(root,fid,agent,counter)
    payloads_by_call={}
    model_audit={e.get('response_id'):e for e in audit.get('model_events',[])}
    tool_audit={(e.get('tool'),e.get('at')):e for e in audit.get('tool_events',[])}
    nodes=[];tokens=Counter();tool_counts=Counter();model_ordinal=0;limits=[];search={}
    for index,event in enumerate(agent.get('events',[])):
        node,payload=event_node(root,fid,event,index,model_ordinal,scorer_reader)
        if event.get('type')=='model_response':
            measured=model_audit.get(event.get('response',{}).get('id'),{})
            node['payload_metrics']=measured.get('tool_context_payloads',{})
            model_ordinal+=1
            for k in ('prompt_tokens','completion_tokens','total_tokens'):tokens[k]+=event.get('response',{}).get('usage',{}).get(k,0) or 0
            tokens['reasoning_tokens']+=(event.get('response',{}).get('usage',{}).get('completion_tokens_details') or {}).get('reasoning_tokens',0) or 0
        if event.get('type')=='tool_result':
            tool_counts[event.get('tool','unknown')]+=1
            node['payload_metrics']=tool_audit.get((event.get('tool'),event.get('at')),{})
            call_id=event.get('tool_call_id') or node['payload_metrics'].get('tool_call_id')
            payload['tool_payload']=tool_payload_view(call_id,event.get('result'),captures,counter,tokenizer_note)
            artifact=unprojected_artifact(root,event.get('result',{}))
            if artifact:
                payload['tool_payload']['unprojected_artifact']=artifact
                payload['tool_payload']['raw_provenance']='Literal recorded MCP transport return (already server-projected for this tool). Full unprojected feedback is linked separately and is not counted as model delivery.'
            if call_id:payloads_by_call[call_id]=payload['tool_payload']
        if node['status'] in ('truncated','recovery','error'):limits.append({k:node[k] for k in ('id','title','turn','status')})
        node['file']=f'data/{fid}/{node["id"]}.json';write(out/node['file'],payload);nodes.append(node);search[node['id']]=json.dumps(payload['raw'],ensure_ascii=False)
    for i,incident in enumerate(audit.get('failure_incidents',[])):
        node=dict(id=f'a{i}',stream='events',kind=incident.get('category','checkpoint_incident'),role='event',tool=None,title='Earlier checkpoint: '+incident.get('category','incident').replace('_',' '),turn=incident.get('last_model_turn'),status='historical incident',usage={},settings={},at=incident.get('finished_at'),search=json.dumps(incident))
        links={}
        for j,source in enumerate(incident.get('snapshot_sources',[])):
            path=safe_source(root,source)
            if path:links[f'Checkpoint snapshot {j+1}']=path
        payload=dict(html='<p class="training-notice">Historical checkpoint incident. The feature’s final status is '+esc(agent.get('status','unknown'))+'. This is not a claim that its final explanation failed.</p>'+markdown(incident.get('error',''))+'<pre>'+pretty(incident)+'</pre>',raw=incident,links=links)
        node['file']=f'data/{fid}/{node["id"]}.json';write(out/node['file'],payload);nodes.append(node);search[node['id']]=json.dumps(incident,ensure_ascii=False)
    tool_names={call.get('id'):call.get('function',{}).get('name') for message in agent.get('messages',[]) for call in message.get('tool_calls',[])}
    for index,message in enumerate(agent.get('messages',[])):
        role=message.get('role','message');tool=tool_names.get(message.get('tool_call_id'))
        title=f'{index+1:02d} · '+(tool or role)
        payload=dict(html=tool_html({'content':message.get('content','')},scorer_reader=scorer_reader) if role=='tool' else message_html(message),raw=message,links={})
        if role=='tool':
            call_id=message.get('tool_call_id')
            payload['tool_payload']=payloads_by_call.get(call_id) or tool_payload_view(call_id,None,captures,counter,tokenizer_note)
        node=dict(id=f'm{index}',stream='messages',kind='message',role=role,tool=tool,title=title,turn=None,status='recorded',usage={},settings={},search=title)
        node['file']=f'data/{fid}/{node["id"]}.json';write(out/node['file'],payload);nodes.append(node);search[node['id']]=json.dumps(payload['raw'],ensure_ascii=False)
    scores={}
    for method in METHODS:
        historical=score.get('comparisons',{}).get('historical',{}).get(method,{})
        old=historical.get('baseline',metric(selected.get('baseline_scores',{}),method));new=historical.get('candidate',metric(score.get('methods',{}),method));fresh=metric(score.get('refreshed_baseline',{}),method)
        if corrected:old=None;fresh=metric(score.get('refreshed_baseline') or baseline.get('methods',{}),method)
        scores[method]=dict(original=old,agent=new,fresh=fresh,delta=new-fresh if isinstance(new,(int,float)) and isinstance(fresh,(int,float)) else None)
    from render_molt_corrected import current_agent_state
    current_status=current_agent_state(agent,run)[0] if corrected else agent.get('status','pending')
    if not audit:
        audit=dict(explainer_usage=dict(tokens),limits=dict(output_length_responses=sum(n.get('finish_reason')=='length' for n in nodes)),tools={name:dict(calls=count) for name,count in tool_counts.items()},scope='Live summary from recorded events; full payload attribution audit not yet available')
    manifest=dict(corrected_run=corrected,active_configuration=run.get('active_configuration','configuration.json'),active_budget=run.get('budget_revision') or config,id=fid,rank=selected.get('rank'),cohort=selected.get('cohort','unassigned'),status=current_status,
                  explanation=agent.get('explanation'),baseline_explanation=baseline.get('explanation') if corrected else selected.get('baseline_explanation'),scores=scores,
                  tokens=dict(tokens),tools=dict(tool_counts),limits=limits,model_responses=model_ordinal,
                  max_prompt_tokens=max((n['usage'].get('prompt_tokens',0) or 0 for n in nodes),default=0),
                  generation=dict(agent.get('generation',{}),context_limit=262144),nodes=nodes,audit=audit,sources=[{'path':p,'url':safe_source(root,p)} for p in audit.get('state_sources',[]) if safe_source(root,p)],
                  links={'Full original agent JSON':f'../agents/{fid}.json','Final scores JSON':f'../scored-features/{fid}.json','Trajectory audit JSON':f'../trajectory-audit/features/{fid}.json'})
    manifest['links']={k:v for k,v in manifest['links'].items() if safe_source(root,v.removeprefix('../'))}
    write(out/'data'/fid/'search.json',search)
    write(out/'data'/f'{fid}.json',manifest)
    return {k:manifest[k] for k in ('id','rank','cohort','status','explanation','scores','tokens','tools','limits','model_responses','max_prompt_tokens')}


def render(root,tokenizer=None,tokenizer_path=TOKENIZER_PATH):
    root=Path(root);out=root/'trajectories';out.mkdir(parents=True,exist_ok=True)
    run=read(root/'run.json');config=read(root/(run.get('active_configuration') or 'configuration.json'))
    selection=read(root/'selection.json');cache=read(out/'.build-cache.json');entries={};features=[]
    if tokenizer is None:
        from tokenizers import Tokenizer
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    counter=PayloadCounter(tokenizer)
    scorer_reader=TrainingScorerInputs(root,out)
    tokenizer_hash=hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest() if Path(tokenizer_path).exists() else None
    tokenizer_note=dict(model='Qwen3.8-27B-NVFP4',path=str(tokenizer_path),sha256=tokenizer_hash,add_special_tokens=False,scope='Standalone text-span tokens, excluding chat/template wrappers. These are not an additive partition of complete prompt tokens.')
    code=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('render_molt_agentic.py').read_bytes()+Path(__file__).with_name('molt_training_scorer_input.py').read_bytes()).hexdigest()
    for selected in selection.get('selected',[]):
        fid=str(selected['id'])
        if not fid.isdecimal():raise ValueError('Invalid feature ID')
        sources=[root/'baseline/features'/f'{fid}.json',root/'agents'/f'{fid}.json',root/'scored-features'/f'{fid}.json',root/'trajectory-audit'/'features'/f'{fid}.json',root/'prompt-debug'/fid]+list((root/'request-bodies'/fid).glob('*.json'))+list((root/'training-scores'/fid).glob('*.jsonl'))
        stamps=[(str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in sources if p.exists()]
        key=hashlib.sha256(json.dumps([VERSION,code,tokenizer_hash,selected,run.get('resumed_at'),run.get('status'),run.get('active_configuration'),stamps],sort_keys=True).encode()).hexdigest()
        old=cache.get(fid,{})
        if old.get('key')==key and (out/'data'/f'{fid}.json').exists():summary=old['summary']
        else:summary=build_feature(root,out,selected,counter,tokenizer_note,scorer_reader)
        features.append(summary);entries[fid]=dict(key=key,summary=summary)
    audit=read(root/'trajectory-audit'/'summary.json')
    write(out/'index.json',dict(version=VERSION,created_at=datetime.now(timezone.utc).isoformat(),features=features,audit=audit,
                               tool_inventory=read(root/'tool-inventory.json'),configuration=config,corrected_run=(root/'baseline/manifest.json').exists(),run_status=run.get('status'),active_configuration=run.get('active_configuration','configuration.json'),budget_revision=run.get('budget_revision'),
                               retry_policy=read(root/'retry-policy.json'),recovery_policy=read(root/'recovery-policy.json')))
    write(out/'.build-cache.json',entries)
    for name in ('index.html','explorer.css','explorer.js'):shutil.copyfile(ASSETS/name,out/name)
    return out/'index.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('results/molt-agentic-pilot-v2'))
    parser.add_argument('--tokenizer',type=Path,default=TOKENIZER_PATH)
    args=parser.parse_args();print(render(args.root,tokenizer_path=args.tokenizer))
