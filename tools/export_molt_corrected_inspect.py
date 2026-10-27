#!/usr/bin/env python3
"""Snapshot corrected MOLT artifacts into immutable Inspect imports; no inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import fcntl
from export_molt_inspect import (read, write, sha, message, model_output, usage, sum_usage,
    publish_log, event_time, ModelEvent, ToolEvent, InfoEvent, ToolCallError, ToolInfo,
    GenerateConfig, EvalSample, EvalLog, EvalSpec, EvalDataset, EvalConfig, EvalPlan,
    EvalPlanStep, EvalResults, EvalStats, EvalScore, EvalMetric, Score, read_eval_log)

NOTE='Imported corrected-context128 saved artifacts; not a native Inspect execution. No inference replayed. Snapshot may contain running/incomplete samples.'


def snapshot(run, storage):
    paths=[run/name for name in ('selection.json','configuration.json','run.json','tool-schemas.json','protocol/frozen.json')]
    for folder,pattern in [('agents','*.json'),('baseline/features','*.json'),('baseline/audit','*.jsonl'),('scoring-audit','*.jsonl'),('scored-features','*.json')]:
        paths.extend(sorted((run/folder).glob(pattern)))
    paths.extend(sorted(run.glob('configuration*.json')))
    paths.extend(sorted(run.glob('tool-schemas*.json')))
    paths.extend(run/name for name in ('deployment.json','pilot-import.json'))
    paths.extend(sorted((run/'protocol').glob('*.json')))
    blobs={str(p.relative_to(run)):p.read_bytes() for p in paths if p.exists()}
    # run.json timestamps are heartbeat-only. Preserve snapshot but do not force exports for them.
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in blobs.items()}
    run_state=json.loads(blobs.get('run.json',b'{}'))
    identity={k:v for k,v in hashes.items() if k!='run.json'}
    identity['active_configuration']=run_state.get('active_configuration','configuration.json')
    identity['concurrency']=run_state.get('concurrency')
    identity['runtime_revisions']={k:v for k,v in run_state.items() if 'revision' in k}
    revision=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    destination=storage/'snapshots'/revision
    if not destination.exists():
        destination.mkdir(parents=True)
        blob_store=storage/'blobs';blob_store.mkdir(exist_ok=True)
        for name,data in blobs.items():
            blob=blob_store/hashes[name]
            if not blob.exists():blob.write_bytes(data)
            p=destination/name;p.parent.mkdir(parents=True,exist_ok=True);os.link(blob,p)
        write(destination/'snapshot-manifest.json',dict(source=str(run),snapshot_at=datetime.now(timezone.utc).isoformat(),sha256=hashes,
            consistency='Each input captured as bytes before building. Files captured sequentially, not an atomic cross-file experiment snapshot; running state remains partial.'))
    return destination,revision


def active_configuration(snap):
    state=read(snap/'run.json') if (snap/'run.json').exists() else {}
    name=state.get('active_configuration','configuration.json')
    path=(snap/name).resolve()
    if not path.is_relative_to(snap.resolve()):
        raise ValueError('Active configuration must be inside snapshot')
    if not path.exists():
        raise ValueError(f'Active configuration missing from snapshot: {name}')
    return read(path),dict(active_configuration=name,active_configuration_sha256=sha(path),active_settings=read(path),
        historical_configuration='configuration.json',
        runtime_revisions={p.name:read(p) for p in sorted((snap/'protocol').glob('*.json')) if 'revision' in p.name or 'parent_protocol_sha256' in read(p)},
        run_revision_metadata={k:v for k,v in state.items() if 'revision' in k})


def jsonl(path):
    if not path.exists():return []
    rows=[]
    for index,line in enumerate(path.read_text().splitlines()):
        try:rows.append(json.loads(line))
        except ValueError:rows.append(dict(type='incomplete_audit_line',line=index,note='Line incomplete at snapshot; not inferred as a response'))
    return rows


def model_event(row, model, tools, scope):
    body=row.get('request_body') or row.get('request_settings') or {}
    template=body.get('chat_template_kwargs',{})
    config=GenerateConfig(max_tokens=body.get('max_tokens'),temperature=body.get('temperature'),top_p=body.get('top_p'),top_k=body.get('top_k'),reasoning_effort=template.get('reasoning_effort'),extra_body={'chat_template_kwargs':template,'min_p':body.get('min_p')})
    stamp=row.get('at') or datetime(2026,10,24,tzinfo=timezone.utc).isoformat()
    return ModelEvent(timestamp=datetime.fromisoformat(stamp),model=model,input=[],tools=tools,tool_choice=body.get('tool_choice','auto'),config=config,
        output=model_output(row['response'],model),metadata=dict(scope=scope,source_record=row,
            input_note='Exact request body/rendered-input path and hash preserved in source_record; repeated prefixes omitted from timeline input. Saved conversation is in Messages.'))


def build_sample(snap,feature,arm,tools,source_url):
    config,configuration_metadata=active_configuration(snap);model=config['model']
    basepath=snap/'baseline/features'/f'{feature}.json';base=read(basepath) if basepath.exists() else {}
    agentpath=snap/'agents'/f'{feature}.json';agent=read(agentpath) if agentpath.exists() else {}
    source=agent if arm=='agent' else base
    events=[];outputs=[];names={}
    for raw in agent.get('messages',[]):
        for call in raw.get('tool_calls') or []:names[call['id']]=call['function']['name']
    if arm=='agent':
        for row in agent.get('events',[]):
            if row.get('type')=='model_response' and row.get('response',{}).get('choices'):
                event=model_event(row,model,tools,'agent_explainer');events.append(event);outputs.append(event.output)
            elif row.get('type')=='tool_result':
                result=row.get('result',{});text='\n'.join(c.get('text','') for c in result.get('content',[]) if isinstance(c,dict) and c.get('type')=='text')
                arguments=row.get('arguments',{})
                if isinstance(arguments,str):
                    try:arguments=json.loads(arguments)
                    except ValueError:arguments={'recorded_unparsed':arguments}
                failed=bool(result.get('isError',result.get('is_error',False)))
                events.append(ToolEvent(timestamp=datetime.fromisoformat(row.get('at') or agent['started_at']),id=row.get('tool_call_id') or 'recorded-'+str(len(events)),function=row['tool'],arguments=arguments,result=text,failed=failed,error=ToolCallError(type='unknown',message=text) if failed else None,metadata={'source_record':row,'scope':'Recorded MCP result; actual model-facing text is in saved conversation.'}))
            else:events.append(InfoEvent(source='Recorded agent lifecycle',data=row))
    else:
        for row in jsonl(snap/'baseline/audit'/f'{feature}.jsonl'):
            if row.get('response',{}).get('choices') and row.get('type')!='cache_reuse':
                event=model_event(row,model,[],'fresh_baseline_explainer');events.append(event);outputs.append(event.output)
            else:events.append(InfoEvent(source='Baseline audit',data=row))
    scoring=[]
    for row in jsonl(snap/'scoring-audit'/f'{feature}.jsonl'):
        if row.get('arm')==arm or (arm=='agent' and row.get('method')=='intruder'):
            if row.get('response',{}).get('choices') and row.get('type')!='cache_reuse':
                events.append(model_event(row,model,[],'heldout_'+row.get('method','unknown')+('_shared_control' if arm=='agent' and row.get('arm')=='baseline' else '')))
                scoring.append(usage(row.get('response',{}).get('usage')))
            else:events.append(InfoEvent(source='Heldout scoring audit',data=row))
    scoredpath=snap/'scored-features'/f'{feature}.json';scored=read(scoredpath) if scoredpath.exists() else {}
    methods=(scored.get('methods',{}) if arm=='agent' else scored.get('refreshed_baseline') or base.get('methods',{}))
    scores={name:Score(value=data['metrics']['accuracy'],explanation='Valid-only held-out metric; missing rows disclosed in metadata. Intruder is shared explanation-independent control.',metadata={'metrics':data['metrics'],'protocol':data.get('protocol')}) for name,data in methods.items() if data.get('metrics',{}).get('accuracy') is not None}
    complete=(scored.get('status')=='complete' if arm=='agent' else all(methods.get(name,{}).get('status')=='complete' for name in ('detection','fuzz','active_fuzz','intruder')))
    status='complete' if source.get('explanation') and complete else 'incomplete'
    messages=[message(raw,names) for raw in agent.get('messages',[])] if arm=='agent' else [message(m) for row in jsonl(snap/'baseline/audit'/f'{feature}.jsonl') for m in (row.get('request_body',{}).get('messages',[])+([row['response']['choices'][0]['message']] if row.get('response',{}).get('choices') else []))]
    total=sum_usage([o.usage for o in outputs]);scorer_usage=sum_usage(scoring)
    return EvalSample(id=feature,epoch=1,input=f'Corrected context128 MOLT layer22 feature {feature}',target='',description=f'Feature {feature} · {arm} · {status}',messages=messages,scores=scores,events=events,
        model_usage={model:total},metadata=dict(provenance=NOTE,configuration=configuration_metadata,status=status,source_status=source.get('status','not_started'),explanation=source.get('explanation'),snapshot=str(snap),source_transcript=f'{source_url}/agents/{feature}.json' if arm=='agent' else f'{source_url}/baseline/features/{feature}.json',scoring_metrics={k:v.get('metrics') for k,v in methods.items()},heldout_scorer_usage=scorer_usage.model_dump(),usage_scope='model_usage counts explainer responses only; held-out scorer usage separate. Agent intruder entries explicitly reuse baseline control; never sum both arms as additional calls.',snapshot_source_hashes=read(snap/'snapshot-manifest.json')['sha256']))


def build_log(snap,arm,source_url):
    config,configuration_metadata=active_configuration(snap)
    schemas=config.get('active_tool_schemas','tool-schemas.json')
    tools=[ToolInfo(**t['function']) for t in read(snap/schemas)]
    samples=[build_sample(snap,row['id'],arm,tools,source_url) for row in read(snap/'selection.json')['selected']]
    complete=sum(s.metadata['status']=='complete' for s in samples)
    metrics=[]
    for name in ('detection','fuzz','active_fuzz','intruder'):
        values=[s.scores[name].value for s in samples if name in s.scores]
        if values:metrics.append(EvalScore(name=name,scorer=name,scored_samples=len(values),metrics={'mean':EvalMetric(name='mean',value=sum(values)/len(values))}))
    model=config['model'];created=read(snap/'snapshot-manifest.json')['snapshot_at']
    source_run=Path(read(snap/'snapshot-manifest.json')['source']).name
    return EvalLog(status='success',eval=EvalSpec(created=created,task=f'{source_run}_{arm}',task_display_name=f'MOLT layer22 · {"agent" if arm=="agent" else "fresh baseline"} · {complete}/{len(samples)} complete',task_description=NOTE,dataset=EvalDataset(name=f'MOLT corrected-context128 {arm}',samples=len(samples),sample_ids=[s.id for s in samples]),model=model,config=EvalConfig(),metadata={'snapshot':str(snap),'provenance':NOTE,'configuration':configuration_metadata,'experiment_status':'complete' if complete==len(samples) else 'incomplete','status_note':'success refers only to the completed IMPORT; experiment completeness is separate.'}),plan=EvalPlan(name='Import saved snapshot',steps=[EvalPlanStep(solver='import_saved_artifacts')],config=GenerateConfig()),results=EvalResults(total_samples=len(samples),completed_samples=complete,scores=metrics),stats=EvalStats(model_usage={model:sum_usage([s.model_usage[model] for s in samples])}),samples=samples,metadata={'provenance':NOTE,'experiment_status':'complete' if complete==len(samples) else 'incomplete', 'status_note':'success means this snapshot IMPORT completed; experiment/sample completeness is reported separately. No native Inspect execution is running.','snapshot_manifest':str(snap/'snapshot-manifest.json')})


def main(args):
    args.storage.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    with (args.storage/'export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        snap,revision=snapshot(args.run.resolve(),args.storage)
        revision=hashlib.sha256((revision+sha(__file__)+args.source_url).encode()).hexdigest()
        state=args.storage/'current.json'
        if state.exists() and read(state).get('revision')==revision:
            print(json.dumps(dict(status='unchanged',**read(state))));return
        paths={}
        for arm in ('agent','baseline'):
            log=build_log(snap,arm,args.source_url)
            physical=publish_log(log,args.storage/'logs',f'{args.run.name}-{arm}_imported')
            served=args.output/physical.name
            if not served.exists():served.symlink_to(physical.resolve())
            restored=read_eval_log(str(served));assert len(restored.samples)==len(read(snap/'selection.json')['selected'])
            paths[arm]=dict(path=str(physical),served=str(served),inspect_url=f'http://100.76.58.1:8770/#/tasks/{served.name}',sha256=sha(physical),bytes=physical.stat().st_size,samples=len(restored.samples),status=restored.status)
        result=dict(revision=revision,snapshot=str(snap),logs=paths)
        write(state,result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('results/molt-agentic-delphi-v3-32-context128'))
    parser.add_argument('--storage',type=Path,default=Path('/mnt/mec-data-ssd/mec-data/molt-runs/agentic-delphi-v3-32-context128/observability/inspect'))
    parser.add_argument('--output',type=Path,default=Path('results/molt-observability/inspect-logs'))
    parser.add_argument('--source-url',default='http://100.76.58.1:8771')
    main(parser.parse_args())
