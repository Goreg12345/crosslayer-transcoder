#!/usr/bin/env python3
"""Import saved MOLT pilot into native Inspect logs, without executing models/tools."""
import argparse
import fcntl
import os
import tempfile
from collections import Counter
from datetime import datetime,timedelta,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean

from inspect_ai.event import InfoEvent,ModelEvent,SampleLimitEvent,ToolEvent
from inspect_ai.log import (EvalConfig,EvalDataset,EvalLog,EvalMetric,EvalPlan,EvalPlanStep,
                            EvalResults,EvalSample,EvalScore,EvalSpec,EvalStats,
                            read_eval_log,write_eval_log)
from inspect_ai.model import (ChatCompletionChoice,ChatMessageAssistant,ChatMessageSystem,
                              ChatMessageTool,ChatMessageUser,ContentReasoning,ContentText,
                              GenerateConfig,ModelOutput,ModelUsage)
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall,ToolCallError,ToolInfo

METHODS=('detection','fuzz','active_fuzz','intruder')
PROVENANCE='Imported from saved MOLT agentic pilot artifacts; NOT a native Inspect execution. No model or tool calls were replayed.'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def usage(source):
    if not source:return None
    reasoning=(source.get('completion_tokens_details') or {}).get('reasoning_tokens')
    return ModelUsage(input_tokens=source.get('prompt_tokens',0),output_tokens=source.get('completion_tokens',0),
        total_tokens=source.get('prompt_tokens',0)+source.get('completion_tokens',0),reasoning_tokens=reasoning)


def sum_usage(values):
    values=[v for v in values if v is not None]
    return ModelUsage(input_tokens=sum(v.input_tokens for v in values),output_tokens=sum(v.output_tokens for v in values),
        total_tokens=sum(v.input_tokens+v.output_tokens for v in values),
        reasoning_tokens=sum(v.reasoning_tokens or 0 for v in values))


def call(raw):
    arguments=raw['function']['arguments'];error=None
    try:parsed=json.loads(arguments) if isinstance(arguments,str) else arguments
    except ValueError:parsed={};error='Unparseable recorded arguments: '+arguments
    return ToolCall(id=raw['id'],function=raw['function']['name'],arguments=parsed,parse_error=error)


def message(raw,call_names=None):
    role=raw.get('role','assistant');content=raw.get('content') or ''
    if role=='assistant':
        reasoning=raw.get('reasoning_content') or raw.get('reasoning')
        if reasoning:
            content=[ContentReasoning(reasoning=reasoning)]+([ContentText(text=content)] if content else [])
        return ChatMessageAssistant(content=content,tool_calls=[call(c) for c in raw.get('tool_calls') or []] or None)
    if role=='tool':return ChatMessageTool(content=content,tool_call_id=raw.get('tool_call_id'),function=(call_names or {}).get(raw.get('tool_call_id')))
    if role=='system':return ChatMessageSystem(content=content)
    if role=='user':return ChatMessageUser(content=content)
    raise ValueError(f'Unexpected source message role: {role}')


def model_output(response,model):
    choices=[]
    for choice in response['choices']:
        reason=choice.get('finish_reason','unknown')
        reason='max_tokens' if reason=='length' else reason
        if reason not in ('stop','max_tokens','model_length','tool_calls','content_filter','unknown'):reason='unknown'
        choices.append(ChatCompletionChoice(message=message(choice['message']),stop_reason=reason))
    return ModelOutput(model=model,choices=choices,completion=response['choices'][0]['message'].get('content') or '',usage=usage(response.get('usage')),response_id=response.get('id'))


def event_time(event,fallback):
    completed=datetime.fromisoformat(event.get('at') or fallback)
    duration=event.get('duration_seconds')
    started=completed-timedelta(seconds=duration) if duration is not None else completed
    return started,completed,duration


def make_scores(result,arm,baseline_explanation=None):
    scores={}
    for name in METHODS:
        comparison=result['comparisons']['refreshed'][name]
        value=comparison['candidate' if arm=='agent' else 'baseline']
        scores[name]=Score(value=value,answer=result['explanation'] if arm=='agent' else baseline_explanation,explanation=(f"Imported matched {comparison['metric']}; identical held-out cases and New York scorer settings in both arms. "
            +('Intruder is explanation-independent and deliberately shared.' if name=='intruder' else '')),
            metadata=dict(metric=comparison['metric'],matched_baseline=comparison['baseline'],agent=comparison['candidate'],delta=comparison['delta'],
                historical_baseline=result['comparisons']['historical'][name]['baseline'],
                source='Saved scored-features; never supplied to the investigating agent.'))
    return scores


def sample_from_files(run,feature,tools,base_url,baseline_directory):
    agent=read(run/'agents'/f'{feature}.json');audit=read(run/'trajectory-audit/features'/f'{feature}.json')
    scored=read(run/'scored-features'/f'{feature}.json');baseline=read(baseline_directory/'features'/f'{feature}.json')
    calls={(r['tool'],r.get('at')):r['tool_call_id'] for r in audit['tool_events']}
    call_names={r['tool_call_id']:r['tool'] for r in audit['tool_events']}
    events=[];ordinal=0;model_outputs=[];started=datetime.fromisoformat(agent['started_at'])
    for original_index,event in enumerate(agent['events']):
        stamp,completed,duration=event_time(event,agent['started_at'])
        clock=dict(timestamp=stamp,working_start=max(0,(stamp-started).total_seconds()))
        common=dict(source_event_index=original_index,provenance=PROVENANCE,
            timing='timestamp derived from recorded completion time minus recorded request duration; durations include queue/network. working_start is wall-clock offset, not measured active compute.')
        if event['type']=='model_response':
            output=model_output(event['response'],agent['model']);model_outputs.append(output)
            config=event['request_settings'];template=config.get('chat_template_kwargs',{})
            settings=GenerateConfig(max_tokens=config.get('max_tokens'),temperature=config.get('temperature'),top_p=config.get('top_p'),top_k=config.get('top_k'),
                reasoning_effort=template.get('reasoning_effort'),extra_body={'chat_template_kwargs':template,'min_p':config.get('min_p')})
            suffix=f"turn-{event['turn']}-truncated-{ordinal}" if event['response']['choices'][0].get('finish_reason')=='length' else f"turn-{event['turn']}"
            links=dict(exact_input_html=f'{base_url}/prompt-debug/{feature}/{suffix}.html',exact_input_json=f'{base_url}/prompt-debug/{feature}/{suffix}.json',
                request_body=f"{base_url}/{event['request_body_path']}",request_body_sha256=event['request_body_sha256'])
            events.append(ModelEvent(**clock,model=agent['model'],input=[],tools=tools,tool_choice=config.get('tool_choice','auto'),config=settings,
                output=output,completed=completed,metadata=dict(**common,**links,turn=event['turn'],attempt=event.get('attempt'),
                    input_omitted=True,input_note='Full repeated input prefixes intentionally omitted from this imported event; follow exact_input_html for rendered text and token IDs. Final saved conversation is in Messages.',
                    recorded_duration_seconds=duration,usage_scope='Actual recorded response only; reasoning is included in output_tokens.')))
            if output.choices[0].stop_reason=='max_tokens':
                events.append(SampleLimitEvent(**clock,type='token',limit=config.get('max_tokens'),message='Recorded response hit its output-token budget; original truncated output retained and later retried.',
                    metadata=dict(recovered=True,scope='single model response, not final sample',turn=event['turn'])))
            ordinal+=1
        elif event['type']=='tool_result':
            result=event['result'];content=result.get('content',[])
            text='\n'.join(c.get('text','') for c in content if isinstance(c,dict) and c.get('type')=='text') if isinstance(content,list) else str(content)
            identifier=event.get('tool_call_id') or calls[(event['tool'],event.get('at'))]
            try:arguments=json.loads(event['arguments'])
            except ValueError:arguments={'unparsed_recorded_arguments':event['arguments']}
            failed=bool(result.get('isError',result.get('is_error',False)))
            events.append(ToolEvent(**clock,id=identifier,function=event['tool'],arguments=arguments,result=text,completed=completed,
                failed=failed,error=ToolCallError(type='unknown',message=text) if failed else None,
                metadata=dict(**common,recorded_duration_seconds=duration,result_scope='Original MCP return text; compacted/escaped model-facing payload is available in the exact input link.',
                    source_tool_result=f'{base_url}/agents/{feature}.json',subject_compute_usage='Qwen4B inference usage not recorded; payload tokens are not subject compute cost.')))
        else:
            events.append(InfoEvent(**clock,source='Imported recovery audit',data=event,metadata=common))
    for failure in audit['failure_incidents']:
        stamp=datetime.fromisoformat(failure.get('finished_at') or agent['started_at'])
        clock=dict(timestamp=stamp,working_start=max(0,(stamp-started).total_seconds()))
        if failure['category'] in ('context_overflow','turn_budget_exhaustion'):
            events.append(SampleLimitEvent(**clock,type='token' if failure['category']=='context_overflow' else 'turn',
                limit=262144 if failure['category']=='context_overflow' else (24 if (failure['last_model_turn'] or 0)<24 else 32),
                message='Recovered historical failure: '+failure['error'],metadata=dict(**failure,recovered=True,source='Deduplicated saved attempt/context-recovery states; no additional usage inferred.')))
        else:events.append(InfoEvent(**clock,source='Saved failed attempt',data=failure,metadata={'recovered':True}))
    events.sort(key=lambda event:event.timestamp)
    total=sum_usage([output.usage for output in model_outputs])
    metadata=dict(provenance=PROVENANCE,feature=feature,layer=22,limits=audit['limits'],
        source_agent_sha256=sha(run/'agents'/f'{feature}.json'),source_score_sha256=sha(run/'scored-features'/f'{feature}.json'),
        trajectory_url=f'{base_url}/trajectories/?feature={feature}',audit_url=f'{base_url}/trajectory-audit/features/{feature}.json',
        exact_inputs_index=f'{base_url}/prompt-debug/index.json',source_transcript=f'{base_url}/agents/{feature}.json',
        baseline_explanation=baseline['explanation'],target_semantics='Original baseline hypothesis, NOT ground truth or a reference answer. Compared under identical held-out scoring.',matched_scores=scored['comparisons']['refreshed'],
        historical_scores=scored['comparisons']['historical'],
        usage_note='CUMULATIVE API tokens summed across all recorded model responses, not unique conversation length or a single context window. Explainer API usage only; reasoning is a subset of completion. Internal training scoring, held-out scoring and subject Qwen4B compute are NOT included in sample model_usage.',
        internal_training_scorer_usage=audit['internal_training_scorer_usage'],
        missing_response_bodies=audit['limits']['request_bodies_without_response'],
        timing_note='Started/completed timestamps include pauses and recovery. Native working_start is wall-clock offset; active working_time is unknown.')
    sample=EvalSample(id=feature,epoch=1,input=agent['messages'][1]['content'],target='Original baseline hypothesis (not ground truth): '+baseline['explanation'],
        description=f'Layer22 MOLT feature{feature}',messages=[message(m,call_names) for m in agent['messages']],
        output=ModelOutput(model=agent['model'],completion=agent['explanation'],choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=agent['explanation']),stop_reason='stop')]),
        scores=make_scores(scored,'agent'),metadata=metadata,events=events,model_usage={agent['model']:total},
        started_at=agent['started_at'],completed_at=agent['finished_at'],total_time=(datetime.fromisoformat(agent['finished_at'])-started).total_seconds(),turn_count=agent.get('turns'))
    reference=EvalSample(id=feature,epoch=1,input=f'Frozen baseline explanation for MOLT layer22 feature{feature}',target='',
        description=f'Layer22 MOLT feature{feature} — matched baseline control',messages=[ChatMessageAssistant(content=baseline['explanation'])],
        output=ModelOutput(model=agent['model'],completion=baseline['explanation'],choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=baseline['explanation']),stop_reason='stop')]),
        scores=make_scores(scored,'baseline',baseline['explanation']),metadata=dict(provenance=PROVENANCE,feature=feature,
            note='Matched baseline re-score only. Original baseline generation trajectory, usage and timing are not reconstructed or claimed.',
            source_baseline=str(baseline_directory/'features'/f'{feature}.json'),source_baseline_sha256=sha(baseline_directory/'features'/f'{feature}.json'),
            matched_scores=scored['comparisons']['refreshed'],historical_scores=scored['comparisons']['historical']))
    return sample,reference


def make_log(samples,config,run,arm):
    model=config['model'];agent=arm=='agent';usage_values=[sample.model_usage[model] for sample in samples if model in sample.model_usage]
    generation=GenerateConfig(max_tokens=config.get('max_tokens'),temperature=config.get('temperature'),top_p=config.get('top_p'),top_k=config.get('top_k'),reasoning_effort=config.get('reasoning_effort'),extra_body={'chat_template_kwargs':{'enable_thinking':config.get('enable_thinking'),'reasoning_effort':config.get('reasoning_effort')},'min_p':config.get('min_p')}) if agent else GenerateConfig()
    summary_scores=[EvalScore(name=name,scorer=name,scored_samples=len(samples),metrics={'mean':EvalMetric(name='mean',value=mean(float(s.scores[name].value) for s in samples))}) for name in METHODS]
    times=[s.started_at for s in samples if s.started_at];ends=[s.completed_at for s in samples if s.completed_at]
    return EvalLog(status='success',eval=EvalSpec(created=min(times) if times else read(run/'configuration.json').get('created_at','2026-10-24T00:00:00+00:00'),
        task='molt_layer22_pilot_agent' if agent else 'molt_layer22_pilot_matched_baseline',
        task_display_name='MOLT pilot · agent trajectories' if agent else 'MOLT pilot · matched baseline control',
        task_description=PROVENANCE,dataset=EvalDataset(name='MOLT layer22 fixed32-feature pilot',samples=len(samples),sample_ids=[s.id for s in samples]),
        model=model,model_base_url=config['agent_url'] if agent else None,model_generate_config=generation,config=EvalConfig(),tags=['imported','no-reexecution',arm],
        packages={'inspect_ai':importlib.metadata.version('inspect_ai')},metadata=dict(provenance=PROVENANCE,arm=arm)),
        plan=EvalPlan(name='Imported saved agent pipeline' if agent else 'Imported matched control',steps=[EvalPlanStep(solver='import_saved_artifacts')],config=generation),
        results=EvalResults(total_samples=len(samples),completed_samples=len(samples),scores=summary_scores),
        stats=EvalStats(started_at=min(times) if times else '',completed_at=max(ends) if ends else '',model_usage={model:sum_usage(usage_values)} if usage_values else {}),
        tags=['imported','no-reexecution','molt','pilot32',arm],
        metadata=dict(provenance=PROVENANCE,source_run=str(run.resolve()),source_configuration_sha256=sha(run/'configuration.json'),
            source_scorer_runtime_sha256=sha(run/'scorer-runtime.json'),target_semantics='TARGET is the original baseline hypothesis, not ground truth. Its visible prefix labels this explicitly; metadata preserves the exact original text.',timing_note='No timing was invented for baseline generation; event active-compute timing is unavailable.',
            usage_scope='Main agent log contains explainer API usage only. Baseline control usage is omitted, not claimed to be zero.'),samples=samples)


def validate(log,expected):
    samples=log.samples or [];models=[e for s in samples for e in s.events if isinstance(e,ModelEvent)];tools=[e for s in samples for e in s.events if isinstance(e,ToolEvent)]
    total=sum_usage([e.output.usage for e in models]);limits=Counter(e.type for s in samples for e in s.events if isinstance(e,SampleLimitEvent))
    actual=dict(samples=len(samples),model_responses=len(models),tool_calls=len(tools),input_tokens=total.input_tokens,
                completion_tokens=total.output_tokens,reasoning_tokens=total.reasoning_tokens,total_tokens=total.total_tokens,
                length_responses=sum(e.output.choices[0].stop_reason=='max_tokens' for e in models),limit_flags=dict(limits))
    for key,value in expected.items():
        if actual.get(key)!=value:raise ValueError(f'Inspect import audit mismatch {key}: {actual.get(key)} != {value}')
    if total.total_tokens!=total.input_tokens+total.output_tokens:raise ValueError('Reasoning double-counted')
    return actual


def publish_log(log,output,stem):
    """Validate privately, then publish once under an immutable content-revision name.

    Inspect's browser ZIP reader caches central-directory offsets by log identity.
    Never replace bytes behind a published filename, even for a metadata-only edit.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    canonical=json.dumps(log.model_dump(mode='json',exclude={'location','etag'}),sort_keys=True,separators=(',',':'))
    revision=hashlib.sha256(canonical.encode()).hexdigest()
    destination=output/f'{stem}-{revision[:20]}.eval'
    manifest_path=output.parent/'inspect-publications.json'
    with (output.parent/'inspect-publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=read(manifest_path) if manifest_path.exists() else {'version':1,'files':{}}
        if destination.exists():
            record=manifest['files'].get(destination.name)
            if not record or record['semantic_sha256']!=revision or record['sha256']!=sha(destination):
                raise ValueError('Existing published log differs from its immutable publication record; refusing overwrite')
            return destination
        prior=manifest['files'].get(destination.name)
        if prior:
            archived=[p for p in (output.parent/'inspect-archive').rglob(destination.name) if sha(p)==prior['sha256']]
            if not archived:raise ValueError('Previously published revision is missing; refusing to regenerate different bytes under its name')
            os.link(archived[0],destination)
            return destination
        with tempfile.TemporaryDirectory(prefix='.inspect-staging-',dir=output.parent) as temporary:
            staged=Path(temporary)/destination.name
            write_eval_log(log.model_copy(deep=True),str(staged))
            restored=read_eval_log(str(staged))
            if len(restored.samples or [])!=len(log.samples or []):raise ValueError('Staged log sample count mismatch')
            # Link is an atomic create-if-absent publication; no intermediate bytes
            # are visible under a served .eval path and no existing file is replaced.
            os.link(staged,destination)
            manifest['files'][destination.name]=dict(semantic_sha256=revision,sha256=sha(destination),bytes=destination.stat().st_size)
            write(manifest_path,manifest)
    return destination


def archive_previous_exports(output,current):
    """Keep prior imports outside the served directory; original experiment untouched."""
    previous=[p for p in output.glob('2026-10-24_molt-pilot-*_imported*.eval') if p not in current]
    if not previous:return []
    archive=output.parent/'inspect-archive'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive.mkdir(parents=True)
    moved=[]
    for path in previous:
        destination=archive/path.name;path.rename(destination);moved.append(str(destination))
    return moved


def main(args):
    run=args.run.resolve();args.output.mkdir(parents=True,exist_ok=True)
    selection=read(run/'selection.json');config=read(run/'configuration.json');audit=read(run/'trajectory-audit/summary.json')
    tools=[ToolInfo(name=t['function']['name'],description=t['function'].get('description',''),parameters=t['function']['parameters']) for t in read(run/'tool-schemas.json')]
    samples=[];reference=[];before={}
    for row in selection['selected']:
        feature=row['id']
        for group in ('agents','scored-features'):
            source=run/group/f'{feature}.json';before[str(source)]=sha(source)
        sample,baseline=sample_from_files(run,feature,tools,args.source_url,Path(selection['baseline']));samples.append(sample);reference.append(baseline)
    agent_log=make_log(samples,config,run,'agent');baseline_log=make_log(reference,config,run,'baseline')
    counts=audit['totals']['explainer_usage'];expected=dict(samples=len(samples),model_responses=counts['response_attempts'],
        tool_calls=sum(t['calls'] for t in audit['tools'].values()),input_tokens=counts['input_tokens'],completion_tokens=counts['completion_tokens'],reasoning_tokens=counts['reasoning_tokens'],length_responses=audit['totals']['limits']['output_length_responses'])
    result=validate(agent_log,expected)
    agent_path=publish_log(agent_log,args.output,'2026-10-24_molt-pilot-agent_imported')
    baseline_path=publish_log(baseline_log,args.output,'2026-10-24_molt-pilot-matched-baseline_imported')
    result['roundtrip']=validate(read_eval_log(str(agent_path)),expected)
    if any(sha(path)!=value for path,value in before.items()):raise ValueError('Frozen source artifact changed during import')
    result.update(status='complete',inspect_ai_version=importlib.metadata.version('inspect_ai'),provenance=PROVENANCE,
        baseline_samples=len(read_eval_log(str(baseline_path)).samples or []),source_hashes=before,
        output_files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (agent_path,baseline_path)})
    result['archived_previous_exports']=archive_previous_exports(args.output,{agent_path,baseline_path})
    write(args.output.parent/'inspect-current-links.json',dict(agent_log=agent_path.name,baseline_log=baseline_path.name,
        agent_url=f'http://100.76.58.1:8770/#/tasks/{agent_path.name}',baseline_url=f'http://100.76.58.1:8770/#/tasks/{baseline_path.name}',
        feature33_url=f'http://100.76.58.1:8770/#/tasks/{agent_path.name}/samples/sample/33/1/'))
    write(args.output.parent/'inspect-import-audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','roundtrip')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,default=Path('results/molt-agentic-pilot-v2'))
    parser.add_argument('--output',type=Path,default=Path('results/molt-observability/inspect-logs'))
    parser.add_argument('--source-url',default='http://100.76.58.1:8766')
    main(parser.parse_args())
