"""Publish compact corrected-run results into the existing read-only dashboard.

No GPU/API calls. Historical annotations are neither changed nor merged with the
corrected protocol. Re-run periodically to publish newly completed results.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

SCRIPT='<script src="assets/molt-corrected-run.js" defer></script>'


def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def metrics(methods):
    return {name:{k:v for k,v in value.items() if k in ('status','metrics','reason','error')}
            for name,value in (methods or {}).items()}


def snapshot(run,dashboard):
    manifest=read(run/'baseline/manifest.json')['settings'];meta=read(dashboard/'metadata.json')
    for key in ('layer','checkpoint','global_step'):
        if manifest[key]!=meta[key]:raise ValueError(f'Corrected run belongs to a different {key}')
    state=read(run/'run.json');selection=read(run/'selection.json');features={}
    for selected in selection['selected']:
        fid=selected['id'];base=read(run/'baseline/features'/f'{fid}.json');agent=read(run/'agents'/f'{fid}.json');score=read(run/'scored-features'/f'{fid}.json')
        events=agent.get('events',[])
        agent_status=agent.get('status','pending')
        if agent_status=='running' and state.get('resumed_at'):
            if (agent.get('updated_at') or agent.get('resumed_at') or agent.get('started_at') or '')<state['resumed_at']:
                agent_status='queued'
        features[str(fid)]=dict(id=fid,rank=selected.get('rank'),cohort=selected.get('cohort'),
            stage=state.get('feature_stages',{}).get(str(fid)),
            baseline=dict(status=base.get('status','pending'),explanation=base.get('explanation'),methods=metrics(score.get('refreshed_baseline',base.get('methods',{})))),
            agent=dict(status=agent_status,explanation=agent.get('explanation') if agent.get('status')=='complete' else None,
                       methods=metrics(score.get('methods',{})),scoring_status=score.get('status','pending'),
                       model_responses=sum(e.get('type')=='model_response' for e in events),
                       tool_calls=sum(e.get('type')=='tool_result' for e in events),updated_at=agent.get('updated_at'),
                       last_activity=events[-1].get('type') if events else None,
                       last_tool=next((e.get('tool') for e in reversed(events) if e.get('type')=='tool_result'),None),
                       error=agent.get('error') if agent.get('status')=='error' else None))
    return dict(version=1,published_at=datetime.now(timezone.utc).isoformat(),run_id=run.name,
        report_url=read(run/'dashboard-settings.json').get('report_url','http://100.76.58.1:8771/'),
        concurrency=state.get('concurrency'),
        status=state.get('status','unknown'),updated_at=state.get('updated_at'),
        profile=dict(context_tokens=manifest['context_tokens'],n_train=manifest['n_train'],n_test_per_class=manifest['n_test_per_class'],
                     cache_sha256=manifest['cache_sha256'],same_cache_as_dashboard=manifest['cache_sha256']==meta.get('cache_sha256')),
        counts=dict(selected=len(features),baseline_explanations=sum(bool(v['baseline']['explanation']) for v in features.values()),
                    agent_explanations=sum(bool(v['agent']['explanation']) for v in features.values()),
                    matched_scores=sum(v['agent']['scoring_status']=='complete' for v in features.values()),
                    active_agents=sum(v['agent']['status']=='running' for v in features.values())),features=features)


def atomic(path,text):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.publish-tmp');temp.write_text(text);temp.replace(path)


def publish(run,dashboard):
    # The8765 root is a redirect to the10M sample; publish where browsers land.
    if (dashboard/"sample-10m/index.html").exists() and "location.replace(destination.href)" in (dashboard/"index.html").read_text():
        dashboard=(dashboard/"sample-10m").resolve()
    payload=snapshot(run,dashboard)
    atomic(dashboard/'corrected-run.json',json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\n')
    atomic(dashboard/'assets/molt-corrected-run.js',Path(__file__).with_name('molt_corrected_run.js').read_text())
    index=dashboard/'index.html';html=index.read_text()
    if SCRIPT not in html:atomic(index,html.replace('</body>',SCRIPT+'\n</body>'))
    return payload['counts']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--dashboard',type=Path,required=True)
    a=p.parse_args();print(json.dumps(publish(a.run,a.dashboard)))
