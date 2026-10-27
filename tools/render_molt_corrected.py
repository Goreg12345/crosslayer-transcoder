#!/usr/bin/env python3
"""Incremental static progress/comparison report for the corrected Delphi run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.render_molt_agentic import STYLE, esc, pct, delta, number, metric, markdown, transcript, pretty

METHODS=('detection','fuzz','active_fuzz','intruder')


def read(path):
    if not path.exists():return {}
    try:return json.loads(path.read_text())
    except (OSError,ValueError) as exc:return {'status':'read_error','error':str(exc)}


def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(text);temp.replace(path)


def document(title,body,base=''):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'+(f'<base href="{esc(base)}">' if base else '')+f'<title>{esc(title)}</title><style>{STYLE}\nmain{{max-width:1600px}} .scroll{{overflow:auto}}td small{{display:block}}.explanations{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:24px}}.protocol{{border-left:4px solid #4685b8;padding:12px 18px}}input{{padding:12px;max-width:100%;box-sizing:border-box}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style></head><body><main>'+body+'</main></body></html>'


def score_cell(methods,name):
    entry=methods.get(name,{}) if isinstance(methods,dict) else {}
    m=entry.get('metrics',{}) if isinstance(entry,dict) else {}
    value=metric(methods,name)
    valid=m.get('valid');total=m.get('total')
    return pct(value)+f'<small>{esc(valid if valid is not None else "—")}/{esc(total if total is not None else "—")} valid</small>'


def comparison(baseline,agent):
    rows=[]
    for name in METHODS:
        b,a=metric(baseline,name),metric(agent,name)
        difference=a-b if number(a) and number(b) and name!='intruder' else None
        rows.append(f'<tr><th>{esc(name)}</th><td>{score_cell(baseline,name)}</td><td>{score_cell(agent,name)}</td><td>{"shared auxiliary" if name=="intruder" else delta(difference)}</td></tr>')
    return '<table><thead><tr><th>Method</th><th>Fresh baseline</th><th>Agent</th><th>Difference</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'



def current_agent_state(agent,run,mtime=0):
    status=agent.get('status','pending');error=agent.get('error')
    try:resumed=datetime.fromisoformat(run.get('resumed_at','')).timestamp()
    except (ValueError,TypeError):resumed=0
    try:recorded=datetime.fromisoformat(agent.get('updated_at') or agent.get('finished_at') or '').timestamp()
    except (ValueError,TypeError):recorded=mtime
    historical=None
    if status in ('error','running','pending') and run.get('status')=='running' and recorded<resumed:
        status='queued for recovery';historical=error;error=None
    elif status!='error' and error:
        historical=error;error=None
    return status,error,historical



def active_budget(root,run):
    name=run.get('active_configuration') or 'configuration.json'
    if not isinstance(name,str) or Path(name).name!=name or not name.endswith('.json'):name='configuration.json'
    config=read(Path(root)/name)
    revision=run.get('budget_revision')
    revision=revision if isinstance(revision,dict) else {}
    recovery=run.get('transport_recovery',{}).get('recovery',{})
    turns=revision.get('max_turns',recovery.get('effective_turn_limit',config.get('max_turns','unknown')))
    policy=config.get('output_budget_policy') or revision.get('output_budget_policy')
    if policy and ('remaining' in str(policy) or 'minus' in str(policy)):
        output='all remaining context: 262,144 minus exact rendered input tokens'
    else:
        budget=config.get('max_tokens')
        output=(f'{budget:,} initial output tokens' if isinstance(budget,int) else 'output budget not recorded')
        output+='; length-only retry allowances remain separately audited'
    return '<section id="active-budget" class="protocol"><strong>Active agent budget: '+esc(turns)+' total turns</strong><p>Output allowance: '+esc(output)+'. Thinking remains enabled.</p><a href="'+esc(name)+'">Active agent configuration · '+esc(name)+'</a>'+(' · <a href="protocol/budget-240.json">Activated budget revision</a>' if revision else '')+'</section>'



def views_nav(root,feature=None):
    current=read(Path(root)/'observability/inspect/current.json')
    query='?feature='+str(feature) if feature is not None else ''
    fragment='#feature='+str(feature) if feature is not None else ''
    main='#'+str(feature) if feature is not None else ''
    body='<nav class="views-nav"><a href="live/'+query+'">Live agent board</a> · <a href="trajectories/'+fragment+'">Trajectory explorer</a> · <a href="http://100.76.58.1:8765/'+main+'">Main MOLT dashboard</a>'
    for name,label in [('agent','Latest agent Inspect log'),('baseline','Latest baseline Inspect log')]:
        value=current.get('logs',{}).get(name,{}).get('inspect_url')
        url=value if isinstance(value,str) and value.startswith(('http://','https://')) else 'http://100.76.58.1:8770/'
        body+=' · <a data-inspect-log="'+name+'" href="'+esc(url)+'">'+(label if value else 'Inspect View · '+name)+'</a>'
    return body+'</nav>'

def inspect_links_script():
    return '''<script>
async function updateInspectLinks(){try{const response=await fetch('observability/inspect/current.json?latest='+Date.now(),{cache:'no-store'});if(!response.ok)return;const data=await response.json();for(const link of document.querySelectorAll('[data-inspect-log]')){const name=link.dataset.inspectLog,value=data.logs?.[name]?.inspect_url;if(typeof value!=='string')continue;const url=new URL(value,location.href);if(!['http:','https:'].includes(url.protocol))continue;link.href=url.href;link.textContent='Latest '+name+' Inspect log';}}catch(e){}}
if(location.protocol!=='file:'){updateInspectLinks();setInterval(updateInspectLinks,60000);}
</script>'''

def feature_status_script(feature):
    return '<script>const featureID='+json.dumps(str(feature))+''';
async function updateFeatureStatus(){
 try{const response=await fetch('monitor.json?feature_status='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const m=await response.json();const row=(m.feature_state||{})[featureID];if(!row)return;
 const status=row.agent_status==='queued_for_recovery'?'queued for recovery':row.agent_status;
 document.getElementById('feature-current-status').textContent='Current agent status: '+status;
 const error=document.getElementById('feature-current-error');error.textContent=row.error||'';error.hidden=!row.error;
 document.getElementById('feature-status-updated').textContent='Monitor checked '+m.heartbeat_at+(Date.now()/1000-m.heartbeat_unix>180?' · heartbeat stale':'');
 const activeResponse=await fetch('index.html?active_budget='+Date.now(),{cache:'no-store'});if(activeResponse.ok){const activeDoc=new DOMParser().parseFromString(await activeResponse.text(),'text/html'),budget=activeDoc.getElementById('active-budget');if(budget)document.getElementById('active-budget').replaceChildren(...Array.from(budget.childNodes));}
 const history=document.getElementById('historical-attempt');if(history&&status!=='queued for recovery')history.querySelector('summary').textContent='Historical attempt error · preserved for audit';
 }catch(e){document.getElementById('feature-status-updated').textContent='Live status unavailable: '+e.message;}
}
if(location.protocol!=='file:'){updateFeatureStatus();setInterval(updateFeatureStatus,15000);}
</script>'''


def render(root):
    root=Path(root);selection=read(root/'selection.json');run=read(root/'run.json')
    state=read(root/'.corrected-render-state.json');cache=state.get('features',{})
    rows=[];counts={'baseline':0,'agent':0,'scored':0};updated={}
    for selected in selection.get('selected',[]):
        feature=str(selected['id'])
        paths=[root/'baseline/features'/f'{feature}.json',root/'agents'/f'{feature}.json',root/'scored-features'/f'{feature}.json']
        input_files=sorted((root/'agent-inputs'/feature).glob('*.json'),key=lambda p:p.stat().st_mtime_ns)
        signature=[(p.stat().st_mtime_ns,p.stat().st_size) if p.exists() else None for p in paths]+[(p.name,p.stat().st_mtime_ns,p.stat().st_size) for p in input_files]
        key=hashlib.sha256(json.dumps(['live-status-v5',run.get('resumed_at'),run.get('status'),run.get('active_configuration'),run.get('budget_revision'),signature]).encode()).hexdigest()
        old=cache.get(feature,{})
        if old.get('signature')==key and (root/'features'/f'{feature}.html').exists():
            meta=old
        else:
            baseline,agent,scored=map(read,paths)
            bmethods=scored.get('refreshed_baseline') or baseline.get('methods',{})
            amethods=scored.get('methods',{})
            status,error,historical=current_agent_state(agent,run,paths[1].stat().st_mtime if paths[1].exists() else 0);explanation=agent.get('explanation') or scored.get('explanation')
            error=error or scored.get('error') or baseline.get('error')
            body=f'<p><a href="index.html">← Corrected experiment</a></p><h1>Feature {esc(feature)}</h1><p id="feature-current-status"><strong>Current agent status: {esc(status)}</strong></p><p id="feature-status-updated">Saved report status; live updates appear here.</p>'
            body+=f'<p id="feature-current-error" class="error"{"" if error else " hidden"}>{esc(error or "")}</p>'
            if historical:body+='<details id="historical-attempt"><summary>Previous attempt failed · queued for recovery</summary><pre><span>'+esc(historical)+'</span></pre><p>This is a historical error, not a new recovery failure.</p></details>'
            body+='<p><strong>Baseline '+('available' if baseline.get('explanation') else 'pending')+' · '+('Agent comparison available' if amethods else 'No completed agent comparison yet')+'</strong></p>'
            body+=views_nav(root,feature)
            body+=active_budget(root,run)
            body+='<div class="explanations"><section><h2>Fresh native-prompt baseline</h2>'+markdown(baseline.get('explanation') or 'Pending')+'</section><section><h2>Agent explanation</h2>'+markdown(explanation or 'Pending')+'</section></div>'+comparison(bmethods,amethods)
            body+=f'<p><a href="agent-inputs/{esc(feature)}/">Exact rendered agent inputs and token IDs</a> · <a href="baseline/evidence/{esc(feature)}.json">Frozen evidence</a> · <a href="baseline/audit/">Baseline requests and exact-input artifacts</a> · <a href="scoring-audit/">Scoring audit</a> · <a href="training-scores/{esc(feature)}/">Training diagnostic audit</a></p>'
            if paths[0].exists():body+=f'<p><a href="baseline/features/{esc(feature)}.json">Full baseline JSON</a></p>'
            if paths[2].exists():body+=f'<p><a href="scored-features/{esc(feature)}.json">Full comparison JSON</a></p>'
            if input_files:
                body+='<details><summary>Exact agent inputs · '+str(len(input_files))+' captured request(s), chronological</summary><ol>'
                for input_file in input_files:
                    prefix='agent-inputs/'+feature+'/'+input_file.stem
                    body+='<li><a href="'+esc(prefix)+'.txt">Full rendered input text</a> · <a href="'+esc(prefix)+'.json">Token IDs, request body and provenance</a> · '+esc(input_file.stem[:12])+'</li>'
                body+='</ol></details>'
            if agent:
                agent.setdefault('id',int(feature));transcript_html=transcript(agent,root,embed_raw=False)
                if status=='queued for recovery':transcript_html=transcript_html.replace('Readable full agent transcript','Prior-attempt transcript · recovery pending',1)
                body+=transcript_html
            else:body+='<p>No agent transcript recorded yet.</p>'
            body+=feature_status_script(feature)+inspect_links_script()
            write(root/'features'/f'{feature}.html',document('Feature '+feature,body,base='../'))
            meta=dict(signature=key,status=status,baseline=bool(baseline.get('explanation')),agent=bool(explanation),scored=bool(amethods),error=error,
                      cells=[(score_cell(bmethods,m),score_cell(amethods,m),delta(metric(amethods,m)-metric(bmethods,m)) if m!='intruder' and number(metric(amethods,m)) and number(metric(bmethods,m)) else '—') for m in METHODS])
        updated[feature]=meta
        for name in counts:counts[name]+=int(meta.get(name,False))
        display_status=meta['status'];display_error=meta.get('error')
        cells=''.join('<td>'+b+' → '+a+'<small>'+('shared auxiliary' if name=='intruder' else d)+'</small></td>' for name,(b,a,d) in zip(METHODS,meta['cells']))
        rows.append(f'<tr data-feature="{esc(feature)}"><th><a href="features/{esc(feature)}.html">{esc(feature)}</a><small>rank {esc(selected.get("rank","—"))}</small></th><td>{esc(selected.get("cohort","—"))}</td><td class="current-status">{esc(display_status)}'+(f'<small class="error">{esc(display_error)}</small>' if display_error else '')+'</td>'+cells+'</tr>')
    body=f'<h1>Corrected Delphi · {len(selection.get("selected",[])):,}-feature comparison</h1><p>Fresh baseline and agent explanations on the same frozen evidence and held-out cases.</p>'
    if (root/'coverage.json').exists():
        body+='<p><a href="coverage.html">Entire layer coverage, including pilot and unavailable features</a> · <a href="queue.json">Persistent queue status</a></p>'
    body+=views_nav(root)
    monitor=read(root/'monitor.json')
    monitor_text=('Monitor: '+monitor.get('health','not initialized')+' · heartbeat '+monitor.get('heartbeat_at','not yet recorded')+' · '+' '.join(monitor.get('alerts',[])))
    body+='<section id="monitor-banner" role="status" class="protocol"><strong id="monitor-status">'+esc(monitor_text)+'</strong><p id="monitor-detail">Local monitoring only; no automatic job restart or external notifications.</p><p><a href="monitor.json">Monitor state</a> · <a href="monitor-events.jsonl">Durable event history</a></p><details id="monitor-errors"><summary>Current feature errors</summary><pre id="monitor-error-text"><span>'+esc(json.dumps(monitor.get('errors',[]),indent=2))+'</span></pre></details></section>'
    body+=f'<p id="availability"><strong>{counts["baseline"]} baselines available · {counts["scored"]} completed agent comparisons</strong></p>'
    body+=f'<p id="run-progress"><strong>{esc(run.get("phase",run.get("status","Awaiting run metadata")))}</strong> · {counts["baseline"]} baseline explanations · {counts["agent"]} agent explanations · {counts["scored"]} comparisons available · Updated {esc(datetime.now(timezone.utc).isoformat(timespec="seconds"))}</p>'
    body+=active_budget(root,run)
    body+='<div class="protocol"><strong>Declared protocol</strong><p>128-token aligned windows; window boundaries are independent of activation position. Native Delphi prompts and parsing, with an explicit uniform incorrect-highlight position repair. Original full-conversation activation cache retained. Cohorts are original historical selection strata, not current scores.</p><p>Baseline and agent: Qwen thinking enabled, xhigh; temperature 0.6. Detection/fuzz/active_fuzz: thinking disabled, greedy, 128 output tokens. Intruder: thinking enabled, medium, up to 8192 tokens within its context limit. Intruder is an explanation-independent shared auxiliary check. Scores show valid/total coverage; missing values are not zero.</p><p>Tool messages shown to the agent are a readable projection: text and marked spans replace large token/gate arrays. Exact scorer shown_text remains unchanged (detection is unmarked). Complete original MCP returns stay in raw agent events; exact delivered messages and rendered input/token IDs are saved separately.</p></div>'
    body+='<p><a href="run.json">Run status JSON</a> · <a href="baseline/manifest.json">Evidence manifest</a> · <a href="http://100.76.58.1:8766/protocol-audit/explainer-findings.json">Prior-run protocol audit</a></p>'
    if (root/'trajectories/index.html').exists():body+='<p><a href="trajectories/index.html">Full trajectory explorer</a></p>'
    if run.get('errors'):body+='<details open><summary>Recorded run errors</summary><pre><span>'+pretty(run['errors'])+'</span></pre></details>'
    body+='<label>Find feature, cohort, or status <input id="filter" type="search" placeholder="Filter rows"></label><div class="scroll"><table id="features"><thead><tr><th>Feature</th><th>Original cohort</th><th>Agent status</th>'+''.join('<th>'+esc(m)+'<small>baseline → agent</small></th>' for m in METHODS)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
    body+='<script>function applyFilter(){let q=document.getElementById("filter").value.toLowerCase();document.querySelectorAll("#features tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}document.getElementById("filter").addEventListener("input",applyFilter)</script>'
    body+='''<script>
async function updateMonitor(){
 const status=document.getElementById('monitor-status'),detail=document.getElementById('monitor-detail');
 try {const response=await fetch('monitor.json?heartbeat='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const m=await response.json();
 const stale=Date.now()/1000-m.heartbeat_unix>180;
 status.textContent=(stale?'MONITOR HEARTBEAT STALE':String(m.health).toUpperCase())+' · last check '+m.heartbeat_at;
 const c=m.counts||{};
 detail.textContent=(m.alerts||[]).join(' ')+' Baselines '+c.baseline_complete+'/'+c.selected+'; agents '+c.agent_complete+'/'+c.selected+'; scored '+c.scored+'/'+c.selected+'; queued for recovery '+(c.queued_recovery||0)+'. Last saved activity '+(m.last_activity_at||'not recorded')+'. Local visibility only: no automatic restart or external notification.';
 document.getElementById('availability').textContent=c.baseline_complete+' baselines available · '+c.scored+' completed agent comparisons';
 for(const row of document.querySelectorAll('#features tbody tr')){const state=(m.feature_state||{})[row.dataset.feature];if(!state)continue;const cell=row.querySelector('.current-status');cell.textContent=state.agent_status==='queued_for_recovery'?'queued for recovery':state.agent_status;if(state.error){const small=document.createElement('small');small.className='error';small.textContent=state.error;cell.appendChild(small);}}
 applyFilter();
 document.getElementById('monitor-banner').style.borderColor=stale||['failed','stopped','no_progress','degraded'].includes(m.health)?'#d55a35':'#4685b8';
 document.getElementById('monitor-error-text').textContent=JSON.stringify(m.errors||[],null,2);
 document.getElementById('monitor-errors').open=['failed','stopped'].includes(m.health)&&Boolean((m.errors||[]).length);
 }catch(e){status.textContent='MONITOR UNAVAILABLE · '+e.message;detail.textContent='Cannot verify the current heartbeat. Refresh or inspect the local monitor service.';}
}
async function refreshDashboard(){
 const x=window.scrollX,y=window.scrollY;
 try{const response=await fetch('index.html?live_rows='+Date.now(),{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);const next=new DOMParser().parseFromString(await response.text(),'text/html');
 const body=next.querySelector('#features tbody');if(body)document.querySelector('#features tbody').replaceChildren(...Array.from(body.children));
 for(const id of ['availability','run-progress','active-budget']){const source=next.getElementById(id);if(source)document.getElementById(id).replaceChildren(...Array.from(source.childNodes));}
 applyFilter();window.scrollTo(x,y);
 }catch(e){document.getElementById('run-progress').textContent='Live table refresh unavailable: '+e.message;}
 await updateMonitor();window.scrollTo(x,y);
}
if(location.protocol!=='file:'){updateMonitor();setInterval(refreshDashboard,15000);}
</script>'''
    body+=inspect_links_script()
    write(root/'index.html',document('Corrected Delphi comparison',body))
    write(root/'.corrected-render-state.json',json.dumps(dict(features=updated)))
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    print(json.dumps(render(p.parse_args().root)))
