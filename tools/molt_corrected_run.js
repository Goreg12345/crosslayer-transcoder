/* Separate, live corrected-protocol results; never modifies historical records. */
(()=>{
 const host=document.getElementById('workspace');if(!host || (typeof META!=='undefined' && META.layer!==22))return;
 const style=document.createElement('style');style.textContent='.corrected-panel{padding:22px 28px;border-bottom:2px solid #158d79;background:#edf8f5}.corrected-arms{display:grid;grid-template-columns:1fr 1fr;gap:16px}.corrected-arm{padding:16px;background:white;border:1px solid #c7d3df;border-radius:8px;min-width:0}.corrected-arm p{white-space:pre-wrap;overflow-wrap:anywhere;color:#253f4a}.corrected-panel select{max-width:280px;margin:8px 14px 8px 0}.corrected-arm table{width:100%;font-size:13px;text-align:left;border-collapse:collapse}.corrected-arm td,.corrected-arm th{padding:6px;border-bottom:1px solid #edf2f5}.corrected-links{display:flex;gap:18px;flex-wrap:wrap}.corrected-warning{color:#895c14}@media(max-width:800px){.corrected-arms{grid-template-columns:1fr}}';document.head.append(style);
 const panel=document.createElement('section');panel.className='corrected-panel';panel.setAttribute('aria-label','Live corrected agentic experiment');host.prepend(panel);
 const historical=document.getElementById('interpretation');if(historical){const label=document.createElement('p');label.className='badge';label.textContent='Historical dashboard explanation and scores · separate protocol';historical.prepend(label)}
 let data=null,lastFeature=null,error='';
 const node=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e};
 const report=new URL(location.href);report.port='8771';report.pathname='/';report.search='';report.hash='';
 function link(text,path){const e=node('a',text);e.href=new URL(path,data?.report_url||report).href;e.target='_blank';e.rel='noopener';return e}
 const names={detection:'Detection',fuzz:'Fuzzing',active_fuzz:'Active fuzzing',intruder:'Intruder'};
 function arm(title,value){const card=node('section',undefined,'corrected-arm');card.append(node('h3',title),node('span',value.status||'pending','status'));card.append(node('p',value.explanation||'Explanation pending. Intermediate drafts are not shown as final explanations.'));
  const table=node('table');const heading=node('tr');heading.append(node('th','Held-out score'),node('th','Score / valid cases'));table.append(heading);
  for(const [key,label] of Object.entries(names)){const m=value.methods?.[key],v=m?.metrics;const row=node('tr');let text=m?.status||'pending';if(Number.isFinite(v?.accuracy)&&v.valid>0)text=`${(v.accuracy*100).toFixed(1)}% · ${v.valid}/${v.total} cases${m.status==='complete'?'':' · '+(m.status||'partial')}`;row.append(node('td',label),node('td',text));table.append(row)}card.append(table);return card}
 function render(){const id=location.hash.slice(1);lastFeature=id;panel.replaceChildren(node('div',`LIVE CORRECTED EXPERIMENT${data?' · '+data.counts.selected+' FEATURES':''}`,'badge'),node('h2','Baseline vs agent · aligned 128-token contexts'));
  if(!data){panel.append(node('p',error||'Loading corrected-run progress…'));return}
  const c=data.counts;panel.append(node('p',`${data.status} · baseline explanations ${c.baseline_explanations}/${c.selected} · agent explanations ${c.agent_explanations}/${c.selected} · matched scores ${c.matched_scores}/${c.selected} · active agents ${c.active_agents}`));
  panel.append(node('p',`Run: ${data.run_id}${data.concurrency?' · '+data.concurrency+' agent slots':''}`,'note'));
  const updated=new Date(data.published_at);panel.append(node('p',`Updates every 30 seconds · published ${updated.toLocaleString()}${Date.now()-updated.getTime()>120000?' · publisher snapshot may be stale':''}`,'note'));
  if(error)panel.append(node('p',error,'corrected-warning'));
  const select=node('select');select.setAttribute('aria-label','Choose corrected-run feature');const option=node('option',`Choose one of the ${c.selected} corrected-run features`);option.value='';select.append(option);for(const f of Object.values(data.features)){const o=node('option',`Feature ${f.id} · ${f.agent.status}`);o.value=f.id;select.append(o)}select.value=data.features[id]?id:'';select.onchange=()=>{if(select.value)location.hash=select.value};panel.append(select);
  const links=node('div',undefined,'corrected-links');links.append(link('Full corrected-run comparison','index.html'),link('Agent trajectories','trajectories/'));panel.append(links);
  const f=data.features[id];if(!f){panel.append(node('p','This feature is not eligible for the current comparison. See the coverage report for its evidence counts; historical results remain below.'));links.append(link('Layer coverage','coverage.html'));return}
  panel.append(node('h3',`Feature ${id} · ${f.stage||f.agent.status} · ${f.agent.model_responses} model responses · ${f.agent.tool_calls} tool calls`));
  if(f.agent.last_tool)panel.append(node('p',`Latest completed tool: ${f.agent.last_tool}${f.agent.updated_at?' · checkpoint '+new Date(f.agent.updated_at).toLocaleString():''}`,'note'));
  if(f.agent.error)panel.append(node('p',String(f.agent.error),'corrected-warning'));
  links.append(link(`Feature ${id}: evidence and trajectory`,`features/${id}.html`));
  const arms=node('div',undefined,'corrected-arms');arms.append(arm('Fresh baseline · same corrected evidence',f.baseline),arm('Agentic explanation',f.agent));panel.append(arms);
  panel.append(node('p',`${data.profile.context_tokens}-token windows; ${data.profile.n_train} initial examples and ${data.profile.n_test_per_class} held-out examples per class. Detection/fuzzing scores are balanced accuracy; intruder is accuracy. Training feedback is not a held-out score. Historical scores below use a different protocol.${data.profile.same_cache_as_dashboard?'':' Corrected results use the 10M-token campaign cache, separate from the activation sample below.'}`,'note'));
 }
 async function refresh(){try{const r=await fetch('corrected-run.json',{cache:'no-store'});if(!r.ok)throw Error('Corrected-run publisher unavailable; retaining last snapshot.');const next=await r.json();if(next.version!==1||!next.features||!next.counts)throw Error('Invalid corrected-run snapshot');data=next;error=''}catch(e){error=e.message}render()}
 window.addEventListener('hashchange',render);setInterval(()=>{if(lastFeature!==location.hash.slice(1))render()},300);refresh();setInterval(refresh,30000);
})();
