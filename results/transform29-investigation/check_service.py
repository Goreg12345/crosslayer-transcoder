import json,urllib.request,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
O=Path('results/transform29-investigation');base='http://100.76.58.1:8766'
def request(path,data=None):
 req=urllib.request.Request(base+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)
t=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B');a=np.load('results/molt-qwen-dashboard/activations.npz');checks=[]
for c in [0,2,25,55]:
 s,en=a['offsets'][c:c+2];ids=a['tokens'][s:en].tolist();r=request('/api/inspect',dict(prompt=t.decode(ids),format='raw',feature=29));assert r['ids']==ids
 g=np.array([x['gate'] for x in r['tokens']]);cache=np.zeros(len(ids));sel=(a['features']==29)&(a['positions']>=s)&(a['positions']<en);cache[a['positions'][sel]-s]=a['values'][sel]
 checks.append(dict(conversation=c,n=len(ids),mean_abs_gate_difference=float(np.abs(g-cache).mean()),max_abs_gate_difference=float(np.abs(g-cache).max()),binary_disagreements=int(((g>0)!=(cache>0)).sum())))
prompt='Explain why the sky is blue in one sentence.';tok=request('/api/tokenize',dict(prompt=prompt,format='chat'));body=dict(prompt=prompt,format='chat',feature=29,mode='multiply',tokenization_id=tok['tokenization_id'],assignments=[dict(position=i,value=1) for i in range(len(tok['ids']))],generated=dict(start=0,end=7,value=1),max_new_tokens=8,temperature=0,top_p=1,seed=42)
job=request('/api/generate',body)
while True:
 result=request('/api/jobs/'+job['id'])
 if result['state'] in ['done','error','cancelled']:break
 time.sleep(.5)
(O/'numerical_checks.json').write_text(json.dumps(dict(cache_checks=checks,noop=result),indent=2));print(checks);print(str(result)[-1000:])
