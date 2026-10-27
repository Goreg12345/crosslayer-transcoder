"""Dose response via the existing local steering server; saves every trace."""
import json,time,urllib.request,urllib.error,os
from pathlib import Path
O=Path('results/transform29-followup');url='http://100.76.58.1:8765';mean=1.6317405700683594

def api(path,data=None):
 req=urllib.request.Request(url+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as f:return json.load(f)
cases=[('active_explanation','The autumn leaves change color because the trees stop producing chlorophyll and','raw','multiply'),('active_story','Once upon a time, a small dog wandered into a forest and discovered','raw','multiply'),('active_reasoning','To calculate seventeen plus twenty-five, we first add the tens and then','raw','multiply'),('inactive_code','def add(a, b):\n    return','raw','set'),('inactive_count','1 2 3 4 5 6 7 8 9 10','raw','set'),('inactive_repeat',' dog'*12,'raw','set'),('inactive_chat_boundary','What is 17 plus 25? Answer briefly.','chat','set')]
matched=os.getenv('MOLT_SWEEP_BOS','0')=='1'
if matched:cases=[(name,'<|endoftext|>'+prompt,fmt,mode) for name,prompt,fmt,mode in cases if name in ['active_explanation','inactive_code','inactive_count']]
output={'reference_mean_active':mean,'training_bos_prefix':matched,'cases':[]}
for name,prompt,fmt,mode in cases:
 inspect=api('/api/inspect',dict(prompt=prompt,format=fmt,feature=29));record=dict(name=name,prompt=prompt,format=fmt,mode=mode,inspection=inspect,runs=[]);output['cases'].append(record)
 # Prompt intervention is limited to naturally active or inactive positions, respectively.
 selected=[x['position'] for x in inspect['tokens'] if (x['gate']>0 if mode=='multiply' else x['gate']==0)]
 if name=='inactive_chat_boundary':selected=[len(inspect['ids'])-1]
 doses=([0,.5,1,2] if matched else [0,.25,.5,1,2]) if mode=='multiply' else ([0,.5,1,2,4] if matched else [0,.25,.5,1,2,4])
 for dose in doses:
  amount=dose if mode=='multiply' else dose*mean
  body=dict(prompt=prompt,format=fmt,feature=29,mode=mode,strength_reference='raw',tokenization_id=inspect['tokenization_id'],assignments=[dict(position=i,value=amount) for i in selected],generated=None if name=='inactive_chat_boundary' else dict(start=0,end=63,value=amount),max_new_tokens=64,temperature=0,top_p=1,seed=42)
  job=api('/api/generate',body)
  while True:
   r=api('/api/jobs/'+job['id'])
   if r['state'] in ['done','error','cancelled']:break
   time.sleep(.5)
  if r['state']!='done':raise RuntimeError(r)
  record['runs'].append(dict(dose=dose,absolute_amount=amount,result=r['result']));(O/('bos_steering_sweep.json' if matched else 'steering_sweep.json')).write_text(json.dumps(output,indent=2,ensure_ascii=False));print(name,dose,r['result']['steered']['text'][:170].replace('\n',' | '),flush=True)
print('DONE',flush=True)
