"""Reproducible role, content, and error-preserving causal probes of transform 29."""
import json,threading,os
from pathlib import Path
import numpy as np
import torch
from crosslayer_transcoder.dashboard.steering import SteeringEngine
O=Path('results/transform29-investigation');torch.set_num_threads(4)
e=SteeringEngine('results/molt-qwen-dashboard',device='cuda:0'); t=e.tokenizer
print('LOADED',flush=True)
r={'checkpoint':{k:v for k,v in e.meta.items() if k!='ranks'},'probes':[],'causal':[],'generations':[]}
def save(): (O/'experiments.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
texts={
'prose':'The small brown dog walked through the quiet garden and settled beneath the old apple tree. The afternoon sunlight warmed the grass around him.',
'question':'How does the author propose to fix the problem of science alienation in our educational system? What changes should be made to science education?',
'math':'1234567890 9876543210 123 + 456 = 579. 2 * 3 = 6. The sum of two and three is five.',
'code':'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n',
'word_salad':'garden science purple because author walked education beside strange computer the dog improve under sunlight and economic tree.',
'repeated':' dog'*35,
'chinese':'一只棕色的小狗穿过安静的花园，在老苹果树下休息。下午的阳光温暖着它周围的草地。',
'french':'Le petit chien brun traversait le jardin tranquille et se reposait sous le vieux pommier. Le soleil réchauffait les fleurs autour de lui.',
'punctuation':'The small brown dog walked through the quiet garden. The small brown dog walked through the quiet garden! The small brown dog walked through the quiet garden?\nThe small brown dog walked through the quiet garden',
'list':'dog\ncat\nbird\nhorse\nfish\nmouse\nThe dog and the cat walked through the garden together.',
}
for name,text in texts.items():
 for role in (['raw','user','assistant','system'] if name in ['prose','question'] else ['user']):
  prompt=text if role=='raw' else f'<|im_start|>{role}\n{text}<|im_end|>\n'
  data=e.inspect(prompt,'raw',29)
  start=0 if role=='raw' else 3;end=len(data['tokens']) if role=='raw' else len(data['tokens'])-2
  z=[x['gate'] for x in data['tokens'][start:end]]
  row=dict(name=name,role=role,rate=float(np.mean(np.array(z)>0)),mean=float(np.mean(z)),tokens=data['tokens']);r['probes'].append(row)
  print('PROBE',name,role,round(row['rate'],3),flush=True)
 save()
# Fixed-sequence interventions: downstream response included, reconstruction error held fixed.
a=np.load('results/molt-qwen-dashboard/activations.npz'); labels=np.load(O/'labels.npz');v,u,std=e.feature_weights(29)
@torch.inference_mode()
def forward(ids,feature=29,mult=None,capture=False):
 batch=torch.tensor([ids],device=e.device);stored={};handles=[]
 def grab(_m,args):stored['x']=args[0][0]
 def modify(_m,args,out):
  x=stored['x'];z=e.natural_gate(x,feature);vv,uu,ss=(v,u,std) if feature==29 else e.feature_weights(feature)
  norm=e.gates.standardizer(x,e.layer).float();delta=((norm@vv)@uu)*((mult-1)*z[:,None])*ss
  stored['delta_norm']=delta.norm(dim=-1).cpu().tolist();stored['mlp_norm']=out[0].float().norm(dim=-1).cpu().tolist()
  return (out.float()+delta[None]).to(out.dtype)
 layer=e.model.model.layers[e.layer];handles.append(layer.post_attention_layernorm.register_forward_pre_hook(grab))
 if mult is not None:handles.append(layer.mlp.register_forward_hook(modify))
 try:
  hidden=e.model.model(input_ids=batch,use_cache=False).last_hidden_state[0]
  # CPU storage and chunks keep the vocabulary readout's GPU footprint bounded.
  logits=torch.cat([e.model.lm_head(h).float().cpu() for h in hidden.split(64)])
  if capture:stored['gate']=e.natural_gate(stored['x'],29).cpu().numpy();stored['x']=stored['x'].cpu()
  else:stored.pop('x',None)
  return logits,stored
 finally:
  for h in handles:h.remove()
for c in [0,2,5,9,16,25,40,55]:
 s,en=a['offsets'][c:c+2];en=min(en,s+384);ids=a['tokens'][s:en].tolist()
 base,st=forward(ids,capture=True);cached=labels['gate'][s:en];print('CACHE CHECK',c,float(np.max(np.abs(st['gate']-cached))),flush=True)
 p=base.log_softmax(-1);prob=p.exp();target=torch.tensor(ids[1:]);nat=st['gate']
 for mult in [0,2]:
  changed,inter=forward(ids,mult=mult);q=changed.log_softmax(-1);kl=(prob*(p-q)).sum(-1).numpy();flip=(base.argmax(-1)!=changed.argmax(-1)).numpy();lossdelta=(p[:-1].gather(1,target[:,None])-q[:-1].gather(1,target[:,None])).squeeze().numpy()
  groups={}
  for name,mask in [('all',np.ones(len(ids),bool)),('active',nat>0),('inactive',nat==0),('user',labels['roles'][s:en]=='user'),('assistant',labels['roles'][s:en]=='assistant')]:
   groups[name]=dict(n=int(mask.sum()),kl_sum=float(kl[mask].sum()),flips=int(flip[mask].sum()),nll_n=int(mask[:-1].sum()),nll_delta_sum=float(lossdelta[mask[:-1]].sum()))
  changes=[]
  for j in np.argsort(-kl)[:8]:
   changes.append(dict(position=int(j),token=t.decode([ids[j]]),context=t.decode(ids[max(0,j-10):j+1]),gate=float(nat[j]),kl=float(kl[j]),baseline_next=t.decode([base[j].argmax().item()]),changed_next=t.decode([changed[j].argmax().item()])))
  r['causal'].append(dict(conversation=c,multiplier=mult,groups=groups,largest_changes=changes,mean_delta_norm=float(np.mean(inter['delta_norm'])),mean_mlp_norm=float(np.mean(inter['mlp_norm'])),cache_max_error=float(np.max(np.abs(nat-cached)))))
  print('CAUSAL',c,mult,groups['all'],flush=True);save()
# Greedy completions with all consumed prompt and generated tokens intervened on.
for prompt in ['Explain why leaves change color in autumn in three sentences.','What is 17 plus 25? Answer briefly.','Write a short story about a dog finding its way home.','Write a Python function that returns the largest number in a list.']:
 tok=e.tokenize(prompt,'chat');cfg=dict(ids=tok['ids'],feature=29,mode='multiply',assignments={i:0 for i in range(len(tok['ids']))},generated={'start':0,'end':63,'value':0},max_new_tokens=64,temperature=0,top_p=1,seed=42)
 b,_=e.generate(cfg,threading.Event(),lambda *args:None,'baseline');off,_=e.generate(cfg,threading.Event(),lambda *args:None,'steered')
 r['generations'].append(dict(prompt=prompt,baseline=b,ablated=off));print('GEN',prompt,b['text'],off['text'],flush=True);save()
print('DONE',flush=True)
