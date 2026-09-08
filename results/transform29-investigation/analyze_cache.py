import json,re
from pathlib import Path
import numpy as np
import sys
P=Path(sys.argv[1]) if len(sys.argv)>1 else Path('results/molt-qwen-dashboard'); O=Path(sys.argv[2]) if len(sys.argv)>2 else Path('results/transform29-investigation')
a=np.load(P/'activations.npz');v=json.loads((P/'vocab.json').read_text());tokens=a['tokens'];texts=np.array([v[str(t)] for t in tokens]);g=np.zeros(len(tokens));sel=a['features']==29;g[a['positions'][sel]]=a['values'][sel];active=g>0
roles=np.full(len(g),'outside',dtype=object);dist=np.full(len(g),-1);sentdist=dist.copy();conv=dist.copy()
for c,(s,e) in enumerate(zip(a['offsets'][:-1],a['offsets'][1:])):
 role='outside';header=False;d=-1;sd=-1
 for i in range(s,e):
  t=texts[i];conv[i]=c
  if t=='<|im_start|>':header=True;role='header';d=-1
  elif header and t in ['user','assistant','system']: nextrole=t
  elif header and '\n' in t:header=False;roles[i]='header';role=nextrole;d=0;sd=0;continue
  elif t=='<|im_end|>':role='outside';d=-1
  roles[i]='special' if t.startswith('<|') else ('header' if header else role)
  if role in ['user','assistant','system'] and not header:
   dist[i]=d;sentdist[i]=sd;d+=1;sd+=1
   if re.search(r'[.!?\n]',t):sd=0
content=np.isin(roles,['user','assistant','system']);punct=np.array([bool(t.strip()) and not any(x.isalnum() for x in t) for t in texts]);space=np.array([not t.strip() for t in texts]);alpha=np.array([any(x.isalpha() for x in t) for t in texts])
def stat(mask):
 z=g[mask];return dict(n=len(z),active=int((z>0).sum()),rate=float((z>0).mean()) if len(z) else None,mean=float(z.mean()) if len(z) else None)
r={'groups':{}}
for role in np.unique(roles):r['groups'][role]=stat(roles==role)
for name,mask in [('content',content),('content_alpha',content&alpha),('content_punctuation',content&punct),('content_whitespace',content&space)]:r['groups'][name]=stat(mask)
r['turn_position']={str(i):stat(content&(dist==i)) for i in range(12)}
r['since_boundary']={str(i):stat(content&(sentdist==i)) for i in range(12)}
r['token_stats']=sorted([dict(token=v[str(t)],**stat(tokens==t)) for t in np.unique(tokens) if (tokens==t).sum()>=50],key=lambda x:x['rate'])
r['conversation_rates']=[dict(conversation=int(c),**stat(conv==c)) for c in np.unique(conv)]
r['examples']={}
for name,mask in [('inactive_words',content&alpha&~active),('active_words',content&alpha&active),('active_punctuation',content&punct&active)]:
 ix=np.where(mask)[0];ix=ix[np.random.default_rng(42).choice(len(ix),min(25,len(ix)),replace=False)]
 r['examples'][name]=[dict(conversation=int(conv[i]),position=int(i-a['offsets'][conv[i]]),token=texts[i],gate=g[i],context=''.join(texts[max(a['offsets'][conv[i]],i-10):i])+'【'+texts[i]+'】'+''.join(texts[i+1:min(a['offsets'][conv[i]+1],i+11)])) for i in sorted(ix)]
np.savez_compressed(O/'labels.npz',roles=roles.astype(str),turn_position=dist,boundary_distance=sentdist,gate=g)
(O/'cache_analysis.json').write_text(json.dumps(r,indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in r.items() if k not in ['token_stats','examples','conversation_rates']},indent=2));print('LOW TOKENS',r['token_stats'][:35]);print('HIGH TOKENS',r['token_stats'][-15:]);print('INACTIVE EXAMPLES',r['examples']['inactive_words'][:12])
