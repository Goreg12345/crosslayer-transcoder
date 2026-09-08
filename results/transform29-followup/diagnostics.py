import json,torch,numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from crosslayer_transcoder.dashboard.gates import MoltGates
O=Path('results/transform29-followup');torch.set_num_threads(4);device='cuda:0';meta=json.load(open('results/molt-qwen-dashboard/metadata.json'));s=torch.load(meta['checkpoint'],mmap=True,map_location='cpu',weights_only=False)['state_dict'];layer=meta['layer'];data=torch.load(O/'xy.pt',weights_only=False);t=AutoTokenizer.from_pretrained(meta['model']);gates=MoltGates(s,layer).to(device).eval();mean=s['model.output_standardizer.mean'][layer].to(device);std=s['model.output_standardizer.std'][layer].to(device);groups={};out=[]
@torch.inference_mode()
def main():
 for ci in range(4):
  lo,hi=data['offsets'][ci:ci+2];x=data['x'][lo:hi].to(device);y=data['y'][lo:hi].to(device);xn=gates.standardizer(x,layer);yn=((y-mean)/std).float()
  with torch.autocast('cuda',dtype=torch.bfloat16):g=gates(x)
  pred=torch.zeros_like(xn,dtype=torch.float32);bf=torch.zeros_like(xn,dtype=torch.float32);offset=0
  for gi in range(5):
   vv=s[f'model.Vs.{gi}'].to(device);uu=s[f'model.Us.{gi}'].to(device)
   for f in torch.where((g[:,offset:offset+len(vv)]>0).any(0))[0].tolist():
    pos=torch.where(g[:,offset+f]>0)[0];c=((xn[pos].float()@vv[f].float())@uu[f].float())*g[pos,offset+f,None].float();pred.index_add_(0,pos,c)
    with torch.autocast('cuda',dtype=torch.bfloat16):cb=((xn[pos]@vv[f])@uu[f])*g[pos,offset+f,None]
    bf.index_add_(0,pos,cb.float())
   offset+=len(vv);del vv,uu
  mse=(pred-yn).square().mean(-1).cpu().numpy();bfmse=(bf-yn).square().mean(-1).cpu().numpy();base=yn.square().mean(-1).cpu().numpy();ids=data['tokens'][lo:hi];txt=[t.decode([z]) for z in ids]
  for name,mask in [('all',np.ones(len(ids),bool)),('first_token',np.arange(len(ids))==0),('after_first_16',np.arange(len(ids))>=16),('special',np.isin(ids,t.all_special_ids)),('letters',np.array([any(c.isalpha() for c in z) for z in txt])),('non_special',~np.isin(ids,t.all_special_ids))]:
   a=groups.setdefault(name,dict(n=0,mse=0.,bf_mse=0.,base=0.));a['n']+=int(mask.sum());a['mse']+=float(mse[mask].sum());a['bf_mse']+=float(bfmse[mask].sum());a['base']+=float(base[mask].sum())
  out.extend(dict(conversation=ci,pos=int(j),token=txt[j],mse=float(mse[j]),base=float(base[j])) for j in np.argsort(-mse)[:8]);print('DIAG',ci,flush=True)
 for v in groups.values():
  for k in ['mse','bf_mse','base']:v[k]/=v['n']
 (O/'diagnostics.json').write_text(json.dumps(dict(groups=groups,largest_errors=out),indent=2));print(groups,flush=True)
main()
