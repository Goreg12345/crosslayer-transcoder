"""Post-hoc SVD rank caps for transform 29, retaining its learned scalar gate."""
import json,torch
from pathlib import Path
from crosslayer_transcoder.dashboard.gates import MoltGates
O=Path('results/transform29-followup');torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');dev='cuda:0';meta=json.load(open('results/molt-qwen-dashboard/metadata.json'));s=torch.load(meta['checkpoint'],mmap=True,map_location='cpu',weights_only=False)['state_dict'];data=torch.load(O/'bos_matched/xy.pt',weights_only=False);gates=MoltGates(s,22).to(dev).eval()
@torch.inference_mode()
def main():
 # SVD of VU through a 512x512 core; caps the learned linear transform itself.
 v=s['model.Vs.0'][29].to(dev).float();u=s['model.Us.0'][29].to(dev).float();qv,rv=torch.linalg.qr(v,mode='reduced');qu,ru=torch.linalg.qr(u.T,mode='reduced');left,sv,right=torch.linalg.svd(rv@ru.T,full_matrices=False);basis=qu@right.T;del qv,rv,qu,ru,left
 caps=[0,1,8,32,64,128,256,512];totals={k:0. for k in caps};energy={k:0. for k in caps};n=0;outmean=s['model.output_standardizer.mean'][22].to(dev).float();std=s['model.output_standardizer.std'][22].to(dev).float();checks=[]
 for ci in range(8):
  lo,hi=data['offsets'][ci:ci+2];x=data['x'][lo:hi].to(dev);yn=(data['y'][lo:hi].to(dev).float()-outmean)/std;xn=gates.standardizer(x,22).float()
  with torch.autocast('cuda',dtype=torch.bfloat16):gg=gates(x)
  pred=torch.zeros_like(xn);c29=((xn@v)@u)*gg[:,29,None].float();offset=0
  for gi in range(5):
   vv=s[f'model.Vs.{gi}'].to(dev).float();uu=s[f'model.Us.{gi}'].to(dev).float()
   for f in torch.where((gg[:,offset:offset+len(vv)]>0).any(0))[0].tolist():
    pos=torch.where(gg[:,offset+f]>0)[0];c=((xn[pos]@vv[f])@uu[f])*gg[pos,offset+f,None].float();pred.index_add_(0,pos,c)
   offset+=len(vv);del vv,uu
  for k in caps:
   approx=(c29@basis[:,:k])@basis[:,:k].T if k else torch.zeros_like(c29);totals[k]+=(pred-c29+approx-yn).square().sum().item();energy[k]+=approx.square().sum().item()
  checks.append((pred-yn).square().sum().item());n+=len(x);print('RANKCAP',ci,flush=True)
 full=sum(checks)/(n*x.shape[1]);r=dict(n=n,conversations=data['conversations'][:8],method='Truncate SVD of learned V29 U29 in standardized output coordinates; preserve gate; no retraining',baseline_mse=full,full_rank_replay_error=totals[512]/(n*x.shape[1])-full,caps=[dict(rank=k,mse=totals[k]/(n*x.shape[1]),mse_increase=totals[k]/(n*x.shape[1])-full,retained_output_energy=energy[k]/energy[512]) for k in caps]);(O/'rank_cap.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
main()
