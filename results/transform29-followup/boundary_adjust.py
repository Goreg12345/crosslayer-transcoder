"""Keep boundary failures visible; separately report all-token and interior importance."""
import json,torch,numpy as np
from pathlib import Path
from crosslayer_transcoder.dashboard.gates import MoltGates
O=Path('results/transform29-followup');torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');dev='cuda:0';r=json.load(open(O/'reconstruction.json'));data=torch.load(O/'xy.pt',weights_only=False);s=torch.load(r['checkpoint'],mmap=True,map_location='cpu',weights_only=False)['state_dict'];gates=MoltGates(s,22).to(dev).eval();d=r['d'];n=r['n'];nf=r['feature_count'];rv=[512,256,128,64,32]
@torch.inference_mode()
def main():
 gs=[]
 for lo,hi in zip(data['offsets'][:-1],data['offsets'][1:]):
  with torch.autocast('cuda',dtype=torch.bfloat16):gs.append(gates(data['x'][lo:hi].to(dev))[0].float())
 gg=torch.stack(gs);idx=torch.tensor(data['offsets'][:-1]);x=gates.standardizer(data['x'][idx].to(dev),22).float();mean=s['model.output_standardizer.mean'][22].to(dev).float();std=s['model.output_standardizer.std'][22].to(dev).float();y=(data['y'][idx].to(dev).float()-mean)/std;b=len(x);pred=torch.zeros_like(x);group=torch.zeros((5,b,d),device=dev);cs=[];offset=0
 for gi in range(5):
  vv=s[f'model.Vs.{gi}'].to(dev).float();uu=s[f'model.Us.{gi}'].to(dev).float()
  for f in torch.where((gg[:,offset:offset+len(vv)]>0).any(0))[0].tolist():
   fid=offset+f;pos=torch.where(gg[:,fid]>0)[0];c=((x[pos]@vv[f])@uu[f])*gg[pos,fid,None];pred.index_add_(0,pos,c);group[gi].index_add_(0,pos,c);cs.append((fid,gi,pos,c))
  offset+=len(vv);del vv,uu
 res=pred-y;bs=y.square().sum().item();fs=res.square().sum().item();boundary_features={};boundary_sh=np.zeros((b,5));boundary_ab=[]
 for fid,gi,pos,c in cs:
  en=c.square().sum().item();ab=en-2*(res[pos]*c).sum().item();sh=(c*(2*y[pos]-pred[pos])).sum(-1);boundary_sh[pos.cpu().numpy(),gi]+=sh.cpu().numpy();boundary_features[fid]=dict(n=len(pos),energy=en,ab=ab,sh=sh.sum().item())
 for gi in range(5):boundary_ab.append(((res-group[gi]).square().sum(-1)-res.square().sum(-1)).cpu().numpy())
 den=(n-b)*d;benefit=(r['mean_baseline_mse']-r['standardized_mse'])*n*d-(bs-fs)
 out=dict(n=n-b,excluded_first_tokens=b,standardized_mse=(r['standardized_mse']*n*d-fs)/den,mean_baseline_mse=(r['mean_baseline_mse']*n*d-bs)/den,boundary_mse=fs/b/d,boundary_baseline_mse=bs/b/d,boundary_error_fraction=fs/(r['standardized_mse']*n*d),groups=[],features=[],per_conversation=[])
 for old in r['features']:
  f=old['id'];bf=boundary_features.get(f,dict(n=0,energy=0,ab=0,sh=0));new=dict(old);new.update(frequency=(old['active_count']-bf['n'])/(n-b),active_count=old['active_count']-bf['n'],ablation_mse_increase=(old['ablation_mse_increase']*n*d-bf['ab'])/den,additive_benefit_share=(old['additive_benefit_share']*(r['mean_baseline_mse']-r['standardized_mse'])*n*d-bf['sh'])/benefit,output_energy=(old['output_energy']*n*d-bf['energy'])/den);out['features'].append(new)
 for gi,old in enumerate(r['groups']):
  new=dict(old);new['shapley_share']=sum(f['additive_benefit_share'] for f in out['features'] if f['rank']==old['rank']);new['ablation_mse_increase']=(old['ablation_mse_increase']*n*d-boundary_ab[gi].sum())/den;new['only_group_mse']=(old['only_group_mse']*n*d-(group[gi]-y).square().sum().item())/den;new['cumulative_largest_first_mse']=(old['cumulative_largest_first_mse']*n*d-(group[:gi+1].sum(0)-y).square().sum().item())/den;new['frequency_sum']=sum(f['frequency'] for f in out['features'] if f['rank']==old['rank']);new['active_rank']=new['frequency_sum']*old['rank'];out['groups'].append(new)
 for ci,old in enumerate(r['per_conversation']):
  new=dict(old);new['n']-=1;new['base']-=y[ci].square().sum().item();new['full']-=res[ci].square().sum().item();new['group_shapley']=[z-boundary_sh[ci,gi] for gi,z in enumerate(old['group_shapley'])];new['group_ab']=[z-boundary_ab[gi][ci].item() for gi,z in enumerate(old['group_ab'])];out['per_conversation'].append(new)
 # Conversation bootstrap, resampling whole conversations rather than correlated tokens.
 rng=np.random.default_rng(42);draw=rng.integers(0,b,(2000,b));B=np.array([q['base']-q['full'] for q in out['per_conversation']]);S=np.array([q['group_shapley'] for q in out['per_conversation']]);shares=S[draw].sum(1)/B[draw].sum(1)[:,None];out['group_share_bootstrap95']={str(rank):np.quantile(shares[:,gi],[.025,.975]).tolist() for gi,rank in enumerate(rv)}
 out['l0']=sum(f['frequency'] for f in out['features']);out['active_rank_per_token']=sum(g['active_rank'] for g in out['groups']);out['boundary_active_features']=boundary_features
 (O/'reconstruction_interior.json').write_text(json.dumps(out,indent=2,default=float));print({k:v for k,v in out.items() if k not in ['features','per_conversation','boundary_active_features']},flush=True)
main()
