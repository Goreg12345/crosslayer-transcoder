"""Rank/feature reconstruction importance on a reproducible held-out token sample."""
import gc,json,time,os
from pathlib import Path
import numpy as np,torch
from datasets import Dataset
from transformers import AutoModel,AutoTokenizer
from crosslayer_transcoder.dashboard.gates import MoltGates
O=Path(os.getenv('MOLT_INV_OUTPUT','results/transform29-followup'));O.mkdir(parents=True,exist_ok=True);limit=int(os.getenv('MOLT_INV_LENGTH','512'));prefix=os.getenv('MOLT_INV_BOS','0')=='1';torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');device='cuda:0'
meta=json.load(open('results/molt-qwen-dashboard/metadata.json'));layer=meta['layer'];ck=torch.load(meta['checkpoint'],mmap=True,map_location='cpu',weights_only=False);state=ck['state_dict'];gates=MoltGates(state,layer).to(device).eval()
@torch.inference_mode()
def main():
 if not (O/'xy.pt').exists():
  t=AutoTokenizer.from_pretrained(meta['model']);model=AutoModel.from_pretrained(meta['model'],torch_dtype=torch.bfloat16,attn_implementation='sdpa');model.layers=torch.nn.ModuleList(list(model.layers)[:layer+1]);model=model.to(device).eval().requires_grad_(False)
  ds=Dataset.from_file('/home/georg/.cache/huggingface/datasets/HuggingFaceH4___ultrachat_200k/default/0.0.0/8049631c405ae6576f93f445c6b8166f76f5505a/ultrachat_200k-test_sft.arrow');indices=np.random.default_rng(20260908).choice(np.arange(192,len(ds)),64,replace=False);xs=[];ys=[];ts=[];offsets=[0];captured={}
  class Done(Exception):pass
  def pre(_m,args):captured['x']=args[0][0].cpu()
  def post(_m,args,out):captured['y']=out[0].cpu();raise Done()
  handles=[model.layers[layer].post_attention_layernorm.register_forward_pre_hook(pre),model.layers[layer].mlp.register_forward_hook(post)]
  for k,i in enumerate(indices):
   original=t.apply_chat_template(ds[int(i)]['messages'],tokenize=True,add_generation_prompt=False);ids=([model.config.bos_token_id]+original[:limit-1])[:min(limit,len(original))] if prefix else original[:limit];batch=torch.tensor([ids],device=device)
   try:model(input_ids=batch,attention_mask=torch.ones_like(batch),use_cache=False)
   except Done:pass
   xs.append(captured.pop('x'));ys.append(captured.pop('y'));ts.extend(ids);offsets.append(len(ts))
   if k%8==7:print('EXTRACT',k+1,len(ts),flush=True)
  for h in handles:h.remove()
  torch.save(dict(x=torch.cat(xs),y=torch.cat(ys),tokens=ts,offsets=offsets,conversations=indices.tolist()),O/'xy.pt');del model,xs,ys,batch;gc.collect();torch.cuda.empty_cache()
 data=torch.load(O/'xy.pt',weights_only=False);x=data['x'];y=data['y'];n,d=x.shape
 ranks=np.concatenate([np.full(state[f'model.Us.{i}'].shape[0],state[f'model.Us.{i}'].shape[1]) for i in range(5)]);nf=len(ranks);rankvals=[512,256,128,64,32]
 totals=dict(n=n,d=d,checkpoint=meta['checkpoint'],step=meta['global_step'],sample_conversations=data['conversations'],max_length=limit,training_bos_prefix=prefix,rank_counts={r:int((ranks==r).sum()) for r in rankvals},feature_count=nf)
 counts=np.zeros(nf);energy=np.zeros(nf);ablation=np.zeros(nf);shapley=np.zeros(nf);group_ab=np.zeros(5);group_only=np.zeros(5);cumulative=np.zeros(5);base=full=rawbase=rawfull=0.;active_rank=0.;rows=[];geometry={};perconv=[]
 weights=[(state[f'model.Vs.{i}'].to(device).float(),state[f'model.Us.{i}'].to(device).float()) for i in range(5)];starts=np.cumsum([0]+[len(w[0]) for w in weights]);outmean=state['model.output_standardizer.mean'][layer].to(device).float();outstd=state['model.output_standardizer.std'][layer].to(device).float()
 # Batches end at conversation boundaries for conversation-level uncertainty.
 for ci,(start,end) in enumerate(zip(data['offsets'][:-1],data['offsets'][1:])):
  xx=x[start:end].to(device);yn=(y[start:end].to(device).float()-outmean)/outstd;xn=gates.standardizer(xx,layer).float()
  with torch.autocast('cuda',dtype=torch.bfloat16):gg=gates(xx)
  b=len(xx);pred=torch.zeros((b,d),device=device);group=torch.zeros((5,b,d),device=device);contributions=[]
  for gi,(vv,uu) in enumerate(weights):
   for f in torch.where((gg[:,starts[gi]:starts[gi+1]]>0).any(0))[0].tolist():
    fid=int(starts[gi]+f);pos=torch.where(gg[:,fid]>0)[0];c=((xn[pos]@vv[f])@uu[f])*gg[pos,fid,None].float();pred.index_add_(0,pos,c);group[gi].index_add_(0,pos,c);contributions.append((fid,pos,c))
    if fid==29 and len(geometry.get('rows',[]))<8:geometry.setdefault('rows',[]).append(c[:128].cpu())
  if ci==0:
   dense=torch.zeros_like(pred[:2])
   for gi,(vv,uu) in enumerate(weights):
    cc=torch.einsum('bd,fdr->bfr',xn[:2],vv);cc=torch.einsum('bfr,frd->bfd',cc,uu);dense+=(cc*gg[:2,starts[gi]:starts[gi+1],None].float()).sum(1)
   totals['dense_sparse_max_abs_error']=(dense-pred[:2]).abs().max().item();assert torch.allclose(dense,pred[:2],rtol=1e-4,atol=1e-4)
  residual=pred-yn;bs=yn.square().sum().item();fs=residual.square().sum().item();base+=bs;full+=fs;rawbase+=(yn*outstd).square().sum().item();rawfull+=(residual*outstd).square().sum().item();active_rank+=(gg.gt(0).sum(0).cpu().numpy()*ranks).sum();record=dict(n=b,base=bs,full=fs,group_ab=[],group_shapley=[0.]*5)
  for fid,pos,c in contributions:
   en=c.square().sum().item();ab=en-2*(residual[pos]*c).sum().item();sh=(c*(2*yn[pos]-pred[pos])).sum().item();counts[fid]+=len(pos);energy[fid]+=en;ablation[fid]+=ab;shapley[fid]+=sh;record['group_shapley'][int(np.where(starts<=fid)[0][-1])]+=sh
  for gi in range(5):
   ab=(residual-group[gi]).square().sum().item()-fs;group_ab[gi]+=ab;group_only[gi]+=(group[gi]-yn).square().sum().item();cumulative[gi]+=(group[:gi+1].sum(0)-yn).square().sum().item();record['group_ab'].append(ab)
  perconv.append(record)
  if ci%8==7:print('RECON',ci+1,'MSE',full/sum(q['n'] for q in perconv)/d,flush=True)
 denom=n*d;benefit=base-full
 totals.update(standardized_mse=full/denom,mean_baseline_mse=base/denom,raw_mse=rawfull/denom,raw_mean_baseline_mse=rawbase/denom,explained_fraction=benefit/base,l0=counts.sum()/n,active_rank_per_token=active_rank/n,shapley_sum_check=float(shapley.sum()-benefit),per_conversation=perconv)
 totals['groups']=[dict(rank=r,n_features=int((ranks==r).sum()),parameter_share=.2,frequency_sum=float(counts[ranks==r].sum()/n),active_rank=float(counts[ranks==r].sum()*r/n),shapley_share=float(shapley[ranks==r].sum()/benefit),ablation_mse_increase=float(group_ab[i]/denom),only_group_mse=float(group_only[i]/denom),cumulative_largest_first_mse=float(cumulative[i]/denom)) for i,r in enumerate(rankvals)]
 totals['features']=[dict(id=f,rank=int(ranks[f]),frequency=float(counts[f]/n),active_count=int(counts[f]),ablation_mse_increase=float(ablation[f]/denom),additive_benefit_share=float(shapley[f]/benefit),output_energy=float(energy[f]/denom)) for f in range(nf)]
 if geometry:
  z=torch.cat(geometry['rows']).to(device);z=z[:1024];sv=torch.linalg.svdvals(z);ev=sv.square();norm=torch.nn.functional.normalize(z,dim=-1);cos=norm@norm.T;ix=torch.triu_indices(len(z),len(z),1,device=device);cos=cos[ix[0],ix[1]];totals['transform29_output_geometry']=dict(n=len(z),first_component_energy=float(ev[0]/ev.sum()),dimensions_90_percent=int(((ev.cumsum(0)/ev.sum())<.9).sum()+1),mean_pairwise_cosine=float(cos.mean()))
 (O/'reconstruction.json').write_text(json.dumps(totals,indent=2));print('DONE',json.dumps({k:v for k,v in totals.items() if k not in ['features','per_conversation']}),flush=True)
main()
