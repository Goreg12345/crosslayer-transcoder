"""Audited scoring-only endpoint overrides; explainer configuration is untouched."""
import asyncio
import hashlib
import json
from pathlib import Path
from urllib.request import Request,urlopen


def load_scorer_runtime(output):
    path=Path(output)/'scorer-runtime.json'
    if not path.exists():return None
    raw=path.read_bytes();data=json.loads(raw)
    required=('base_url','model','context_limit')
    if any(k not in data for k in required):raise ValueError('Scorer runtime requires base_url, model, context_limit')
    if not data['base_url'].startswith(('http://','https://')):raise ValueError('Invalid scorer endpoint')
    if type(data['context_limit']) is not int or data['context_limit']<64:raise ValueError('Invalid scorer context limit')
    cap=data.get('intruder_max_tokens',8192)
    if type(cap) is not int or not 1<=cap<data['context_limit']:raise ValueError('Invalid scorer output cap')
    effort=data.get('intruder_reasoning_effort','xhigh')
    if effort not in ('low','medium','high','xhigh'):raise ValueError('Scorer reasoning effort must preserve thinking')
    return dict(data,base_url=data['base_url'].rstrip('/'),intruder_max_tokens=cap,intruder_reasoning_effort=effort,
                runtime_sha256=hashlib.sha256(raw).hexdigest())


def count_prompt_tokens(runtime,body):
    probe=dict(body,max_tokens=1)
    request=Request(runtime['base_url']+'/chat/completions/render',data=json.dumps(probe).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=120) as response:result=json.load(response)
    ids=result.get('token_ids')
    if not isinstance(ids,list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError('Scorer render endpoint did not return token IDs')
    if len(ids)>=runtime['context_limit']:raise ValueError('Scorer prompt leaves no output capacity')
    return len(ids)


def bounded_budgets(runtime,prompt_tokens,thinking,requested=32):
    remaining=runtime['context_limit']-prompt_tokens
    if remaining<1:raise ValueError('Scorer prompt exhausts context')
    if not thinking:
        if requested>remaining:raise ValueError('Scorer prompt leaves insufficient room for fixed binary output budget')
        return [requested]
    first=min(runtime['intruder_max_tokens'],remaining)
    return [first] if first==remaining else [first,remaining]


def training_endpoint(endpoint_class,runtime,concurrency=16):
    class BoundedTrainingEndpoint(endpoint_class):
        async def _request(self,body):
            prompt_tokens=await asyncio.to_thread(count_prompt_tokens,runtime,body)
            metadata=dict(scoring_context_limit=runtime['context_limit'],scoring_prompt_tokens_counted=prompt_tokens,
                          scoring_effective_output_budget=body['max_tokens'],scorer_runtime_sha256=runtime['runtime_sha256'])
            if body['max_tokens']==1:
                # The legacy adapter's conservative byte-bound probe needs only a
                # count; satisfy it via render, never an extra model completion.
                return {'usage':dict(prompt_tokens=prompt_tokens,completion_tokens=0,**metadata),'render_only':True}
            bounded_budgets(runtime,prompt_tokens,False,body['max_tokens'])
            result=await super()._request(body)
            result.setdefault('usage',{}).update(metadata)
            return result
    return BoundedTrainingEndpoint(runtime['model'],runtime['base_url'],'local-not-required',concurrency,
                                   context_limit=runtime['context_limit'])
